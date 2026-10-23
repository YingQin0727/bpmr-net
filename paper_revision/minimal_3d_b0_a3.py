"""Run the frozen minimal WF2/GI Design A B0/A3 3-D comparison."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "revision_mtcomm_2026"
DATA = BASE / "REVISION_REPLICATION_P1_dataset"
TRAINING = BASE / "REVISION_REPLICATION_training" / "DesignA"
B0_MANIFEST = TRAINING / "REVISION_DESIGNA_B0_TRAIN_R0_CHECKPOINT_MANIFEST.json"
A0A5_MANIFEST = TRAINING / "REVISION_DESIGNA_A0A5_R0_CHECKPOINT_MANIFEST.json"
PLANE_REGISTRY = BASE / "R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv"
DATASET_AUDIT = BASE / "R1_RECONSTITUTION_P1_DATASET_AUDIT.json"
# The interrupted pre-fix run is preserved under MINIMAL_B0_A3_R0.  Formal
# trajectories are written into a separate namespace so they cannot be
# confused with that invalid attempt.
OUT = BASE / "REVISION_REPLICATION_3D" / "MINIMAL_B0_A3_R0" / "FORMAL_RUN_R1"
HIST = Path(r"C:\Users\admin\Documents\博士申请论文")
EXTERNAL = (HIST / "r2_lite_3d_pilot.py", HIST / "step65b_condition_cpdf_predictor.py", HIST / "fu_style_cpdf_runtime.py", HIST / ".remote_src" / "fu_style_senn_core.py")
MATERIALS = ("WF2", "GI")
ORIENTATIONS = ("x", "y", "z")
TEMPLATES = (0, 1, 2, 3)
PHASES = (0, 1, 2, 3)
SHAPE = (48, 48, 48)
ROI = 0
GLOBAL_FIT_UM = 250
LOCAL_OBSERVATION_UM = 25
TRAINING_SEED = 1701
ROLLOUT_SEED = 1701
SWEEPS = (0, 3, 5, 10)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(str(arr.dtype).encode()); h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_path(relative: str) -> Path:
    value = Path(relative)
    if value.parts and value.parts[0].lower() == DATA.name.lower():
        return BASE / value
    return DATA / value


def extract(path: Path, name: str, namespace: dict[str, Any]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def pure_predictor_functions():
    ns: dict[str, Any] = {"np": np, "torch": torch, "Any": Any, "Iterable": Any, "PROJECT_LABEL_COUNT": 4, "PHASES": PHASES, "ORIENTATIONS": ORIENTATIONS, "_AXIS": {"x": (1, 2), "y": (2, 0), "z": (0, 1)}, "full_square_neighbour_offsets": lambda r: [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if (a, b) != (0, 0)]}
    ns["normalize_categorical_probabilities"] = extract(EXTERNAL[3], "normalize_categorical_probabilities", ns)
    template = extract(EXTERNAL[2], "integrate_exactly_four_phase_template_outputs", ns)
    orientation = extract(EXTERNAL[2], "integrate_exactly_three_orientation_cpdfs", ns)
    encode = extract(EXTERNAL[1], "encode_periodic_3d_neighbourhood", ns)
    return encode, (template, orientation)


def manifests():
    b0, a0a5 = load_json(B0_MANIFEST), load_json(A0A5_MANIFEST)
    if len(b0["checkpoints"]) != 120 or len(a0a5["checkpoints"]) != 720:
        raise RuntimeError("FROZEN_MODEL_MANIFEST_COUNT_MISMATCH")
    return b0, a0a5, load_json(DATASET_AUDIT)


def load_bank(material: str, variant: str, b0: dict[str, Any], a0a5: dict[str, Any]):
    rows = b0["checkpoints"] if variant == "B0_SOURCE" else [r for r in a0a5["checkpoints"] if r["variant"] == "A3"]
    bank: dict[str, dict[int, torch.nn.Module]] = {axis: {} for axis in ORIENTATIONS}; bindings = []
    for axis in ORIENTATIONS:
        for template in TEMPLATES:
            unit = f"{material}/G{GLOBAL_FIT_UM}/{axis}/seed{TRAINING_SEED}/template{template}"
            found = [r for r in rows if r["unit"] == unit]
            if len(found) != 1:
                raise RuntimeError(f"CHECKPOINT_BINDING_NOT_UNIQUE:{unit}:{variant}")
            row = found[0]; rel = row.get("path") or row.get("checkpoint_path")
            path = ROOT / rel
            expected = row.get("sha256") or row.get("checkpoint_sha256")
            actual = sha_file(path)
            if not path.is_file() or actual != expected:
                raise RuntimeError(f"CHECKPOINT_SHA_MISMATCH:{path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["metadata"].get("unit") != unit or payload["metadata"].get("kind") != "EPOCH50":
                raise RuntimeError(f"CHECKPOINT_METADATA_MISMATCH:{unit}")
            model = B0Network() if variant == "B0_SOURCE" else build_revision_bpmr(B0Network(), "A3")
            model.load_state_dict(payload["state"], strict=True); model.eval(); bank[axis][template] = model
            source = actual
            if variant == "A3":
                source = row["source_checkpoint_sha256"]
                if not any(r["unit"] == unit and r["sha256"] == source for r in b0["checkpoints"]):
                    raise RuntimeError(f"A3_SOURCE_B0_MISMATCH:{unit}")
            bindings.append({"unit": unit, "variant": variant, "orientation": axis, "template_phase": template, "path": str(path), "sha256": actual, "source_b0_sha256": source})
    return bank, bindings


def train_prior(material: str, rows: list[dict[str, str]]) -> list[float]:
    counts = np.zeros(4, dtype=np.int64)
    for row in rows:
        if row["material"] != material or row["role"] not in {"train", "validation"} or "25" not in row["regimes_um"].split(";"):
            continue
        lp, sp = payload_path(row["labels_path"]), payload_path(row["support_path"])
        if sha_file(lp) != row["labels_sha256"] or sha_file(sp) != row["support_sha256"]:
            raise RuntimeError(f"PLANE_INPUT_SHA_MISMATCH:{lp}")
        shape = tuple(json.loads(row["shape"])); labels = np.fromfile(lp, dtype=np.uint8).reshape(shape); support = np.fromfile(sp, dtype=np.uint8).reshape(shape).astype(bool)
        for phase in PHASES: counts[phase] += int(np.count_nonzero(support & (labels == phase)))
    if counts.sum() == 0 or np.any(counts == 0): raise RuntimeError(f"TRAIN_PRIOR_INVALID:{material}:{counts.tolist()}")
    return (counts / counts.sum()).tolist()


def observation(material: str, audit: dict[str, Any]):
    roi = next(x for x in audit["rois"] if x["material"] == material and int(x["roi"]) == ROI); item = roi["observations"][str(LOCAL_OBSERVATION_UM)]; path = payload_path(item["path"])
    if sha_file(path) != item["sha256"]: raise RuntimeError(f"OBSERVATION_SHA_MISMATCH:{path}")
    raw = np.fromfile(path, dtype=np.uint8).reshape(SHAPE); labels = np.transpose(np.where(raw == 255, -1, raw).astype(np.int16), (2, 1, 0)); valid = np.transpose(raw != 255, (2, 1, 0))
    return labels, valid, {"path": str(path), "file_sha256": item["sha256"], "mask_sha256": item["mask_sha256"], "observed_count": int(valid.sum()), "updates_per_sweep": int((~valid).sum()), "origin_zyx": roi["origin_zyx"], "shape_zyx": roi["shape_zyx"]}


def initialize(prior, labels, valid):
    values = np.random.default_rng(ROLLOUT_SEED).normal(size=SHAPE); thresholds = np.quantile(values, np.cumsum(np.asarray(prior))[:-1]); state = np.digitize(values, thresholds).astype(np.int16); state[valid] = labels[valid]; return state


def save_state(path: Path, state: np.ndarray, rng: np.random.Generator, meta: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp"); payload = dict(meta); payload["rng_state"] = rng.bit_generator.state
    with tmp.open("wb") as f: np.savez_compressed(f, state=state, metadata_json=np.asarray(json.dumps(payload, sort_keys=True))); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def components(state):
    structure = ndimage.generate_binary_structure(3, 1); out = {}
    for phase in PHASES:
        labelled, count = ndimage.label(state == phase, structure=structure); sizes = np.bincount(labelled.ravel())[1:]; out[str(phase)] = {"components_6_connected": int(count), "largest_component": int(sizes.max()) if len(sizes) else 0}
    return out


def metrics(state, truth):
    fractions = [float(np.mean(state == p)) for p in PHASES]; truth_f = [float(np.mean(truth == p)) for p in PHASES]
    return {"phase_fractions": fractions, "truth_phase_fractions": truth_f, "phase_fraction_error": [a - b for a, b in zip(fractions, truth_f)], "extinct_phases": [p for p, x in zip(PHASES, fractions) if x == 0], "components": components(state), "slice_sha256": {"XY_z24": sha_array(state[:, :, 24]), "XZ_y24": sha_array(state[:, 24, :]), "YZ_x24": sha_array(state[24, :, :])}}


def trajectory(material, variant, bank, binding, labels, valid, prior, encode, fusion):
    template, orientation = fusion
    def probability(state, index):
        axis_values = []
        with torch.no_grad():
            for axis in ORIENTATIONS:
                feature = encode(state, index, axis, 4); tensor = torch.from_numpy(np.ascontiguousarray(feature)).unsqueeze(0); outputs = []
                for phase in PHASES:
                    raw = bank[axis][phase](tensor); p = raw[1] if isinstance(raw, tuple) else raw; p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=1e-12); outputs.append(p.cpu().numpy()[0])
                axis_values.append(template(outputs))
        p = orientation(axis_values)
        if p.shape != (4,) or not np.all(np.isfinite(p)) or not np.isclose(float(p.sum()), 1.0, atol=1e-8): raise RuntimeError("INVALID_CONDITIONAL_PROBABILITY")
        return p
    state = initialize(prior, labels, valid); initial = state.copy(); rng = np.random.default_rng(ROLLOUT_SEED); run_dir = OUT / material / variant; started = time.perf_counter(); peak = psutil.Process().memory_info().rss; state_files = []
    save_state(run_dir / "sweep_0.npz", state, rng, {"material": material, "variant": variant, "sweep": 0, "global_fit_um": GLOBAL_FIT_UM, "local_observation_um": LOCAL_OBSERVATION_UM, "observed_mask_sha256": sha_array(valid), "state_sha256": sha_array(state)})
    state_files.append({"sweep": 0, "path": str(run_dir / "sweep_0.npz"), "sha256": sha_file(run_dir / "sweep_0.npz")})
    for sweep in range(1, max(SWEEPS) + 1):
        for index in np.ndindex(SHAPE):
            if not valid[index]: state[index] = int(rng.choice(4, p=probability(state, index)))
        if np.any(state[valid] != labels[valid]): raise RuntimeError(f"HARD_CLAMP_VIOLATION:{material}:{variant}:{sweep}")
        peak = max(peak, psutil.Process().memory_info().rss)
        if sweep in SWEEPS:
            save_state(run_dir / f"sweep_{sweep}.npz", state, rng, {"material": material, "variant": variant, "sweep": sweep, "global_fit_um": GLOBAL_FIT_UM, "local_observation_um": LOCAL_OBSERVATION_UM, "observed_mask_sha256": sha_array(valid), "state_sha256": sha_array(state)})
            for suffix, array in (("XY_z24", state[:, :, 24]), ("XZ_y24", state[:, 24, :]), ("YZ_x24", state[24, :, :])): np.save(run_dir / f"sweep_{sweep}_{suffix}.npy", array)
            state_files.append({"sweep": sweep, "path": str(run_dir / f"sweep_{sweep}.npz"), "sha256": sha_file(run_dir / f"sweep_{sweep}.npz")})
    result = {"material": material, "variant": variant, "roi": ROI, "global_fit_um": GLOBAL_FIT_UM, "local_observation_um": LOCAL_OBSERVATION_UM, "clamp_mode": "hard_observed", "training_seed": TRAINING_SEED, "rollout_seed": ROLLOUT_SEED, "sweeps_saved": list(SWEEPS), "observation_mask_sha256": sha_array(valid), "initial_state_sha256": sha_array(initial), "final_state_sha256": sha_array(state), "checkpoint_bindings": binding, "state_files": state_files, "elapsed_wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak, "truth_read": False}
    run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "trajectory.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"); return result


def evaluate_frozen_trajectories(summary: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Read test truth only after all four reconstruction trajectories are frozen."""
    truth_records = {}
    for material in MATERIALS:
        truth_path = DATA / f"test_truth/{material}_ROI{ROI}.truth.bin"
        if not truth_path.is_file():
            raise RuntimeError(f"MISSING_TRUTH_FOR_EVALUATION:{truth_path}")
        truth = np.transpose(np.fromfile(truth_path, dtype=np.uint8).reshape(SHAPE).astype(np.int16), (2, 1, 0)).copy()
        if np.any((truth < 0) | (truth > 3)):
            raise RuntimeError(f"TRUTH_LABEL_RANGE:{truth_path}")
        truth_records[material] = {"path": str(truth_path), "sha256": sha_file(truth_path), "array": truth}
    for result in summary["results"]:
        truth_record = truth_records[result["material"]]
        metrics_by_sweep = {}
        for item in result["state_files"]:
            state_path = Path(item["path"])
            if sha_file(state_path) != item["sha256"]:
                raise RuntimeError(f"STATE_SHA_MISMATCH:{state_path}")
            with np.load(state_path, allow_pickle=False) as archive:
                state = archive["state"].astype(np.int16, copy=False)
            if state.shape != SHAPE or np.any((state < 0) | (state > 3)):
                raise RuntimeError(f"INVALID_SAVED_STATE:{state_path}")
            metrics_by_sweep[str(item["sweep"])] = metrics(state, truth_record["array"])
        result["metrics"] = metrics_by_sweep
        result["truth_file_sha256"] = truth_record["sha256"]
        result["truth_read"] = True
    summary["truth_read_after_reconstruction"] = True
    summary["truth_files"] = {m: {"path": v["path"], "sha256": v["sha256"]} for m, v in truth_records.items()}
    return summary


def main():
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    b0, a0a5, audit = manifests(); registry = list(csv.DictReader(PLANE_REGISTRY.open(encoding="utf-8-sig"))); encode, fusion = pure_predictor_functions(); summary = {"stage": "MINIMAL_3D_B0_A3_COMPARISON", "scientific_evidence": True, "selection_rule": "ROI0 and first matrix condition; all registry-ordered x/y/z x template0..3 checkpoints; no outcome selection", "global_fit_um": GLOBAL_FIT_UM, "local_observation_um": LOCAL_OBSERVATION_UM, "results": [], "formal_trajectory_count": 0, "truth_read_after_reconstruction": False}
    for material in MATERIALS:
        labels, valid, obs_meta = observation(material, audit); prior = train_prior(material, registry)
        for variant in ("B0_SOURCE", "A3"):
            bank, binding = load_bank(material, variant, b0, a0a5); result = trajectory(material, variant, bank, binding, labels, valid, prior, encode, fusion); result["observation_metadata"] = obs_meta; result["train_prior"] = prior; summary["results"].append(result); summary["formal_trajectory_count"] += 1
    OUT.mkdir(parents=True, exist_ok=True); summary["status"] = "RECONSTRUCTION_FROZEN" if summary["formal_trajectory_count"] == 4 else "PARTIAL"; summary["reconstruction_summary_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); (OUT / "MINIMAL_3D_B0_A3_RECONSTRUCTION_LOCK.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary = evaluate_frozen_trajectories(summary, audit); summary["status"] = "PASS" if summary["formal_trajectory_count"] == 4 else "PARTIAL"; summary["summary_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); (OUT / "MINIMAL_3D_B0_A3_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "MINIMAL_3D_B0_A3_METRICS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["material", "variant", "sweep", "phase_fractions", "truth_phase_fractions", "phase_fraction_error", "extinct_phases", "components"])
        for result in summary["results"]:
            for sweep, m in result["metrics"].items(): w.writerow([result["material"], result["variant"], sweep, json.dumps(m["phase_fractions"]), json.dumps(m["truth_phase_fractions"]), json.dumps(m["phase_fraction_error"]), json.dumps(m["extinct_phases"]), json.dumps(m["components"], sort_keys=True)])
    print(json.dumps({"status": summary["status"], "formal_trajectory_count": summary["formal_trajectory_count"], "output": str(OUT)}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
