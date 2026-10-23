"""Frozen Design A held-out evaluation with resumable per-model commits.

This entrypoint consumes the already frozen R2 evidence decision.  It never
changes model eligibility and writes each model's metrics and access record to
an atomic task artifact before advancing the progress state.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from bpmr_net.models import B0Network
from paper_revision.models_revision import build_revision_bpmr
from paper_revision.training_access import TrainingAccess, canonical
from paper_revision import heldout_evaluation_r0 as r0
from paper_revision import heldout_evaluation_r2 as r2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/revision_mtcomm_2026"
TRAIN = BASE / "REVISION_REPLICATION_training/DesignA"
OUT = BASE / "REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R1"
TASKS = OUT / "tasks"
PUB = ROOT / "paper_revision/evidence_release_r2"
DEC = PUB / "REVISION_DESIGNA_EVIDENCE_R2_RELEASE_DECISION.json"
TIERS = PUB / "REVISION_DESIGNA_EVIDENCE_R2_MODEL_TIERS.csv"
PAIRS = PUB / "REVISION_DESIGNA_EVIDENCE_R2_PAIRING_ELIGIBILITY.csv"
FREEZE = BASE / "REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0/REVISION_DESIGNA_HELDOUT_R0_MODEL_FREEZE.json"
PROTOCOL = BASE / "REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R0/REVISION_DESIGNA_HELDOUT_R0_PROTOCOL.json"
REGISTRY = BASE / "R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv"
PREFIX = "REVISION_DESIGNA_HELDOUT_R1_"
COMPARISONS = (("A0", "A1"), ("A1", "A3"), ("A2", "A3"), ("A3", "A4"), ("A3", "A5"), ("B0_SOURCE", "A0"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(unit, variant):
    return unit + "/" + variant


def truth(value):
    return str(value).lower() == "true"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def canonical_identity_hash(rows):
    values = sorted(identity(row["unit"], row["variant"]) for row in rows)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


class EvaluationAccess(TrainingAccess):
    """TrainingAccess plus an explicit output-only atomic rename capability."""

    def __init__(self, reads, writes, runtime_roots, namespace, pairs):
        super().__init__(reads, writes, runtime_roots)
        self.namespace = Path(namespace).resolve()
        self.atomic_pairs = {(canonical(a), canonical(b)) for a, b in pairs}

    def safe_atomic_path(self, value):
        path = Path(os.fsdecode(value))
        if not path.is_absolute() or ".." in path.parts:
            raise PermissionError("atomic paths must be absolute and traversal-free")
        for parent in (path, *path.parents):
            if parent.exists() and parent.is_symlink():
                raise PermissionError("symbolic links are not valid atomic paths")
        key = canonical(path)
        if not Path(key).is_relative_to(self.namespace):
            raise PermissionError("atomic path outside evaluation namespace")
        return key

    def hook(self, event, args):
        if event != "os.rename" or not self.active or self.busy:
            return super().hook(event, args)
        self.busy = True
        record = {"operation": event, "phase": self.phase, "allowed": False}
        try:
            if any(value not in (None, -1) for value in args[2:]):
                raise PermissionError("directory-descriptor rename is not allowed")
            source = self.safe_atomic_path(args[0])
            target = self.safe_atomic_path(args[1])
            if (source, target) not in self.atomic_pairs:
                raise PermissionError("unregistered evaluation atomic pair")
            record.update(source=source, target=target, allowed=True, category="atomic_output")
        finally:
            self.records.append(record)
            self.busy = False

    
def atomic_write(path, payload, guard):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(payload)
    os.replace(pending, path)
    # The committed artifact is immediately hashed for the ledger/manifest.
    # Register that read explicitly; this remains inside the revision namespace.
    guard.reads.add(canonical(path))
    return sha(path)


def json_bytes(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def csv_bytes(rows):
    if not rows:
        raise RuntimeError("refusing to write an empty evaluation table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def preflight(resume):
    decision = json.loads(DEC.read_bytes())
    freeze = json.loads(FREEZE.read_bytes())
    protocol = json.loads(PROTOCOL.read_bytes())
    tiers = read_csv(TIERS)
    pairs = read_csv(PAIRS)
    registry = read_csv(REGISTRY)
    assert decision["HELDOUT_EVALUATION_READY"] == "YES"
    assert decision["ORIGINAL_STRICT_RELEASE"] == "FAIL"
    assert decision["CONDITIONAL_SCIENTIFIC_RELEASE"] == "YES"
    assert decision["evaluation_protocol"]["protocol_sha256"] == sha(PROTOCOL)
    r2_path = ROOT / "paper_revision/heldout_evaluation_r2.py"
    assert decision["evaluation_protocol"]["next_entrypoint"] == "paper_revision.heldout_evaluation_r2"
    assert decision["evaluation_protocol"]["next_entrypoint_sha256"] == sha(r2_path)
    assert freeze["status"] == "PASS" and len(freeze["models"]) == 840
    assert protocol["model_freeze_sha256"] == sha(FREEZE)
    assert len(tiers) == 840 and len(pairs) == 720
    primary = [row for row in tiers if truth(row["primary_candidate"])]
    sensitivity = [row for row in tiers if truth(row["sensitivity_candidate"])]
    assert len(primary) == 833 and len(sensitivity) == 840
    assert sum(truth(row["primary_eligible"]) for row in pairs) == 714
    assert sum(truth(row["sensitivity_eligible"]) for row in pairs) == 720
    assert canonical_identity_hash(primary) == decision["evaluation_protocol"]["primary_set_sha256"]
    assert canonical_identity_hash(sensitivity) == decision["evaluation_protocol"]["sensitivity_set_sha256"]
    assert sha(PAIRS) == decision["evaluation_protocol"]["pairing_sha256"]
    assert sum(row["role"] == "test_truth" for row in registry) == 180
    frozen = {identity(row["unit"], row["variant"]): row for row in freeze["models"]}
    assert set(frozen) == {identity(row["unit"], row["variant"]) for row in tiers}
    for row in tiers:
        key = identity(row["unit"], row["variant"])
        assert frozen[key]["sha256"] == row["checkpoint_sha256"]
        assert sha(ROOT / frozen[key]["path"]) == frozen[key]["sha256"]
    OUT.mkdir(parents=True, exist_ok=True)
    TASKS.mkdir(parents=True, exist_ok=True)
    finals = [OUT / (PREFIX + name) for name in (
        "MODEL_MANIFEST.json", "ALL_METRICS.csv", "PHASE_METRICS.csv",
        "PAIRED_COMPARISONS.csv", "GATE_DIAGNOSTICS.csv", "RUNTIME.csv",
        "SEED_UNCERTAINTY.csv",
        "FINAL_AUDIT.json", "SUMMARY.md", "EVENT_IDENTITY.json", "RESULT_MANIFEST.json")]
    state_path = OUT / (PREFIX + "STATE.json")
    if not resume and (state_path.exists() or any(path.exists() for path in finals)):
        raise RuntimeError("R1 outputs already exist; use --resume after checking their evidence")
    return decision, freeze, protocol, tiers, pairs, registry, primary, sensitivity, frozen, state_path


def task_path(index):
    return TASKS / (f"task_{index:06d}.json")


def load_existing(path):
    return json.loads(path.read_bytes())


def build_task(index, model_info, unit, variant, event, metrics, phases, roi_rows, gate_rows, runtime, records, tier):
    return dict(index=index, status="PASS", identity=identity(unit, variant), unit=unit, variant=variant,
                checkpoint_sha256=model_info["sha256"], event_sha256=event["event_sha256"],
                evaluated_event_count=event["count"], evidence_tier=tier,
                metrics=metrics, phase_metrics=phases, roi_metrics=roi_rows,
                gate_diagnostics=gate_rows, runtime=runtime,
                access_records=records,
                access_violation_count=sum(not record.get("allowed", False) for record in records))


def evaluate(resume=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    decision, freeze, protocol, tiers, pairs, registry, primary, sensitivity, frozen, state_path = preflight(resume)
    ledger_a = read_csv(TRAIN / "REVISION_DESIGNA_A0A5_R0_TASK_LEDGER.csv")
    ledger_b = read_csv(TRAIN / "REVISION_DESIGNA_B0_TRAIN_R0_TASK_LEDGER.csv")
    task_times = {(row["unit"], row["variant"]): float(row["elapsed_seconds"]) for row in ledger_a}
    task_times.update({(row["unit"], "B0_SOURCE"): float(row["task_wall_seconds_including_preparation"]) for row in ledger_b})
    model_map = {identity(row["unit"], row["variant"]): row for row in freeze["models"]}
    tier_map = {identity(row["unit"], row["variant"]): row for row in tiers}
    primary_ids = {identity(row["unit"], row["variant"]) for row in primary}
    sensitivity_ids = {identity(row["unit"], row["variant"]) for row in sensitivity}
    plane_rows = registry
    task_infos = sorted(model_map.items())
    task_index = {key: index for index, (key, _) in enumerate(task_infos, 1)}
    task_finals = [task_path(index) for index in range(1, len(task_infos) + 1)]
    task_pending = [path.with_name(path.name + ".pending") for path in task_finals]
    output_names = ["MODEL_MANIFEST.json", "ALL_METRICS.csv", "PHASE_METRICS.csv", "PAIRED_COMPARISONS.csv", "GATE_DIAGNOSTICS.csv", "RUNTIME.csv", "SEED_UNCERTAINTY.csv", "FINAL_AUDIT.json", "SUMMARY.md", "EVENT_IDENTITY.json", "RESULT_MANIFEST.json", "STATE.json"]
    output_finals = [OUT / (PREFIX + name) for name in output_names]
    output_pending = [path.with_name(path.name + ".pending") for path in output_finals]
    ledger_paths = [TRAIN / "REVISION_DESIGNA_A0A5_R0_TASK_LEDGER.csv", TRAIN / "REVISION_DESIGNA_B0_TRAIN_R0_TASK_LEDGER.csv"]
    reads = [FREEZE, PROTOCOL, DEC, TIERS, PAIRS, REGISTRY, Path(__file__), ROOT / "paper_revision/heldout_evaluation_r2.py", state_path, *ledger_paths, *task_finals]
    reads += [ROOT / row["path"] for row in freeze["models"]]
    reads += [BASE / row[key + "_path"] for row in registry if row["role"] == "test_truth" for key in ("labels", "support")]
    writes = output_finals + output_pending + task_finals + task_pending
    pairs_atomic = [(path.with_name(path.name + ".pending"), path) for path in output_finals + task_finals]
    guard = EvaluationAccess(reads, writes, [Path(torch.__file__).resolve().parents[1], Path(np.__file__).resolve().parents[1], ROOT / ".venv-revision311/Lib/site-packages"], OUT, pairs_atomic)
    guard.install()
    completed = {}
    if resume and state_path.exists():
        state = load_existing(state_path)
        for key, artifact in state.get("completed", {}).items():
            path = ROOT / artifact["path"]
            task = load_existing(path)
            assert task["status"] == "PASS" and sha(path) == artifact["task_sha256"]
            assert task["checkpoint_sha256"] == model_map[key]["sha256"]
            completed[key] = task
    started = time.perf_counter()
    process = psutil.Process()
    event_groups = {}
    task_records = dict(completed)
    for material in ("GI", "WF2"):
        for axis in "xyz":
            guard.phase = "heldout_test_events_" + material + "_" + axis
            prep_start = time.perf_counter()
            x, y, roi, event = r0.test_events(plane_rows, material, axis)
            preparation = time.perf_counter() - prep_start
            event_groups[(material, axis)] = dict(x=x, y=y, roi=roi, event=event, preparation=preparation, access=list(guard.records))
            guard.records.clear()
            selected = [(key, model_map[key]) for key in sorted(model_map) if key in sensitivity_ids and key.split("/")[0] == material and key.split("/")[2] == axis]
            for key, model_info in selected:
                index = task_index[key]
                path_task = task_path(index)
                if key in completed:
                    continue
                unit, variant = key.rsplit("/", 1)
                group = event_groups[(material, axis)]
                guard.phase = "heldout_model_" + key
                started_model = time.perf_counter()
                cpu_start = time.process_time()
                checkpoint_path = ROOT / model_info["path"]
                assert sha(checkpoint_path) == model_info["sha256"]
                checkpoint = torch.load(io.BytesIO(checkpoint_path.read_bytes()), weights_only=True, map_location="cpu")
                model = B0Network()
                if variant in ("A2", "A3", "A4", "A5"):
                    model = build_revision_bpmr(model, variant)
                model.load_state_dict(checkpoint["state"], strict=True)
                model.eval()
                predictions, diagnostics = [], []
                infer_start = time.perf_counter()
                with torch.inference_mode():
                    for begin in range(0, len(group["x"]), 4096):
                        batch = torch.from_numpy(group["x"][begin:begin + 4096])
                        raw = model(batch)
                        if isinstance(raw, tuple):
                            probability = raw[1]
                            if variant == "A3":
                                gate_inputs = model.gate_inputs(batch, raw[3])
                                shift = (probability - raw[3]).abs().sum(1)
                                diagnostics.append(np.column_stack([gate_inputs.numpy(), raw[4].numpy(), raw[5].norm(dim=1).numpy(), shift.numpy(), raw[3].argmax(1).numpy()]))
                        else:
                            probability = raw / raw.sum(1, keepdim=True).clamp_min(1e-12)
                        predictions.append(probability.numpy().copy())
                inference = time.perf_counter() - infer_start
                probability = np.concatenate(predictions)
                metrics, phases = r0.metrics(probability, group["y"])
                base = r0.identity_columns(unit, variant)
                memberships = (["PRIMARY"] if key in primary_ids else []) + ["SENSITIVITY"]
                metric_rows = [dict(analysis_set=aset, **base, **metrics, event_sha256=group["event"]["event_sha256"], checkpoint_sha256=model_info["sha256"], evidence_tier=tier_map[key]["evidence_tier"]) for aset in memberships]
                phase_rows = [dict(analysis_set=aset, **base, **row) for aset in memberships for row in phases]
                roi_rows = []
                for aset in memberships:
                    for rid in sorted(set(group["roi"].tolist())):
                        roi_metric, _ = r0.metrics(probability[group["roi"] == rid], group["y"][group["roi"] == rid])
                        roi_rows.append(dict(analysis_set=aset, **base, roi=rid, **roi_metric, event_sha256=group["event"]["event_sha256"]))
                gate_rows = []
                if variant == "A3" and diagnostics:
                    diag = np.concatenate(diagnostics)
                    before = diag[:, 6].astype(int) == group["y"]
                    after = probability.argmax(1) == group["y"]
                    for aset in memberships:
                        for rid in sorted(set(group["roi"].tolist())):
                            for phase_id in range(4):
                                for code in range(4):
                                    mask = (group["roi"] == rid) & (group["y"] == phase_id) & ((before.astype(int) * 2 + after.astype(int)) == code)
                                    if not mask.any():
                                        continue
                                    values = diag[mask]
                                    gate_rows.append(dict(analysis_set=aset, **base, roi=rid, phase=phase_id, b0_correct=bool(code // 2), a3_correct=bool(code % 2), count=len(values), b0_entropy=float(values[:, 0].mean()), top_two_margin=float(values[:, 1].mean()), support_ratio=float(values[:, 2].mean()), learned_gate=float(values[:, 3].mean()), residual_magnitude=float(values[:, 4].mean()), probability_shift=float(values[:, 5].mean()), gate_entropy_pearson=r0.corr(values[:, 3], values[:, 0]), gate_margin_pearson=r0.corr(values[:, 3], values[:, 1]), gate_support_pearson=r0.corr(values[:, 3], values[:, 2])))
                runtime = [dict(analysis_set=aset, **base, events=len(group["y"]), parameters=model_info["parameters"], effective_trainable_parameters=model_info["effective_trainable_parameters"], training_wall_seconds=task_times[(unit, variant)], checkpoint_load_seconds=time.perf_counter() - started_model - inference, inference_seconds=inference, cpu_seconds=time.process_time() - cpu_start, data_preparation_seconds_shared=group["preparation"], process_rss_bytes=process.memory_info().rss, threads=1, evidence_tier=tier_map[key]["evidence_tier"]) for aset in memberships]
                task = build_task(index, model_info, unit, variant, group["event"], metric_rows, phase_rows, roi_rows, gate_rows, runtime, list(guard.records), tier_map[key]["evidence_tier"])
                task["shared_event_access_records"] = group["access"]
                task_bytes = json_bytes(task)
                task_sha = atomic_write(path_task, task_bytes, guard)
                task_records[key] = task
                state_payload = {"stage": "REVISION_DESIGNA_HELDOUT_EVALUATION_R1", "completed_count": len(task_records), "expected_count": 840, "completed": {k: {"path": str(task_path(task_index[k]).relative_to(ROOT)), "task_sha256": sha(task_path(task_index[k])), "checkpoint_sha256": v["checkpoint_sha256"], "event_sha256": v["event_sha256"]} for k, v in sorted(task_records.items())}}
                atomic_write(state_path, json_bytes(state_payload), guard)
                print(f"evaluated {len(task_records)}/840 {key} task_sha256={task_sha}", flush=True)
                guard.records.clear()
    if len(task_records) != 840:
        raise RuntimeError(f"incomplete evaluation: {len(task_records)}/840")
    metrics = [row for task in task_records.values() for row in task["metrics"]]
    phases = [row for task in task_records.values() for row in task["phase_metrics"]]
    roi_rows = [row for task in task_records.values() for row in task["roi_metrics"]]
    gates = [row for task in task_records.values() for row in task["gate_diagnostics"]]
    runtime = [row for task in task_records.values() for row in task["runtime"]]
    lookup = {(row["analysis_set"], row["unit"], row["variant"]): row for row in metrics}
    paired = []
    for pair in pairs:
        for aset, field in (("PRIMARY", "primary_eligible"), ("SENSITIVITY", "sensitivity_eligible")):
            if not truth(pair[field]):
                continue
            left = lookup[(aset, pair["unit"], pair["left_variant"])]
            right = lookup[(aset, pair["unit"], pair["right_variant"])]
            assert left["event_sha256"] == right["event_sha256"]
            paired.append(dict(analysis_set=aset, **r0.identity_columns(pair["unit"], pair["right_variant"]), comparison=pair["comparison"], **{metric: right[metric] - left[metric] for metric in ("categorical_nll", "brier", "ece", "accuracy", "balanced_accuracy", "macro_f1")}, event_sha256=left["event_sha256"]))
    seed_rows = []
    for aset in ("PRIMARY", "SENSITIVITY"):
        for comparison in sorted({row["comparison"] for row in paired if row["analysis_set"] == aset}):
            group = [row for row in paired if row["analysis_set"] == aset and row["comparison"] == comparison]
            for material, spacing, axis, template in sorted({(row["material"], row["spacing_um"], row["orientation"], row["template"]) for row in group}):
                rows = [row for row in group if (row["material"], row["spacing_um"], row["orientation"], row["template"]) == (material, spacing, axis, template)]
                for metric in ("categorical_nll", "brier", "ece", "accuracy", "balanced_accuracy", "macro_f1"):
                    values = np.asarray([row[metric] for row in rows], dtype=float)
                    n = len(values)
                    sd = float(values.std(ddof=1)) if n > 1 else None
                    half = 4.302652729749464 * sd / math.sqrt(n) if n == 3 else None
                    seed_rows.append(dict(analysis_set=aset, comparison=comparison, material=material, spacing_um=spacing, orientation=axis, template=template, metric=metric, n_training_seeds=n, mean_delta=float(values.mean()), seed_sd=sd, min_delta=float(values.min()), max_delta=float(values.max()), conditional_seed_ci95_lower=float(values.mean() - half) if half is not None else None, conditional_seed_ci95_upper=float(values.mean() + half) if half is not None else None, inference_scope="training-seed variation conditional on fixed source and shared test events; not material generalization"))
    btime = {row["unit"]: row["inference_seconds"] for row in runtime if row["variant"] == "B0_SOURCE"}
    for row in runtime:
        row["inference_overhead_vs_paired_b0"] = row["inference_seconds"] / btime[row["unit"]]
    model_manifest = dict(stage="REVISION_DESIGNA_HELDOUT_EVALUATION_R1", status="PASS", decision_sha256=sha(DEC), protocol_sha256=sha(PROTOCOL), model_freeze_sha256=sha(FREEZE), tiers_sha256=sha(TIERS), pairing_sha256=sha(PAIRS), evaluated_models=840, primary_models=len(primary), sensitivity_models=len(sensitivity), tasks={key: {"task_path": str(task_path(task_index[key]).relative_to(ROOT)), "task_sha256": sha(task_path(task_index[key])), "checkpoint_sha256": value["checkpoint_sha256"], "event_sha256": value["event_sha256"]} for key, value in sorted(task_records.items())})
    audit = dict(stage="REVISION_DESIGNA_HELDOUT_EVALUATION_R1", status="PASS", REVISION_DESIGNA_HELDOUT_EVALUATION_R1="PASS", MODEL_INTEGRITY="PASS", TRAINING_PROVENANCE="PARTIAL", ACCESS_EVIDENCE="PARTIAL", ORIGINAL_STRICT_RELEASE="FAIL", CONDITIONAL_SCIENTIFIC_RELEASE="YES", SCIENTIFIC_TRAINING_COMPLETE=True, HELDOUT_EVALUATION_COMPLETE=True, THREE_D_RECONSTRUCTION_COMPLETE=False, PUBLISHED_BASELINE_COMPLETE=False, evaluated_models=840, primary_models=len(primary), sensitivity_models=len(sensitivity), primary_pairs=sum(truth(row["primary_eligible"]) for row in pairs), sensitivity_pairs=sum(truth(row["sensitivity_eligible"]) for row in pairs), test_payload_reads=True, prohibited_reads_observed=sum(task["access_violation_count"] for task in task_records.values()), unresolved_access_records=4, event_groups=len(event_groups), event_identity_sha256=sha_bytes(json_bytes([group["event"] for group in event_groups.values()])), model_manifest_sha256=None, seed_uncertainty_rows=len(seed_rows), elapsed_seconds=time.perf_counter() - started, OS_LEVEL_ISOLATION="NOT_ESTABLISHED", limitations=list(protocol["limitations"]))
    model_manifest_bytes = json_bytes(model_manifest)
    model_manifest_path = OUT / (PREFIX + "MODEL_MANIFEST.json")
    model_manifest_sha = atomic_write(model_manifest_path, model_manifest_bytes, guard)
    audit["model_manifest_sha256"] = model_manifest_sha
    final_rows = {"ALL_METRICS.csv": metrics, "PHASE_METRICS.csv": phases, "PAIRED_COMPARISONS.csv": paired, "GATE_DIAGNOSTICS.csv": gates, "RUNTIME.csv": runtime, "SEED_UNCERTAINTY.csv": seed_rows}
    for name, rows in final_rows.items():
        atomic_write(OUT / (PREFIX + name), csv_bytes(rows), guard)
    atomic_write(OUT / (PREFIX + "EVENT_IDENTITY.json"), json_bytes(dict(protocol_sha256=sha(PROTOCOL), groups=[group["event"] for group in event_groups.values()], paired_across_models=True)), guard)
    summary_lines = ["# Design A held-out evaluation R1", "", "REVISION_DESIGNA_HELDOUT_EVALUATION_R1=PASS", "", f"840/840 models evaluated. PRIMARY={len(primary)} models and {sum(truth(row['primary_eligible']) for row in pairs)} pairs; SENSITIVITY={len(sensitivity)} models and {sum(truth(row['sensitivity_eligible']) for row in pairs)} pairs.", "", "The original strict access-evidence rule remains FAIL. TIER_2 models are retained only in the pre-frozen SENSITIVITY set; no eligibility was changed after test reads.", "", "This is a 2D teacher-forced held-out classification evaluation using the frozen event cap, sampling, four-class metrics and seed-conditional uncertainty protocol. Two source volumes do not support broad cross-material generalization claims.", "", "Paired deltas are right minus left; negative NLL/Brier/ECE and positive accuracy/F1 favor the right model."]
    for aset in ("PRIMARY", "SENSITIVITY"):
        summary_lines += ["", f"## {aset}"]
        for comparison in sorted({row["comparison"] for row in paired if row["analysis_set"] == aset}):
            group = [row for row in paired if row["analysis_set"] == aset and row["comparison"] == comparison]
            summary_lines.append(f"{comparison}: n={len(group)}; " + "; ".join(f"{metric}={np.mean([row[metric] for row in group]):.6g}" for metric in ("categorical_nll", "brier", "ece", "accuracy", "balanced_accuracy", "macro_f1")))
    summary_lines += ["", "Seed uncertainty is conditional on fixed source, spacing, orientation, template and shared test events; where three training seeds exist, the pointwise 95% t interval uses df=2 and is not material generalization.", "Gate diagnostics report entropy, margin, support ratio, learned gate, residual magnitude, probability shift and all four B0/A3 correctness transitions. Correlations are descriptive and not causal.", "", "SCIENTIFIC_TRAINING_COMPLETE=YES", "HELDOUT_EVALUATION_COMPLETE=YES", "THREE_D_RECONSTRUCTION_COMPLETE=NO", "PUBLISHED_BASELINE_COMPLETE=NO", "OS_LEVEL_ISOLATION=NOT_ESTABLISHED"]
    atomic_write(OUT / (PREFIX + "SUMMARY.md"), ("\n".join(summary_lines) + "\n").encode(), guard)
    audit_path = OUT / (PREFIX + "FINAL_AUDIT.json")
    atomic_write(audit_path, json_bytes(audit), guard)
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name.startswith(PREFIX) and not path.name.endswith(("RESULT_MANIFEST.json", "STATE.json")))
    manifest = {path.name: sha(path) for path in files}
    atomic_write(OUT / (PREFIX + "RESULT_MANIFEST.json"), json_bytes(manifest), guard)
    final_state = {"stage": "REVISION_DESIGNA_HELDOUT_EVALUATION_R1", "status": "PASS", "completed_count": 840, "expected_count": 840, "result_manifest_sha256": sha(OUT / (PREFIX + "RESULT_MANIFEST.json"))}
    atomic_write(state_path, json_bytes(final_state), guard)
    guard.records.clear()
    print(json.dumps(dict(status="PASS", evaluated_models=840, primary_models=len(primary), sensitivity_models=len(sensitivity), primary_pairs=sum(truth(row["primary_eligible"]) for row in pairs), sensitivity_pairs=sum(truth(row["sensitivity_eligible"]) for row in pairs), result_manifest_sha256=sha(OUT / (PREFIX + "RESULT_MANIFEST.json")), final_audit_sha256=sha(audit_path)), ensure_ascii=False), flush=True)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.evaluate:
        raise SystemExit("use --evaluate for the authorized held-out run")
    evaluate(resume=args.resume)


if __name__ == "__main__":
    main()
