"""Create compact, revision-only outputs for the completed minimal 3-D run."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "revision_mtcomm_2026"
DATA = BASE / "REVISION_REPLICATION_P1_dataset"
OUT = BASE / "REVISION_REPLICATION_3D" / "MINIMAL_B0_A3_R0" / "FORMAL_RUN_R1"
SUMMARY_PATH = OUT / "MINIMAL_3D_B0_A3_SUMMARY.json"
MATERIALS = ("WF2", "GI")
VARIANTS = ("B0_SOURCE", "A3")
SWEEPS = (0, 3, 5, 10)
PHASES = (0, 1, 2, 3)
PHASE_NAMES = ("Phase 0", "Phase 1", "Phase 2", "Phase 3")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("formal_trajectory_count") != 4:
        raise RuntimeError("FORMAL_TRAJECTORY_SET_INCOMPLETE")
    if not summary.get("truth_read_after_reconstruction"):
        raise RuntimeError("FROZEN_EVALUATION_NOT_COMPLETED")

    by_key = {(r["material"], r["variant"]): r for r in summary["results"]}
    if set(by_key) != {(m, v) for m in MATERIALS for v in VARIANTS}:
        raise RuntimeError("TRAJECTORY_IDENTITY_SET_MISMATCH")

    # Compact numeric table, retaining the full metric payload in the JSON summary.
    metrics_path = OUT / "MINIMAL_3D_B0_A3_RESULTS.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "variant", "sweep", "phase", "predicted_fraction", "truth_fraction", "fraction_error", "extinct_phase", "components_6_connected", "largest_component"])
        for key in sorted(by_key):
            result = by_key[key]
            for sweep in SWEEPS:
                metric = result["metrics"][str(sweep)]
                for phase in PHASES:
                    component = metric["components"][str(phase)]
                    writer.writerow([result["material"], result["variant"], sweep, phase, metric["phase_fractions"][phase], metric["truth_phase_fractions"][phase], metric["phase_fraction_error"][phase], int(phase in metric["extinct_phases"]), component["components_6_connected"], component["largest_component"]])

    runtime_path = OUT / "MINIMAL_3D_B0_A3_RUNTIME.csv"
    with runtime_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "variant", "wall_seconds", "peak_rss_bytes", "updates_per_sweep", "observed_voxels", "initial_state_sha256", "final_state_sha256"])
        for key in sorted(by_key):
            result = by_key[key]
            obs = result["observation_metadata"]
            writer.writerow([result["material"], result["variant"], result["elapsed_wall_seconds"], result["peak_rss_bytes"], obs["updates_per_sweep"], obs["observed_count"], result["initial_state_sha256"], result["final_state_sha256"]])

    roi_registry = {
        "stage": "MINIMAL_3D_B0_A3_COMPARISON",
        "roi_selection": "ROI0, first frozen matrix condition; selected before reconstruction quality was read",
        "roi": {
            m: {
                "origin_zyx": by_key[(m, "B0_SOURCE")]["observation_metadata"]["origin_zyx"],
                "shape_zyx": by_key[(m, "B0_SOURCE")]["observation_metadata"]["shape_zyx"],
                "observed_count": by_key[(m, "B0_SOURCE")]["observation_metadata"]["observed_count"],
                "updates_per_sweep": by_key[(m, "B0_SOURCE")]["observation_metadata"]["updates_per_sweep"],
                "observation_file_sha256": by_key[(m, "B0_SOURCE")]["observation_metadata"]["file_sha256"],
                "mask_sha256": by_key[(m, "B0_SOURCE")]["observation_metadata"]["mask_sha256"],
            }
            for m in MATERIALS
        },
        "control": {"global_fit_um": 250, "local_observation_um": 25, "clamp_mode": "hard_observed", "shape_xyz": [48, 48, 48], "training_seed": 1701, "rollout_seed": 1701, "update_order": "forward", "initialization": "train_prior"},
        "train_validation_test_separation": {"test_roi_z_origin": 2300, "train_z_max": 1000, "validation_z_min": 1128, "guard_band_voxels": 128, "direct_label_overlap": False},
        "observation_mask_sha256": by_key[("WF2", "B0_SOURCE")]["observation_mask_sha256"],
        "materials_observation_sha256": {m: by_key[(m, "B0_SOURCE")]["observation_metadata"]["file_sha256"] for m in MATERIALS},
        "truth_sha256": {m: {"sha256": summary["truth_files"][m]["sha256"]} for m in MATERIALS},
    }
    (OUT / "MINIMAL_3D_B0_A3_ROI_REGISTRY.json").write_text(json.dumps(roi_registry, indent=2, sort_keys=True), encoding="utf-8")

    # Python-only publication preview: same phase palette for every panel.
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False})
    cmap = ListedColormap(["#f3c7a6", "#9ecae1", "#74c476", "#bcbddc"])
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.4), constrained_layout=True)
    for row, material in enumerate(MATERIALS):
        for variant_index, variant in enumerate(VARIANTS):
            result = by_key[(material, variant)]
            path = Path(result["state_files"][-1]["path"])
            with np.load(path, allow_pickle=False) as archive:
                state = archive["state"]
            images = (state[:, :, 24], state[:, 24, :], state[24, :, :])
            panel_row = row * 2 + variant_index
            for col, (ax, image, label) in enumerate(zip(axes[panel_row], images, ("XY z=24", "XZ y=24", "YZ x=24"))):
                ax.imshow(image.T, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{material} {variant}", rotation=90, labelpad=8)
                ax.set_title(label if panel_row == 0 else "")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=cmap(i), markersize=6, label=PHASE_NAMES[i]) for i in PHASES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Minimal Design A 3-D reconstruction comparison (sweep 10)", y=1.01)
    fig.savefig(OUT / "MINIMAL_3D_B0_A3_SLICES.svg", bbox_inches="tight")
    fig.savefig(OUT / "MINIMAL_3D_B0_A3_SLICES.pdf", bbox_inches="tight")
    fig.savefig(OUT / "MINIMAL_3D_B0_A3_SLICES.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    checks = {
        "trajectory_count": len(by_key) == 4,
        "saved_sweeps": all(r["sweeps_saved"] == list(SWEEPS) for r in by_key.values()),
        "paired_initial_state": all(by_key[(m, "B0_SOURCE")]["initial_state_sha256"] == by_key[(m, "A3")]["initial_state_sha256"] for m in MATERIALS),
        "paired_observation_mask": all(by_key[(m, "B0_SOURCE")]["observation_mask_sha256"] == by_key[(m, "A3")]["observation_mask_sha256"] for m in MATERIALS),
        "hard_clamp": True,
        "state_hashes": True,
        "all_phases_present": True,
    }
    for key, result in by_key.items():
        obs = result["observation_metadata"]
        raw = np.fromfile(Path(obs["path"]), dtype=np.uint8).reshape((48, 48, 48))
        observed = np.transpose(raw != 255, (2, 1, 0))
        labels = np.transpose(np.where(raw == 255, -1, raw).astype(np.int16), (2, 1, 0))
        for item in result["state_files"]:
            state_path = Path(item["path"])
            checks["state_hashes"] = checks["state_hashes"] and sha_file(state_path) == item["sha256"]
            with np.load(state_path, allow_pickle=False) as archive:
                state = archive["state"]
            checks["hard_clamp"] = checks["hard_clamp"] and bool(np.array_equal(state[observed], labels[observed]))
        checks["all_phases_present"] = checks["all_phases_present"] and all(not result["metrics"][str(s)]["extinct_phases"] for s in SWEEPS)
    if not all(checks.values()):
        raise RuntimeError(f"MINIMAL_3D_AUDIT_FAILED:{checks}")

    release_results = []
    for key in sorted(by_key):
        result = by_key[key]
        release_results.append({
            "material": result["material"],
            "variant": result["variant"],
            "roi": result["roi"],
            "global_fit_um": result["global_fit_um"],
            "local_observation_um": result["local_observation_um"],
            "clamp_mode": result["clamp_mode"],
            "training_seed": result["training_seed"],
            "rollout_seed": result["rollout_seed"],
            "sweeps_saved": result["sweeps_saved"],
            "observation_mask_sha256": result["observation_mask_sha256"],
            "initial_state_sha256": result["initial_state_sha256"],
            "final_state_sha256": result["final_state_sha256"],
            "state_files": [{"sweep": x["sweep"], "sha256": x["sha256"]} for x in result["state_files"]],
            "elapsed_wall_seconds": result["elapsed_wall_seconds"],
            "peak_rss_bytes": result["peak_rss_bytes"],
            "metrics": result["metrics"],
            "checkpoint_bindings": [{"unit": x["unit"], "variant": x["variant"], "sha256": x["sha256"], "source_b0_sha256": x["source_b0_sha256"]} for x in result["checkpoint_bindings"]],
            "observation": {"file_sha256": result["observation_metadata"]["file_sha256"], "mask_sha256": result["observation_metadata"]["mask_sha256"], "observed_count": result["observation_metadata"]["observed_count"], "updates_per_sweep": result["observation_metadata"]["updates_per_sweep"]},
            "truth_file_sha256": result["truth_file_sha256"],
        })
    release_summary = {
        "stage": "MINIMAL_3D_B0_A3_COMPARISON",
        "status": "PASS",
        "formal_trajectory_count": 4,
        "controls": {"global_fit_um": 250, "local_observation_um": 25, "clamp_mode": "hard_observed", "shape_xyz": [48, 48, 48], "training_seed": 1701, "rollout_seed": 1701, "update_order": "forward", "initialization": "train_prior", "sweeps": list(SWEEPS)},
        "results": release_results,
        "truth_files_sha256": {m: summary["truth_files"][m]["sha256"] for m in MATERIALS},
        "truth_read_after_reconstruction": True,
        "checks": checks,
    }
    release_path = OUT / "MINIMAL_3D_B0_A3_SUMMARY_RELEASE.json"
    release_path.write_text(json.dumps(release_summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "stage": "MINIMAL_3D_B0_A3_COMPARISON",
        "status": "PASS",
        "formal_trajectories": 4,
        "materials": list(MATERIALS),
        "variants": list(VARIANTS),
        "sweeps": list(SWEEPS),
        "model_identity": "Design A B0_SOURCE/A3, G250, seed1701, registry-ordered x/y/z and template0..3 fusion",
        "truth_read_policy": "truth loaded only after all four trajectory state files and reconstruction lock were written",
        "checks": checks,
        "files": {p.name: sha_file(p) for p in (metrics_path, runtime_path, OUT / "MINIMAL_3D_B0_A3_ROI_REGISTRY.json", OUT / "MINIMAL_3D_B0_A3_SLICES.svg", OUT / "MINIMAL_3D_B0_A3_SLICES.pdf", OUT / "MINIMAL_3D_B0_A3_SLICES.png", release_path)},
        "summary_sha256": sha_file(release_path),
    }
    (OUT / "MINIMAL_3D_B0_A3_FINAL_AUDIT.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Minimal Design A 3-D B0/A3 comparison",
        "",
        "Status: PASS for the four pre-registered trajectories only.",
        "",
        "The fixed condition was ROI0, 48^3, global fit spacing 250 um, local observation spacing 25 um, hard observed clamping, train-prior initialization, forward sequential Gibbs updates, rollout seed 1701, and saved sweeps 0, 3, 5, and 10. ROI and model identities were fixed before reconstruction quality was inspected.",
        "",
        "WF2 and GI used their registered 25 um sparse observations. The ROI origin is ZYX [2300, 800, 200], and the registered test slab is separated from the train and validation z slabs by the frozen split and guard-band rule. The observation mask hash is shared by geometry; material observation file hashes are distinct.",
        "",
        "The four trajectories completed with the reference sequential, immediate-writeback Gibbs implementation. B0 and A3 used the same initial state, observed mask, update order, and NumPy rollout stream within each material. The A3 trajectory is slower because it evaluates the frozen residual model at every unobserved voxel.",
        "",
        "## Runtime",
        "",
        "| Material | Variant | Wall time (s) | Peak RSS (MB) |",
        "|---|---:|---:|---:|",
    ]
    for key in sorted(by_key):
        result = by_key[key]
        lines.append(f"| {result['material']} | {result['variant']} | {result['elapsed_wall_seconds']:.1f} | {result['peak_rss_bytes'] / 1024**2:.1f} |")
    total = sum(float(r["elapsed_wall_seconds"]) for r in by_key.values())
    lines += [
        f"| **Total** | 4 trajectories | **{total:.1f}** | serial CPU |",
        "",
        "## Scientific reading",
        "",
        "Phase fractions, extinction flags, six-connected component counts and largest-component sizes for every saved sweep are in `MINIMAL_3D_B0_A3_RESULTS.csv`. The sweep-10 orthogonal slices are in `MINIMAL_3D_B0_A3_SLICES.png` (with editable SVG/PDF copies). These four trajectories provide a direct B0/A3 comparison for WF2 and GI under one fixed registered condition; they do not establish broad cross-material or general 3-D superiority.",
        "",
        "The observed outputs retain all four phases in these four trajectories at sweeps 0, 3, 5 and 10; no phase-extinction flag was produced. This is a statement about these registered ROIs and this fixed condition only. It does not convert the 2-D minority-phase findings into a general 3-D claim.",
        "",
        "Reviewer issues partially addressed: Design A B0 versus A3 can now be compared in 3-D with hard observation clamping, fixed sparse observations, and explicit runtime/topology outputs. Remaining gaps are additional ROI/spacing/sweep sensitivity, initialization/update-order sensitivity, longer-sweep behavior beyond 10, and external comparator baselines.",
        "",
        f"Summary SHA256: `{sha_file(SUMMARY_PATH)}`",
    ]
    (OUT / "MINIMAL_3D_B0_A3_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "trajectories": 4, "total_wall_seconds": total, "output": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
