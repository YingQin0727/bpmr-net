"""Prepare mask-aware paper tables and slice comparisons from frozen 3-D states."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "revision_mtcomm_2026"
DATA = BASE / "REVISION_REPLICATION_P1_dataset"
FORMAL = BASE / "REVISION_REPLICATION_3D" / "MINIMAL_B0_A3_R0" / "FORMAL_RUN_R1"
OUT = BASE / "REVISION_REPLICATION_3D" / "MINIMAL_B0_A3_R0" / "PAPER_COMPARISON_R1"
MATERIALS = ("WF2", "GI")
VARIANTS = ("B0_SOURCE", "A3")
SWEEPS = (3, 5, 10)
PHASES = (0, 1, 2, 3)
PHASE_NAMES = ("Phase 0", "Phase 1", "Phase 2", "Phase 3")
SHAPE = (48, 48, 48)
PHASE_COLORS = ("#f3c7a6", "#9ecae1", "#74c476", "#bcbddc")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_state(material: str, variant: str, sweep: int) -> np.ndarray:
    path = FORMAL / material / variant / f"sweep_{sweep}.npz"
    if not path.is_file():
        raise RuntimeError(f"MISSING_FROZEN_STATE:{path}")
    with np.load(path, allow_pickle=False) as archive:
        state = archive["state"].astype(np.int16, copy=False)
    if state.shape != SHAPE or np.any((state < 0) | (state > 3)):
        raise RuntimeError(f"INVALID_FROZEN_STATE:{path}")
    return state.copy()


def load_inputs(material: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_path = DATA / "observations" / f"{material}_ROI0_25um.bin"
    truth_path = DATA / "test_truth" / f"{material}_ROI0.truth.bin"
    raw = np.fromfile(obs_path, dtype=np.uint8).reshape(SHAPE)
    observed = np.transpose(raw != 255, (2, 1, 0))
    observation = np.transpose(np.where(raw == 255, -1, raw).astype(np.int16), (2, 1, 0))
    truth = np.transpose(np.fromfile(truth_path, dtype=np.uint8).reshape(SHAPE), (2, 1, 0)).astype(np.int16)
    if np.any((truth < 0) | (truth > 3)):
        raise RuntimeError(f"TRUTH_LABEL_RANGE:{material}")
    return truth, observation, observed


def plane_coverage(observed: np.ndarray, axis: int) -> np.ndarray:
    other = tuple(i for i in range(3) if i != axis)
    return np.mean(observed, axis=other)


def choose_plane(coverage: np.ndarray, center: int = 24) -> int:
    # Mask-only rule: lowest observed coverage; ties nearest the center, then lower index.
    return min(range(len(coverage)), key=lambda i: (float(coverage[i]), abs(i - center), i))


def components(values: np.ndarray, region: np.ndarray, phase: int) -> tuple[int, int, float, int]:
    mask = region & (values == phase)
    labelled, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labelled.ravel())[1:]
    voxels = int(mask.sum())
    largest = int(sizes.max()) if len(sizes) else 0
    ratio = float(largest / voxels) if voxels else 0.0
    return int(count), largest, ratio, voxels


def region_metrics(values: np.ndarray, truth: np.ndarray, region: np.ndarray) -> list[dict[str, object]]:
    denominator = int(region.sum())
    if denominator == 0:
        raise RuntimeError("EMPTY_UNOBSERVED_REGION")
    truth_fraction = [float(np.sum(region & (truth == p)) / denominator) for p in PHASES]
    rows = []
    for p in PHASES:
        fraction = float(np.sum(region & (values == p)) / denominator)
        count, largest, ratio, voxels = components(values, region, p)
        rows.append({"phase": p, "fraction": fraction, "truth_fraction": truth_fraction[p], "abs_error": abs(fraction - truth_fraction[p]), "component_count": count, "largest_component_size": largest, "largest_component_fraction": ratio, "phase_voxels": voxels})
    return rows


def image_slice(values: np.ndarray, axis: int, index: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = index
    return values[tuple(sl)]


def draw_comparison(material: str, truth: np.ndarray, observed: np.ndarray, states: dict[tuple[str, int], np.ndarray], selected: dict[str, int]) -> list[str]:
    names = ("Truth", "Observed mask", "B0", "A3")
    axes_info = ((2, "XY", "z"), (1, "XZ", "y"), (0, "YZ", "x"))
    phase_cmap = ListedColormap(PHASE_COLORS)
    mask_cmap = ListedColormap(("#d9d9d9", "#2b8cbe"))
    fig, axes = plt.subplots(4, 3, figsize=(7.5, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.11, wspace=0.03, hspace=0.08)
    for col, (axis, plane_name, coordinate_name) in enumerate(axes_info):
        idx = selected[coordinate_name]
        coverage = float(plane_coverage(observed, axis)[idx])
        axes[0, col].set_title(f"{plane_name} {coordinate_name}={idx}\nobserved {coverage:.1%}")
        truth_slice = image_slice(truth, axis, idx)
        mask_slice = image_slice(observed, axis, idx)
        axes[0, col].imshow(truth_slice.T, origin="lower", interpolation="nearest", cmap=phase_cmap, vmin=0, vmax=3)
        axes[1, col].imshow(mask_slice.T, origin="lower", interpolation="nearest", cmap=mask_cmap, vmin=0, vmax=1)
        for row, (label, state) in enumerate((("B0", states[("B0_SOURCE", 10)]), ("A3", states[("A3", 10)])), start=2):
            axes[row, col].imshow(image_slice(state, axis, idx).T, origin="lower", interpolation="nearest", cmap=phase_cmap, vmin=0, vmax=3)
        boundary = (~mask_slice).astype(float)
        for row in (0, 2, 3):
            axes[row, col].contour(boundary.T, levels=[0.5], colors="white", linewidths=0.45)
    for row, label in enumerate(names):
        axes[row, 0].set_ylabel(label, labelpad=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    legend = [Patch(facecolor=PHASE_COLORS[p], edgecolor="none", label=PHASE_NAMES[p]) for p in PHASES]
    legend += [Patch(facecolor="#2b8cbe", edgecolor="none", label="Observed"), Patch(facecolor="#d9d9d9", edgecolor="none", label="Unobserved")]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(f"{material}: mask-aware 3-D comparison at sweep 10", y=0.98)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("png", {"dpi": 300}), ("svg", {}), ("pdf", {})):
        fig.savefig(OUT / f"MASK_AWARE_{material}_B0_A3.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    return [str(OUT / f"MASK_AWARE_{material}_B0_A3.{ext}") for ext in ("png", "svg", "pdf")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    figure_paths = []
    selected_by_material: dict[str, dict[str, int]] = {}
    center_coverage_by_material: dict[str, dict[str, float]] = {}
    observation_fraction_by_material: dict[str, float] = {}
    input_hashes_by_material: dict[str, dict[str, str]] = {}
    topology_assessment = {}
    for material in MATERIALS:
        truth, observation, observed = load_inputs(material)
        observation_fraction_by_material[material] = float(observed.mean())
        obs_path = DATA / "observations" / f"{material}_ROI0_25um.bin"
        truth_path = DATA / "test_truth" / f"{material}_ROI0.truth.bin"
        input_hashes_by_material[material] = {
            "observation_file_sha256": sha_file(obs_path),
            "truth_file_sha256": sha_file(truth_path),
            "observation_mask_sha256": hashlib.sha256(np.ascontiguousarray(observed.astype(np.uint8)).tobytes()).hexdigest(),
        }
        center_coverage_by_material[material] = {
            axis: float(plane_coverage(observed, axis_index)[24])
            for axis, axis_index in (("x", 0), ("y", 1), ("z", 2))
        }
        selected = {}
        for axis, coordinate_name in ((2, "z"), (1, "y"), (0, "x")):
            coverage = plane_coverage(observed, axis)
            index = choose_plane(coverage)
            selected[coordinate_name] = index
            for i, value in enumerate(coverage):
                slice_rows.append({"material": material, "axis": coordinate_name, "plane_index": i, "observed_coverage": float(value), "selected": i == index, "center_plane": i == 24})
        selected_by_material[material] = selected
        states = {(variant, sweep): load_state(material, variant, sweep) for variant in VARIANTS for sweep in SWEEPS}
        figure_paths.extend(draw_comparison(material, truth, observed, states, selected))

        for region_name, region in (("overall", np.ones(SHAPE, dtype=bool)), ("unobserved", ~observed)):
            for source, values, sweep in [("Truth", truth, "truth")]:
                for row in region_metrics(values, truth, region):
                    row.update({"material": material, "source": source, "variant": "Truth", "sweep": sweep, "region": region_name})
                    all_metrics.append(row)
            for variant in VARIANTS:
                for sweep in SWEEPS:
                    for row in region_metrics(states[(variant, sweep)], truth, region):
                        row.update({"material": material, "source": "Prediction", "variant": variant, "sweep": sweep, "region": region_name})
                        all_metrics.append(row)

        # Topology assessment is based on sweep 10 and is descriptive only.
        topology_assessment[material] = {}
        for region_name, region in (("overall", np.ones(SHAPE, dtype=bool)), ("unobserved", ~observed)):
            truth_rows = region_metrics(truth, truth, region)
            b0_rows = region_metrics(states[("B0_SOURCE", 10)], truth, region)
            a3_rows = region_metrics(states[("A3", 10)], truth, region)
            truth_counts = [r["component_count"] for r in truth_rows]
            b0_counts = [r["component_count"] for r in b0_rows]
            a3_counts = [r["component_count"] for r in a3_rows]
            topology_assessment[material][region_name] = {
                "truth_component_count": truth_counts,
                "b0_component_count": b0_counts,
                "a3_component_count": a3_counts,
                "b0_abs_component_count_difference_sum": int(sum(abs(a - b) for a, b in zip(b0_counts, truth_counts))),
                "a3_abs_component_count_difference_sum": int(sum(abs(a - b) for a, b in zip(a3_counts, truth_counts))),
                "truth_largest_component_fraction": [r["largest_component_fraction"] for r in truth_rows],
                "b0_largest_component_fraction": [r["largest_component_fraction"] for r in b0_rows],
                "a3_largest_component_fraction": [r["largest_component_fraction"] for r in a3_rows],
            }

    coverage_path = OUT / "MASK_SLICE_COVERAGE.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["material", "axis", "plane_index", "observed_coverage", "selected", "center_plane"])
        writer.writeheader(); writer.writerows(slice_rows)
    metrics_path = OUT / "MASK_AWARE_3D_METRICS.csv"
    fields = ["material", "source", "variant", "sweep", "region", "phase", "fraction", "truth_fraction", "abs_error", "component_count", "largest_component_size", "largest_component_fraction", "phase_voxels"]
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(all_metrics)

    summary = {
        "stage": "MINIMAL_3D_PAPER_COMPARISON_R1",
        "status": "PASS",
        "source_commit": "fa72c0c",
        "materials": list(MATERIALS),
        "fixed_sweeps": list(SWEEPS),
        "slice_selection_rule": "minimum observed-mask coverage; ties nearest center index 24, then lower index; no state or truth values used",
        "selected_slices": selected_by_material,
        "center_slice_coverage": center_coverage_by_material,
        "selected_slice_coverage": {m: {axis: float(next(r["observed_coverage"] for r in slice_rows if r["material"] == m and r["axis"] == axis and r["selected"])) for axis in ("x", "y", "z")} for m in MATERIALS},
        "observation_fraction": observation_fraction_by_material,
        "input_hashes": input_hashes_by_material,
        "topology_assessment_sweep10": topology_assessment,
        "files": {p.name: sha_file(p) for p in (coverage_path, metrics_path)},
        "figures": {},
    }
    for path in figure_paths:
        summary["figures"][Path(path).name] = sha_file(Path(path))
    summary_path = OUT / "MASK_AWARE_3D_COMPARISON_AUDIT.json"
    summary["audit_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Mask-aware 3-D paper comparison",
        "",
        "Source trajectories are frozen from commit `fa72c0c`; no Gibbs updates were rerun.",
        "",
        "## Why the former center slices were identical",
        "",
        "The measured observation mask coverage at XY z=24, XZ y=24 and YZ x=24 is 100% for both materials. These planes are fully observed, and the reconstruction uses hard observed clamping after every sequential Gibbs update. The B0 and A3 slices therefore remain identical by construction. The observed values on these center planes also match the registered truth values exactly in this ROI.",
        "",
        "## Mask-only slice selection",
        "",
        "The comparison figures use index 23 in each direction. This is the minimum-coverage plane (33.984375% observed, 66.015625% unobserved); ties were resolved by nearest distance to the center index 24 and then the lower index. No model or truth values entered this choice. The mask is geometrically identical for WF2 and GI, so the selected indices are identical, while the material observations and reconstructed phase fields remain distinct.",
        "",
        "## Topology interpretation",
        "",
        "At sweep 10, A3 reduces the large B0 component-count excess for both materials and is closer to the truth in the sum of absolute component-count differences, both overall and in the unobserved region. This is evidence of reduced B0 fragmentation under this fixed condition. It is not proof of topology preservation: A3 still has more components than truth for several phases, and the dominant-component fraction for GI phase 0 in the unobserved region is higher than truth, consistent with possible over-connection. Therefore the defensible statement is conditional improvement in fragmentation, with residual smoothing/over-connection risk.",
        "",
        "All phase fractions, absolute errors, six-connected component counts and largest-component fractions are in `MASK_AWARE_3D_METRICS.csv`. `Truth` rows are reported once per region; B0 and A3 rows cover sweeps 3, 5 and 10.",
        "",
        "## Scope",
        "",
        "The results support a fixed-ROI, fixed-mask B0/A3 3-D comparison for WF2 and GI. They do not establish broad 3-D generalization, superiority across ROI or spacing choices, or equivalence of topology to the truth. The figures are descriptive and do not replace the unresolved sensitivity matrix or external baseline comparisons.",
    ]
    (OUT / "MASK_AWARE_3D_COMPARISON_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["files"]["MASK_AWARE_3D_COMPARISON_SUMMARY.md"] = sha_file(OUT / "MASK_AWARE_3D_COMPARISON_SUMMARY.md")
    summary["audit_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(OUT), "selected_slices": selected_by_material}, sort_keys=True))


if __name__ == "__main__":
    main()
