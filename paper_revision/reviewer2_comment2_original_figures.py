"""Generate Reviewer #2 Comment 2 figures from the original submission data.

This script deliberately reads only the original dense segmentation ROIs and the
original global-plane/role registries.  It does not import Design A data, model
checkpoints, reconstruction states, or test labels.

The raw source arrays are memory-mapped and only the two registered 48^3 ROIs
are materialized.  Labels follow the original protocol's canonicalization:
raw labels 1..4 -> canonical phases 0..3 (already-canonical 0..3 are retained).
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle


REPO = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO / "figures"
RAW_ROOT_DEFAULT = Path(
    r"D:\Paper2\paper2_stage02a_source_recovery\official_download"
)
AUTHORITY_ROOT_DEFAULT = Path(
    r"C:\Users\admin\Documents\博士申请论文\factorial_v2_supervisor_handoff_v1\core_extracted_r2\supervisor_handoff_v1_r2\14_PAPER_FACTBOOK_AND_CLAIMS\methods"
)
REGISTERED_ROOT_DEFAULT = Path(r"D:\Paper2\paper2_stage02a_registered_data")

FULL_SHAPE_ZYX = (3051, 1033, 1030)
VOXEL_PITCH_UM = 5
ROI = {
    "WF2": (1376, 1424, 576, 624, 580, 628),
    "GI": (1576, 1624, 376, 424, 580, 628),
}
RAW_FILENAMES = {
    "WF2": "WF2_5um_1030x1033x3051_8b_segmented_corrected.raw",
    "GI": "GI_5um_1030x1033x3051_8b_segmented.raw",
}
RAW_SHA256 = {
    "WF2": "932853c36761504216c3a734a852c08f6bf451ba6a5c4da9367a2ec8b724d22a",
    "GI": "35f336b0446a0bd8107197fdde697283b450d67aabb9fd59d4f104d8b97f8c94",
}

# The colour sequence is the one used by the original manuscript/reproduction
# plotting code.  Labels are intentionally called Phase 0..3 only.
PHASE_COLORS = ("#0072B2", "#E69F00", "#8C8C8C", "#009E73")
PHASE_CMAP = ListedColormap(PHASE_COLORS, name="original_four_phase")
PHASE_NORM = BoundaryNorm(np.arange(-0.5, 4.5, 1), PHASE_CMAP.N)
MASK_CMAP = ListedColormap(("#f2f2f2", "#2b8cbe"), name="registered_mask")
ROLE_COLORS = {
    "fitting": "#0072B2",
    "validation": "#D55E00",
    "held-out": "#CC79A7",
    "unassigned": "#8c8c8c",
}
SPACINGS = (25, 250, 1000)
ORIENTATIONS = ("x", "y", "z")
ROLE_FIELDS = {
    "fitting": "fitting_plane_indices",
    "validation": "validation_plane_indices",
    "held-out": "withheld_plane_indices",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_list(value: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)!r}")
    return [int(item) for item in parsed]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonicalize_labels(labels: np.ndarray) -> tuple[np.ndarray, str, list[int]]:
    values = np.unique(labels)
    if len(values) and int(values.min()) >= 1 and int(values.max()) <= 4:
        return labels.astype(np.int16) - 1, "raw 1..4 -> canonical 0..3", values.tolist()
    if len(values) and int(values.min()) >= 0 and int(values.max()) <= 3:
        return labels.astype(np.int16), "already canonical 0..3", values.tolist()
    raise RuntimeError(f"UNSUPPORTED_PHASE_ENCODING:{values.tolist()}")


def load_original_data(
    raw_root: Path, authority_root: Path, registered_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    roi_json_path = authority_root / "source_roi_registry_v2r2.json"
    global_csv_path = authority_root / "global_plane_sets_v2r2.csv"
    role_csv_path = authority_root / "plane_role_registry_v2r2.csv"
    master_ledger_path = registered_root / "PAPER2_REGISTERED_MASTER_PLANE_LEDGER.csv"
    for path in (roi_json_path, global_csv_path, role_csv_path, master_ledger_path):
        if not path.is_file():
            raise FileNotFoundError(f"ORIGINAL_REGISTRATION_FILE_MISSING:{path}")

    roi_registry = read_json(roi_json_path)
    roi_records = {record["material"]: record for record in roi_registry["records"]}
    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, dict] = {}
    for material, bounds in ROI.items():
        record = roi_records.get(material)
        if record is None:
            raise RuntimeError(f"ROI_REGISTRY_RECORD_MISSING:{material}")
        expected_start = list(bounds[::2])
        expected_end = list(bounds[1::2])
        if record["roi_start_zyx_inclusive"] != expected_start:
            raise RuntimeError(f"ROI_START_REGISTRY_MISMATCH:{material}")
        if record["roi_end_zyx_exclusive"] != expected_end:
            raise RuntimeError(f"ROI_END_REGISTRY_MISMATCH:{material}")
        if record["roi_shape_zyx"] != [48, 48, 48]:
            raise RuntimeError(f"ROI_SHAPE_REGISTRY_MISMATCH:{material}")
        raw_path = raw_root / RAW_FILENAMES[material]
        if not raw_path.is_file():
            raise FileNotFoundError(f"ORIGINAL_ARRAY_MISSING:{raw_path}")
        expected_bytes = int(np.prod(FULL_SHAPE_ZYX))
        if raw_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"RAW_BYTE_SIZE_MISMATCH:{material}:{raw_path.stat().st_size}")
        raw_sha = sha256_file(raw_path)
        if raw_sha != RAW_SHA256[material] or raw_sha != record["source_sha256"]:
            raise RuntimeError(f"RAW_SHA256_MISMATCH:{material}:{raw_sha}")
        raw = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=FULL_SHAPE_ZYX, order="C")
        z0, z1, y0, y1, x0, x1 = bounds
        crop = np.asarray(raw[z0:z1, y0:y1, x0:x1]).copy()
        canonical, mapping, raw_values = canonicalize_labels(crop)
        arrays[material] = canonical
        provenance[material] = {
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha,
            "raw_values": raw_values,
            "canonical_mapping": mapping,
            "roi_zyx_half_open": [[z0, z1], [y0, y1], [x0, x1]],
            "roi_shape_zyx": list(crop.shape),
            "voxel_pitch_um": VOXEL_PITCH_UM,
        }

    global_rows = read_csv(global_csv_path)
    role_rows = read_csv(role_csv_path)
    globals_by_key: dict[tuple[str, int, str], dict] = {}
    for row in global_rows:
        key = (row["material"], int(row["spacing_um"]), row["orientation"])
        globals_by_key[key] = {
            "voxel_indices": parse_list(row["voxel_indices"]),
            "physical_coordinates_um": parse_list(row["physical_coordinates_um"]),
            "roi_intersecting": parse_list(row["roi_intersecting_plane_indices"]),
            "anchor_voxel_index": int(row["anchor_voxel_index"]),
        }
    roles_by_key: dict[tuple[str, int, str], dict[int, str]] = {}
    for row in role_rows:
        key = (row["material"], int(row["spacing_um"]), row["orientation"])
        role_map: dict[int, str] = {}
        for role, field in ROLE_FIELDS.items():
            for index in parse_list(row[field]):
                if index in role_map:
                    raise RuntimeError(f"ROLE_OVERLAP:{key}:{index}")
                role_map[index] = role
        roles_by_key[key] = role_map

    registration = {
        "global_rows": globals_by_key,
        "role_maps": roles_by_key,
        "source_roi_registry": str(roi_json_path),
        "global_plane_registry": str(global_csv_path),
        "plane_role_registry": str(role_csv_path),
        "registered_master_plane_ledger": str(master_ledger_path),
        "registry_sha256": {
            "source_roi_registry_v2r2.json": sha256_file(roi_json_path),
            "global_plane_sets_v2r2.csv": sha256_file(global_csv_path),
            "plane_role_registry_v2r2.csv": sha256_file(role_csv_path),
            "PAPER2_REGISTERED_MASTER_PLANE_LEDGER.csv": sha256_file(master_ledger_path),
        },
        "provenance": provenance,
    }
    for material in ROI:
        for spacing in SPACINGS:
            for orientation in ORIENTATIONS:
                key = (material, spacing, orientation)
                if key not in globals_by_key or key not in roles_by_key:
                    raise RuntimeError(f"ORIGINAL_PLANE_OR_ROLE_RECORD_MISSING:{key}")
                indices = set(globals_by_key[key]["voxel_indices"])
                if not indices.issubset(roles_by_key[key]):
                    missing = sorted(indices - roles_by_key[key].keys())
                    raise RuntimeError(f"GLOBAL_PLANE_ROLE_BINDING_MISSING:{key}:{missing[:10]}")
    return arrays, registration, provenance


def registered_mask(material: str, spacing: int, registration: dict) -> np.ndarray:
    mask = np.zeros((48, 48, 48), dtype=bool)
    starts = {"z": ROI[material][0], "y": ROI[material][2], "x": ROI[material][4]}
    for orientation in ORIENTATIONS:
        key = (material, spacing, orientation)
        data = registration["global_rows"][key]
        for global_index in data["roi_intersecting"]:
            local_index = int(global_index) - starts[orientation]
            if not 0 <= local_index < 48:
                raise RuntimeError(f"ROI_INTERSECTING_PLANE_OUTSIDE_ROI:{key}:{global_index}")
            if orientation == "x":
                mask[:, :, local_index] = True
            elif orientation == "y":
                mask[:, local_index, :] = True
            else:
                mask[local_index, :, :] = True
    return mask


def assert_mask_counts(registration: dict) -> dict[tuple[str, int], int]:
    expected = {25: 51273, 250: 6769, 1000: 6769}
    counts: dict[tuple[str, int], int] = {}
    for material in ROI:
        for spacing in SPACINGS:
            mask = registered_mask(material, spacing, registration)
            counts[(material, spacing)] = int(mask.sum())
            if counts[(material, spacing)] != expected[spacing]:
                raise RuntimeError(
                    f"REGISTERED_MASK_COUNT_MISMATCH:{material}:{spacing}:"
                    f"{counts[(material, spacing)]}!={expected[spacing]}"
                )
    return counts


def _style_axes(ax) -> None:
    ax.tick_params(labelsize=7, width=0.6, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def add_phase_legend(fig) -> None:
    fig.legend(
        handles=[Patch(facecolor=PHASE_COLORS[p], edgecolor="none", label=f"Phase {p}") for p in range(4)],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.002),
    )


def plot_truth_overview(arrays: dict[str, np.ndarray], provenance: dict[str, dict], out_pdf: Path, out_png: Path) -> None:
    fig = plt.figure(figsize=(12.6, 6.6), constrained_layout=False)
    gs = GridSpec(2, 5, figure=fig, width_ratios=(1.35, 1, 1, 1, 1), wspace=0.28, hspace=0.36, bottom=0.12, top=0.91, left=0.035, right=0.985)
    fig.suptitle("Original-submission ground-truth ROIs", fontsize=13, fontweight="bold")
    for row, material in enumerate(("WF2", "GI")):
        arr = arrays[material]
        ax3d = fig.add_subplot(gs[row, 0], projection="3d")
        # A deterministic stride keeps the editable vector PDF practical; all
        # fractions and all 2D slices below use the complete 48^3 ROI.
        sample = arr[::2, ::2, ::2]
        zz, yy, xx = np.indices(sample.shape)
        for phase in range(4):
            select = sample == phase
            if not np.any(select):
                continue
            ax3d.scatter(
                (xx[select] * 2 + 0.5) * VOXEL_PITCH_UM,
                (yy[select] * 2 + 0.5) * VOXEL_PITCH_UM,
                (zz[select] * 2 + 0.5) * VOXEL_PITCH_UM,
                s=3.2,
                c=PHASE_COLORS[phase],
                linewidths=0,
                alpha=0.78,
                depthshade=False,
            )
        ax3d.set_title(f"{material}\n3D reference", fontsize=9, loc="left")
        ax3d.set_xlabel("X (μm)", fontsize=7, labelpad=1)
        ax3d.set_ylabel("Y (μm)", fontsize=7, labelpad=1)
        ax3d.set_zlabel("Z (μm)", fontsize=7, labelpad=1)
        ax3d.set_xlim(0, 240); ax3d.set_ylim(0, 240); ax3d.set_zlim(0, 240)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.view_init(elev=25, azim=-58)
        ax3d.tick_params(labelsize=6, pad=0)
        ax3d.text2D(0.02, 0.02, "display stride 2; full-ROI fractions", transform=ax3d.transAxes, fontsize=6)

        slices = (
            (arr[24, :, :], "XY · z=24"),
            (arr[:, 24, :], "XZ · y=24"),
            (arr[:, :, 24], "YZ · x=24"),
        )
        for col, (image, title) in enumerate(slices, start=1):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image, cmap=PHASE_CMAP, norm=PHASE_NORM, origin="lower", interpolation="none", aspect="equal")
            ax.set_title(title, fontsize=8)
            ax.set_xticks((0, 24, 47), labels=("0", "24", "47"), fontsize=6)
            ax.set_yticks((0, 24, 47), labels=("0", "24", "47"), fontsize=6)
            ax.set_xlabel("local voxel index", fontsize=6)
            if col == 1:
                ax.set_ylabel("local voxel index", fontsize=6)
            _style_axes(ax)

        counts = np.bincount(arr.ravel(), minlength=4).astype(float)
        fractions = counts / counts.sum()
        axf = fig.add_subplot(gs[row, 4])
        axf.bar(np.arange(4), fractions, color=PHASE_COLORS, width=0.72)
        axf.set_ylim(0, 1.0)
        axf.set_xticks(range(4), labels=["0", "1", "2", "3"], fontsize=7)
        axf.set_ylabel("volume fraction", fontsize=7)
        axf.set_title("Full ROI fractions", fontsize=8)
        axf.grid(axis="y", color="#dddddd", linewidth=0.5)
        for phase, fraction in enumerate(fractions):
            axf.text(phase, min(0.97, fraction + 0.03), f"{fraction:.3f}", ha="center", va="bottom", fontsize=7)
        axf.text(0.02, -0.20, "n=110,592 voxels", transform=axf.transAxes, fontsize=6)
        _style_axes(axf)
        # Material-level provenance is embedded in the PDF metadata and this
        # unobtrusive note; no physical phase names are inferred.
        axf.text(0.02, 1.02, provenance[material]["canonical_mapping"], transform=axf.transAxes, fontsize=5.5, color="#555555")

    add_phase_legend(fig)
    fig.savefig(out_pdf, metadata={"Title": "Original-submission ground-truth ROIs", "Creator": "reviewer2_comment2_original_figures.py"})
    fig.savefig(out_png, dpi=450)
    plt.close(fig)


def plot_registered_masks(registration: dict, counts: dict[tuple[str, int], int], out_pdf: Path, out_png: Path) -> None:
    fig = plt.figure(figsize=(15.6, 8.2), constrained_layout=False)
    gs = GridSpec(2, 4, figure=fig, width_ratios=(1, 1, 1, 1.58), wspace=0.30, hspace=0.48, left=0.045, right=0.985, top=0.90, bottom=0.14)
    fig.suptitle("Original registered global planes and 48³ observation masks", fontsize=13, fontweight="bold")
    roi_axis_bounds = {
        "x": {"WF2": (580, 628), "GI": (580, 628)},
        "y": {"WF2": (576, 624), "GI": (376, 424)},
        "z": {"WF2": (1376, 1424), "GI": (1576, 1624)},
    }
    for row, material in enumerate(("WF2", "GI")):
        for col, orientation in enumerate(ORIENTATIONS):
            ax = fig.add_subplot(gs[row, col])
            for lane, spacing in enumerate(SPACINGS):
                key = (material, spacing, orientation)
                data = registration["global_rows"][key]
                role_map = registration["role_maps"][key]
                y = 2 - lane
                indices = np.asarray(data["voxel_indices"], dtype=int)
                roles = np.asarray([role_map.get(int(index), "unassigned") for index in indices], dtype=object)
                for role in ("fitting", "validation", "held-out", "unassigned"):
                    sel = roles == role
                    if np.any(sel):
                        ax.scatter(indices[sel], np.full(int(sel.sum()), y), s=8, color=ROLE_COLORS[role], alpha=0.55, linewidths=0)
                roi_sel = np.isin(indices, np.asarray(data["roi_intersecting"], dtype=int))
                if np.any(roi_sel):
                    ax.scatter(indices[roi_sel], np.full(int(roi_sel.sum()), y), s=30, facecolors="none", edgecolors="black", linewidths=0.7, zorder=4)
            lo, hi = roi_axis_bounds[orientation][material]
            ax.axvspan(lo, hi, color="#000000", alpha=0.06, zorder=0)
            ax.set_yticks((2, 1, 0), labels=("25", "250", "1000"), fontsize=7)
            ax.set_xlabel(f"{orientation.upper()} source voxel index", fontsize=7)
            ax.set_title(f"{material} · global {orientation}-planes", fontsize=8)
            ax.set_ylim(-0.55, 2.55)
            ax.grid(axis="x", color="#e5e5e5", linewidth=0.45)
            _style_axes(ax)

        mask_spec = GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[row, 3], hspace=0.35, wspace=0.12)
        for r, spacing in enumerate(SPACINGS):
            mask = registered_mask(material, spacing, registration)
            # Local index 22 is deliberately an unobserved slice for all three
            # regimes; the center plane itself would be fully occupied by the
            # registered central plane and would hide the nine-vs-one contrast.
            images = (mask[22, :, :], mask[:, 22, :], mask[:, :, 22])
            for c, image in enumerate(images):
                axm = fig.add_subplot(mask_spec[r, c])
                axm.imshow(image.astype(np.uint8), cmap=MASK_CMAP, vmin=0, vmax=1, origin="lower", interpolation="none", aspect="equal")
                axm.set_xticks([]); axm.set_yticks([])
                for spine in axm.spines.values():
                    spine.set_linewidth(0.45)
                if r == 0:
                    axm.set_title(("XY" if c == 0 else "XZ" if c == 1 else "YZ"), fontsize=7, pad=2)
                if c == 0:
                    axm.set_ylabel(f"{spacing} μm\n{counts[(material, spacing)]:,}", fontsize=6, rotation=0, ha="right", va="center", labelpad=16)
        # Make the local-panel title and boundary explicit without filling the panel.
        fig.text(0.77, 0.925 - 0.42 * row, f"{material} · target-domain masks (slice index 22; observed=blue)", fontsize=8, ha="center")

    role_handles = [Patch(facecolor=ROLE_COLORS["fitting"], edgecolor="none", label="Fitting"), Patch(facecolor=ROLE_COLORS["validation"], edgecolor="none", label="Validation"), Patch(facecolor=ROLE_COLORS["held-out"], edgecolor="none", label="Held-out (withheld)")]
    role_handles.append(Patch(facecolor="none", edgecolor="black", label="ROI-intersecting plane"))
    fig.legend(handles=role_handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.055))
    fig.text(0.5, 0.018, "Global rugs use the original registered plane sets; target masks use their ROI-intersecting positions. Counts are 51,273 (25 μm) and 6,769 (250/1000 μm).", ha="center", fontsize=7, color="#444444")
    fig.savefig(out_pdf, metadata={"Title": "Original registered global planes and observation masks", "Creator": "reviewer2_comment2_original_figures.py"})
    fig.savefig(out_png, dpi=450)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path(os.environ.get("BPMR_ORIGINAL_RAW_ROOT", RAW_ROOT_DEFAULT)))
    parser.add_argument("--authority-root", type=Path, default=Path(os.environ.get("BPMR_ORIGINAL_AUTHORITY_ROOT", AUTHORITY_ROOT_DEFAULT)))
    parser.add_argument("--registered-root", type=Path, default=Path(os.environ.get("BPMR_ORIGINAL_REGISTERED_ROOT", REGISTERED_ROOT_DEFAULT)))
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, registration, provenance = load_original_data(args.raw_root, args.authority_root, args.registered_root)
    counts = assert_mask_counts(registration)
    plot_truth_overview(arrays, provenance, args.output_dir / "dataset_ground_truth_overview.pdf", args.output_dir / "dataset_ground_truth_overview.png")
    plot_registered_masks(registration, counts, args.output_dir / "registered_observation_masks.pdf", args.output_dir / "registered_observation_masks.png")

    audit = {
        "source": "original submission registries and original raw segmentation arrays",
        "roi": {material: list(bounds) for material, bounds in ROI.items()},
        "voxel_pitch_um": VOXEL_PITCH_UM,
        "phase_fractions": {
            material: (np.bincount(arrays[material].ravel(), minlength=4) / arrays[material].size).tolist()
            for material in arrays
        },
        "mask_counts": {f"{material}_{spacing}um": count for (material, spacing), count in counts.items()},
        "provenance": provenance,
        "registry_sha256": registration["registry_sha256"],
        "outputs": [
            "dataset_ground_truth_overview.pdf",
            "dataset_ground_truth_overview.png",
            "registered_observation_masks.pdf",
            "registered_observation_masks.png",
        ],
    }
    (args.output_dir / "reviewer2_comment2_original_figures_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
