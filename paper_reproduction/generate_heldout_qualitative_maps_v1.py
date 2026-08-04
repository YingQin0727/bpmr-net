#!/usr/bin/env python3
"""Export real held-out per-pixel probability maps from frozen factorial models."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm


REPO = Path(
    os.environ.get("BPMR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(REPO / "src"))
import factorial_protocol_and_training_v2 as f  # noqa: E402


OUT = Path(
    os.environ.get(
        "BPMR_HELDOUT_OUTPUT_ROOT",
        REPO / "outputs/factorial_v2/heldout_section_qualitative_maps_v1",
    )
).resolve()
CELLS = (("WF2", 25), ("WF2", 250), ("GI", 25), ("GI", 250))
ORIENTATION = "z"
RADIUS = 4
FEATURE_CHUNK = 16384
EPSILON = 1e-12
PHASE_COLORS = ("#0072B2", "#E69F00", "#8C8C8C", "#009E73")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def selected_plane(material: str, spacing: int) -> dict:
    planes = f.load_orientation_planes(material, spacing, ORIENTATION)
    heldout = sorted(
        (plane for plane in planes if plane["role"] == "withheld"),
        key=lambda plane: (int(plane["index"]), str(plane["plane_id"])),
    )
    if not heldout:
        raise RuntimeError(f"NO_WITHHELD_PLANES:{material}:{spacing}:{ORIENTATION}")
    return heldout[(len(heldout) - 1) // 2]


def event_positions(plane: dict) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(plane["support"], dtype=bool)
    inner = support[RADIUS:-RADIUS, RADIUS:-RADIUS]
    selected = np.flatnonzero(inner.ravel())
    width = inner.shape[1]
    rows = selected // width + RADIUS
    columns = selected % width + RADIUS
    return rows.astype(np.int64), columns.astype(np.int64)


def feature_chunk(
    labels: np.ndarray,
    support: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    neighbour_rows = rows[:, None] + offsets[None, :, 0]
    neighbour_columns = columns[:, None] + offsets[None, :, 1]
    valid = support[neighbour_rows, neighbour_columns]
    neighbour_labels = labels[neighbour_rows, neighbour_columns]
    if np.any(valid & ((neighbour_labels < 0) | (neighbour_labels >= 4))):
        raise RuntimeError("ACTIVE_ILLEGAL_PHASE_IN_FEATURE")
    local = np.zeros((len(rows), 80, 5), dtype=np.float32)
    event_indices, slot_indices = np.nonzero(valid)
    local[
        event_indices,
        slot_indices,
        neighbour_labels[event_indices, slot_indices],
    ] = 1.0
    local[:, :, 4] = valid
    return local.reshape(len(rows), 400)


def load_models(material: str, spacing: int, paired_seed: int):
    b0_models = []
    bpmr_models = []
    cell = f.condition_id(material, spacing)
    for template_phase in range(4):
        seed = f.model_seed(paired_seed, cell, ORIENTATION, template_phase)
        b0_dir = f.checkpoint_dir(
            material, spacing, ORIENTATION, paired_seed, template_phase, "B0"
        )
        bpmr_dir = f.checkpoint_dir(
            material,
            spacing,
            ORIENTATION,
            paired_seed,
            template_phase,
            "ORIGINAL_R2",
        )
        b0_models.append(f.load_b0_model(b0_dir, seed))
        bpmr_models.append(f.load_r2_model(bpmr_dir, b0_dir, seed))
    return b0_models, bpmr_models


def predict_models(models, features: np.ndarray, method: str) -> np.ndarray:
    predictions = []
    x = torch.from_numpy(np.ascontiguousarray(features))
    with torch.no_grad():
        for model in models:
            model.eval()
            if method == "B0":
                raw = model(x)
                probability = raw / torch.clamp(
                    raw.sum(dim=1, keepdim=True), min=EPSILON
                )
            else:
                _, probability, _, _ = model(x)
            predictions.append(probability.cpu().numpy())
    return f.eq9_average(predictions).astype(np.float32, copy=False)


def infer_cell(material: str, spacing: int) -> tuple[dict, dict]:
    plane = selected_plane(material, spacing)
    labels = np.asarray(plane["labels"], dtype=np.int16)
    support = np.asarray(plane["support"], dtype=bool)
    rows, columns = event_positions(plane)
    _, offsets = f.production_event_helpers()
    height, width = labels.shape
    seed_b0 = np.full(
        (len(f.TRAINING_SEEDS), height, width, 4), np.nan, dtype=np.float32
    )
    seed_bpmr = np.full_like(seed_b0, np.nan)
    checkpoint_records = []

    for seed_index, paired_seed in enumerate(f.TRAINING_SEEDS):
        b0_models, bpmr_models = load_models(material, spacing, paired_seed)
        for template_phase in range(4):
            for method in ("B0", "ORIGINAL_R2"):
                directory = f.checkpoint_dir(
                    material,
                    spacing,
                    ORIENTATION,
                    paired_seed,
                    template_phase,
                    method,
                )
                checkpoint = directory / "best_checkpoint.pt"
                checkpoint_records.append(
                    {
                        "paired_seed": paired_seed,
                        "template_phase": template_phase,
                        "method": method,
                        "relative_path": str(checkpoint.relative_to(REPO)),
                        "sha256": sha256(checkpoint),
                    }
                )
        for start in range(0, len(rows), FEATURE_CHUNK):
            stop = min(start + FEATURE_CHUNK, len(rows))
            features = feature_chunk(
                labels,
                support,
                rows[start:stop],
                columns[start:stop],
                offsets,
            )
            seed_b0[
                seed_index, rows[start:stop], columns[start:stop], :
            ] = predict_models(b0_models, features, "B0")
            seed_bpmr[
                seed_index, rows[start:stop], columns[start:stop], :
            ] = predict_models(bpmr_models, features, "BPMR")
        print(
            f"INFERRED {material} {spacing}um seed={paired_seed} "
            f"events={len(rows)}",
            flush=True,
        )

    b0 = np.nanmean(seed_b0, axis=0)
    bpmr = np.nanmean(seed_bpmr, axis=0)
    b0 /= np.nansum(b0, axis=2, keepdims=True)
    bpmr /= np.nansum(bpmr, axis=2, keepdims=True)
    valid = np.zeros((height, width), dtype=bool)
    valid[rows, columns] = True
    target = labels[rows, columns]
    b0_true = np.full((height, width), np.nan, dtype=np.float32)
    bpmr_true = np.full_like(b0_true, np.nan)
    b0_true[rows, columns] = b0[rows, columns, target]
    bpmr_true[rows, columns] = bpmr[rows, columns, target]
    improvement = np.full_like(b0_true, np.nan)
    improvement[rows, columns] = np.log(
        np.clip(bpmr_true[rows, columns], EPSILON, 1.0)
    ) - np.log(np.clip(b0_true[rows, columns], EPSILON, 1.0))

    cell_id = f"{material}_{spacing}um_z_{int(plane['index'])}"
    cell_dir = OUT / "arrays" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cell_dir / "heldout_pixel_probabilities.npz",
        truth_labels=labels,
        valid_center_mask=valid,
        support_mask=support,
        b0_class_probability=b0,
        bpmr_net_class_probability=bpmr,
        b0_class_probability_by_training_seed=seed_b0,
        bpmr_net_class_probability_by_training_seed=seed_bpmr,
        b0_true_label_probability=b0_true,
        bpmr_net_true_label_probability=bpmr_true,
        log_probability_improvement=improvement,
        row_coordinates=rows,
        column_coordinates=columns,
        training_seeds=np.asarray(f.TRAINING_SEEDS, dtype=np.int32),
    )
    np.save(cell_dir / "truth_phase_label.npy", labels)
    np.save(cell_dir / "b0_true_label_probability.npy", b0_true)
    np.save(cell_dir / "bpmr_net_true_label_probability.npy", bpmr_true)
    np.save(cell_dir / "log_probability_improvement.npy", improvement)

    metadata = {
        "schema": "heldout_section_pixel_probabilities_v1",
        "material": material,
        "spacing_um": spacing,
        "orientation": ORIENTATION,
        "plane_id": plane["plane_id"],
        "global_plane_index": int(plane["index"]),
        "split_role": plane["role"],
        "selection_rule": (
            "lower median by global_plane_index among z-orientation withheld "
            "planes; selection is independent of predictions and metrics"
        ),
        "valid_center_count": int(valid.sum()),
        "shape": [height, width],
        "training_seeds": list(f.TRAINING_SEEDS),
        "template_phase_aggregation": "arithmetic mean of four phase templates",
        "training_seed_aggregation": "arithmetic mean of paired seeds 1701 and 1702",
        "epsilon_for_log": EPSILON,
        "label_sha256": plane["label_sha256"],
        "support_sha256": plane["support_sha256"],
        "checkpoints": checkpoint_records,
    }
    atomic_json(cell_dir / "metadata.json", metadata)
    metrics = {
        "cell_id": cell_id,
        "material": material,
        "spacing_um": spacing,
        "orientation": ORIENTATION,
        "global_plane_index": int(plane["index"]),
        "valid_center_count": int(valid.sum()),
        "b0_nll": float(-np.mean(np.log(np.clip(b0_true[valid], EPSILON, 1.0)))),
        "bpmr_net_nll": float(
            -np.mean(np.log(np.clip(bpmr_true[valid], EPSILON, 1.0)))
        ),
        "mean_log_probability_improvement": float(np.mean(improvement[valid])),
        "fraction_pixels_improved": float(np.mean(improvement[valid] > 0)),
        "npz_relative_path": str(
            (cell_dir / "heldout_pixel_probabilities.npz").relative_to(OUT)
        ),
    }
    return metadata, metrics


def plot_preview(metrics: list[dict]) -> None:
    phase_cmap = ListedColormap(PHASE_COLORS)
    phase_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), phase_cmap.N)
    probability_cmap = "viridis"
    improvement_cmap = "RdBu_r"
    loaded = []
    maximum = 0.0
    for row in metrics:
        path = OUT / row["npz_relative_path"]
        data = np.load(path)
        loaded.append(data)
        values = data["log_probability_improvement"]
        maximum = max(maximum, float(np.nanpercentile(np.abs(values), 99)))
    maximum = max(maximum, 0.1)
    fig, axes = plt.subplots(
        len(metrics),
        4,
        figsize=(7.2, 7.5),
        constrained_layout=True,
        dpi=200,
    )
    titles = (
        "Truth phase",
        "B0 true-label p",
        "BPMR-Net true-label p",
        "log p improvement",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=9)
    phase_image = None
    probability_image = None
    improvement_image = None
    for row_index, (row, data) in enumerate(zip(metrics, loaded)):
        truth = np.ma.masked_where(
            ~data["valid_center_mask"], data["truth_labels"]
        )
        phase_image = axes[row_index, 0].imshow(
            truth, cmap=phase_cmap, norm=phase_norm, interpolation="nearest"
        )
        probability_image = axes[row_index, 1].imshow(
            data["b0_true_label_probability"],
            cmap=probability_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[row_index, 2].imshow(
            data["bpmr_net_true_label_probability"],
            cmap=probability_cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        improvement_image = axes[row_index, 3].imshow(
            data["log_probability_improvement"],
            cmap=improvement_cmap,
            norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
            interpolation="nearest",
        )
        axes[row_index, 0].set_ylabel(
            f"{row['material']} | {row['spacing_um']} um\nz={row['global_plane_index']}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
    phase_bar = fig.colorbar(
        phase_image,
        ax=axes[:, 0],
        orientation="horizontal",
        fraction=0.025,
        pad=0.02,
        ticks=(0, 1, 2, 3),
    )
    phase_bar.set_label("Phase label", fontsize=8)
    probability_bar = fig.colorbar(
        probability_image,
        ax=axes[:, 1:3],
        orientation="horizontal",
        fraction=0.025,
        pad=0.02,
    )
    probability_bar.set_label("Probability", fontsize=8)
    improvement_bar = fig.colorbar(
        improvement_image,
        ax=axes[:, 3],
        orientation="horizontal",
        fraction=0.025,
        pad=0.02,
    )
    improvement_bar.set_label("True-label log p gain", fontsize=8)
    preview_dir = OUT / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_dir / "heldout_section_qualitative_maps.png", dpi=300)
    fig.savefig(preview_dir / "heldout_section_qualitative_maps.pdf")
    plt.close(fig)
    for data in loaded:
        data.close()


def write_readme(metrics: list[dict]) -> None:
    lines = [
        "# Real held-out section pixel probability maps",
        "",
        "This package contains newly generated inference outputs from the frozen",
        "factorial-v2 B0 and ORIGINAL_R2 (BPMR-Net) checkpoints. It does not",
        "contain reconstructed or illustrative probability values.",
        "",
        "For each of WF2/GI at 25/250 um, the displayed plane is the lower median",
        "global plane index among z-orientation planes assigned the formal",
        "`withheld` split role. This rule is independent of model predictions.",
        "",
        "Each `arrays/<cell>/heldout_pixel_probabilities.npz` contains:",
        "",
        "- `truth_labels`: integer phase label image (0-3)",
        "- `valid_center_mask`: pixels where the model was evaluated",
        "- `b0_class_probability`: mean B0 probabilities, shape H x W x 4",
        "- `bpmr_net_class_probability`: mean BPMR-Net probabilities, H x W x 4",
        "- per-training-seed class-probability arrays (seed axis: 1701, 1702)",
        "- B0/BPMR-Net true-label probability maps",
        "- pixelwise log-probability improvement",
        "",
        "NaN denotes pixels outside the valid evaluation-center mask. Separate",
        "`.npy` files expose the four requested maps directly.",
        "",
        "The preview is descriptive visualization, not an additional inferential",
        "test. Formal aggregate held-out metrics remain the primary evaluation.",
        "",
        "## Included cells",
        "",
    ]
    for row in metrics:
        lines.append(
            f"- {row['cell_id']}: n={row['valid_center_count']}, "
            f"B0 NLL={row['b0_nll']:.6f}, "
            f"BPMR-Net NLL={row['bpmr_net_nll']:.6f}"
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    f.require_feasible()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    source_dir = OUT / "source"
    source_dir.mkdir()
    shutil.copy2(Path(__file__), source_dir / Path(__file__).name)
    metrics = []
    metadata = []
    for material, spacing in CELLS:
        cell_metadata, cell_metrics = infer_cell(material, spacing)
        metadata.append(cell_metadata)
        metrics.append(cell_metrics)
    with (OUT / "selected_sections_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    atomic_json(
        OUT / "generation_manifest.json",
        {
            "schema": "heldout_section_qualitative_maps_generation_v1",
            "cells": metadata,
            "formal_method_names": {"baseline": "B0", "bpmr_net": "ORIGINAL_R2"},
            "source_script": "source/generate_heldout_qualitative_maps_v1.py",
        },
    )
    plot_preview(metrics)
    write_readme(metrics)
    manifest_rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(OUT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    with (OUT / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("relative_path", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"COMPLETE {OUT}", flush=True)


if __name__ == "__main__":
    main()
