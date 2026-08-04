from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


REPO = Path(
    os.environ.get("BPMR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
ROOT = Path(
    os.environ.get("BPMR_OUTPUT_ROOT", REPO / "outputs" / "factorial_v2")
).resolve()
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import factorial_multiseed_reconstruction_v2 as reconstruction  # noqa: E402
import factorial_protocol_and_training_v2 as protocol  # noqa: E402


MATERIALS = protocol.MATERIALS
SPACINGS = protocol.SPACINGS
METHODS = protocol.METHODS
SEEDS = protocol.RECONSTRUCTION_SEEDS
PHASES = (0, 1, 2, 3)
SWEEPS = (0, 1, 2, 3)
DOMAINS = ("native", "common")
NEAR_EXTINCTION = 0.001
REPRODUCIBLE_SEED_COUNT = 3

NATIVE_CSV = (
    ROOT / "three_d_evaluation" / "factorial_3d_metrics_native_domain_v2.csv"
)
COMMON_CSV = (
    ROOT / "three_d_evaluation" / "factorial_3d_metrics_common_domain_v2.csv"
)
PHASE_CSV = (
    ROOT / "three_d_evaluation" / "factorial_phasewise_survival_v2.csv"
)
SELECTOR_CSV = (
    ROOT / "three_d_evaluation" / "factorial_selector_usage_v2.csv"
)
TPCF_CSV = ROOT / "supplementary" / "factorial_phasewise_tpcf_v2.csv"
TOPOLOGY_CSV = ROOT / "supplementary" / "factorial_topology_v2.csv"
TRANSITIONS_JSON = (
    ROOT / "supplementary" / "factorial_endpoint_transition_matrices_v2.json"
)
EVALUATION_JSON = (
    ROOT / "three_d_evaluation" / "factorial_3d_evaluation_v2.json"
)
STATS_CSV = ROOT / "statistics" / "factorial_paired_seed_statistics_v2.csv"
INTERACTION_JSON = (
    ROOT / "statistics" / "factorial_interaction_summary_v2.json"
)
TABLE_TEX = ROOT / "tables" / "factorial_main_results_v2.tex"
CLAIM_JSON = ROOT / "tables" / "factorial_claim_matrix_v2.json"
SUMMARY_MD = ROOT / "factorial_publication_ready_summary_v2.md"
SHA_CSV = ROOT / "reproducibility" / "sha256_manifest_v2.csv"
MASTER_JSON = ROOT / "manifests" / "factorial_master_manifest_v2.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).view(np.uint8)
    ).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise RuntimeError(f"FROZEN_TEXT_CONFLICT:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"EMPTY_CSV:{path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
        )
    lines.append(buffer.getvalue())
    atomic_text(path, "".join(lines))


def require_frozen_inputs() -> dict[str, Any]:
    required = (
        protocol.ANALYSIS_PLAN,
        protocol.ROI_REGISTRY,
        protocol.CHECKPOINT_FREEZE,
        reconstruction.STATE_FREEZE,
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"REQUIRED_FREEZE_MISSING:{path}")
    checkpoint = load_json(protocol.CHECKPOINT_FREEZE)
    states = load_json(reconstruction.STATE_FREEZE)
    if checkpoint.get("checkpoint_matrix_complete") is not True:
        raise RuntimeError("CHECKPOINT_MATRIX_NOT_FROZEN")
    for script, expected_sha in checkpoint.get(
        "main_script_shas", {}
    ).items():
        path = Path(script)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"FACTORIAL_MAIN_SCRIPT_SHA_CONFLICT:{path}")
    if (
        states.get("dense_truth_evaluation_authorized") is not True
        or states.get("completed_trajectories") != 90
        or states.get("valid_sweep_boundary_states") != 360
    ):
        raise RuntimeError("STATE_MATRIX_NOT_FROZEN")
    return states


def roi_record(material: str) -> dict[str, Any]:
    registry = load_json(protocol.ROI_REGISTRY)
    return next(
        row for row in registry["materials"] if row["material"] == material
    )


def load_truth(material: str) -> np.ndarray:
    record = roi_record(material)
    path = Path(record["source_file"])
    if sha256_file(path) != record["source_sha256"]:
        raise RuntimeError(f"DENSE_SOURCE_SHA_MISMATCH:{material}")
    shape = tuple(int(value) for value in record["source_shape_zyx"])
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=shape, order="C")
    z0, y0, x0 = map(int, record["roi_origin_zyx"])
    dz, dy, dx = map(int, record["roi_shape_zyx"])
    crop_zyx = np.asarray(
        raw[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]
    ).copy()
    canonical_zyx = protocol.canonicalize_labels(
        crop_zyx, np.ones(crop_zyx.shape, dtype=bool)
    ).astype(np.int16)
    if array_sha(canonical_zyx) != record["truth_sha256"]:
        raise RuntimeError(f"REGISTERED_TRUTH_SHA_MISMATCH:{material}")
    return np.transpose(canonical_zyx, (2, 1, 0)).copy()


def load_state(
    material: str, spacing: int, method: str, seed: int, sweep: int
) -> tuple[np.ndarray, dict[str, Any]]:
    path = reconstruction.state_path(material, spacing, method, seed, sweep)
    arrays, metadata = reconstruction.load_npz(path)
    state = arrays["state"].astype(np.int16, copy=False)
    if reconstruction.array_sha(state) != metadata["state_sha256"]:
        raise RuntimeError(f"STATE_SHA_MISMATCH:{path}")
    return state, metadata


def evaluation_domains(material: str) -> dict[int, dict[str, np.ndarray]]:
    observations = {}
    for spacing in SPACINGS:
        arrays, _ = reconstruction.load_observations(material, spacing)
        observations[spacing] = arrays["observed_mask"].astype(bool)
    support = reconstruction.load_observations(material, 25)[0][
        "support_mask"
    ].astype(bool)
    common = support.copy()
    for spacing in SPACINGS:
        common &= ~observations[spacing]
    densest_equivalent = support & ~observations[25]
    if not np.array_equal(common, densest_equivalent):
        raise RuntimeError(f"COMMON_DOMAIN_NOT_DENSEST_EQUIVALENT:{material}")
    return {
        spacing: {
            "native": support & ~observations[spacing],
            "common": common,
        }
        for spacing in SPACINGS
    }


def masked_fractions(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.mean(volume[mask] == phase) for phase in PHASES], dtype=float
    )


def topology(volume: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    structure = ndimage.generate_binary_structure(3, 1)
    result: dict[str, Any] = {}
    for phase in PHASES:
        mask = domain & (volume == phase)
        labels, count = ndimage.label(mask, structure=structure)
        sizes = np.bincount(labels.ravel())
        largest = int(sizes[1:].max()) if len(sizes) > 1 else 0
        percolates = []
        for axis in range(3):
            first = {
                int(value)
                for value in np.unique(np.take(labels, 0, axis=axis))
                if value
            }
            last = {
                int(value)
                for value in np.unique(
                    np.take(labels, labels.shape[axis] - 1, axis=axis)
                )
                if value
            }
            percolates.append(bool(first & last))
        result[str(phase)] = {
            "component_count": int(count),
            "largest_component_fraction": largest
            / max(int(mask.sum()), 1),
            "percolates": percolates,
        }
    return result


def tpcf_metrics(
    truth: np.ndarray, state: np.ndarray, domain: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for orientation, normal_axis in zip(protocol.ORIENTATIONS, range(3)):
        directions = [axis for axis in range(3) if axis != normal_axis]
        for phase in PHASES:
            for direction in directions:
                for lag in range(1, 13):
                    left = [slice(None)] * 3
                    right = [slice(None)] * 3
                    left[direction] = slice(0, -lag)
                    right[direction] = slice(lag, None)
                    valid = domain[tuple(left)] & domain[tuple(right)]
                    pairs = int(np.count_nonzero(valid))
                    if pairs == 0:
                        raise RuntimeError("TPCF_EMPTY_VALID_PAIR_SET")
                    truth_s2 = float(
                        np.mean(
                            (
                                (truth[tuple(left)] == phase)
                                & (truth[tuple(right)] == phase)
                            )[valid]
                        )
                    )
                    predicted_s2 = float(
                        np.mean(
                            (
                                (state[tuple(left)] == phase)
                                & (state[tuple(right)] == phase)
                            )[valid]
                        )
                    )
                    denominator = max(abs(truth_s2), 1.0 / pairs)
                    rows.append(
                        {
                            "orientation": orientation,
                            "phase": phase,
                            "direction_axis": direction,
                            "lag": lag,
                            "valid_pairs": pairs,
                            "truth_s2": truth_s2,
                            "predicted_s2": predicted_s2,
                            "normalized_error": abs(
                                predicted_s2 - truth_s2
                            )
                            / denominator,
                        }
                    )
    return (
        {
            "overall": float(
                np.mean([row["normalized_error"] for row in rows])
            ),
            "per_phase": {
                str(phase): float(
                    np.mean(
                        [
                            row["normalized_error"]
                            for row in rows
                            if row["phase"] == phase
                        ]
                    )
                )
                for phase in PHASES
            },
            "per_orientation": {
                orientation: float(
                    np.mean(
                        [
                            row["normalized_error"]
                            for row in rows
                            if row["orientation"] == orientation
                        ]
                    )
                )
                for orientation in protocol.ORIENTATIONS
            },
        },
        rows,
    )


def interface_signature(
    volume: np.ndarray, domain: np.ndarray
) -> dict[str, Any]:
    pair_counts = {
        (left, right): 0
        for left in PHASES
        for right in PHASES
        if left < right
    }
    differing = 0
    total = 0
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        valid = domain[tuple(left_slice)] & domain[tuple(right_slice)]
        left = volume[tuple(left_slice)][valid]
        right = volume[tuple(right_slice)][valid]
        total += len(left)
        differing += int(np.count_nonzero(left != right))
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        for pair in pair_counts:
            pair_counts[pair] += int(
                np.count_nonzero((lower == pair[0]) & (upper == pair[1]))
            )
    return {
        "total_density": differing / max(total, 1),
        "pairwise_density": {
            f"{left}-{right}": count / max(total, 1)
            for (left, right), count in pair_counts.items()
        },
        "valid_edges": total,
    }


def transition_matrix(
    before: np.ndarray, after: np.ndarray, domain: np.ndarray
) -> list[list[int]]:
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (before[domain], after[domain]), 1)
    return matrix.tolist()


def evaluate_state(
    truth: np.ndarray, state: np.ndarray, domain: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth_fraction = masked_fractions(truth, domain)
    predicted_fraction = masked_fractions(state, domain)
    tpcf, tpcf_rows = tpcf_metrics(truth, state, domain)
    truth_interface = interface_signature(truth, domain)
    predicted_interface = interface_signature(state, domain)
    truth_topology = topology(truth, domain)
    predicted_topology = topology(state, domain)
    survival = np.divide(
        predicted_fraction,
        truth_fraction,
        out=np.full(4, np.nan),
        where=truth_fraction > 0,
    )
    return (
        {
            "truth_phase_fractions": truth_fraction.tolist(),
            "predicted_phase_fractions": predicted_fraction.tolist(),
            "phasewise_signed_error": (
                predicted_fraction - truth_fraction
            ).tolist(),
            "phasewise_absolute_error": np.abs(
                predicted_fraction - truth_fraction
            ).tolist(),
            "phase_fraction_mae": float(
                np.mean(np.abs(predicted_fraction - truth_fraction))
            ),
            "maximum_phasewise_absolute_error": float(
                np.max(np.abs(predicted_fraction - truth_fraction))
            ),
            "phase_survival_ratio": [
                float(value) if math.isfinite(float(value)) else None
                for value in survival
            ],
            "exact_extinctions": [
                phase
                for phase in PHASES
                if np.count_nonzero(state[domain] == phase) == 0
            ],
            "near_extinctions": [
                phase
                for phase in PHASES
                if predicted_fraction[phase] < NEAR_EXTINCTION
            ],
            "severe_collapses": [
                phase
                for phase in PHASES
                if truth_fraction[phase] > 0 and survival[phase] < 0.25
            ],
            "tpcf_error": tpcf["overall"],
            "tpcf_per_phase": tpcf["per_phase"],
            "tpcf_per_orientation": tpcf["per_orientation"],
            "truth_interface": truth_interface,
            "predicted_interface": predicted_interface,
            "interface_error": abs(
                predicted_interface["total_density"]
                - truth_interface["total_density"]
            ),
            "truth_topology": truth_topology,
            "topology": predicted_topology,
        },
        tpcf_rows,
    )


def flatten_metrics(
    base: Mapping[str, Any], values: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "evaluation_voxels": base["evaluation_voxels"],
            "phase_fraction_mae": values["phase_fraction_mae"],
            "maximum_phasewise_absolute_error": values[
                "maximum_phasewise_absolute_error"
            ],
            "exact_extinction_count": len(values["exact_extinctions"]),
            "near_extinction_count": len(values["near_extinctions"]),
            "severe_collapse_count": len(values["severe_collapses"]),
            "exact_extinctions": values["exact_extinctions"],
            "near_extinctions": values["near_extinctions"],
            "severe_collapses": values["severe_collapses"],
            "tpcf_error": values["tpcf_error"],
            "interface_error": values["interface_error"],
            "runtime_seconds": metadata.get("runtime_seconds", 0.0),
            "peak_rss_kb": metadata.get("peak_rss_kb", 0),
        }
    )
    return row


def cumulative_runtime(
    material: str, spacing: int, method: str, seed: int, sweep: int
) -> float:
    total = 0.0
    for boundary in range(1, sweep + 1):
        _, metadata = load_state(
            material, spacing, method, seed, boundary
        )
        total += float(metadata.get("runtime_seconds", 0.0))
    return total


def evaluate() -> dict[str, Any]:
    freeze = require_frozen_inputs()
    all_rows: dict[str, list[dict[str, Any]]] = {
        "native": [],
        "common": [],
    }
    phase_rows: list[dict[str, Any]] = []
    tpcf_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    truth_shas: dict[str, str] = {}
    common_shas: dict[str, str] = {}

    for material in MATERIALS:
        truth = load_truth(material)
        truth_shas[material] = array_sha(truth)
        domains = evaluation_domains(material)
        common_values = [domains[s]["common"] for s in SPACINGS]
        if not all(
            np.array_equal(common_values[0], value)
            for value in common_values[1:]
        ):
            raise RuntimeError(f"COMMON_DOMAIN_MISMATCH:{material}")
        common_shas[material] = array_sha(common_values[0])
        for spacing in SPACINGS:
            initial, _ = load_state(material, spacing, "B0", SEEDS[0], 0)
            for method, seed, sweep, domain_name in itertools.product(
                METHODS, SEEDS, SWEEPS, DOMAINS
            ):
                state, metadata = load_state(
                    material, spacing, method, seed, sweep
                )
                domain = domains[spacing][domain_name]
                values, tpcf_detail = evaluate_state(truth, state, domain)
                base = {
                    "material": material,
                    "spacing_um": spacing,
                    "method": method,
                    "seed": seed,
                    "sweep": sweep,
                    "domain": domain_name,
                    "evaluation_voxels": int(domain.sum()),
                }
                metric_row = flatten_metrics(base, values, metadata)
                metric_row["cumulative_runtime_seconds"] = cumulative_runtime(
                    material, spacing, method, seed, sweep
                )
                if method == "B0" or sweep == 0:
                    metric_row["method_b0_runtime_ratio"] = (
                        1.0 if method == "B0" and sweep > 0 else None
                    )
                else:
                    b0_runtime = cumulative_runtime(
                        material, spacing, "B0", seed, sweep
                    )
                    metric_row["method_b0_runtime_ratio"] = (
                        metric_row["cumulative_runtime_seconds"]
                        / max(b0_runtime, 1e-12)
                    )
                all_rows[domain_name].append(metric_row)
                for phase in PHASES:
                    phase_rows.append(
                        {
                            **base,
                            "phase": phase,
                            "truth_fraction": values[
                                "truth_phase_fractions"
                            ][phase],
                            "predicted_fraction": values[
                                "predicted_phase_fractions"
                            ][phase],
                            "signed_error": values[
                                "phasewise_signed_error"
                            ][phase],
                            "absolute_error": values[
                                "phasewise_absolute_error"
                            ][phase],
                            "survival_ratio": values[
                                "phase_survival_ratio"
                            ][phase],
                            "exact_extinction": int(
                                phase in values["exact_extinctions"]
                            ),
                            "near_extinction": int(
                                phase in values["near_extinctions"]
                            ),
                            "severe_collapse": int(
                                phase in values["severe_collapses"]
                            ),
                        }
                    )
                    topo = values["topology"][str(phase)]
                    truth_topo = values["truth_topology"][str(phase)]
                    topology_rows.append(
                        {
                            **base,
                            "phase": phase,
                            "component_count": topo["component_count"],
                            "truth_component_count": truth_topo[
                                "component_count"
                            ],
                            "largest_component_fraction": topo[
                                "largest_component_fraction"
                            ],
                            "truth_largest_component_fraction": truth_topo[
                                "largest_component_fraction"
                            ],
                            "percolates_x": int(topo["percolates"][0]),
                            "percolates_y": int(topo["percolates"][1]),
                            "percolates_z": int(topo["percolates"][2]),
                            "truth_percolates_x": int(
                                truth_topo["percolates"][0]
                            ),
                            "truth_percolates_y": int(
                                truth_topo["percolates"][1]
                            ),
                            "truth_percolates_z": int(
                                truth_topo["percolates"][2]
                            ),
                        }
                    )
                for detail in tpcf_detail:
                    tpcf_rows.append({**base, **detail})
                if method == "SELECTIVE_V2":
                    b0_state, _ = load_state(
                        material, spacing, "B0", seed, sweep
                    )
                    r2_state, _ = load_state(
                        material, spacing, "ORIGINAL_R2", seed, sweep
                    )
                    selector_rows.append(
                        {
                            **base,
                            **metadata.get("selector", {}),
                            "v2_to_r2_hamming_fraction": float(
                                np.mean(state[domain] != r2_state[domain])
                            ),
                            "v2_to_b0_hamming_fraction": float(
                                np.mean(state[domain] != b0_state[domain])
                            ),
                            "v2_r2_exact_state_identity": int(
                                np.array_equal(state, r2_state)
                            ),
                            "v2_b0_exact_state_identity": int(
                                np.array_equal(state, b0_state)
                            ),
                        }
                    )
                if sweep == 3:
                    initial_seed, _ = load_state(
                        material, spacing, method, seed, 0
                    )
                    transitions.append(
                        {
                            **base,
                            "matrix": transition_matrix(
                                initial_seed, state, domain
                            ),
                            "endpoint_net_transition_not_full_update_path": True,
                        }
                    )

    write_csv(NATIVE_CSV, all_rows["native"])
    write_csv(COMMON_CSV, all_rows["common"])
    write_csv(PHASE_CSV, phase_rows)
    write_csv(SELECTOR_CSV, selector_rows)
    write_csv(TPCF_CSV, tpcf_rows)
    write_csv(TOPOLOGY_CSV, topology_rows)
    atomic_json(
        TRANSITIONS_JSON,
        {
            "schema": "factorial_endpoint_transition_matrices_v2",
            "records": transitions,
        },
    )
    result = {
        "schema": "factorial_3d_evaluation_v2",
        "native_metric_rows": len(all_rows["native"]),
        "common_metric_rows": len(all_rows["common"]),
        "phasewise_rows": len(phase_rows),
        "selector_rows": len(selector_rows),
        "tpcf_detail_rows": len(tpcf_rows),
        "topology_rows": len(topology_rows),
        "truth_array_shas_xyz": truth_shas,
        "common_domain_shas": common_shas,
        "COMMON_EVALUATION_DOMAIN_IDENTICAL_ACROSS_SPACINGS": True,
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
        "dense_truth_reads_during_evaluation": 2,
        "state_freeze_sha256": sha256_file(reconstruction.STATE_FREEZE),
        "finite_metrics": all(
            math.isfinite(float(row[key]))
            for domain_rows in all_rows.values()
            for row in domain_rows
            for key in (
                "phase_fraction_mae",
                "tpcf_error",
                "interface_error",
            )
        ),
        "state_freeze": {
            "completed_trajectories": freeze["completed_trajectories"],
            "valid_sweep_boundary_states": freeze[
                "valid_sweep_boundary_states"
            ],
            "state_sha_mismatches": freeze["state_sha_mismatches"],
            "fixed_observation_violations": freeze[
                "fixed_observation_violations"
            ],
            "invalid_or_sentinel_voxels": freeze[
                "invalid_or_sentinel_voxels"
            ],
            "resume_chain_violations": freeze[
                "resume_chain_violations"
            ],
            "invalid_updates": freeze["invalid_updates"],
            "inactive_phase_samples": freeze["inactive_phase_samples"],
        },
    }
    atomic_json(EVALUATION_JSON, result)
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def common_endpoint_rows() -> list[dict[str, Any]]:
    return [
        {
            **row,
            "spacing_um": int(row["spacing_um"]),
            "seed": int(row["seed"]),
            "sweep": int(row["sweep"]),
            "phase_fraction_mae": float(row["phase_fraction_mae"]),
            "tpcf_error": float(row["tpcf_error"]),
            "interface_error": float(row["interface_error"]),
        }
        for row in read_csv(COMMON_CSV)
        if int(row["sweep"]) == 3
    ]


def paired_statistics() -> list[dict[str, Any]]:
    rows = common_endpoint_rows()
    output: list[dict[str, Any]] = []
    metric_names = ("phase_fraction_mae", "tpcf_error", "interface_error")
    for material in MATERIALS:
        for method in METHODS:
            for left, right in ((25, 250), (25, 1000), (250, 1000)):
                for metric in metric_names:
                    a = {
                        row["seed"]: row[metric]
                        for row in rows
                        if row["material"] == material
                        and row["method"] == method
                        and row["spacing_um"] == left
                    }
                    b = {
                        row["seed"]: row[metric]
                        for row in rows
                        if row["material"] == material
                        and row["method"] == method
                        and row["spacing_um"] == right
                    }
                    differences = [b[seed] - a[seed] for seed in SEEDS]
                    output.append(
                        {
                            "comparison_type": "spacing",
                            "material": material,
                            "fixed_method": method,
                            "left": left,
                            "right": right,
                            "metric": metric,
                            "left_values_by_seed": [a[s] for s in SEEDS],
                            "right_values_by_seed": [b[s] for s in SEEDS],
                            **{
                                f"paired_difference_{key}": value
                                for key, value in numeric_summary(
                                    differences
                                ).items()
                            },
                            "right_improved_seed_count": sum(
                                value < 0 for value in differences
                            ),
                            "right_worsened_seed_count": sum(
                                value > 0 for value in differences
                            ),
                            "sign_consistent": int(
                                all(value >= 0 for value in differences)
                                or all(value <= 0 for value in differences)
                            ),
                        }
                    )
    for material in MATERIALS:
        for spacing in SPACINGS:
            for left, right in (
                ("B0", "ORIGINAL_R2"),
                ("B0", "SELECTIVE_V2"),
                ("ORIGINAL_R2", "SELECTIVE_V2"),
            ):
                for metric in metric_names:
                    a = {
                        row["seed"]: row[metric]
                        for row in rows
                        if row["material"] == material
                        and row["method"] == left
                        and row["spacing_um"] == spacing
                    }
                    b = {
                        row["seed"]: row[metric]
                        for row in rows
                        if row["material"] == material
                        and row["method"] == right
                        and row["spacing_um"] == spacing
                    }
                    differences = [b[seed] - a[seed] for seed in SEEDS]
                    output.append(
                        {
                            "comparison_type": "method",
                            "material": material,
                            "fixed_spacing_um": spacing,
                            "left": left,
                            "right": right,
                            "metric": metric,
                            "left_values_by_seed": [a[s] for s in SEEDS],
                            "right_values_by_seed": [b[s] for s in SEEDS],
                            **{
                                f"paired_difference_{key}": value
                                for key, value in numeric_summary(
                                    differences
                                ).items()
                            },
                            "right_improved_seed_count": sum(
                                value < 0 for value in differences
                            ),
                            "right_worsened_seed_count": sum(
                                value > 0 for value in differences
                            ),
                            "sign_consistent": int(
                                all(value >= 0 for value in differences)
                                or all(value <= 0 for value in differences)
                            ),
                        }
                    )
    write_csv(STATS_CSV, output)
    return output


def collapse_counts(
    method: str = "SELECTIVE_V2",
) -> dict[str, dict[str, dict[str, int]]]:
    phase_rows = read_csv(PHASE_CSV)
    result: dict[str, dict[str, dict[str, int]]] = {}
    for material in MATERIALS:
        result[material] = {}
        for spacing in SPACINGS:
            result[material][str(spacing)] = {}
            for phase in PHASES:
                seed_count = len(
                    {
                        int(row["seed"])
                        for row in phase_rows
                        if row["domain"] == "common"
                        and row["material"] == material
                        and int(row["spacing_um"]) == spacing
                        and row["method"] == method
                        and int(row["sweep"]) == 3
                        and int(row["phase"]) == phase
                        and int(row["severe_collapse"]) == 1
                    }
                )
                result[material][str(spacing)][str(phase)] = seed_count
    return result


def material_classification(counts: Mapping[str, Mapping[str, int]]) -> str:
    collapsed = {
        int(spacing): any(
            int(value) >= REPRODUCIBLE_SEED_COUNT
            for value in phases.values()
        )
        for spacing, phases in counts.items()
    }
    if not collapsed[25] and (collapsed[250] or collapsed[1000]):
        return "FINE_SAMPLING_STABLE_COARSE_SAMPLING_COLLAPSE"
    if all(collapsed.values()):
        return "COLLAPSE_PERSISTS_ACROSS_ALL_SAMPLING_REGIMES"
    if not any(collapsed.values()):
        return "NO_REPRODUCIBLE_COLLAPSE_ACROSS_SAMPLING_REGIMES"
    return "SEED_SENSITIVE_SAMPLING_RESPONSE"


def selector_saturation() -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in read_csv(SELECTOR_CSV)
        if row["domain"] == "common" and int(row["sweep"]) == 3
    ]
    output: dict[str, dict[str, Any]] = {}
    for material in MATERIALS:
        output[material] = {}
        for spacing in SPACINGS:
            values = [
                float(row["r2_usage_fraction"])
                for row in rows
                if row["material"] == material
                and int(row["spacing_um"]) == spacing
            ]
            output[material][str(spacing)] = {
                **numeric_summary(values),
                "saturated_seed_count_at_or_above_0_95": sum(
                    value >= 0.95 for value in values
                ),
                "reproducibly_saturated": sum(value >= 0.95 for value in values)
                >= REPRODUCIBLE_SEED_COUNT,
            }
    return output


def cross_classification(
    wf2: str, gi: str
) -> str:
    fine = "FINE_SAMPLING_STABLE_COARSE_SAMPLING_COLLAPSE"
    all_regimes = "COLLAPSE_PERSISTS_ACROSS_ALL_SAMPLING_REGIMES"
    none = "NO_REPRODUCIBLE_COLLAPSE_ACROSS_SAMPLING_REGIMES"
    if wf2 == fine and gi == fine:
        return "COARSE_SAMPLING_ASSOCIATED_FAILURE_ACROSS_MATERIALS"
    if wf2 == none and gi == fine:
        return "GI_SPECIFIC_SENSITIVITY_TO_COARSE_SAMPLING"
    if wf2 == none and gi == all_regimes:
        return "MATERIAL_DOMINANT_GI_INSTABILITY"
    if wf2 != gi and (wf2 != none or gi != none):
        return "MATERIAL_BY_SAMPLING_REGIME_INTERACTION"
    return "NO_REPRODUCIBLE_FACTORIAL_PATTERN"


def interaction_analysis() -> dict[str, Any]:
    paired_statistics()
    counts = collapse_counts()
    counts_by_method = {
        method: collapse_counts(method) for method in METHODS
    }
    wf2 = material_classification(counts["WF2"])
    gi = material_classification(counts["GI"])
    selector = selector_saturation()
    rows = common_endpoint_rows()
    phase_rows = [
        row
        for row in read_csv(PHASE_CSV)
        if row["domain"] == "common" and int(row["sweep"]) == 3
    ]
    topology_rows = [
        row
        for row in read_csv(TOPOLOGY_CSV)
        if row["domain"] == "common" and int(row["sweep"]) == 3
    ]
    descriptive_means = []
    for material, spacing, method in itertools.product(
        MATERIALS, SPACINGS, METHODS
    ):
        subset = [
            row
            for row in rows
            if row["material"] == material
            and row["spacing_um"] == spacing
            and row["method"] == method
        ]
        descriptive_means.append(
            {
                "material": material,
                "spacing_um": spacing,
                "method": method,
                **{
                    metric: numeric_summary([row[metric] for row in subset])
                    for metric in (
                        "phase_fraction_mae",
                        "tpcf_error",
                        "interface_error",
                    )
                },
            }
        )
    result = {
        "schema": "factorial_interaction_summary_v2",
        "statistical_unit": "reconstruction seed",
        "voxel_level_independence_claim": False,
        "source_material_count": 2,
        "external_material_population_inference": False,
        "collapse_definition": (
            "truth fraction > 0 and predicted/truth fraction < 0.25"
        ),
        "reproducible_definition": "at least 3 of 5 paired seeds",
        "collapse_seed_counts": counts,
        "collapse_seed_counts_by_method": counts_by_method,
        "primary_collapse_classification_method": "SELECTIVE_V2",
        "selector_saturation": selector,
        "descriptive_cell_summaries": descriptive_means,
        "phasewise_signed_error_summary": [
            {
                "material": material,
                "spacing_um": spacing,
                "method": method,
                "phase": phase,
                **numeric_summary(
                    [
                        float(row["signed_error"])
                        for row in phase_rows
                        if row["material"] == material
                        and int(row["spacing_um"]) == spacing
                        and row["method"] == method
                        and int(row["phase"]) == phase
                    ]
                ),
            }
            for material, spacing, method, phase in itertools.product(
                MATERIALS, SPACINGS, METHODS, PHASES
            )
        ],
        "new_topology_failures_vs_b0": topology_failure_summary(
            topology_rows
        ),
        "WF2_SAMPLING_RESPONSE_CLASSIFICATION": wf2,
        "GI_SAMPLING_RESPONSE_CLASSIFICATION": gi,
        "MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION": cross_classification(
            wf2, gi
        ),
        "interpretation_limit": (
            "SPARSE_SECTION_SAMPLING_REGIME jointly changes physical interval, "
            "section count, and local observation density; it is not an "
            "isolated physical-distance effect."
        ),
    }
    atomic_json(INTERACTION_JSON, result)
    return result


def topology_failure_summary(
    rows: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    output = []
    for material, spacing, method in itertools.product(
        MATERIALS, SPACINGS, ("ORIGINAL_R2", "SELECTIVE_V2")
    ):
        connectivity = 0
        percolation = 0
        for seed, phase in itertools.product(SEEDS, PHASES):
            candidate = next(
                row
                for row in rows
                if row["material"] == material
                and int(row["spacing_um"]) == spacing
                and row["method"] == method
                and int(row["seed"]) == seed
                and int(row["phase"]) == phase
            )
            b0 = next(
                row
                for row in rows
                if row["material"] == material
                and int(row["spacing_um"]) == spacing
                and row["method"] == "B0"
                and int(row["seed"]) == seed
                and int(row["phase"]) == phase
            )
            truth_present = int(candidate["truth_component_count"]) > 0
            b0_present = int(b0["component_count"]) > 0
            candidate_present = int(candidate["component_count"]) > 0
            connectivity += int(
                b0_present == truth_present
                and candidate_present != truth_present
            )
            for axis in ("x", "y", "z"):
                truth_value = int(candidate[f"truth_percolates_{axis}"])
                percolation += int(
                    int(b0[f"percolates_{axis}"]) == truth_value
                    and int(candidate[f"percolates_{axis}"])
                    != truth_value
                )
        output.append(
            {
                "material": material,
                "spacing_um": spacing,
                "method": method,
                "new_connectivity_failures": connectivity,
                "new_percolation_failures": percolation,
            }
        )
    return output


def grouped_endpoint() -> list[dict[str, Any]]:
    rows = common_endpoint_rows()
    output = []
    for material, spacing, method in itertools.product(
        MATERIALS, SPACINGS, METHODS
    ):
        subset = [
            row
            for row in rows
            if row["material"] == material
            and row["spacing_um"] == spacing
            and row["method"] == method
        ]
        output.append(
            {
                "material": material,
                "spacing_um": spacing,
                "method": method,
                **{
                    f"{metric}_{stat}": value
                    for metric in (
                        "phase_fraction_mae",
                        "tpcf_error",
                        "interface_error",
                    )
                    for stat, value in numeric_summary(
                        [row[metric] for row in subset]
                    ).items()
                },
            }
        )
    return output


def figure_data_path(letter: str) -> Path:
    return ROOT / "figures" / f"figure_{letter.lower()}_data_v2.csv"


def save_figure(fig: plt.Figure, letter: str, rows: Sequence[Mapping[str, Any]]) -> None:
    directory = ROOT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(figure_data_path(letter), rows)
    for suffix in ("pdf", "svg", "png", "tiff"):
        path = directory / f"figure_{letter.lower()}_v2.{suffix}"
        fig.savefig(
            path,
            dpi=600 if suffix == "tiff" else 300,
            bbox_inches="tight",
        )
    plt.close(fig)


def figures() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    atomic_json(
        ROOT / "figures" / "figure_contracts_v2.json",
        {
            "backend": "Python/matplotlib only",
            "journal_export": {
                "width_inches": 7.1,
                "editable_text": True,
                "formats": ["SVG", "PDF", "TIFF", "PNG"],
                "tiff_dpi": 600,
            },
            "statistical_unit": "paired reconstruction seed",
            "n_per_cell": 5,
            "spread": "sample standard deviation across seeds",
            "source_data": "one CSV per figure",
            "contracts": {
                "A": {
                    "conclusion": "The benchmark crosses both materials with all three frozen sampling regimes.",
                    "archetype": "schematic-led composite",
                },
                "B": {
                    "conclusion": "Nested regimes alter sparse-section observation density within a fixed material ROI.",
                    "archetype": "image plate + quant",
                },
                "C": {
                    "conclusion": "Two-dimensional validation behavior is reported across the complete factorial.",
                    "archetype": "quantitative grid",
                },
                "D": {
                    "conclusion": "WF2 three-dimensional response and selector usage vary across sampling regimes.",
                    "archetype": "quantitative grid",
                },
                "E": {
                    "conclusion": "GI three-dimensional response and selector usage vary across sampling regimes.",
                    "archetype": "quantitative grid",
                },
                "F": {
                    "conclusion": "The paired-seed evidence tests a descriptive material-by-regime interaction.",
                    "archetype": "quantitative grid",
                },
                "G": {
                    "conclusion": "Fixed-seed reconstructions reveal endpoint morphology and GI minority-phase retention.",
                    "archetype": "image plate + quant",
                },
            },
        },
    )
    colors = {"B0": "#4D4D4D", "ORIGINAL_R2": "#0072B2", "SELECTIVE_V2": "#D55E00"}

    design_rows = [
        {
            "material": material,
            "spacing_um": spacing,
            "method": method,
            "seed_count": len(SEEDS),
            "cumulative_sweeps": 3,
        }
        for material, spacing, method in itertools.product(
            MATERIALS, SPACINGS, METHODS
        )
    ]
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    matrix = np.arange(6).reshape(2, 3)
    ax.imshow(matrix, cmap="Greys", alpha=0.18, aspect="auto")
    for i, material in enumerate(MATERIALS):
        for j, spacing in enumerate(SPACINGS):
            ax.text(j, i, f"{material}\n{spacing} um\n3 methods x 5 seeds", ha="center", va="center")
    ax.set_xticks(range(3), [f"{value} um" for value in SPACINGS])
    ax.set_yticks(range(2), MATERIALS)
    ax.set_title("Factorial benchmark design")
    save_figure(fig, "A", design_rows)

    observation_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5))
    for i, material in enumerate(MATERIALS):
        for j, spacing in enumerate(SPACINGS):
            arrays, _ = reconstruction.load_observations(material, spacing)
            mask = arrays["observed_mask"].astype(bool)
            projection = mask.sum(axis=2)
            axes[i, j].imshow(projection.T, cmap="magma", origin="lower")
            axes[i, j].set_title(f"{material}, {spacing} um")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            observation_rows.append(
                {
                    "material": material,
                    "spacing_um": spacing,
                    "observed_voxels": int(mask.sum()),
                    "observed_fraction": float(mask.mean()),
                    "observed_mask_sha256": array_sha(mask),
                }
            )
    save_figure(fig, "B", observation_rows)

    two_d = read_csv(protocol.TWO_D_ROWS)
    two_d_rows = []
    for material, spacing, method in itertools.product(
        MATERIALS, SPACINGS, METHODS
    ):
        subset = [
            row
            for row in two_d
            if row["material"] == material
            and int(row["spacing_um"]) == spacing
            and row["method"] == method
        ]
        two_d_rows.append(
            {
                "material": material,
                "spacing_um": spacing,
                "method": method,
                "nll": float(np.mean([float(r["nll"]) for r in subset])),
                "brier": float(np.mean([float(r["brier"]) for r in subset])),
                "ece": float(np.mean([float(r["ece"]) for r in subset])),
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    for ax, metric in zip(axes, ("nll", "brier", "ece")):
        for material, marker in zip(MATERIALS, ("o", "s")):
            for method in METHODS:
                values = [
                    next(
                        row[metric]
                        for row in two_d_rows
                        if row["material"] == material
                        and row["spacing_um"] == spacing
                        and row["method"] == method
                    )
                    for spacing in SPACINGS
                ]
                ax.plot(SPACINGS, values, marker=marker, color=colors[method], label=f"{material} {method}")
        ax.set_xscale("log")
        ax.set_xlabel("Sampling regime (um)")
        ax.set_ylabel(metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", fontsize=6)
    fig.subplots_adjust(top=0.72)
    save_figure(fig, "C", two_d_rows)

    grouped = grouped_endpoint()
    selector = selector_saturation()
    for material, letter in (("WF2", "D"), ("GI", "E")):
        rows = []
        fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.25))
        for ax, metric in zip(
            axes,
            ("phase_fraction_mae", "tpcf_error", "interface_error"),
        ):
            for method in METHODS:
                subset = [
                    row
                    for row in grouped
                    if row["material"] == material and row["method"] == method
                ]
                ax.errorbar(
                    [row["spacing_um"] for row in subset],
                    [row[f"{metric}_mean"] for row in subset],
                    yerr=[row[f"{metric}_standard_deviation"] for row in subset],
                    marker="o",
                    color=colors[method],
                    capsize=2,
                    label=method,
                )
                rows.extend(
                    {
                        "material": material,
                        "spacing_um": row["spacing_um"],
                        "method": method,
                        "metric": metric,
                        "mean": row[f"{metric}_mean"],
                        "standard_deviation": row[
                            f"{metric}_standard_deviation"
                        ],
                    }
                    for row in subset
                )
            ax.set_xscale("log")
            ax.set_title(metric.replace("_", " "))
        axes[3].plot(
            SPACINGS,
            [selector[material][str(s)]["mean"] for s in SPACINGS],
            marker="o",
            color=colors["SELECTIVE_V2"],
        )
        axes[3].axhline(0.95, color="#777777", ls="--", lw=0.8)
        axes[3].set_xscale("log")
        axes[3].set_title("V2 R2 usage")
        axes[0].legend(frameon=False, fontsize=6)
        fig.suptitle(material)
        save_figure(fig, letter, rows)

    interaction = load_json(INTERACTION_JSON)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    interaction_rows = []
    for material, marker in zip(MATERIALS, ("o", "s")):
        values = []
        for spacing in SPACINGS:
            cell = next(
                row
                for row in grouped
                if row["material"] == material
                and row["spacing_um"] == spacing
                and row["method"] == "SELECTIVE_V2"
            )
            values.append(cell["phase_fraction_mae_mean"])
            interaction_rows.append(
                {
                    "material": material,
                    "spacing_um": spacing,
                    "selective_v2_phase_fraction_mae_mean": values[-1],
                }
            )
        ax.plot(SPACINGS, values, marker=marker, label=material)
    ax.set_xscale("log")
    ax.set_xlabel("Sampling regime (um)")
    ax.set_ylabel("Phase-fraction MAE")
    ax.legend(frameon=False)
    ax.set_title(interaction["MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION"].replace("_", " "))
    save_figure(fig, "F", interaction_rows)

    rendering_rows = []
    fig, axes = plt.subplots(12, 4, figsize=(7.1, 16.0))
    for i, (material, spacing) in enumerate(
        itertools.product(MATERIALS, SPACINGS)
    ):
        truth = load_truth(material)
        images = [truth]
        labels = ["Truth"]
        for method in METHODS:
            state, _ = load_state(material, spacing, method, 2701, 3)
            images.append(state)
            labels.append(method)
        for j, (image, label) in enumerate(zip(images, labels)):
            axes[i, j].imshow(
                image[:, :, image.shape[2] // 2].T,
                cmap=matplotlib.colors.ListedColormap(
                    ["#0072B2", "#E69F00", "#999999", "#009E73"]
                ),
                vmin=0,
                vmax=3,
                origin="lower",
                interpolation="nearest",
            )
            axes[i, j].set_title(
                f"{material} {spacing} um\n{label}", fontsize=7
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            rendering_rows.append(
                {
                    "material": material,
                    "spacing_um": spacing,
                    "method": label,
                    "seed": "truth" if label == "Truth" else 2701,
                    "sweep": "truth" if label == "Truth" else 3,
                    "central_z_slice": image.shape[2] // 2,
                    "array_sha256": array_sha(image),
                }
            )
    minority_row = 6
    for spacing, phase in itertools.product(SPACINGS, (0, 1)):
        truth = load_truth("GI")
        images = [truth]
        labels = ["Truth"]
        for method in METHODS:
            state, _ = load_state("GI", spacing, method, 2701, 3)
            images.append(state)
            labels.append(method)
        for j, (image, label) in enumerate(zip(images, labels)):
            mask_slice = (
                image[:, :, image.shape[2] // 2].T == phase
            ).astype(int)
            axes[minority_row, j].imshow(
                mask_slice,
                cmap=matplotlib.colors.ListedColormap(
                    ["#000000", "#CC3311" if phase == 0 else "#33BBEE"]
                ),
                vmin=0,
                vmax=1,
                origin="lower",
                interpolation="nearest",
            )
            axes[minority_row, j].set_title(
                f"GI {spacing} um, phase {phase}\n{label}", fontsize=7
            )
            axes[minority_row, j].set_xticks([])
            axes[minority_row, j].set_yticks([])
            rendering_rows.append(
                {
                    "material": "GI",
                    "spacing_um": spacing,
                    "method": label,
                    "seed": "truth" if label == "Truth" else 2701,
                    "sweep": "truth" if label == "Truth" else 3,
                    "separate_phase_render": phase,
                    "central_z_slice": image.shape[2] // 2,
                    "phase_pixels_in_slice": int(mask_slice.sum()),
                    "array_sha256": array_sha(image),
                }
            )
        minority_row += 1
    save_figure(fig, "G", rendering_rows)


def package() -> dict[str, Any]:
    evaluation = load_json(EVALUATION_JSON)
    interaction = interaction_analysis()
    figures()
    grouped = grouped_endpoint()
    tex_lines = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Material & Regime ($\mu$m) & Method & Composition MAE & TPCF error & Interface error \\",
        r"\midrule",
    ]
    for row in grouped:
        method_tex = row["method"].replace("_", r"\_")
        tex_lines.append(
            f"{row['material']} & {row['spacing_um']} & "
            f"{method_tex} & "
            f"{row['phase_fraction_mae_mean']:.4f} $\\pm$ "
            f"{row['phase_fraction_mae_standard_deviation']:.4f} & "
            f"{row['tpcf_error_mean']:.4f} $\\pm$ "
            f"{row['tpcf_error_standard_deviation']:.4f} & "
            f"{row['interface_error_mean']:.4f} $\\pm$ "
            f"{row['interface_error_standard_deviation']:.4f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_text(TABLE_TEX, "\n".join(tex_lines) + "\n")

    counts = interaction["collapse_seed_counts"]
    counts_by_method = interaction["collapse_seed_counts_by_method"]
    cell_summaries = interaction["descriptive_cell_summaries"]
    phase_summary = interaction["phasewise_signed_error_summary"]

    def cell_metric(
        material: str, spacing: int, method: str, metric: str, stat: str
    ) -> float:
        row = next(
            item
            for item in cell_summaries
            if item["material"] == material
            and item["spacing_um"] == spacing
            and item["method"] == method
        )
        return float(row[metric][stat])

    def phase_signed_mean(
        material: str, spacing: int, method: str, phase: int
    ) -> float:
        row = next(
            item
            for item in phase_summary
            if item["material"] == material
            and item["spacing_um"] == spacing
            and item["method"] == method
            and item["phase"] == phase
        )
        return float(row["mean"])

    wf2_interface_gain = {
        str(spacing): cell_metric(
            "WF2", spacing, "SELECTIVE_V2", "interface_error", "mean"
        )
        - cell_metric("WF2", spacing, "B0", "interface_error", "mean")
        for spacing in SPACINGS
    }
    wf2_tpcf_seed_sd = {
        str(spacing): cell_metric(
            "WF2",
            spacing,
            "SELECTIVE_V2",
            "tpcf_error",
            "standard_deviation",
        )
        for spacing in SPACINGS
    }
    phase2_bias = {
        material: {
            str(spacing): phase_signed_mean(
                material, spacing, "ORIGINAL_R2", 2
            )
            for spacing in SPACINGS
        }
        for material in MATERIALS
    }
    claims = {
        "schema": "factorial_claim_matrix_v2",
        "primary_domain": "common",
        "questions": {
            "GI_25_PHASE0_COLLAPSE_SEEDS": counts["GI"]["25"]["0"],
            "GI_25_PHASE1_COLLAPSE_SEEDS": counts["GI"]["25"]["1"],
            "GI_250_PHASE0_COLLAPSE_SEEDS": counts["GI"]["250"]["0"],
            "GI_250_PHASE1_COLLAPSE_SEEDS": counts["GI"]["250"]["1"],
            "GI_1000_PHASE0_COLLAPSE_SEEDS": counts["GI"]["1000"]["0"],
            "GI_1000_PHASE1_COLLAPSE_SEEDS": counts["GI"]["1000"]["1"],
            "WF2_25_COLLAPSE_SEEDS": max(counts["WF2"]["25"].values()),
            "WF2_250_COLLAPSE_SEEDS": max(counts["WF2"]["250"].values()),
            "WF2_1000_COLLAPSE_SEEDS": max(
                counts["WF2"]["1000"].values()
            ),
            "COLLAPSE_SEED_COUNTS_BY_METHOD": counts_by_method,
            "WF2_SELECTOR_SATURATION_BY_SPACING": interaction[
                "selector_saturation"
            ]["WF2"],
            "GI_SELECTOR_SATURATION_BY_SPACING": interaction[
                "selector_saturation"
            ]["GI"],
            "WF2_INTERFACE_ERROR_V2_MINUS_B0_BY_SPACING": wf2_interface_gain,
            "WF2_V2_TPCF_SEED_SD_BY_SPACING": wf2_tpcf_seed_sd,
            "ORIGINAL_R2_PHASE2_SIGNED_BIAS_BY_SPACING": phase2_bias,
            "GI_FAILURE_ATTRIBUTION": (
                "sampling-associated within this source condition"
                if interaction["GI_SAMPLING_RESPONSE_CLASSIFICATION"]
                == "FINE_SAMPLING_STABLE_COARSE_SAMPLING_COLLAPSE"
                else "not attributable to sparse sampling regime alone"
            ),
            "MATERIAL_BY_SAMPLING_INTERACTION_PRESENT": (
                interaction["MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION"]
                == "MATERIAL_BY_SAMPLING_REGIME_INTERACTION"
            ),
        },
        "WF2_SAMPLING_RESPONSE_CLASSIFICATION": interaction[
            "WF2_SAMPLING_RESPONSE_CLASSIFICATION"
        ],
        "GI_SAMPLING_RESPONSE_CLASSIFICATION": interaction[
            "GI_SAMPLING_RESPONSE_CLASSIFICATION"
        ],
        "MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION": interaction[
            "MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION"
        ],
        "limitations": [
            "Only two source material conditions are evaluated.",
            "Reconstruction seeds are paired algorithmic repeats, not independent material specimens.",
            "Sampling regime changes interval, section count, and observation density together.",
            "The 48^3 ROI can make 250 and 1000 um local observation masks coincide even though their model-fitting plane domains differ.",
            "No claim of reproducing original-paper performance is made.",
        ],
    }
    atomic_json(CLAIM_JSON, claims)

    q = claims["questions"]
    summary = f"""# Factorial material-by-sampling-regime control v2

This package evaluates two fixed source materials under three nested sparse-section sampling regimes, three frozen methods, five paired reconstruction seeds, and three cumulative pseudo-Gibbs sweeps. Cross-regime conclusions use the common unobserved-valid domain.

## Direct answers

1. GI phase-0/phase-1 severe-collapse seeds at 25 um: {q['GI_25_PHASE0_COLLAPSE_SEEDS']}/{q['GI_25_PHASE1_COLLAPSE_SEEDS']} of 5.
2. GI phase-0/phase-1 severe-collapse seeds at 250 um: {q['GI_250_PHASE0_COLLAPSE_SEEDS']}/{q['GI_250_PHASE1_COLLAPSE_SEEDS']} of 5.
3. GI phase-0/phase-1 severe-collapse seeds at 1000 um: {q['GI_1000_PHASE0_COLLAPSE_SEEDS']}/{q['GI_1000_PHASE1_COLLAPSE_SEEDS']} of 5.
4. WF2 maximum phase-collapse seed counts at 25/250/1000 um: {q['WF2_25_COLLAPSE_SEEDS']}/{q['WF2_250_COLLAPSE_SEEDS']}/{q['WF2_1000_COLLAPSE_SEEDS']} of 5.
5. WF2 V2-minus-B0 interface-error differences at 25/250/1000 um: {q['WF2_INTERFACE_ERROR_V2_MINUS_B0_BY_SPACING']}.
6. WF2 V2 TPCF seed-level standard deviations at 25/250/1000 um: {q['WF2_V2_TPCF_SEED_SD_BY_SPACING']}.
7. WF2/GI V2 selector saturation summaries: {q['WF2_SELECTOR_SATURATION_BY_SPACING']} / {q['GI_SELECTOR_SATURATION_BY_SPACING']}.
8. Original-R2 phase-2 signed marginal errors by regime: {q['ORIGINAL_R2_PHASE2_SIGNED_BIAS_BY_SPACING']}.
9. GI@1000 attribution: {q['GI_FAILURE_ATTRIBUTION']}; GI classification is {claims['GI_SAMPLING_RESPONSE_CLASSIFICATION']}.
10. Material-by-sampling classification: {claims['MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION']}.

## Interpretation boundary

`SPARSE_SECTION_SAMPLING_REGIME` is the experimental factor. It jointly changes physical section interval, the available number of sections, and observation density. With only WF2 and GI source conditions, the material-by-regime result is descriptive and does not support population-level material inference. Reconstruction seeds are not independent specimens.
"""
    atomic_text(SUMMARY_MD, summary)

    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != SHA_CSV
        and not path.name.endswith(".tmp")
    )
    sha_rows = [
        {
            "relative_path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    write_csv(SHA_CSV, sha_rows)
    feasibility = load_json(protocol.FEASIBILITY)
    reuse = load_json(protocol.REUSE_AUDIT)
    master = {
        "schema": "factorial_master_manifest_v2",
        "task": "PUBLICATION_FACTORIAL_MATERIAL_BY_SAMPLING_REGIME_CONTROL_V2",
        "factorial_root": str(ROOT),
        "FULL_2X3_FACTORIAL_FEASIBLE": feasibility[
            "FULL_2X3_FACTORIAL_FEASIBLE"
        ],
        "NESTED_SAMPLING_VERIFIED": feasibility[
            "NESTED_SAMPLING_VERIFIED"
        ],
        "SPLIT_ROLE_CONSISTENCY_VERIFIED": feasibility[
            "SPLIT_ROLE_CONSISTENCY_VERIFIED"
        ],
        "COMMON_EVALUATION_DOMAIN_IDENTICAL_ACROSS_SPACINGS": evaluation[
            "COMMON_EVALUATION_DOMAIN_IDENTICAL_ACROSS_SPACINGS"
        ],
        "FINAL_V1_REUSED_CELLS": reuse["reused_cells"],
        "FINAL_V1_RERUN_CELLS": reuse["rerun_cells"],
        "EXPECTED_TRAJECTORIES": 90,
        "COMPLETED_TRAJECTORIES": 90,
        "EXPECTED_SWEEP_BOUNDARY_STATES": 360,
        "VALID_SWEEP_BOUNDARY_STATES": 360,
        "cell_completion": {
            f"{material}_{spacing}": True
            for material in MATERIALS
            for spacing in SPACINGS
        },
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
        "WF2_SAMPLING_RESPONSE_CLASSIFICATION": interaction[
            "WF2_SAMPLING_RESPONSE_CLASSIFICATION"
        ],
        "GI_SAMPLING_RESPONSE_CLASSIFICATION": interaction[
            "GI_SAMPLING_RESPONSE_CLASSIFICATION"
        ],
        "MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION": interaction[
            "MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION"
        ],
        "GI_25_PHASE0_COLLAPSE_SEEDS": q[
            "GI_25_PHASE0_COLLAPSE_SEEDS"
        ],
        "GI_25_PHASE1_COLLAPSE_SEEDS": q[
            "GI_25_PHASE1_COLLAPSE_SEEDS"
        ],
        "GI_250_PHASE0_COLLAPSE_SEEDS": q[
            "GI_250_PHASE0_COLLAPSE_SEEDS"
        ],
        "GI_250_PHASE1_COLLAPSE_SEEDS": q[
            "GI_250_PHASE1_COLLAPSE_SEEDS"
        ],
        "GI_1000_PHASE0_COLLAPSE_SEEDS": q[
            "GI_1000_PHASE0_COLLAPSE_SEEDS"
        ],
        "GI_1000_PHASE1_COLLAPSE_SEEDS": q[
            "GI_1000_PHASE1_COLLAPSE_SEEDS"
        ],
        "WF2_25_COLLAPSE_SEEDS": q["WF2_25_COLLAPSE_SEEDS"],
        "WF2_250_COLLAPSE_SEEDS": q["WF2_250_COLLAPSE_SEEDS"],
        "WF2_1000_COLLAPSE_SEEDS": q["WF2_1000_COLLAPSE_SEEDS"],
        "WF2_SELECTOR_SATURATION_BY_SPACING": q[
            "WF2_SELECTOR_SATURATION_BY_SPACING"
        ],
        "GI_SELECTOR_SATURATION_BY_SPACING": q[
            "GI_SELECTOR_SATURATION_BY_SPACING"
        ],
        "factorial_3d_metrics": str(COMMON_CSV),
        "factorial_statistics": str(STATS_CSV),
        "factorial_claim_matrix": str(CLAIM_JSON),
        "factorial_publication_summary": str(SUMMARY_MD),
        "sha256_manifest": str(SHA_CSV),
        "NEW_METHOD_DEVELOPMENT": False,
        "ALGORITHM_OR_THRESHOLD_CHANGED": False,
        "ORIGINAL_PAPER_PERFORMANCE_REPRODUCED": False,
        "TECHNICAL_INTEGRITY_PASS": (
            evaluation["finite_metrics"]
            and evaluation[
                "COMMON_EVALUATION_DOMAIN_IDENTICAL_ACROSS_SPACINGS"
            ]
            and all(
                evaluation["state_freeze"][key] == 0
                for key in (
                    "state_sha_mismatches",
                    "fixed_observation_violations",
                    "invalid_or_sentinel_voxels",
                    "resume_chain_violations",
                    "invalid_updates",
                    "inactive_phase_samples",
                )
            )
        ),
        "FACTORIAL_PUBLICATION_PACKAGE_COMPLETE": True,
        "CLASSIFICATION": interaction[
            "MATERIAL_BY_SAMPLING_FINAL_CLASSIFICATION"
        ],
    }
    atomic_json(MASTER_JSON, master)
    return master


def status() -> dict[str, Any]:
    return {
        "evaluation_complete": EVALUATION_JSON.is_file(),
        "statistics_complete": INTERACTION_JSON.is_file(),
        "figures_complete": all(
            (ROOT / "figures" / f"figure_{letter}_v2.png").is_file()
            for letter in "abcdefg"
        ),
        "package_complete": MASTER_JSON.is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("evaluate", "package", "all", "status")
    )
    args = parser.parse_args()
    if args.command == "evaluate":
        result = evaluate()
    elif args.command == "package":
        result = package()
    elif args.command == "all":
        evaluate()
        result = package()
    else:
        result = status()
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
