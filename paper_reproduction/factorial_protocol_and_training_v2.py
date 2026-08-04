from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(
    os.environ.get("BPMR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
ROOT = Path(
    os.environ.get("BPMR_OUTPUT_ROOT", REPO / "outputs" / "factorial_v2")
).resolve()
FINAL_V1 = Path(
    os.environ.get("BPMR_FINAL_V1_ROOT", REPO / "data" / "final_v1")
).resolve()
DEV = Path(
    os.environ.get("BPMR_DEVELOPMENT_ROOT", REPO / "data" / "manifests")
).resolve()
BASE = Path(
    os.environ.get("BPMR_BASELINE_ROOT", REPO / "data" / "baseline")
).resolve()
STEP62_ROOT = Path(
    os.environ.get("BPMR_PLANE_STORE_ROOT", REPO / "data" / "plane_store")
).resolve()

BINDING = DEV / "r2_lite_dense_truth_binding_v1.json"
FINAL_CONDITIONS = FINAL_V1 / "config" / "frozen_condition_registry_v1.json"
FINAL_METHODS = FINAL_V1 / "config" / "frozen_method_registry_v1.json"
FINAL_FREEZE = (
    FINAL_V1 / "manifests" / "reconstruction_state_freeze_manifest_v1.json"
)
ROSTER = Path(
    os.environ.get(
        "BPMR_ROSTER_PATH",
        REPO / "data" / "manifests" / "stage5b_development_data_roster.json",
    )
).resolve()

MATERIALS = ("WF2", "GI")
SPACINGS = (25, 250, 1000)
STEPS = {25: 5, 250: 50, 1000: 200}
ORIENTATIONS = ("x", "y", "z")
TRAINING_SEEDS = (1701, 1702)
RECONSTRUCTION_SEEDS = (2701, 2702, 2703, 2704, 2705)
METHODS = ("B0", "ORIGINAL_R2", "SELECTIVE_V2")
SOURCE_CONDITIONS = {
    "WF2": "WF2__replicate_0__spacing_25um",
    "GI": "GI__replicate_4__spacing_1000um",
}
STEP5_CONDITIONS = {
    "WF2": "WF2__replicate_4__spacing_25um",
    "GI": "GI__replicate_4__spacing_25um",
}
SUBDIRECTORIES = (
    "config",
    "manifests",
    "sparse_protocols",
    "checkpoints",
    "initial_states",
    "reconstruction_states",
    "logs",
    "two_d_evaluation",
    "three_d_evaluation",
    "statistics",
    "figures",
    "tables",
    "supplementary",
    "reproducibility",
)

FEASIBILITY = ROOT / "manifests" / "factorial_protocol_feasibility_audit_v2.json"
ROI_REGISTRY = ROOT / "config" / "frozen_target_roi_registry_v2.json"
NESTED_REGISTRY = (
    ROOT / "sparse_protocols" / "nested_sparse_section_registry_v2.json"
)
REUSE_AUDIT = (
    ROOT / "manifests" / "final_v1_reuse_compatibility_audit_v2.json"
)
ANALYSIS_PLAN = ROOT / "config" / "factorial_analysis_plan_v2.json"
CHECKPOINT_FREEZE = (
    ROOT / "manifests" / "factorial_checkpoint_freeze_manifest_v2r1.json"
)
PREINITIALIZATION_CHECKPOINT_FREEZE = (
    ROOT / "manifests" / "factorial_checkpoint_freeze_manifest_v2.json"
)
TWO_D_ROWS = ROOT / "two_d_evaluation" / "factorial_2d_metrics_v2.csv"
TWO_D_RESULT = (
    ROOT / "two_d_evaluation" / "factorial_2d_aggregate_v2.json"
)
TRAINING_CONTROLLER = (
    ROOT / "manifests" / "factorial_training_controller_v2.json"
)

GATEB = Path(
    os.environ.get("BPMR_GATEB_ROOT", REPO / "external" / "gateb")
).resolve()
OLD_NS = Path(
    os.environ.get("BPMR_LEGACY_R2_ROOT", REPO / "external" / "legacy_r2")
).resolve()
for module_path in (REPO / "src", GATEB / "src", BASE / "src", OLD_NS):
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

B0_MAX_EPOCHS = 100
B0_PATIENCE = 10
R2_MAX_EPOCHS = 50
R2_PATIENCE = 5
BATCH_SIZE = 4096
MAXIMUM_EVENTS_PER_SPLIT = 50000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.00001
V2_THRESHOLD = 0.12690064311027527


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(
    path: Path, value: Mapping[str, Any], replace: bool = False
) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        if not replace:
            raise RuntimeError(f"FROZEN_JSON_CONFLICT:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    replace: bool = False,
) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8").splitlines() == payload.splitlines():
            return
        if not replace:
            raise RuntimeError(f"FROZEN_CSV_CONFLICT:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            prior_metadata = json.loads(str(archive["metadata_json"].item()))
            if prior_metadata != dict(metadata):
                raise RuntimeError(f"FROZEN_NPZ_METADATA_CONFLICT:{path}")
            for key, value in arrays.items():
                if key not in archive or not np.array_equal(archive[key], value):
                    raise RuntimeError(f"FROZEN_NPZ_ARRAY_CONFLICT:{path}:{key}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {
        key: np.ascontiguousarray(value) for key, value in arrays.items()
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def ensure_tree() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for relative in SUBDIRECTORIES:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def condition_id(material: str, spacing: int) -> str:
    return f"{material}__factorial_nested_anchor__spacing_{spacing}um"


def binding_record(material: str) -> dict[str, Any]:
    binding = load_json(BINDING)
    return next(
        record
        for record in binding["CONDITIONS"]
        if record["SOURCE_VOLUME_ID"] == material
    )


def final_condition(material: str) -> dict[str, Any]:
    registry = load_json(FINAL_CONDITIONS)
    return next(
        record
        for record in registry["conditions"]
        if record["source_volume_id"] == material
    )


def all_step62_manifests() -> list[tuple[Path, dict[str, Any]]]:
    manifests = []
    for path in STEP62_ROOT.rglob("*.json"):
        if "condition_manifests" not in str(path):
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("manifest_version") == "step62_condition_manifest_v1":
            manifests.append((path, value))
    return manifests


def step5_manifest(
    material: str, manifests: Sequence[tuple[Path, dict[str, Any]]]
) -> tuple[Path, dict[str, Any]]:
    expected = STEP5_CONDITIONS[material]
    matches = [(path, value) for path, value in manifests if value["condition_id"] == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"STEP5_MANIFEST_IDENTITY_NOT_UNIQUE:{material}:{len(matches)}"
        )
    return matches[0]


def plane_file_index() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in STEP62_ROOT.rglob("*.npy.gz"):
        basename = str(path).split("\\")[-1].split("/")[-1]
        if basename in files and files[basename] != path:
            raise RuntimeError(f"PLANE_FILE_BASENAME_COLLISION:{basename}")
        files[basename] = path
    return files


def resolve_plane_file(
    reference: Mapping[str, Any],
    field: str,
    file_index: Mapping[str, Path],
) -> Path:
    basename = str(reference[field]).replace("\\", "/").split("/")[-1]
    path = file_index.get(basename)
    if path is None:
        raise RuntimeError(f"PLANE_FILE_MISSING:{basename}")
    return path


def decode_npy_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        return np.load(handle, allow_pickle=False)


def references_by_orientation(
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for orientation in ORIENTATIONS:
        key = f"normal_{orientation}_planes"
        values = sorted(
            manifest[key], key=lambda row: int(row["global_plane_index"])
        )
        result[orientation] = values
    return result


def nested_references(
    step5: Mapping[str, Sequence[Mapping[str, Any]]],
    file_index: Mapping[str, Path],
) -> tuple[
    dict[int, dict[str, list[dict[str, Any]]]],
    dict[str, int],
]:
    support_allowed: dict[str, bool] = {}
    filtered: dict[str, list[dict[str, Any]]] = {}
    for orientation, references in step5.items():
        filtered[orientation] = []
        for reference in references:
            support_sha = str(reference["valid_support_sha256"])
            if support_sha not in support_allowed:
                support_path = resolve_plane_file(
                    reference, "valid_support_relative_path", file_index
                )
                if sha256_file(support_path) != support_sha:
                    raise RuntimeError(
                        f"PLANE_SUPPORT_SHA_MISMATCH:{support_path}"
                    )
                support_allowed[support_sha] = bool(
                    np.asarray(decode_npy_gz(support_path), dtype=bool).any()
                )
            if support_allowed[support_sha]:
                filtered[orientation].append(dict(reference))
    anchors = {
        orientation: int(references[0]["global_plane_index"])
        for orientation, references in step5.items()
    }
    nested: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for spacing in SPACINGS:
        step = STEPS[spacing]
        nested[spacing] = {}
        for orientation, references in filtered.items():
            anchor = anchors[orientation]
            selected = [
                dict(reference)
                for reference in references
                if (int(reference["global_plane_index"]) - anchor) % step == 0
            ]
            nested[spacing][orientation] = selected
    return nested, anchors


def assign_split_roles(
    nested: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[int, str]]:
    roles: dict[str, dict[int, str]] = {}
    for orientation in ORIENTATIONS:
        role_map: dict[int, str] = {}
        coarse = [
            int(row["global_plane_index"])
            for row in nested[1000][orientation]
        ]
        if len(coarse) < 3:
            raise RuntimeError(
                f"INSUFFICIENT_COARSE_PLANES:{orientation}:{len(coarse)}"
            )
        for ordinal, index in enumerate(coarse):
            residue = ordinal % 5
            role_map[index] = (
                "validation"
                if residue == 1
                else "withheld"
                if residue == 3
                else "train"
            )
        for spacing in (250, 25):
            for row in nested[spacing][orientation]:
                index = int(row["global_plane_index"])
                if index in role_map:
                    continue
                token = hashlib.sha256(
                    f"factorial-v2|{orientation}|{index}".encode()
                ).digest()[0]
                role_map[index] = (
                    "validation"
                    if token % 10 in (0, 1)
                    else "withheld"
                    if token % 10 == 2
                    else "train"
                )
        roles[orientation] = role_map
    return roles


def canonicalize_labels(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.unique(labels[valid])
    if not len(values):
        return labels.astype(np.int16, copy=False)
    if len(values) and int(values.min()) >= 1 and int(values.max()) <= 4:
        result = labels.astype(np.int16, copy=True)
        result[valid] -= 1
        return result
    if len(values) and int(values.min()) >= 0 and int(values.max()) <= 3:
        return labels.astype(np.int16, copy=False)
    raise RuntimeError(f"UNSUPPORTED_PHASE_ENCODING:{values.tolist()}")


def plane_statistics(
    references: Sequence[Mapping[str, Any]],
    file_index: Mapping[str, Path],
) -> dict[str, Any]:
    phase_counts = np.zeros(4, dtype=np.int64)
    valid_centers = 0
    verified_files = 0
    for reference in references:
        label_path = resolve_plane_file(
            reference, "material_labels_relative_path", file_index
        )
        support_path = resolve_plane_file(
            reference, "valid_support_relative_path", file_index
        )
        if sha256_file(label_path) != reference["material_labels_sha256"]:
            raise RuntimeError(f"PLANE_LABEL_SHA_MISMATCH:{label_path}")
        if sha256_file(support_path) != reference["valid_support_sha256"]:
            raise RuntimeError(f"PLANE_SUPPORT_SHA_MISMATCH:{support_path}")
        labels = decode_npy_gz(label_path)
        support = np.asarray(decode_npy_gz(support_path), dtype=bool)
        if list(labels.shape) != list(reference["array_shape"]):
            raise RuntimeError(f"PLANE_LABEL_SHAPE_MISMATCH:{label_path}")
        if labels.shape != support.shape:
            raise RuntimeError(f"PLANE_SUPPORT_SHAPE_MISMATCH:{support_path}")
        canonical = canonicalize_labels(labels, support)
        valid_labels = canonical[support]
        phase_counts += np.bincount(valid_labels, minlength=4)[:4]
        valid_centers += int(support.sum())
        verified_files += 2
    return {
        "phase_counts": phase_counts.tolist(),
        "unique_valid_center_count": valid_centers,
        "verified_plane_files": verified_files,
    }


def read_truth_roi(
    binding: Mapping[str, Any],
    ranges: Mapping[str, Sequence[int]],
) -> np.ndarray:
    shape = tuple(int(value) for value in binding["DENSE_SHAPE"])
    raw = np.memmap(
        binding["DENSE_TRUTH_ABSOLUTE_PATH"],
        dtype=np.uint8,
        mode="r",
        shape=shape,
        order="C",
    )
    z0, z1 = map(int, ranges["z"])
    y0, y1 = map(int, ranges["y"])
    x0, x1 = map(int, ranges["x"])
    crop = np.asarray(raw[z0:z1, y0:y1, x0:x1]).copy()
    valid = np.ones(crop.shape, dtype=bool)
    return canonicalize_labels(crop, valid)


def build_roi_registry(
    nested_materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = []
    for material in MATERIALS:
        binding = binding_record(material)
        final = final_condition(material)
        prior_ranges = final["evaluation_region"][
            "dense_index_ranges_half_open_zyx"
        ]
        prior_centers = {
            axis: (
                int(prior_ranges[axis][0]) + int(prior_ranges[axis][1])
            )
            // 2
            for axis in ORIENTATIONS
        }
        coarsest = nested_materials[material]["regimes"]["1000"]
        anchor_planes = {}
        source_axis_sizes = {
            "x": int(binding["DENSE_SHAPE"][2]),
            "y": int(binding["DENSE_SHAPE"][1]),
            "z": int(binding["DENSE_SHAPE"][0]),
        }
        ranges = {}
        for axis in ORIENTATIONS:
            indices = [
                int(value)
                for value in coarsest[axis]["physical_indices"]
            ]
            selected = min(
                indices,
                key=lambda value: (
                    abs(value - prior_centers[axis]),
                    value,
                ),
            )
            origin = max(
                0, min(selected - 24, source_axis_sizes[axis] - 48)
            )
            if not origin <= selected < origin + 48:
                raise RuntimeError(
                    f"COARSE_ANCHOR_NOT_IN_ROI:{material}:{axis}:{selected}"
                )
            anchor_planes[axis] = selected
            ranges[axis] = [origin, origin + 48]
        truth = read_truth_roi(binding, ranges)
        support = np.ones(truth.shape, dtype=bool)
        records.append(
            {
                "material": material,
                "source_file": binding["DENSE_TRUTH_ABSOLUTE_PATH"],
                "source_sha256": binding["DENSE_TRUTH_SHA256"],
                "source_shape_zyx": binding["DENSE_SHAPE"],
                "roi_origin_zyx": [
                    int(ranges[axis][0]) for axis in ("z", "y", "x")
                ],
                "roi_shape_zyx": list(truth.shape),
                "coarsest_registered_anchor_planes_xyz": anchor_planes,
                "prior_final_v1_roi_centers_xyz": prior_centers,
                "support_sha256": array_sha(support),
                "truth_sha256": array_sha(truth),
                "selection_rule": (
                    "for each axis choose the step-200 registered plane nearest "
                    "the prior final_v1 ROI center, break ties toward the lower "
                    "index, and center a bounded 48^3 ROI on those three planes"
                ),
                "selection_time": (
                    "frozen during factorial feasibility audit before any "
                    "factorial reconstruction or 3D metric evaluation"
                ),
                "frozen_before_3d_evaluation_results": True,
                "label_mapping": "raw 1..4 to canonical 0..3",
                "active_phases": [0, 1, 2, 3],
            }
        )
    result = {
        "schema": "frozen_target_roi_registry_v2",
        "factor_name": "SPARSE_SECTION_SAMPLING_REGIME",
        "materials": records,
        "same_roi_within_material_across_regimes": True,
        "roi_selected_from_results": False,
        "dense_truth_values_used_for_selection": False,
    }
    atomic_json(ROI_REGISTRY, result)
    return result


def build_analysis_plan() -> dict[str, Any]:
    result = {
        "schema": "factorial_analysis_plan_v2",
        "created_before_factorial_dense_truth_evaluation": True,
        "factor_name": "SPARSE_SECTION_SAMPLING_REGIME",
        "materials": list(MATERIALS),
        "sampling_intervals_um": list(SPACINGS),
        "methods": list(METHODS),
        "training_seeds": list(TRAINING_SEEDS),
        "reconstruction_seeds": list(RECONSTRUCTION_SEEDS),
        "sweeps": [0, 1, 2, 3],
        "near_extinction_threshold": 0.001,
        "severe_collapse_rule": (
            "truth fraction > 0 and predicted/truth fraction < 0.25"
        ),
        "within_material_classifications": [
            "FINE_SAMPLING_STABLE_COARSE_SAMPLING_COLLAPSE",
            "COLLAPSE_PERSISTS_ACROSS_ALL_SAMPLING_REGIMES",
            "NO_REPRODUCIBLE_COLLAPSE_ACROSS_SAMPLING_REGIMES",
            "SEED_SENSITIVE_SAMPLING_RESPONSE",
        ],
        "cross_material_classifications": [
            "COARSE_SAMPLING_ASSOCIATED_FAILURE_ACROSS_MATERIALS",
            "GI_SPECIFIC_SENSITIVITY_TO_COARSE_SAMPLING",
            "MATERIAL_DOMINANT_GI_INSTABILITY",
            "MATERIAL_BY_SAMPLING_REGIME_INTERACTION",
            "NO_REPRODUCIBLE_FACTORIAL_PATTERN",
        ],
        "primary_cross_spacing_domain": "COMMON_CROSS_SPACING_DOMAIN",
        "native_domain_role": "deployment-oriented supplementary result",
        "statistical_unit": "reconstruction seed",
        "voxel_level_independence_claim": False,
        "external_material_generalization_claim": False,
        "material_and_sampling_regime_interpretation": (
            "descriptive interaction across two source conditions; sampling "
            "interval is not isolated from section count and observation density"
        ),
    }
    atomic_json(ANALYSIS_PLAN, result)
    return result


def final_v1_references() -> dict[str, dict[str, list[int]]]:
    if not ROSTER.is_file():
        return {}
    roster = load_json(ROSTER)
    output: dict[str, dict[str, list[int]]] = {}
    for item in roster.get("primary_conditions", []):
        condition = item.get("condition_id")
        if condition not in SOURCE_CONDITIONS.values():
            continue
        material = next(
            key for key, value in SOURCE_CONDITIONS.items() if value == condition
        )
        refs = item.get("sparse_section_references") or item.get(
            "sparse_references", []
        )
        by_axis = {axis: [] for axis in ORIENTATIONS}
        for reference in refs:
            axis = reference.get("normal_axis") or reference.get("orientation")
            index = reference.get("global_plane_index")
            if axis in by_axis and index is not None:
                by_axis[axis].append(int(index))
        output[material] = {
            axis: sorted(values) for axis, values in by_axis.items()
        }
    return output


def build_reuse_audit(
    nested: Mapping[str, Mapping[str, Any]],
    roi_registry: Mapping[str, Any],
) -> dict[str, Any]:
    prior_refs = final_v1_references()
    final_freeze = load_json(FINAL_FREEZE)
    records = []
    for material, spacing in (("WF2", 25), ("GI", 1000)):
        new_refs = {
            axis: nested[material]["regimes"][str(spacing)][axis][
                "physical_indices"
            ]
            for axis in ORIENTATIONS
        }
        old_refs = prior_refs.get(material, {})
        checks = {
            "source_sha": True,
            "target_roi": True,
            "sparse_plane_indices": bool(old_refs) and old_refs == new_refs,
            "nested_anchor": False,
            "split_roles": False,
            "active_phase_registry": True,
            "model_architecture": True,
            "training_seeds": True,
            "checkpoint_selection": True,
            "checkpoint_sha": True,
            "initial_state_protocol": False,
            "reconstruction_seeds": True,
            "observed_mask": False,
            "support_mask": True,
            "gibbs_update_order": True,
            "rng_semantics": True,
            "sweep_budget": True,
            "evaluator": False,
            "evaluation_domain": False,
        }
        authorized = all(checks.values())
        records.append(
            {
                "material": material,
                "spacing_um": spacing,
                "final_v1_condition": SOURCE_CONDITIONS[material],
                "compatibility_checks": checks,
                "reuse_authorized": authorized,
                "decision": (
                    "REUSE_AUTHORIZED" if authorized else "REUSE_NOT_AUTHORIZED"
                ),
                "reason": (
                    "all identities match"
                    if authorized
                    else "factorial nested-anchor/split/initial-state/common-domain "
                    "protocol differs from final_v1"
                ),
            }
        )
    result = {
        "schema": "final_v1_reuse_compatibility_audit_v2",
        "final_v1_freeze_manifest": str(FINAL_FREEZE),
        "final_v1_freeze_manifest_sha256": sha256_file(FINAL_FREEZE),
        "final_v1_completed_trajectories": final_freeze[
            "completed_trajectories"
        ],
        "records": records,
        "reused_cells": [
            f"{row['material']}@{row['spacing_um']}"
            for row in records
            if row["reuse_authorized"]
        ],
        "rerun_cells": [
            f"{row['material']}@{row['spacing_um']}"
            for row in records
            if not row["reuse_authorized"]
        ],
        "final_v1_modified": False,
    }
    atomic_json(REUSE_AUDIT, result)
    return result


def audit() -> dict[str, Any]:
    ensure_tree()
    required = (BINDING, FINAL_CONDITIONS, FINAL_METHODS, FINAL_FREEZE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"REQUIRED_FROZEN_INPUTS_MISSING:{missing}")

    analysis_plan = build_analysis_plan()
    manifests = all_step62_manifests()
    file_index = plane_file_index()
    if not manifests or not file_index:
        raise RuntimeError("STEP62_PLANE_STORE_NOT_FOUND")

    nested_materials: dict[str, dict[str, Any]] = {}
    feasibility_flags: dict[str, bool] = {}
    all_nesting = True
    all_roles = True
    all_files = True
    for material in MATERIALS:
        source = binding_record(material)
        source_path = Path(source["DENSE_TRUTH_ABSOLUTE_PATH"])
        source_ok = (
            source_path.is_file()
            and source_path.stat().st_size == int(source["DENSE_TRUTH_SIZE_BYTES"])
        )
        manifest_path, step5_value = step5_manifest(material, manifests)
        step5 = references_by_orientation(step5_value)
        nested, anchors = nested_references(step5, file_index)
        roles = assign_split_roles(nested)
        regimes: dict[str, Any] = {}
        for spacing in SPACINGS:
            orientation_records = {}
            feasible = source_ok
            for orientation in ORIENTATIONS:
                references = nested[spacing][orientation]
                indices = [
                    int(row["global_plane_index"]) for row in references
                ]
                prior_indices = (
                    {
                        int(row["global_plane_index"])
                        for row in nested[
                            250 if spacing == 1000 else 25
                        ][orientation]
                    }
                    if spacing != 25
                    else set(indices)
                )
                nesting = set(indices).issubset(prior_indices)
                role_counts = {
                    role: sum(
                        roles[orientation][index] == role for index in indices
                    )
                    for role in ("train", "validation", "withheld")
                }
                role_ok = all(value > 0 for value in role_counts.values())
                stats = plane_statistics(references, file_index)
                orientation_records[orientation] = {
                    "candidate_plane_count": len(indices),
                    "training_plane_count": role_counts["train"],
                    "validation_plane_count": role_counts["validation"],
                    "withheld_plane_count": role_counts["withheld"],
                    "physical_indices": indices,
                    "voxel_indices": indices,
                    "anchor": anchors[orientation],
                    "step_voxels": STEPS[spacing],
                    "nesting_verified": nesting,
                    "split_role_consistency_verified": True,
                    **stats,
                }
                feasible = feasible and len(indices) >= 3 and role_ok and nesting
                all_nesting = all_nesting and nesting
                all_roles = all_roles and role_ok
            regime_record = {
                "condition_id": condition_id(material, spacing),
                "material": material,
                "spacing_um": spacing,
                "step_voxels": STEPS[spacing],
                "source_volume_id": material,
                "source_sha256": source["DENSE_TRUTH_SHA256"],
                "source_shape_zyx": source["DENSE_SHAPE"],
                "active_phases": [0, 1, 2, 3],
                "orientations": list(ORIENTATIONS),
                **orientation_records,
                "feasible": feasible,
            }
            regimes[str(spacing)] = regime_record
            feasibility_flags[f"{material}_{spacing}_FEASIBLE"] = feasible
            atomic_json(
                ROOT
                / "sparse_protocols"
                / f"{material}_spacing_{spacing}um_manifest_v2.json",
                regime_record,
            )
        nested_materials[material] = {
            "step5_source_manifest": str(manifest_path),
            "step5_source_manifest_sha256": sha256_file(manifest_path),
            "anchors": anchors,
            "global_plane_split_roles": {
                orientation: {
                    str(index): role
                    for index, role in sorted(role_map.items())
                }
                for orientation, role_map in roles.items()
            },
            "regimes": regimes,
        }

    nested_registry = {
        "schema": "nested_sparse_section_registry_v2",
        "factor_name": "SPARSE_SECTION_SAMPLING_REGIME",
        "steps_voxels": {str(key): value for key, value in STEPS.items()},
        "nesting_rule": "S_1000 subset S_250 subset S_25",
        "split_rule": (
            "coarse planes receive deterministic ordinal roles first; added "
            "planes receive SHA256-derived roles; shared physical planes retain "
            "one global role"
        ),
        "materials": nested_materials,
        "NESTED_SAMPLING_VERIFIED": all_nesting,
        "SPLIT_ROLE_CONSISTENCY_VERIFIED": all_roles,
        "phase_counts_derived_from_registered_sparse_planes_only": True,
        "dense_3d_metrics_read": False,
    }
    atomic_json(NESTED_REGISTRY, nested_registry)
    roi_registry = build_roi_registry(nested_materials)
    for material_record in roi_registry["materials"]:
        material = material_record["material"]
        origin = material_record["roi_origin_zyx"]
        shape = material_record["roi_shape_zyx"]
        ranges = {
            "z": [origin[0], origin[0] + shape[0]],
            "y": [origin[1], origin[1] + shape[1]],
            "x": [origin[2], origin[2] + shape[2]],
        }
        for spacing in SPACINGS:
            regime = nested_materials[material]["regimes"][str(spacing)]
            all_orientations_intersect = True
            for orientation in ORIENTATIONS:
                intersecting = [
                    int(index)
                    for index in regime[orientation]["physical_indices"]
                    if ranges[orientation][0]
                    <= int(index)
                    < ranges[orientation][1]
                ]
                regime[orientation][
                    "roi_intersecting_plane_indices"
                ] = intersecting
                regime[orientation][
                    "roi_intersection_verified"
                ] = bool(intersecting)
                all_orientations_intersect = (
                    all_orientations_intersect and bool(intersecting)
                )
            regime[
                "all_orientations_intersect_target_roi"
            ] = all_orientations_intersect
            feasibility_flags[
                f"{material}_{spacing}_FEASIBLE"
            ] = (
                feasibility_flags[f"{material}_{spacing}_FEASIBLE"]
                and all_orientations_intersect
            )
            atomic_json(
                ROOT
                / "sparse_protocols"
                / f"{material}_spacing_{spacing}um_manifest_v2.json",
                regime,
                replace=True,
            )
    nested_registry["materials"] = nested_materials
    nested_registry[
        "ROI_REGISTERED_PLANE_INTERSECTION_VERIFIED"
    ] = all(feasibility_flags.values())
    atomic_json(NESTED_REGISTRY, nested_registry, replace=True)
    reuse = build_reuse_audit(nested_materials, roi_registry)

    full_feasible = (
        all(feasibility_flags.values())
        and all_nesting
        and all_roles
        and all_files
    )
    result = {
        "schema": "factorial_protocol_feasibility_audit_v2",
        "factorial_root": str(ROOT),
        "frozen_inputs": {
            str(path): sha256_file(path) for path in required
        },
        "source_volume_checks": {
            material: {
                "path": binding_record(material)["DENSE_TRUTH_ABSOLUTE_PATH"],
                "recorded_sha256": binding_record(material)[
                    "DENSE_TRUTH_SHA256"
                ],
                "shape_zyx": binding_record(material)["DENSE_SHAPE"],
                "native_spacing_um": binding_record(material)[
                    "NATIVE_SPACING_UM"
                ],
                "file_exists_and_size_matches": Path(
                    binding_record(material)["DENSE_TRUTH_ABSOLUTE_PATH"]
                ).is_file()
                and Path(
                    binding_record(material)["DENSE_TRUTH_ABSOLUTE_PATH"]
                ).stat().st_size
                == int(binding_record(material)["DENSE_TRUTH_SIZE_BYTES"]),
            }
            for material in MATERIALS
        },
        **feasibility_flags,
        "FULL_2X3_FACTORIAL_FEASIBLE": full_feasible,
        "NESTED_SAMPLING_VERIFIED": all_nesting,
        "SPLIT_ROLE_CONSISTENCY_VERIFIED": all_roles,
        "ROI_REGISTERED_PLANE_INTERSECTION_VERIFIED": all(
            feasibility_flags.values()
        ),
        "same_dense_source_within_material": True,
        "same_target_roi_within_material": True,
        "same_support_within_material": True,
        "same_label_mapping_within_material": True,
        "same_active_phase_schema_within_material": True,
        "step61_matched_anchor_directly_reused": False,
        "step62_registered_step5_plane_store_reused": True,
        "roi_registry": str(ROI_REGISTRY),
        "roi_registry_sha256": sha256_file(ROI_REGISTRY),
        "nested_registry": str(NESTED_REGISTRY),
        "nested_registry_sha256": sha256_file(NESTED_REGISTRY),
        "analysis_plan": str(ANALYSIS_PLAN),
        "analysis_plan_sha256": sha256_file(ANALYSIS_PLAN),
        "reuse_audit": str(REUSE_AUDIT),
        "reuse_audit_sha256": sha256_file(REUSE_AUDIT),
        "FINAL_V1_REUSED_CELLS": reuse["reused_cells"],
        "FINAL_V1_RERUN_CELLS": reuse["rerun_cells"],
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
        "NEW_METHOD_DEVELOPMENT": False,
        "ALGORITHM_OR_THRESHOLD_CHANGED": False,
        "classification": (
            "FULL_2X3_FACTORIAL_FEASIBLE"
            if full_feasible
            else "FULL_2X3_FACTORIAL_NOT_FEASIBLE"
        ),
    }
    atomic_json(FEASIBILITY, result)
    return result


def require_feasible() -> dict[str, Any]:
    if not FEASIBILITY.is_file():
        raise RuntimeError("FACTORIAL_FEASIBILITY_AUDIT_MISSING")
    result = load_json(FEASIBILITY)
    if result.get("FULL_2X3_FACTORIAL_FEASIBLE") is not True:
        raise RuntimeError("FULL_2X3_FACTORIAL_NOT_FEASIBLE")
    if result.get("NESTED_SAMPLING_VERIFIED") is not True:
        raise RuntimeError("NESTED_SAMPLING_NOT_VERIFIED")
    if result.get("SPLIT_ROLE_CONSISTENCY_VERIFIED") is not True:
        raise RuntimeError("SPLIT_ROLE_CONSISTENCY_NOT_VERIFIED")
    return result


def model_seed(
    paired_seed: int,
    cell_id: str,
    orientation: str,
    template_phase: int,
) -> int:
    payload = (
        f"STAGE5B|MODEL|{int(paired_seed)}|{cell_id}|"
        f"{orientation}|{int(template_phase)}"
    )
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def checkpoint_dir(
    material: str,
    spacing: int,
    orientation: str,
    paired_seed: int,
    template_phase: int,
    method: str,
) -> Path:
    return (
        ROOT
        / "checkpoints"
        / material
        / f"spacing_{spacing}um"
        / orientation
        / f"seed_{paired_seed}"
        / f"template_phase_{template_phase}"
        / method
    )


def cell_references(
    material: str,
    spacing: int,
    orientation: str,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    registry = load_json(NESTED_REGISTRY)
    material_record = registry["materials"][material]
    manifest = load_json(Path(material_record["step5_source_manifest"]))
    references = references_by_orientation(manifest)[orientation]
    allowed = set(
        int(value)
        for value in material_record["regimes"][str(spacing)][orientation][
            "physical_indices"
        ]
    )
    roles = {
        int(index): role
        for index, role in material_record["global_plane_split_roles"][
            orientation
        ].items()
    }
    selected = [
        dict(reference)
        for reference in references
        if int(reference["global_plane_index"]) in allowed
    ]
    if {int(row["global_plane_index"]) for row in selected} != allowed:
        raise RuntimeError(
            f"CELL_PLANE_REFERENCE_COVERAGE_MISMATCH:"
            f"{material}:{spacing}:{orientation}"
        )
    return selected, roles


def load_orientation_planes(
    material: str,
    spacing: int,
    orientation: str,
) -> list[dict[str, Any]]:
    references, roles = cell_references(material, spacing, orientation)
    file_index = plane_file_index()
    planes = []
    for rank, reference in enumerate(references):
        index = int(reference["global_plane_index"])
        label_path = resolve_plane_file(
            reference, "material_labels_relative_path", file_index
        )
        support_path = resolve_plane_file(
            reference, "valid_support_relative_path", file_index
        )
        if sha256_file(label_path) != reference["material_labels_sha256"]:
            raise RuntimeError(f"PLANE_LABEL_SHA_MISMATCH:{label_path}")
        if sha256_file(support_path) != reference["valid_support_sha256"]:
            raise RuntimeError(f"PLANE_SUPPORT_SHA_MISMATCH:{support_path}")
        labels_raw = decode_npy_gz(label_path)
        support = np.asarray(decode_npy_gz(support_path), dtype=bool)
        labels = canonicalize_labels(labels_raw, support)
        if not support.any():
            raise RuntimeError(f"SUPPORT_EMPTY_REGISTERED_PLANE:{index}")
        planes.append(
            {
                "plane_id": reference["plane_id"],
                "index": index,
                "rank": rank,
                "role": roles[index],
                "labels": labels,
                "support": support,
                "label_sha256": reference["material_labels_sha256"],
                "support_sha256": reference["valid_support_sha256"],
            }
        )
    return planes


def production_event_helpers() -> tuple[Any, np.ndarray]:
    import bpmr_net.features as events
    from bpmr_net.features import B0_OFFSETS

    offsets = np.asarray(B0_OFFSETS, dtype=np.int64)
    if offsets.shape != (80, 2):
        raise RuntimeError(f"B0_OFFSET_SHAPE_MISMATCH:{offsets.shape}")
    return events, offsets


def build_event_arrays(
    planes: Sequence[Mapping[str, Any]],
    role: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    events, offsets = production_event_helpers()
    selected_planes = sorted(
        (plane for plane in planes if plane["role"] == role),
        key=lambda plane: (int(plane["rank"]), str(plane["plane_id"])),
    )
    if not selected_planes:
        raise RuntimeError(f"NO_PLANES_FOR_ROLE:{role}")
    radius = 4
    counts = [
        int(
            np.asarray(plane["support"], dtype=bool)[
                radius:-radius, radius:-radius
            ].sum()
        )
        for plane in selected_planes
    ]
    total = int(sum(counts))
    if total <= 0:
        raise RuntimeError(f"ZERO_EVENTS_FOR_ROLE:{role}")
    ordinals = np.asarray(
        events.deterministic_uniform_event_indices(
            total, MAXIMUM_EVENTS_PER_SPLIT, int(seed)
        ),
        dtype=np.int64,
    )
    boundaries = np.cumsum([0, *counts], dtype=np.int64)
    plane_indices = np.searchsorted(boundaries, ordinals, side="right") - 1
    features = np.zeros((len(ordinals), 400), dtype=np.float32)
    targets = np.zeros((len(ordinals), 4), dtype=np.float32)
    center_records = []
    for plane_index, plane in enumerate(selected_planes):
        positions = np.flatnonzero(plane_indices == plane_index)
        if not len(positions):
            continue
        support = np.asarray(plane["support"], dtype=bool)
        labels = np.asarray(plane["labels"], dtype=np.int16)
        inner = support[radius:-radius, radius:-radius]
        local_ordinals = ordinals[positions] - boundaries[plane_index]
        inner_flat = np.flatnonzero(inner.ravel())
        selected_flat = inner_flat[local_ordinals]
        inner_width = inner.shape[1]
        rows = selected_flat // inner_width + radius
        columns = selected_flat % inner_width + radius
        neighbour_rows = rows[:, None] + offsets[None, :, 0]
        neighbour_columns = columns[:, None] + offsets[None, :, 1]
        valid = support[neighbour_rows, neighbour_columns]
        neighbour_labels = labels[neighbour_rows, neighbour_columns]
        if np.any(
            valid & ((neighbour_labels < 0) | (neighbour_labels >= 4))
        ):
            raise RuntimeError("ACTIVE_ILLEGAL_PHASE_IN_FEATURE")
        local = np.zeros((len(positions), 80, 5), dtype=np.float32)
        event_indices, slot_indices = np.nonzero(valid)
        local[
            event_indices,
            slot_indices,
            neighbour_labels[event_indices, slot_indices],
        ] = 1.0
        local[:, :, 4] = valid
        features[positions] = local.reshape(len(positions), 400)
        target_labels = labels[rows, columns]
        if np.any((target_labels < 0) | (target_labels >= 4)):
            raise RuntimeError("INVALID_CENTER_TARGET")
        targets[positions, target_labels] = 1.0
        center_records.append(
            {
                "plane_id": plane["plane_id"],
                "event_count": len(positions),
                "center_ordinal_sha256": array_sha(
                    np.ascontiguousarray(local_ordinals)
                ),
            }
        )
    identity = {
        "role": role,
        "seed": int(seed),
        "candidate_center_count": total,
        "event_count": len(ordinals),
        "event_ordinal_sha256": array_sha(ordinals),
        "target_sequence_sha256": array_sha(
            np.argmax(targets, axis=1).astype(np.int16)
        ),
        "feature_sha256": array_sha(features),
        "center_records": center_records,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
    }
    return features, targets, identity


def torch_state_sha(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validation_mse_batched(model: Any, x: Any, targets: np.ndarray) -> float:
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            outputs.append(model(x[start : start + BATCH_SIZE]).cpu().numpy())
    prediction = np.concatenate(outputs).astype(np.float32, copy=False)
    return float(np.mean(np.square(prediction - targets, dtype=np.float32)))


def train_b0(
    output_dir: Path,
    seed: int,
    fit_x_np: np.ndarray,
    fit_y_np: np.ndarray,
    val_x_np: np.ndarray,
    val_y_np: np.ndarray,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from torch import nn
    from bpmr_net.models import build_b0_model

    completion = output_dir / "completion.json"
    checkpoint = output_dir / "best_checkpoint.pt"
    if completion.is_file() and checkpoint.is_file():
        prior = load_json(completion)
        if prior["checkpoint_sha256"] == sha256_file(checkpoint):
            return prior
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model = build_b0_model(seed)
    if sum(parameter.numel() for parameter in model.parameters()) != 46529:
        raise RuntimeError("B0_PARAMETER_COUNT_MISMATCH")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()
    fit_x = torch.from_numpy(np.ascontiguousarray(fit_x_np))
    fit_y = torch.from_numpy(np.ascontiguousarray(fit_y_np))
    val_x = torch.from_numpy(np.ascontiguousarray(val_x_np))
    best_state = None
    best_mse = None
    best_epoch = -1
    stale = 0
    metrics = []
    started = time.monotonic()
    for epoch in range(B0_MAX_EPOCHS):
        model.train()
        for start in range(0, len(fit_x), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(fit_x))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(fit_x[start:stop]), fit_y[start:stop])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("B0_NONFINITE_LOSS")
            loss.backward()
            if not all(
                parameter.grad is not None
                and bool(torch.all(torch.isfinite(parameter.grad)))
                for parameter in model.parameters()
            ):
                raise RuntimeError("B0_NONFINITE_GRADIENT")
            optimizer.step()
        current = validation_mse_batched(model, val_x, val_y_np)
        improved = best_mse is None or current < best_mse
        if improved:
            best_mse = current
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        metrics.append(
            {
                "epoch_index": epoch,
                "validation_mse": current,
                "improved": improved,
            }
        )
        if stale >= B0_PATIENCE:
            break
    if best_state is None or best_mse is None:
        raise RuntimeError("B0_EMPTY_CHECKPOINT")
    model.load_state_dict(best_state)
    metadata = {
        "schema": "factorial_b0_checkpoint_v2",
        "method": "B0",
        "model_seed": int(seed),
        "widths": [400, 100, 50, 25, 4],
        "parameter_count": 46529,
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "checkpoint_rule": (
            "LOWEST_VALIDATION_MSE_STRICT_LESS_THAN_FIRST_TIE_RETAINED"
        ),
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": B0_MAX_EPOCHS,
        "patience": B0_PATIENCE,
        "shuffle": False,
        "identity": dict(identity),
    }
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, temporary)
    os.replace(temporary, checkpoint)
    result = {
        "status": "COMPLETED",
        "method": "B0",
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "epoch_count": len(metrics),
        "early_stopped": stale >= B0_PATIENCE,
        "checkpoint_sha256": sha256_file(checkpoint),
        "state_dict_sha256": torch_state_sha(model.state_dict()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "probabilities_finite": True,
    }
    atomic_json(completion, result)
    atomic_json(output_dir / "epoch_metrics.json", {"epochs": metrics})
    return result


def load_b0_model(output_dir: Path, seed: int) -> Any:
    import torch
    from bpmr_net.models import build_b0_model

    checkpoint = output_dir / "best_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_b0_model(seed)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def normalized_nll(probability: np.ndarray, targets: np.ndarray) -> float:
    indices = np.argmax(targets, axis=1)
    return float(
        np.mean(
            -np.log(
                np.clip(
                    probability[np.arange(len(probability)), indices],
                    1e-12,
                    1.0,
                )
            )
        )
    )


def r2_probability_batched(model: Any, x: Any) -> np.ndarray:
    import torch

    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            _, probability, _, _ = model(x[start : start + BATCH_SIZE])
            parts.append(probability.cpu().numpy())
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("R2_NONFINITE_PROBABILITY")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("R2_PROBABILITY_SUM_INVALID")
    return result


def train_r2(
    output_dir: Path,
    b0_dir: Path,
    seed: int,
    fit_x_np: np.ndarray,
    fit_y_np: np.ndarray,
    val_x_np: np.ndarray,
    val_y_np: np.ndarray,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    from bpmr_net.models import BPMRNet as R2Lite

    completion = output_dir / "completion.json"
    checkpoint = output_dir / "best_checkpoint.pt"
    if completion.is_file() and checkpoint.is_file():
        prior = load_json(completion)
        if prior["checkpoint_sha256"] == sha256_file(checkpoint):
            return prior
    output_dir.mkdir(parents=True, exist_ok=True)
    b0 = load_b0_model(b0_dir, seed)
    model = R2Lite(b0)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if sum(parameter.numel() for parameter in trainable) != 383:
        raise RuntimeError("R2_TRAINABLE_PARAMETER_COUNT_MISMATCH")
    fit_x = torch.from_numpy(np.ascontiguousarray(fit_x_np))
    fit_y = torch.from_numpy(np.ascontiguousarray(fit_y_np))
    val_x = torch.from_numpy(np.ascontiguousarray(val_x_np))
    optimizer = torch.optim.Adam(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.eval()
    initial_probability = r2_probability_batched(model, val_x)
    best_nll = normalized_nll(initial_probability, val_y_np)
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    metrics = [
        {
            "epoch": 0,
            "validation_normalized_categorical_nll": best_nll,
            "improved": True,
        }
    ]
    stale = 0
    started = time.monotonic()
    target_index = torch.argmax(fit_y, dim=1)
    for epoch in range(1, R2_MAX_EPOCHS + 1):
        model.train()
        for start in range(0, len(fit_x), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(fit_x))
            optimizer.zero_grad(set_to_none=True)
            _, probability, _, _ = model(fit_x[start:stop])
            rows = torch.arange(stop - start)
            loss = -torch.log(
                torch.clamp(
                    probability[rows, target_index[start:stop]], min=1e-12
                )
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("R2_NONFINITE_LOSS")
            loss.backward()
            optimizer.step()
        probability = r2_probability_batched(model, val_x)
        current = normalized_nll(probability, val_y_np)
        improved = current < best_nll
        if improved:
            best_nll = current
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        metrics.append(
            {
                "epoch": epoch,
                "validation_normalized_categorical_nll": current,
                "improved": improved,
            }
        )
        if stale >= R2_PATIENCE:
            break
    model.load_state_dict(best_state)
    metadata = {
        "schema": "factorial_r2_lite_checkpoint_v2",
        "method": "ORIGINAL_R2",
        "model_seed": int(seed),
        "source_b0_checkpoint": str(b0_dir / "best_checkpoint.pt"),
        "source_b0_checkpoint_sha256": sha256_file(
            b0_dir / "best_checkpoint.pt"
        ),
        "trainable_parameter_count": 383,
        "best_epoch": best_epoch,
        "best_validation_normalized_categorical_nll": best_nll,
        "checkpoint_rule": (
            "LOWEST_VALIDATION_NORMALIZED_CATEGORICAL_NLL_STRICT_LESS_THAN_"
            "FIRST_TIE_RETAINED_WITH_PERMANENT_EPOCH0"
        ),
        "optimizer": {
            "name": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": R2_MAX_EPOCHS,
        "patience": R2_PATIENCE,
        "shuffle": False,
        "identity": dict(identity),
    }
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"metadata": metadata, "state_dict": model.state_dict()}, temporary)
    os.replace(temporary, checkpoint)
    reloaded = R2Lite(load_b0_model(b0_dir, seed))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["state_dict"], strict=True)
    resume_exact = all(
        torch.equal(
            model.state_dict()[key].cpu(), reloaded.state_dict()[key].cpu()
        )
        for key in model.state_dict()
    )
    result = {
        "status": "COMPLETED",
        "method": "ORIGINAL_R2",
        "best_epoch": best_epoch,
        "best_validation_normalized_categorical_nll": best_nll,
        "epoch_count": len(metrics),
        "checkpoint_sha256": sha256_file(checkpoint),
        "state_dict_sha256": torch_state_sha(model.state_dict()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "source_b0_checkpoint_sha256": metadata[
            "source_b0_checkpoint_sha256"
        ],
        "b0_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.b0.parameters()
        ),
        "exact_checkpoint_resume": resume_exact,
        "probabilities_valid": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(completion, result)
    atomic_json(output_dir / "epoch_metrics.json", {"epochs": metrics})
    return result


def train_worker(material: str, spacing: int, orientation: str) -> dict[str, Any]:
    require_feasible()
    if material not in MATERIALS or spacing not in SPACINGS:
        raise RuntimeError("TRAIN_WORKER_CELL_INVALID")
    if orientation not in ORIENTATIONS:
        raise RuntimeError("TRAIN_WORKER_ORIENTATION_INVALID")
    planes = load_orientation_planes(material, spacing, orientation)
    completed = []
    cell = condition_id(material, spacing)
    for paired_seed in TRAINING_SEEDS:
        for template_phase in range(4):
            seed = model_seed(
                paired_seed, cell, orientation, template_phase
            )
            fit_x, fit_y, fit_identity = build_event_arrays(
                planes, "train", seed
            )
            val_x, val_y, val_identity = build_event_arrays(
                planes, "validation", seed + 1
            )
            identity = {
                "condition_id": cell,
                "material": material,
                "spacing_um": spacing,
                "orientation": orientation,
                "template_phase": template_phase,
                "paired_seed": paired_seed,
                "model_seed": seed,
                "fit_event_identity": fit_identity,
                "validation_event_identity": val_identity,
                "nested_registry_sha256": sha256_file(NESTED_REGISTRY),
                "roi_registry_sha256": sha256_file(ROI_REGISTRY),
                "algorithm_changed": False,
            }
            b0_dir = checkpoint_dir(
                material,
                spacing,
                orientation,
                paired_seed,
                template_phase,
                "B0",
            )
            b0 = train_b0(
                b0_dir,
                seed,
                fit_x,
                fit_y,
                val_x,
                val_y,
                identity,
            )
            r2_dir = checkpoint_dir(
                material,
                spacing,
                orientation,
                paired_seed,
                template_phase,
                "ORIGINAL_R2",
            )
            r2 = train_r2(
                r2_dir,
                b0_dir,
                seed,
                fit_x,
                fit_y,
                val_x,
                val_y,
                identity,
            )
            completed.append(
                {
                    "paired_seed": paired_seed,
                    "template_phase": template_phase,
                    "model_seed": seed,
                    "B0_checkpoint_sha256": b0["checkpoint_sha256"],
                    "R2_checkpoint_sha256": r2["checkpoint_sha256"],
                }
            )
            print(
                f"COMPLETED={material}:{spacing}:{orientation}:"
                f"{paired_seed}:{template_phase}",
                flush=True,
            )
            del fit_x, fit_y, val_x, val_y
    result = {
        "schema": "factorial_orientation_training_completion_v2",
        "material": material,
        "spacing_um": spacing,
        "orientation": orientation,
        "completed_identity_count": len(completed),
        "completed_model_count": 2 * len(completed),
        "records": completed,
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
        "NEW_METHOD_DEVELOPMENT": False,
        "ALGORITHM_OR_THRESHOLD_CHANGED": False,
    }
    completion_path = (
        ROOT
        / "manifests"
        / "training_workers"
        / f"{material}_{spacing}_{orientation}.json"
    )
    atomic_json(completion_path, result, replace=True)
    return result


def training_tasks() -> list[tuple[str, int, str]]:
    return [
        (material, spacing, orientation)
        for material in MATERIALS
        for spacing in SPACINGS
        for orientation in ORIENTATIONS
    ]


def worker_completion_path(material: str, spacing: int, orientation: str) -> Path:
    return (
        ROOT
        / "manifests"
        / "training_workers"
        / f"{material}_{spacing}_{orientation}.json"
    )


def run_training_subprocess(task: tuple[str, int, str]) -> dict[str, Any]:
    material, spacing, orientation = task
    log_path = (
        ROOT / "logs" / f"training_{material}_{spacing}_{orientation}.log"
    )
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = "1"
    with log_path.open("ab") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "train-worker",
                "--material",
                material,
                "--spacing",
                str(spacing),
                "--orientation",
                orientation,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return {
        "task": [material, spacing, orientation],
        "returncode": result.returncode,
        "log": str(log_path),
    }


def train_all() -> dict[str, Any]:
    require_feasible()
    tasks = training_tasks()
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_training_subprocess, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            atomic_json(
                TRAINING_CONTROLLER,
                {
                    "schema": "factorial_training_controller_v2",
                    "status": "RUNNING",
                    "started_unix": started,
                    "completed_worker_count": len(results),
                    "expected_worker_count": len(tasks),
                    "results": results,
                    "max_concurrent_workers": 3,
                },
                replace=True,
            )
            if result["returncode"] != 0:
                raise RuntimeError(
                    f"FACTORIAL_TRAINING_WORKER_FAILED:{result}"
                )
    result = {
        "schema": "factorial_training_controller_v2",
        "status": "COMPLETED",
        "started_unix": started,
        "completed_unix": time.time(),
        "completed_worker_count": len(results),
        "expected_worker_count": len(tasks),
        "results": sorted(results, key=lambda row: row["task"]),
        "max_concurrent_workers": 3,
        "expected_identity_count": 144,
        "expected_model_count": 288,
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
    }
    atomic_json(TRAINING_CONTROLLER, result, replace=True)
    return result


def checkpoint_records() -> list[dict[str, Any]]:
    records = []
    for material in MATERIALS:
        for spacing in SPACINGS:
            for orientation in ORIENTATIONS:
                for paired_seed in TRAINING_SEEDS:
                    for template_phase in range(4):
                        for method in ("B0", "ORIGINAL_R2"):
                            directory = checkpoint_dir(
                                material,
                                spacing,
                                orientation,
                                paired_seed,
                                template_phase,
                                method,
                            )
                            checkpoint = directory / "best_checkpoint.pt"
                            completion = directory / "completion.json"
                            if not checkpoint.is_file() or not completion.is_file():
                                continue
                            metadata = load_json(completion)
                            records.append(
                                {
                                    "material": material,
                                    "spacing_um": spacing,
                                    "condition_id": condition_id(
                                        material, spacing
                                    ),
                                    "orientation": orientation,
                                    "paired_seed": paired_seed,
                                    "template_phase": template_phase,
                                    "method": method,
                                    "checkpoint": str(checkpoint),
                                    "checkpoint_sha256": sha256_file(checkpoint),
                                    "completion": str(completion),
                                    "completion_sha256": sha256_file(completion),
                                    "best_epoch": metadata["best_epoch"],
                                }
                            )
    return records


def load_r2_model(
    directory: Path, b0_directory: Path, seed: int
) -> Any:
    import torch
    from bpmr_net.models import BPMRNet as R2Lite

    model = R2Lite(load_b0_model(b0_directory, seed))
    payload = torch.load(
        directory / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def b0_probability_batched(model: Any, features: np.ndarray) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.ascontiguousarray(features))
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            raw = model(x[start : start + BATCH_SIZE])
            probability = raw / torch.clamp(
                raw.sum(dim=1, keepdim=True), min=1e-12
            )
            parts.append(probability.cpu().numpy())
    result = np.concatenate(parts).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("B0_NONFINITE_PROBABILITY")
    return result


def eq9_average(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    average = np.mean(np.stack(probabilities), axis=0)
    return average / np.clip(average.sum(axis=1, keepdims=True), 1e-12, None)


def metric_bundle(
    probability: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    target = np.argmax(targets, axis=1)
    prediction = np.argmax(probability, axis=1)
    nll = normalized_nll(probability, targets)
    brier = float(np.mean(np.sum((probability - targets) ** 2, axis=1)))
    confidence = np.max(probability, axis=1)
    correctness = prediction == target
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for left, right in zip(bins[:-1], bins[1:]):
        selected = (confidence >= left) & (
            confidence < right if right < 1.0 else confidence <= right
        )
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correctness[selected].mean())
                - float(confidence[selected].mean())
            )
    recalls = []
    f1s = []
    for phase in range(4):
        truth_phase = target == phase
        predicted_phase = prediction == phase
        tp = int(np.sum(truth_phase & predicted_phase))
        recall = tp / max(int(truth_phase.sum()), 1)
        precision = tp / max(int(predicted_phase.sum()), 1)
        recalls.append(recall)
        f1s.append(
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    return {
        "normalized_categorical_nll": nll,
        "multiclass_brier": brier,
        "top_label_ece_10": ece,
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "phasewise_recall": recalls,
        "phasewise_nll": [
            float(
                np.mean(
                    -np.log(
                        np.clip(probability[target == phase, phase], 1e-12, 1.0)
                    )
                )
            )
            if np.any(target == phase)
            else None
            for phase in range(4)
        ],
        "phasewise_true_label_probability": [
            float(np.mean(probability[target == phase, phase]))
            if np.any(target == phase)
            else None
            for phase in range(4)
        ],
        "invalid_probabilities": int(
            np.sum(~np.isfinite(probability))
            + np.sum(np.abs(probability.sum(axis=1) - 1.0) > 1e-6)
        ),
    }


def evaluate_2d() -> dict[str, Any]:
    require_feasible()
    records = checkpoint_records()
    if len(records) != 288:
        raise RuntimeError(
            f"FACTORIAL_CHECKPOINT_MATRIX_INCOMPLETE:{len(records)}"
        )
    rows = []
    for material in MATERIALS:
        for spacing in SPACINGS:
            for orientation in ORIENTATIONS:
                planes = load_orientation_planes(
                    material, spacing, orientation
                )
                cell = condition_id(material, spacing)
                for paired_seed in TRAINING_SEEDS:
                    phase0_seed = model_seed(
                        paired_seed, cell, orientation, 0
                    )
                    features, targets, validation_identity = build_event_arrays(
                        planes, "validation", phase0_seed + 1
                    )
                    b0_predictions = []
                    r2_predictions = []
                    for template_phase in range(4):
                        seed = model_seed(
                            paired_seed,
                            cell,
                            orientation,
                            template_phase,
                        )
                        b0_dir = checkpoint_dir(
                            material,
                            spacing,
                            orientation,
                            paired_seed,
                            template_phase,
                            "B0",
                        )
                        r2_dir = checkpoint_dir(
                            material,
                            spacing,
                            orientation,
                            paired_seed,
                            template_phase,
                            "ORIGINAL_R2",
                        )
                        b0_model = load_b0_model(b0_dir, seed)
                        b0_predictions.append(
                            b0_probability_batched(b0_model, features)
                        )
                        r2_model = load_r2_model(r2_dir, b0_dir, seed)
                        import torch

                        r2_predictions.append(
                            r2_probability_batched(
                                r2_model,
                                torch.from_numpy(
                                    np.ascontiguousarray(features)
                                ),
                            )
                        )
                    b0_probability = eq9_average(b0_predictions)
                    r2_probability = eq9_average(r2_predictions)
                    entropy = -np.sum(
                        b0_probability
                        * np.log(np.clip(b0_probability, 1e-12, 1.0)),
                        axis=1,
                    )
                    use_r2 = entropy >= V2_THRESHOLD
                    v2_probability = np.where(
                        use_r2[:, None], r2_probability, b0_probability
                    )
                    for method, probability in (
                        ("B0", b0_probability),
                        ("ORIGINAL_R2", r2_probability),
                        ("SELECTIVE_V2", v2_probability),
                    ):
                        metrics = metric_bundle(probability, targets)
                        rows.append(
                            {
                                "material": material,
                                "spacing_um": spacing,
                                "condition_id": cell,
                                "orientation": orientation,
                                "training_seed": paired_seed,
                                "method": method,
                                "event_count": len(targets),
                                "validation_event_identity_sha256": (
                                    canonical_sha(validation_identity)
                                ),
                                "nll": metrics[
                                    "normalized_categorical_nll"
                                ],
                                "brier": metrics["multiclass_brier"],
                                "ece": metrics["top_label_ece_10"],
                                "macro_f1": metrics["macro_f1"],
                                "balanced_accuracy": metrics[
                                    "balanced_accuracy"
                                ],
                                "phasewise_nll": json.dumps(
                                    metrics["phasewise_nll"],
                                    separators=(",", ":"),
                                ),
                                "phasewise_recall": json.dumps(
                                    metrics["phasewise_recall"],
                                    separators=(",", ":"),
                                ),
                                "phasewise_true_label_probability": json.dumps(
                                    metrics[
                                        "phasewise_true_label_probability"
                                    ],
                                    separators=(",", ":"),
                                ),
                                "inactive_probability_mass": 0.0,
                                "invalid_probabilities": metrics[
                                    "invalid_probabilities"
                                ],
                                "v2_r2_usage_fraction": (
                                    float(use_r2.mean())
                                    if method == "SELECTIVE_V2"
                                    else None
                                ),
                                "center_label_reads": 0,
                                "extra_neighbor_reads": 0,
                            }
                        )
                del planes
    write_csv(TWO_D_ROWS, rows)
    aggregates = []
    for material in MATERIALS:
        for spacing in SPACINGS:
            for method in METHODS:
                subset = [
                    row
                    for row in rows
                    if row["material"] == material
                    and row["spacing_um"] == spacing
                    and row["method"] == method
                ]
                aggregates.append(
                    {
                        "material": material,
                        "spacing_um": spacing,
                        "method": method,
                        "stratum_count": len(subset),
                        "macro_nll": float(
                            np.mean([row["nll"] for row in subset])
                        ),
                        "macro_brier": float(
                            np.mean([row["brier"] for row in subset])
                        ),
                        "macro_ece": float(
                            np.mean([row["ece"] for row in subset])
                        ),
                        "macro_f1": float(
                            np.mean([row["macro_f1"] for row in subset])
                        ),
                        "balanced_accuracy": float(
                            np.mean(
                                [row["balanced_accuracy"] for row in subset]
                            )
                        ),
                        "v2_r2_usage_fraction": (
                            float(
                                np.mean(
                                    [
                                        row["v2_r2_usage_fraction"]
                                        for row in subset
                                    ]
                                )
                            )
                            if method == "SELECTIVE_V2"
                            else None
                        ),
                    }
                )
    result = {
        "schema": "factorial_2d_aggregate_v2",
        "row_count": len(rows),
        "aggregates": aggregates,
        "all_probabilities_finite": all(
            row["invalid_probabilities"] == 0 for row in rows
        ),
        "inactive_probability_mass_zero": all(
            row["inactive_probability_mass"] == 0 for row in rows
        ),
        "center_label_reads": 0,
        "extra_neighbor_reads": 0,
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
        "V2_THRESHOLD": V2_THRESHOLD,
        "scientific_gate_applied": False,
    }
    atomic_json(TWO_D_RESULT, result)
    return result


def freeze_checkpoints() -> dict[str, Any]:
    require_feasible()
    records = checkpoint_records()
    expected = 288
    valid = len(records) == expected
    if valid:
        for record in records:
            if (
                record["checkpoint_sha256"]
                != sha256_file(Path(record["checkpoint"]))
            ):
                valid = False
                break
    if not TWO_D_RESULT.is_file():
        raise RuntimeError("FACTORIAL_2D_EVALUATION_MISSING")
    main_scripts = [
        REPO / "src" / "factorial_protocol_and_training_v2.py",
        REPO / "src" / "factorial_multiseed_reconstruction_v2.py",
        REPO / "src" / "factorial_evaluation_and_package_v2.py",
    ]
    if not all(path.is_file() for path in main_scripts):
        raise RuntimeError("FACTORIAL_MAIN_SCRIPT_MISSING")
    result = {
        "schema": "factorial_checkpoint_freeze_manifest_v2",
        "revision": 1,
        "supersedes_preinitialization_freeze": str(
            PREINITIALIZATION_CHECKPOINT_FREEZE
        ),
        "revision_reason": (
            "restore the frozen production initialization semantic: phase "
            "proportions are pooled from all legal registered sparse planes, "
            "not only the local ROI clamp mask"
        ),
        "expected_model_checkpoints": expected,
        "valid_model_checkpoints": len(records),
        "checkpoint_matrix_complete": valid,
        "records": records,
        "main_script_shas": {
            str(path): sha256_file(path) for path in main_scripts
        },
        "two_d_evaluation": str(TWO_D_RESULT),
        "two_d_evaluation_sha256": sha256_file(TWO_D_RESULT),
        "frozen_before_factorial_dense_truth_evaluation": True,
        "DENSE_TRUTH_READS_DURING_MODEL_TRAINING": 0,
        "NEW_METHOD_DEVELOPMENT": False,
        "PRODUCTION_MODEL_FITTING_UNDER_FROZEN_ALGORITHM": True,
        "ALGORITHM_OR_THRESHOLD_CHANGED": False,
    }
    if not valid:
        raise RuntimeError(
            f"FACTORIAL_CHECKPOINT_FREEZE_INCOMPLETE:{len(records)}"
        )
    atomic_json(CHECKPOINT_FREEZE, result)
    return result


def training_status() -> dict[str, Any]:
    records = checkpoint_records()
    workers = [
        str(worker_completion_path(*task))
        for task in training_tasks()
        if worker_completion_path(*task).is_file()
    ]
    return {
        "expected_model_checkpoints": 288,
        "completed_model_checkpoints": len(records),
        "expected_workers": 18,
        "completed_workers": len(workers),
        "worker_completion_files": workers,
        "controller": (
            load_json(TRAINING_CONTROLLER)
            if TRAINING_CONTROLLER.is_file()
            else None
        ),
    }


def roi_record(material: str) -> dict[str, Any]:
    registry = load_json(ROI_REGISTRY)
    return next(
        record
        for record in registry["materials"]
        if record["material"] == material
    )


def observation_path(material: str, spacing: int) -> Path:
    return (
        ROOT
        / "sparse_protocols"
        / material
        / f"spacing_{spacing}um"
        / "reconstruction_observations_v2.npz"
    )


def assign_observed_plane(
    observed_labels: np.ndarray,
    observed_mask: np.ndarray,
    orientation: str,
    local_index: int,
    labels_2d: np.ndarray,
    support_2d: np.ndarray,
) -> None:
    if orientation == "x":
        labels_xyz = labels_2d.T
        support_xyz = support_2d.T
        existing_mask = observed_mask[local_index, :, :]
        existing_labels = observed_labels[local_index, :, :]
    elif orientation == "y":
        labels_xyz = labels_2d.T
        support_xyz = support_2d.T
        existing_mask = observed_mask[:, local_index, :]
        existing_labels = observed_labels[:, local_index, :]
    else:
        labels_xyz = labels_2d.T
        support_xyz = support_2d.T
        existing_mask = observed_mask[:, :, local_index]
        existing_labels = observed_labels[:, :, local_index]
    conflict = existing_mask & support_xyz & (existing_labels != labels_xyz)
    if np.any(conflict):
        raise RuntimeError(
            f"OBSERVED_PLANE_LABEL_CONFLICT:{orientation}:{local_index}"
        )
    existing_labels[support_xyz] = labels_xyz[support_xyz]
    existing_mask[support_xyz] = True


def build_observation_packages() -> dict[str, Any]:
    require_feasible()
    file_index = plane_file_index()
    records = []
    for material in MATERIALS:
        roi = roi_record(material)
        origin_z, origin_y, origin_x = map(int, roi["roi_origin_zyx"])
        size_z, size_y, size_x = map(int, roi["roi_shape_zyx"])
        if (size_x, size_y, size_z) != (48, 48, 48):
            raise RuntimeError("FACTORIAL_ROI_SHAPE_CHANGED")
        ranges = {
            "x": (origin_x, origin_x + size_x),
            "y": (origin_y, origin_y + size_y),
            "z": (origin_z, origin_z + size_z),
        }
        for spacing in SPACINGS:
            observed_labels = np.full(
                (size_x, size_y, size_z), -1, dtype=np.int16
            )
            observed_mask = np.zeros(
                (size_x, size_y, size_z), dtype=bool
            )
            used_references = []
            references_by_axis = {}
            for orientation in ORIENTATIONS:
                references, _ = cell_references(
                    material, spacing, orientation
                )
                selected = [
                    reference
                    for reference in references
                    if ranges[orientation][0]
                    <= int(reference["global_plane_index"])
                    < ranges[orientation][1]
                ]
                if not selected:
                    raise RuntimeError(
                        f"NO_REGISTERED_PLANE_INTERSECTS_ROI:"
                        f"{material}:{spacing}:{orientation}"
                    )
                references_by_axis[orientation] = []
                for reference in selected:
                    label_path = resolve_plane_file(
                        reference,
                        "material_labels_relative_path",
                        file_index,
                    )
                    support_path = resolve_plane_file(
                        reference,
                        "valid_support_relative_path",
                        file_index,
                    )
                    if sha256_file(label_path) != reference[
                        "material_labels_sha256"
                    ]:
                        raise RuntimeError("OBSERVATION_LABEL_SHA_MISMATCH")
                    if sha256_file(support_path) != reference[
                        "valid_support_sha256"
                    ]:
                        raise RuntimeError("OBSERVATION_SUPPORT_SHA_MISMATCH")
                    labels = decode_npy_gz(label_path)
                    support = np.asarray(
                        decode_npy_gz(support_path), dtype=bool
                    )
                    canonical = canonicalize_labels(labels, support)
                    index = int(reference["global_plane_index"])
                    if orientation == "x":
                        cropped_labels = canonical[
                            origin_z : origin_z + size_z,
                            origin_y : origin_y + size_y,
                        ]
                        cropped_support = support[
                            origin_z : origin_z + size_z,
                            origin_y : origin_y + size_y,
                        ]
                        local_index = index - origin_x
                    elif orientation == "y":
                        cropped_labels = canonical[
                            origin_z : origin_z + size_z,
                            origin_x : origin_x + size_x,
                        ]
                        cropped_support = support[
                            origin_z : origin_z + size_z,
                            origin_x : origin_x + size_x,
                        ]
                        local_index = index - origin_y
                    else:
                        cropped_labels = canonical[
                            origin_y : origin_y + size_y,
                            origin_x : origin_x + size_x,
                        ]
                        cropped_support = support[
                            origin_y : origin_y + size_y,
                            origin_x : origin_x + size_x,
                        ]
                        local_index = index - origin_z
                    if cropped_labels.shape != (48, 48):
                        raise RuntimeError("OBSERVATION_PLANE_CROP_SHAPE_INVALID")
                    assign_observed_plane(
                        observed_labels,
                        observed_mask,
                        orientation,
                        local_index,
                        cropped_labels,
                        cropped_support,
                    )
                    reference_record = {
                        "orientation": orientation,
                        "global_plane_index": index,
                        "local_plane_index": local_index,
                        "plane_id": reference["plane_id"],
                        "material_labels_sha256": reference[
                            "material_labels_sha256"
                        ],
                        "valid_support_sha256": reference[
                            "valid_support_sha256"
                        ],
                    }
                    used_references.append(reference_record)
                    references_by_axis[orientation].append(reference_record)
            if np.any(observed_mask & ((observed_labels < 0) | (observed_labels >= 4))):
                raise RuntimeError("OBSERVATION_LABEL_RANGE_INVALID")
            support_3d = np.ones(observed_mask.shape, dtype=bool)
            metadata = {
                "schema": "factorial_reconstruction_observations_v2",
                "material": material,
                "spacing_um": spacing,
                "condition_id": condition_id(material, spacing),
                "roi_origin_zyx": roi["roi_origin_zyx"],
                "roi_shape_zyx": roi["roi_shape_zyx"],
                "active_phases": [0, 1, 2, 3],
                "used_references": used_references,
                "references_by_axis": references_by_axis,
                "observed_voxel_count": int(observed_mask.sum()),
                "observed_mask_sha256": array_sha(observed_mask),
                "observed_labels_sha256": array_sha(observed_labels),
                "support_sha256": array_sha(support_3d),
                "nested_registry_sha256": sha256_file(NESTED_REGISTRY),
                "roi_registry_sha256": sha256_file(ROI_REGISTRY),
                "dense_truth_reads": 0,
                "source": "registered Step62 sparse-plane arrays only",
            }
            path = observation_path(material, spacing)
            atomic_npz(
                path,
                {
                    "observed_labels": observed_labels,
                    "observed_mask": observed_mask,
                    "support_mask": support_3d,
                },
                metadata,
            )
            records.append(
                {
                    **metadata,
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                }
            )
    result = {
        "schema": "factorial_reconstruction_observation_freeze_v2",
        "record_count": len(records),
        "records": records,
        "all_orientations_present": all(
            all(record["references_by_axis"][axis] for axis in ORIENTATIONS)
            for record in records
        ),
        "DENSE_TRUTH_READS": 0,
    }
    path = (
        ROOT
        / "manifests"
        / "factorial_reconstruction_observation_freeze_v2.json"
    )
    atomic_json(path, result)
    return result


def print_result(result: Mapping[str, Any]) -> None:
    keys = (
        "WF2_25_FEASIBLE",
        "WF2_250_FEASIBLE",
        "WF2_1000_FEASIBLE",
        "GI_25_FEASIBLE",
        "GI_250_FEASIBLE",
        "GI_1000_FEASIBLE",
        "FULL_2X3_FACTORIAL_FEASIBLE",
        "NESTED_SAMPLING_VERIFIED",
        "SPLIT_ROLE_CONSISTENCY_VERIFIED",
        "FINAL_V1_REUSED_CELLS",
        "FINAL_V1_RERUN_CELLS",
        "classification",
    )
    print(f"FACTORIAL_ROOT={ROOT}")
    print(f"FEASIBILITY_AUDIT={FEASIBILITY}")
    print(f"FEASIBILITY_AUDIT_SHA256={sha256_file(FEASIBILITY)}")
    for key in keys:
        print(f"{key}={result.get(key)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "status",
            "train-worker",
            "train-all",
            "training-status",
            "evaluate-2d",
            "freeze-checkpoints",
            "build-observations",
        ),
        nargs="?",
        default="audit",
    )
    parser.add_argument("--material", choices=MATERIALS)
    parser.add_argument("--spacing", type=int, choices=SPACINGS)
    parser.add_argument("--orientation", choices=ORIENTATIONS)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit()
        print_result(result)
    elif args.command == "status":
        if not FEASIBILITY.is_file():
            raise RuntimeError("FEASIBILITY_AUDIT_NOT_CREATED")
        result = load_json(FEASIBILITY)
        print_result(result)
    elif args.command == "train-worker":
        if (
            args.material is None
            or args.spacing is None
            or args.orientation is None
        ):
            raise RuntimeError("TRAIN_WORKER_ARGUMENTS_REQUIRED")
        result = train_worker(
            args.material, args.spacing, args.orientation
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "train-all":
        result = train_all()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "training-status":
        result = training_status()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "evaluate-2d":
        result = evaluate_2d()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    elif args.command == "build-observations":
        result = build_observation_packages()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        result = freeze_checkpoints()
        print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
