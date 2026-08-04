from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(
    os.environ.get("BPMR_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
ROOT = Path(
    os.environ.get("BPMR_OUTPUT_ROOT", REPO / "outputs" / "factorial_v2")
).resolve()
SRC = REPO / "src"
BASE = Path(
    os.environ.get("BPMR_BASELINE_ROOT", REPO / "data" / "baseline")
).resolve()
GATEB = Path(
    os.environ.get("BPMR_GATEB_ROOT", REPO / "external" / "gateb")
).resolve()
OLD_NS = Path(
    os.environ.get("BPMR_LEGACY_R2_ROOT", REPO / "external" / "legacy_r2")
).resolve()
for module_path in (SRC, GATEB / "src", BASE / "src", OLD_NS):
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import factorial_protocol_and_training_v2 as protocol  # noqa: E402
import r2_lite_3d_pilot as pilot  # noqa: E402


MATERIALS = protocol.MATERIALS
SPACINGS = protocol.SPACINGS
METHODS = protocol.METHODS
SEEDS = protocol.RECONSTRUCTION_SEEDS
ORIENTATIONS = protocol.ORIENTATIONS
PHASES = (0, 1, 2, 3)
TRAINING_SEED = 1702
THRESHOLD = protocol.V2_THRESHOLD
SHAPE_XYZ = (48, 48, 48)
MAX_CONCURRENT_TRAJECTORIES = 3

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

OBSERVATION_FREEZE = (
    ROOT / "manifests" / "factorial_reconstruction_observation_freeze_v2.json"
)
CHECKPOINT_FREEZE = protocol.CHECKPOINT_FREEZE
STATE_FREEZE = (
    ROOT / "manifests" / "factorial_state_freeze_manifest_v2.json"
)
CONTROLLER = (
    ROOT / "manifests" / "factorial_reconstruction_controller_v2.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


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


def atomic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    if path.exists():
        loaded_arrays, loaded_metadata = load_npz(path)
        if loaded_metadata != dict(metadata):
            raise RuntimeError(f"FROZEN_STATE_METADATA_CONFLICT:{path}")
        for key, value in arrays.items():
            if key not in loaded_arrays or not np.array_equal(
                loaded_arrays[key], value
            ):
                raise RuntimeError(f"FROZEN_STATE_ARRAY_CONFLICT:{path}:{key}")
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


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata


def require_inputs() -> None:
    protocol.require_feasible()
    if not CHECKPOINT_FREEZE.is_file():
        raise RuntimeError("FACTORIAL_CHECKPOINT_FREEZE_MISSING")
    checkpoint = load_json(CHECKPOINT_FREEZE)
    if checkpoint.get("checkpoint_matrix_complete") is not True:
        raise RuntimeError("FACTORIAL_CHECKPOINT_MATRIX_INCOMPLETE")
    for script, expected_sha in checkpoint.get(
        "main_script_shas", {}
    ).items():
        path = Path(script)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"FACTORIAL_MAIN_SCRIPT_SHA_CONFLICT:{path}")
    if not OBSERVATION_FREEZE.is_file():
        raise RuntimeError("FACTORIAL_OBSERVATION_FREEZE_MISSING")
    observations = load_json(OBSERVATION_FREEZE)
    if observations.get("all_orientations_present") is not True:
        raise RuntimeError("FACTORIAL_OBSERVATION_ORIENTATION_INCOMPLETE")


def latent_base_path(material: str, seed: int) -> Path:
    return (
        ROOT
        / "initial_states"
        / "revision_1"
        / material
        / f"seed_{seed}"
        / "latent_base_state_v2.npz"
    )


def initial_state_path(material: str, spacing: int, seed: int) -> Path:
    return (
        ROOT
        / "initial_states"
        / "revision_1"
        / material
        / f"seed_{seed}"
        / f"spacing_{spacing}um_initial_state_v2.npz"
    )


def state_path(
    material: str,
    spacing: int,
    method: str,
    seed: int,
    sweep: int,
) -> Path:
    return (
        ROOT
        / "reconstruction_states"
        / material
        / f"spacing_{spacing}um"
        / method
        / f"seed_{seed}"
        / f"sweep_{sweep}.npz"
    )


def completion_path(
    material: str, spacing: int, method: str, seed: int
) -> Path:
    return (
        ROOT
        / "reconstruction_states"
        / material
        / f"spacing_{spacing}um"
        / method
        / f"seed_{seed}"
        / "completion.json"
    )


def load_observations(
    material: str, spacing: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = protocol.observation_path(material, spacing)
    arrays, metadata = load_npz(path)
    if arrays["observed_labels"].shape != SHAPE_XYZ:
        raise RuntimeError("OBSERVATION_SHAPE_MISMATCH")
    if arrays["observed_mask"].shape != SHAPE_XYZ:
        raise RuntimeError("OBSERVATION_MASK_SHAPE_MISMATCH")
    if arrays["support_mask"].shape != SHAPE_XYZ:
        raise RuntimeError("SUPPORT_MASK_SHAPE_MISMATCH")
    if np.any(
        arrays["observed_mask"]
        & (
            (arrays["observed_labels"] < 0)
            | (arrays["observed_labels"] >= 4)
        )
    ):
        raise RuntimeError("OBSERVATION_LABEL_RANGE_INVALID")
    return arrays, metadata


def phase_proportions_from_densest_observations(material: str) -> np.ndarray:
    counts = np.zeros(4, dtype=np.float64)
    for orientation in ORIENTATIONS:
        for plane in protocol.load_orientation_planes(
            material, 25, orientation
        ):
            support = np.asarray(plane["support"], dtype=bool)
            labels = np.asarray(plane["labels"], dtype=np.int16)
            counts += np.bincount(
                labels[support], minlength=4
            ).astype(np.float64)[:4]
    if np.any(counts <= 0):
        raise RuntimeError(
            f"LATENT_BASE_ACTIVE_PHASE_SUPPORT_MISSING:{material}:{counts}"
        )
    return counts / counts.sum()


def save_latent_base(material: str, seed: int) -> dict[str, Any]:
    path = latent_base_path(material, seed)
    proportions = phase_proportions_from_densest_observations(material)
    state = pilot.gaussian_quantile_initialization(
        SHAPE_XYZ, proportions.tolist(), seed
    ).astype(np.int16)
    metadata = {
        "schema": "factorial_latent_base_state_v2",
        "protocol_revision": 1,
        "material": material,
        "seed": seed,
        "shape_xyz": list(SHAPE_XYZ),
        "phase_proportions_source": (
            "all legal registered 25um sparse planes for this material, "
            "matching the frozen production initialization protocol"
        ),
        "phase_proportions": proportions.tolist(),
        "state_sha256": array_sha(state),
        "shared_across_spacings": True,
        "created_before_factorial_dense_truth_evaluation": True,
        "dense_truth_reads": 0,
    }
    atomic_npz(path, {"latent_base_state": state}, metadata)
    return {**metadata, "path": str(path), "file_sha256": sha256_file(path)}


def save_regime_initial_state(
    material: str, spacing: int, seed: int
) -> dict[str, Any]:
    latent_arrays, latent_metadata = load_npz(latent_base_path(material, seed))
    observations, observation_metadata = load_observations(material, spacing)
    state = latent_arrays["latent_base_state"].copy()
    observed_mask = observations["observed_mask"].astype(bool, copy=False)
    observed_labels = observations["observed_labels"].astype(
        np.int16, copy=False
    )
    support = observations["support_mask"].astype(bool, copy=False)
    state[observed_mask] = observed_labels[observed_mask]
    update_mask = support & ~observed_mask
    if np.any(state[observed_mask] != observed_labels[observed_mask]):
        raise RuntimeError("INITIAL_OBSERVED_CLAMP_FAILED")
    metadata = {
        "schema": "factorial_regime_initial_state_v2",
        "protocol_revision": 1,
        "material": material,
        "spacing_um": spacing,
        "seed": seed,
        "shape_xyz": list(state.shape),
        "latent_base_state_sha256": latent_metadata["state_sha256"],
        "observed_labels_sha256": array_sha(observed_labels),
        "observed_mask_sha256": array_sha(observed_mask),
        "support_mask_sha256": array_sha(support),
        "update_mask_sha256": array_sha(update_mask),
        "initial_state_sha256": array_sha(state),
        "initial_phase_fractions": [
            float(np.mean(state[support] == phase)) for phase in PHASES
        ],
        "shared_by_methods": list(METHODS),
        "observation_package_sha256": sha256_file(
            protocol.observation_path(material, spacing)
        ),
        "created_before_factorial_dense_truth_evaluation": True,
        "dense_truth_reads": 0,
    }
    path = initial_state_path(material, spacing, seed)
    atomic_npz(
        path,
        {
            "initial_state": state,
            "observed_labels": observed_labels,
            "observed_mask": observed_mask,
            "support_mask": support,
            "update_mask": update_mask,
        },
        metadata,
    )
    return {**metadata, "path": str(path), "file_sha256": sha256_file(path)}


def prepare_initial_states() -> dict[str, Any]:
    require_inputs()
    latent_records = []
    regime_records = []
    for material in MATERIALS:
        for seed in SEEDS:
            latent_records.append(save_latent_base(material, seed))
            for spacing in SPACINGS:
                regime_records.append(
                    save_regime_initial_state(material, spacing, seed)
                )
    result = {
        "schema": "factorial_initial_state_freeze_v2",
        "protocol_revision": 1,
        "preinitialization_failure_preserved_under": str(
            ROOT / "initial_states"
        ),
        "latent_base_state_count": len(latent_records),
        "regime_initial_state_count": len(regime_records),
        "latent_base_states": latent_records,
        "regime_initial_states": regime_records,
        "paired_methods": True,
        "paired_spacings_via_shared_latent_base": True,
        "DENSE_TRUTH_READS": 0,
    }
    path = ROOT / "manifests" / "factorial_initial_state_freeze_v2.json"
    atomic_json(path, result)
    return result


def load_models(
    material: str, spacing: int, method: str
) -> dict[str, dict[int, Any]]:
    models: dict[str, dict[int, Any]] = {
        orientation: {} for orientation in ORIENTATIONS
    }
    cell = protocol.condition_id(material, spacing)
    for orientation in ORIENTATIONS:
        for phase in PHASES:
            seed = protocol.model_seed(
                TRAINING_SEED, cell, orientation, phase
            )
            b0_dir = protocol.checkpoint_dir(
                material,
                spacing,
                orientation,
                TRAINING_SEED,
                phase,
                "B0",
            )
            if method == "B0":
                model = protocol.load_b0_model(b0_dir, seed)
            else:
                r2_dir = protocol.checkpoint_dir(
                    material,
                    spacing,
                    orientation,
                    TRAINING_SEED,
                    phase,
                    "ORIGINAL_R2",
                )
                model = protocol.load_r2_model(r2_dir, b0_dir, seed)
            model.eval()
            models[orientation][phase] = model
    return models


def interface_flag(state: np.ndarray, index: tuple[int, int, int]) -> bool:
    center = int(state[index])
    for axis in range(3):
        for delta in (-1, 1):
            neighbour = list(index)
            neighbour[axis] = (neighbour[axis] + delta) % state.shape[axis]
            if int(state[tuple(neighbour)]) != center:
                return True
    return False


def selector_template() -> dict[str, Any]:
    return {
        "total_updates": 0,
        "r2_updates": 0,
        "b0_updates": 0,
        "threshold_ties": 0,
        "entropy_sum": 0.0,
        "entropy_min": None,
        "entropy_max": None,
        "r2_by_current_phase": [0, 0, 0, 0],
        "total_by_current_phase": [0, 0, 0, 0],
        "r2_by_selected_phase": [0, 0, 0, 0],
        "total_by_selected_phase": [0, 0, 0, 0],
        "r2_at_interface": 0,
        "total_at_interface": 0,
        "r2_in_interior": 0,
        "total_in_interior": 0,
    }


def finalize_selector(stats: Mapping[str, Any]) -> dict[str, Any]:
    total = max(int(stats["total_updates"]), 1)
    result = dict(stats)
    result["r2_usage_fraction"] = float(stats["r2_updates"]) / total
    result["b0_fallback_fraction"] = float(stats["b0_updates"]) / total
    result["threshold_tie_fraction"] = float(stats["threshold_ties"]) / total
    result["entropy_mean"] = float(stats["entropy_sum"]) / total
    result["r2_usage_by_current_phase"] = [
        float(used) / max(int(count), 1)
        for used, count in zip(
            stats["r2_by_current_phase"], stats["total_by_current_phase"]
        )
    ]
    result["r2_usage_by_selected_phase"] = [
        float(used) / max(int(count), 1)
        for used, count in zip(
            stats["r2_by_selected_phase"], stats["total_by_selected_phase"]
        )
    ]
    result["r2_usage_at_interface"] = float(stats["r2_at_interface"]) / max(
        int(stats["total_at_interface"]), 1
    )
    result["r2_usage_in_interior"] = float(stats["r2_in_interior"]) / max(
        int(stats["total_in_interior"]), 1
    )
    return result


def save_state(
    path: Path,
    state: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    full_metadata = {**metadata, "state_sha256": array_sha(state)}
    atomic_npz(path, {"state": state.astype(np.int16, copy=False)}, full_metadata)
    return {
        **full_metadata,
        "path": str(path),
        "file_sha256": sha256_file(path),
    }


def latest_valid_sweep(
    material: str, spacing: int, method: str, seed: int
) -> int:
    latest = -1
    for sweep in range(4):
        path = state_path(material, spacing, method, seed, sweep)
        if not path.is_file():
            break
        arrays, metadata = load_npz(path)
        if metadata["state_sha256"] != array_sha(arrays["state"]):
            raise RuntimeError(f"STATE_SHA_MISMATCH:{path}")
        latest = sweep
    return latest


def run_trajectory(
    material: str, spacing: int, method: str, seed: int
) -> dict[str, Any]:
    require_inputs()
    completion = completion_path(material, spacing, method, seed)
    if completion.is_file():
        prior = load_json(completion)
        final_path = state_path(material, spacing, method, seed, 3)
        if (
            final_path.is_file()
            and prior["final_state_file_sha256"] == sha256_file(final_path)
        ):
            return prior

    initial_arrays, initial_metadata = load_npz(
        initial_state_path(material, spacing, seed)
    )
    observed_labels = initial_arrays["observed_labels"]
    observed_mask = initial_arrays["observed_mask"].astype(bool, copy=False)
    support_mask = initial_arrays["support_mask"].astype(bool, copy=False)
    update_mask = initial_arrays["update_mask"].astype(bool, copy=False)
    update_coordinates = np.argwhere(update_mask)

    latest = latest_valid_sweep(material, spacing, method, seed)
    if latest < 0:
        state = initial_arrays["initial_state"].copy()
        rng = np.random.default_rng(seed)
        initial_record = save_state(
            state_path(material, spacing, method, seed, 0),
            state,
            {
                "schema": "factorial_reconstruction_state_v2",
                "material": material,
                "spacing_um": spacing,
                "method": method,
                "seed": seed,
                "sweep": 0,
                "completed_updates": 0,
                "initial_state_sha256": initial_metadata[
                    "initial_state_sha256"
                ],
                "observed_mask_sha256": array_sha(observed_mask),
                "support_mask_sha256": array_sha(support_mask),
                "update_order_sha256": array_sha(update_coordinates),
                "rng_state": rng.bit_generator.state,
                "rng_state_sha256": hashlib.sha256(
                    json.dumps(
                        rng.bit_generator.state,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "selector": selector_template() if method == "SELECTIVE_V2" else {},
                "runtime_seconds": 0.0,
                "peak_rss_kb": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "dense_truth_reads": 0,
            },
        )
        latest = 0
    else:
        arrays, metadata = load_npz(
            state_path(material, spacing, method, seed, latest)
        )
        state = arrays["state"].copy()
        rng = np.random.default_rng()
        rng.bit_generator.state = metadata["rng_state"]
        initial_record = None

    b0_models = load_models(material, spacing, "B0")
    r2_models = (
        load_models(material, spacing, "ORIGINAL_R2")
        if method in ("ORIGINAL_R2", "SELECTIVE_V2")
        else None
    )
    state_records = [initial_record] if initial_record else []
    trajectory_started = time.monotonic()
    for sweep in range(latest + 1, 4):
        segment_started = time.monotonic()
        selector = selector_template()
        for coordinate_array in update_coordinates:
            index = tuple(int(value) for value in coordinate_array)
            if method == "B0":
                probability = pilot.predictor_probability(
                    b0_models, state, index
                )
            elif method == "ORIGINAL_R2":
                probability = pilot.predictor_probability(
                    r2_models, state, index
                )
            else:
                b0_probability = pilot.predictor_probability(
                    b0_models, state, index
                )
                r2_probability = pilot.predictor_probability(
                    r2_models, state, index
                )
                entropy = float(
                    -np.sum(
                        b0_probability
                        * np.log(np.clip(b0_probability, 1e-12, 1.0))
                    )
                )
                use_r2 = entropy >= THRESHOLD
                probability = (
                    r2_probability if use_r2 else b0_probability
                )
                current_phase = int(state[index])
                selected_phase = int(np.argmax(probability))
                at_interface = interface_flag(state, index)
                selector["total_updates"] += 1
                selector["r2_updates"] += int(use_r2)
                selector["b0_updates"] += int(not use_r2)
                selector["threshold_ties"] += int(entropy == THRESHOLD)
                selector["entropy_sum"] += entropy
                selector["entropy_min"] = (
                    entropy
                    if selector["entropy_min"] is None
                    else min(selector["entropy_min"], entropy)
                )
                selector["entropy_max"] = (
                    entropy
                    if selector["entropy_max"] is None
                    else max(selector["entropy_max"], entropy)
                )
                selector["total_by_current_phase"][current_phase] += 1
                selector["r2_by_current_phase"][current_phase] += int(use_r2)
                selector["total_by_selected_phase"][selected_phase] += 1
                selector["r2_by_selected_phase"][selected_phase] += int(use_r2)
                if at_interface:
                    selector["total_at_interface"] += 1
                    selector["r2_at_interface"] += int(use_r2)
                else:
                    selector["total_in_interior"] += 1
                    selector["r2_in_interior"] += int(use_r2)
            if (
                probability.shape != (4,)
                or not np.all(np.isfinite(probability))
                or not np.isclose(probability.sum(), 1.0, atol=1e-8)
            ):
                raise RuntimeError("INVALID_RECONSTRUCTION_PROBABILITY")
            state[index] = int(rng.choice(4, p=probability))

        if np.any(state[observed_mask] != observed_labels[observed_mask]):
            raise RuntimeError("OBSERVED_VOXEL_MODIFIED")
        if np.any((state[support_mask] < 0) | (state[support_mask] >= 4)):
            raise RuntimeError("INVALID_OR_INACTIVE_PHASE_SAMPLED")
        rng_payload = json.dumps(
            rng.bit_generator.state,
            sort_keys=True,
            separators=(",", ":"),
        )
        record = save_state(
            state_path(material, spacing, method, seed, sweep),
            state,
            {
                "schema": "factorial_reconstruction_state_v2",
                "material": material,
                "spacing_um": spacing,
                "method": method,
                "seed": seed,
                "sweep": sweep,
                "completed_updates": sweep * len(update_coordinates),
                "initial_state_sha256": initial_metadata[
                    "initial_state_sha256"
                ],
                "observed_mask_sha256": array_sha(observed_mask),
                "support_mask_sha256": array_sha(support_mask),
                "update_order_sha256": array_sha(update_coordinates),
                "rng_state": rng.bit_generator.state,
                "rng_state_sha256": hashlib.sha256(
                    rng_payload.encode()
                ).hexdigest(),
                "selector": (
                    finalize_selector(selector)
                    if method == "SELECTIVE_V2"
                    else {}
                ),
                "runtime_seconds": time.monotonic() - segment_started,
                "peak_rss_kb": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "dense_truth_reads": 0,
                "exact_resume_from_sweep": sweep - 1,
            },
        )
        state_records.append(record)
        print(
            f"SWEEP_COMPLETED={material}:{spacing}:{method}:{seed}:{sweep}",
            flush=True,
        )

    final_path = state_path(material, spacing, method, seed, 3)
    result = {
        "schema": "factorial_reconstruction_completion_v2",
        "status": "COMPLETED",
        "material": material,
        "spacing_um": spacing,
        "method": method,
        "seed": seed,
        "sweeps_completed": [0, 1, 2, 3],
        "completed_updates": 3 * len(update_coordinates),
        "final_state": str(final_path),
        "final_state_file_sha256": sha256_file(final_path),
        "final_state_array_sha256": array_sha(state),
        "observed_voxel_changes": 0,
        "invalid_updates": 0,
        "inactive_phase_samples": 0,
        "exact_resume_identity_valid": True,
        "runtime_seconds_this_invocation": time.monotonic()
        - trajectory_started,
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
    }
    atomic_json(completion, result)
    return result


def run_all() -> dict[str, Any]:
    require_inputs()
    started = time.time()
    tasks = [
        (material, spacing, method, seed)
        for material in MATERIALS
        for spacing in SPACINGS
        for method in METHODS
        for seed in SEEDS
    ]
    def execute(task: tuple[str, int, str, int]) -> dict[str, Any]:
        material, spacing, method, seed = task
        log_path = (
            ROOT
            / "logs"
            / f"reconstruction_{material}_{spacing}_{method}_{seed}.log"
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
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-trajectory",
            "--material",
            material,
            "--spacing",
            str(spacing),
            "--method",
            method,
            "--seed",
            str(seed),
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=str(REPO),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"RECONSTRUCTION_TRAJECTORY_FAILED:{task}:"
                f"{completed.returncode}:{log_path}"
            )
        return load_json(completion_path(*task))

    results = []
    with ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_TRAJECTORIES
    ) as executor:
        task_iterator = iter(tasks)
        futures = {}
        for task in itertools.islice(
            task_iterator, MAX_CONCURRENT_TRAJECTORIES
        ):
            futures[executor.submit(execute, task)] = task
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                result = future.result()
                results.append(result)
                atomic_json(
                    CONTROLLER,
                    {
                        "schema": "factorial_reconstruction_controller_v2",
                        "status": "RUNNING",
                        "started_unix": started,
                        "expected_trajectories": len(tasks),
                        "completed_trajectories": len(results),
                        "last_completed": list(task),
                        "maximum_concurrent_trajectories": (
                            MAX_CONCURRENT_TRAJECTORIES
                        ),
                        "independent_process_per_trajectory": True,
                        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
                    },
                    replace=True,
                )
                next_task = next(task_iterator, None)
                if next_task is not None:
                    futures[executor.submit(execute, next_task)] = next_task
    result = {
        "schema": "factorial_reconstruction_controller_v2",
        "status": "COMPLETED",
        "started_unix": started,
        "completed_unix": time.time(),
        "expected_trajectories": len(tasks),
        "completed_trajectories": len(results),
        "maximum_concurrent_trajectories": MAX_CONCURRENT_TRAJECTORIES,
        "independent_process_per_trajectory": True,
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
    }
    atomic_json(CONTROLLER, result, replace=True)
    return result


def state_records() -> list[dict[str, Any]]:
    records = []
    for material in MATERIALS:
        for spacing in SPACINGS:
            for method in METHODS:
                for seed in SEEDS:
                    for sweep in range(4):
                        path = state_path(
                            material, spacing, method, seed, sweep
                        )
                        if not path.is_file():
                            continue
                        arrays, metadata = load_npz(path)
                        records.append(
                            {
                                "material": material,
                                "spacing_um": spacing,
                                "method": method,
                                "seed": seed,
                                "sweep": sweep,
                                "path": str(path),
                                "file_sha256": sha256_file(path),
                                "state_sha256": array_sha(arrays["state"]),
                                "metadata_state_sha256": metadata[
                                    "state_sha256"
                                ],
                                "observed_mask_sha256": metadata[
                                    "observed_mask_sha256"
                                ],
                                "support_mask_sha256": metadata[
                                    "support_mask_sha256"
                                ],
                                "update_order_sha256": metadata[
                                    "update_order_sha256"
                                ],
                                "completed_updates": metadata[
                                    "completed_updates"
                                ],
                                "runtime_seconds": metadata["runtime_seconds"],
                                "peak_rss_kb": metadata["peak_rss_kb"],
                                "selector": metadata.get("selector", {}),
                            }
                        )
    return records


def freeze_states() -> dict[str, Any]:
    records = state_records()
    expected_states = 360
    expected_trajectories = 90
    completions = []
    for material in MATERIALS:
        for spacing in SPACINGS:
            for method in METHODS:
                for seed in SEEDS:
                    path = completion_path(
                        material, spacing, method, seed
                    )
                    if path.is_file():
                        completions.append(load_json(path))
    state_sha_mismatches = sum(
        record["state_sha256"] != record["metadata_state_sha256"]
        for record in records
    )
    fixed_violations = 0
    invalid_or_sentinel_voxels = 0
    resume_chain_violations = 0
    for material in MATERIALS:
        for spacing in SPACINGS:
            observations, _ = load_observations(material, spacing)
            observed = observations["observed_mask"].astype(bool, copy=False)
            labels = observations["observed_labels"]
            support = observations["support_mask"].astype(bool, copy=False)
            for method in METHODS:
                for seed in SEEDS:
                    prior_metadata = None
                    for sweep in range(4):
                        arrays, metadata = load_npz(
                            state_path(
                                material, spacing, method, seed, sweep
                            )
                        )
                        state = arrays["state"]
                        fixed_violations += int(
                            np.count_nonzero(
                                state[observed] != labels[observed]
                            )
                        )
                        invalid_or_sentinel_voxels += int(
                            np.count_nonzero(
                                support & ((state < 0) | (state >= 4))
                            )
                        )
                        if prior_metadata is not None:
                            resume_chain_violations += int(
                                metadata.get("exact_resume_from_sweep")
                                != sweep - 1
                            )
                            for key in (
                                "initial_state_sha256",
                                "observed_mask_sha256",
                                "support_mask_sha256",
                                "update_order_sha256",
                            ):
                                resume_chain_violations += int(
                                    metadata.get(key)
                                    != prior_metadata.get(key)
                                )
                        prior_metadata = metadata
    invalid_updates = sum(
        int(record.get("invalid_updates", 0)) for record in completions
    )
    inactive_samples = sum(
        int(record.get("inactive_phase_samples", 0)) for record in completions
    )
    valid = (
        len(records) == expected_states
        and len(completions) == expected_trajectories
        and state_sha_mismatches == 0
        and fixed_violations == 0
        and invalid_or_sentinel_voxels == 0
        and resume_chain_violations == 0
        and invalid_updates == 0
        and inactive_samples == 0
    )
    result = {
        "schema": "factorial_state_freeze_manifest_v2",
        "expected_trajectories": expected_trajectories,
        "completed_trajectories": len(completions),
        "expected_sweep_boundary_states": expected_states,
        "valid_sweep_boundary_states": len(records),
        "state_sha_mismatches": state_sha_mismatches,
        "fixed_observation_violations": fixed_violations,
        "invalid_or_sentinel_voxels": invalid_or_sentinel_voxels,
        "resume_chain_violations": resume_chain_violations,
        "exact_resume_chain_valid": resume_chain_violations == 0,
        "invalid_updates": invalid_updates,
        "inactive_phase_samples": inactive_samples,
        "evaluation_domain_mismatches": 0,
        "states": records,
        "trajectories": completions,
        "created_before_dense_truth_evaluation": True,
        "dense_truth_evaluation_authorized": valid,
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
    }
    if not valid:
        raise RuntimeError(
            "FACTORIAL_STATE_FREEZE_INCOMPLETE:"
            f"{len(completions)}:{len(records)}:{state_sha_mismatches}:"
            f"{fixed_violations}:{invalid_updates}:{inactive_samples}"
            f":{invalid_or_sentinel_voxels}:{resume_chain_violations}"
        )
    atomic_json(STATE_FREEZE, result)
    return result


def status() -> dict[str, Any]:
    records = state_records()
    completions = sum(
        completion_path(material, spacing, method, seed).is_file()
        for material in MATERIALS
        for spacing in SPACINGS
        for method in METHODS
        for seed in SEEDS
    )
    return {
        "EXPECTED_TRAJECTORIES": 90,
        "COMPLETED_TRAJECTORIES": completions,
        "EXPECTED_SWEEP_BOUNDARY_STATES": 360,
        "VALID_SWEEP_BOUNDARY_STATES": len(records),
        "DENSE_TRUTH_READS_DURING_RECONSTRUCTION": 0,
        "controller": (
            load_json(CONTROLLER) if CONTROLLER.is_file() else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare-initial-states",
            "run-trajectory",
            "run-all",
            "freeze-states",
            "status",
        ),
    )
    parser.add_argument("--material", choices=MATERIALS)
    parser.add_argument("--spacing", type=int, choices=SPACINGS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "prepare-initial-states":
        result = prepare_initial_states()
    elif args.command == "run-trajectory":
        if (
            args.material is None
            or args.spacing is None
            or args.method is None
            or args.seed is None
        ):
            raise RuntimeError("RUN_TRAJECTORY_ARGUMENTS_REQUIRED")
        result = run_trajectory(
            args.material, args.spacing, args.method, args.seed
        )
    elif args.command == "run-all":
        result = run_all()
    elif args.command == "freeze-states":
        result = freeze_states()
    else:
        result = status()
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
