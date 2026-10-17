"""Pre-flight audit for the isolated MTCOMM revision workflow.

This audit is deliberately data-presence only: it never reconstructs frozen
manifests from dense truth and never reads dense truth.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "revision_mtcomm_2026" / "manifests" / "R1_EXEC_P0_PREFLIGHT.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main() -> None:
    expected = {
        "paper_revision/README.md": "required frozen workflow README",
        "paper_revision/revision_protocol.json": "frozen revision protocol",
        "src/bpmr_net/models.py": "submitted-paper model",
        "paper_reproduction/factorial_protocol_and_training_v2.py": "training workflow",
        "paper_reproduction/factorial_multiseed_reconstruction_v2.py": "reconstruction workflow",
        "paper_reproduction/factorial_evaluation_and_package_v2.py": "evaluation workflow",
    }
    tracked = run("git", "ls-files").splitlines()
    modified = set(run("git", "diff", "--name-only").splitlines())
    branch = run("git", "branch", "--show-current")
    checks = {}
    for relative, purpose in expected.items():
        path = ROOT / relative
        checks[relative] = {
            "purpose": purpose,
            "exists": path.is_file(),
            "sha256": sha256(path),
            "tracked": relative in tracked,
        }
    protected = [
        path for path in modified if path.startswith("src/bpmr_net/") or path.startswith("paper_reproduction/")
    ]
    required_inputs = {
        "frozen_final_v1_registry": ROOT / "data" / "final_v1",
        "registered_sparse_plane_store": ROOT / "data" / "plane_store",
        "baseline_archive": ROOT / "data" / "baseline",
        "legacy_3d_runtime": ROOT / "external" / "legacy_r2",
        "source_b0_checkpoints": ROOT / "checkpoints",
    }
    inputs = {
        name: {"path": str(path), "exists": path.exists(), "nonempty": path.is_dir() and any(path.iterdir())}
        for name, path in required_inputs.items()
    }
    report = {
        "schema": "mtcomm_r1_exec_p0_preflight_v1",
        "branch": {"expected": "revision/mtcomm-major-2026", "actual": branch, "pass": branch == "revision/mtcomm-major-2026"},
        "head_sha": run("git", "rev-parse", "HEAD"),
        "working_tree_clean": not bool(run("git", "status", "--short")),
        "protected_path_modifications": protected,
        "protected_paths_unchanged": not protected,
        "required_files": checks,
        "revision_output_root": str(ROOT / "outputs" / "revision_mtcomm_2026"),
        "factorial_output_root": str(ROOT / "outputs" / "factorial_v2"),
        "output_roots_nonoverlapping": (ROOT / "outputs" / "revision_mtcomm_2026") != (ROOT / "outputs" / "factorial_v2"),
        "research_inputs": inputs,
        "research_archive_complete": all(item["exists"] and item["nonempty"] for item in inputs.values()),
        "classification": "BLOCKED",
        "blocking_items": [
            *(["paper_revision/README.md is missing"] if not checks["paper_revision/README.md"]["exists"] else []),
            *[f"{name} is unavailable or empty" for name, item in inputs.items() if not (item["exists"] and item["nonempty"])],
            "source B0 checkpoint and frozen event manifests are unavailable; no pilot may run",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
