#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", ".venv", ".venv-test", "__pycache__", "dist", "build"}
EXCLUDED_NAMES = {"release_manifest.csv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output = root / "release_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("relative_path", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output} files={len(rows)}")


if __name__ == "__main__":
    main()
