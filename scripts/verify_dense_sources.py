#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
    "WF2_5um_1030x1033x3051_8b_segmented_corrected.raw": (
        3246233490,
        "932853c36761504216c3a734a852c08f6bf451ba6a5c4da9367a2ec8b724d22a",
    ),
    "GI_5um_1030x1033x3051_8b_segmented.raw": (
        3246233490,
        "35f336f0a4afea34110fe1f5cd4d1cca70ef128c7e06dad41370d652b97f8c94",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    failed = False
    for filename, (expected_bytes, expected_sha) in EXPECTED.items():
        path = args.data_root / filename
        if not path.is_file():
            print(f"MISSING {path}")
            failed = True
            continue
        actual_bytes = path.stat().st_size
        actual_sha = sha256(path)
        valid = actual_bytes == expected_bytes and actual_sha == expected_sha
        print(
            f"{'PASS' if valid else 'FAIL'} {filename} "
            f"bytes={actual_bytes} sha256={actual_sha}"
        )
        failed = failed or not valid
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
