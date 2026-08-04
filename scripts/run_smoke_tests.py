#!/usr/bin/env python3
"""Run the public assert-based tests without requiring pytest."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    count = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            value = getattr(module, name)
            if name.startswith("test_") and callable(value):
                value()
                count += 1
                print(f"PASS {path.name}::{name}")
    print(f"SMOKE_TESTS_PASS count={count}")


if __name__ == "__main__":
    main()
