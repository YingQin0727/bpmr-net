#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as data:
        required = {
            "truth_labels",
            "valid_center_mask",
            "b0_class_probability",
            "bpmr_net_class_probability",
            "log_probability_improvement",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"missing arrays: {missing}")
        valid = np.asarray(data["valid_center_mask"], dtype=bool)
        b0 = np.asarray(data["b0_class_probability"])[valid]
        bpmr = np.asarray(data["bpmr_net_class_probability"])[valid]
        improvement = np.asarray(data["log_probability_improvement"])[valid]
        result = {
            "valid_pixels": int(valid.sum()),
            "b0_probability_sum_max_error": float(
                np.max(np.abs(b0.sum(axis=1) - 1.0))
            ),
            "bpmr_probability_sum_max_error": float(
                np.max(np.abs(bpmr.sum(axis=1) - 1.0))
            ),
            "mean_log_probability_improvement": float(improvement.mean()),
            "fraction_pixels_improved": float(np.mean(improvement > 0)),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
