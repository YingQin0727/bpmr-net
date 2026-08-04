"""Command-line synthetic smoke demo for the public package."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .features import encode_b0_features, valid_center_coordinates
from .fusion import entropy_select
from .models import BPMRNet, build_b0_model


def run_demo(seed: int = 1701, size: int = 24) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=(size, size), dtype=np.int16)
    yy, xx = np.ogrid[:size, :size]
    support = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.46) ** 2
    rows, columns = valid_center_coordinates(support)
    features = encode_b0_features(labels, support, rows, columns)
    b0 = build_b0_model(seed)
    model = BPMRNet(b0)
    tensor = torch.from_numpy(features)
    with torch.no_grad():
        _, bpmr_probability, _, b0_probability = model(tensor)
    selected, selector = entropy_select(
        b0_probability.numpy(), bpmr_probability.numpy()
    )
    return {
        "event_count": int(len(features)),
        "feature_shape": list(features.shape),
        "bpmr_trainable_parameters": model.trainable_parameter_count,
        "b0_probability_sum_max_error": float(
            np.max(np.abs(b0_probability.numpy().sum(axis=1) - 1.0))
        ),
        "bpmr_probability_sum_max_error": float(
            np.max(np.abs(bpmr_probability.numpy().sum(axis=1) - 1.0))
        ),
        "selector_usage_fraction": float(selector.mean()),
        "selected_probability_shape": list(selected.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--size", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.seed, args.size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
