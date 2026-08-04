"""Probability normalization, fusion, and ES-BPMR selection rules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ES_BPMR_THRESHOLD = 0.12690064311027527


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, 1e-12, None)
    return clipped / clipped.sum(axis=-1, keepdims=True)


def template_average(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    if len(probabilities) != 4:
        raise ValueError("template fusion requires four phase-template outputs")
    normalized = [normalize_probability(value) for value in probabilities]
    return normalize_probability(np.mean(np.stack(normalized), axis=0))


def orientation_average(probabilities: Sequence[np.ndarray]) -> np.ndarray:
    if len(probabilities) != 3:
        raise ValueError("orientation fusion requires x, y, and z outputs")
    normalized = [normalize_probability(value) for value in probabilities]
    return normalize_probability(np.mean(np.stack(normalized), axis=0))


def categorical_entropy(probability: np.ndarray) -> np.ndarray:
    probability = normalize_probability(probability)
    return -(probability * np.log(np.clip(probability, 1e-12, 1.0))).sum(axis=-1)


def entropy_select(
    b0_probability: np.ndarray,
    bpmr_probability: np.ndarray,
    threshold: float = ES_BPMR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    b0_probability = normalize_probability(b0_probability)
    bpmr_probability = normalize_probability(bpmr_probability)
    if b0_probability.shape != bpmr_probability.shape:
        raise ValueError("B0 and BPMR-Net probabilities must have equal shapes")
    use_bpmr = categorical_entropy(b0_probability) >= float(threshold)
    selected = np.where(use_bpmr[..., None], bpmr_probability, b0_probability)
    return normalize_probability(selected), use_bpmr
