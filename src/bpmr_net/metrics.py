"""Small dependency-free metric set used by examples and smoke tests."""

from __future__ import annotations

import numpy as np


def normalized_categorical_nll(
    probability: np.ndarray, target_labels: np.ndarray
) -> float:
    probability = np.asarray(probability)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    selected = probability[np.arange(len(target_labels)), target_labels]
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def multiclass_brier(probability: np.ndarray, target_labels: np.ndarray) -> float:
    probability = np.asarray(probability)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    targets = np.eye(probability.shape[1], dtype=np.float64)[target_labels]
    return float(np.mean(np.sum((probability - targets) ** 2, axis=1)))


def true_label_probability(
    probability: np.ndarray, target_labels: np.ndarray
) -> np.ndarray:
    probability = np.asarray(probability)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    return probability[np.arange(len(target_labels)), target_labels]


def log_probability_improvement(
    b0_probability: np.ndarray,
    bpmr_probability: np.ndarray,
    target_labels: np.ndarray,
) -> np.ndarray:
    b0_true = true_label_probability(b0_probability, target_labels)
    bpmr_true = true_label_probability(bpmr_probability, target_labels)
    return np.log(np.clip(bpmr_true, 1e-12, 1.0)) - np.log(
        np.clip(b0_true, 1e-12, 1.0)
    )
