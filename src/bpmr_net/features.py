"""Feature construction for four-phase masked 2-D neighborhoods."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


PHASE_COUNT = 4
B0_OFFSETS = tuple(
    (row, column)
    for row in range(-4, 5)
    for column in range(-4, 5)
    if (row, column) != (0, 0)
)
SLOT_WIDTH = PHASE_COUNT + 1
FEATURE_DIMENSION = len(B0_OFFSETS) * SLOT_WIDTH


def deterministic_uniform_event_indices(
    event_count: int, maximum_events: int, seed: int
) -> np.ndarray:
    if event_count < 0 or maximum_events < 1:
        raise ValueError("invalid event count or cap")
    if event_count <= maximum_events:
        return np.arange(event_count, dtype=np.int64)
    return np.sort(
        np.random.default_rng(int(seed)).choice(
            event_count, size=maximum_events, replace=False
        )
    )


def valid_center_coordinates(
    support: np.ndarray, radius: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(support, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("support must be a 2-D boolean array")
    if radius < 1 or min(valid.shape) <= 2 * radius:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    inner = valid[radius:-radius, radius:-radius]
    flat = np.flatnonzero(inner.ravel())
    rows = flat // inner.shape[1] + radius
    columns = flat % inner.shape[1] + radius
    return rows.astype(np.int64), columns.astype(np.int64)


def encode_b0_features(
    labels: np.ndarray,
    support: np.ndarray,
    rows: Sequence[int] | np.ndarray,
    columns: Sequence[int] | np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int16)
    support = np.asarray(support, dtype=bool)
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if labels.ndim != 2 or labels.shape != support.shape:
        raise ValueError("labels and support must be matching 2-D arrays")
    if rows.shape != columns.shape or rows.ndim != 1:
        raise ValueError("rows and columns must be matching 1-D arrays")
    offsets = np.asarray(B0_OFFSETS, dtype=np.int64)
    neighbor_rows = rows[:, None] + offsets[None, :, 0]
    neighbor_columns = columns[:, None] + offsets[None, :, 1]
    if len(rows) and (
        neighbor_rows.min() < 0
        or neighbor_columns.min() < 0
        or neighbor_rows.max() >= labels.shape[0]
        or neighbor_columns.max() >= labels.shape[1]
    ):
        raise ValueError("center neighborhood is out of bounds")
    valid = support[neighbor_rows, neighbor_columns]
    neighbor_labels = labels[neighbor_rows, neighbor_columns]
    if np.any(valid & ((neighbor_labels < 0) | (neighbor_labels >= PHASE_COUNT))):
        raise ValueError("active support contains an illegal phase label")
    slots = np.zeros((len(rows), len(B0_OFFSETS), SLOT_WIDTH), dtype=np.float32)
    event_indices, slot_indices = np.nonzero(valid)
    slots[
        event_indices,
        slot_indices,
        neighbor_labels[event_indices, slot_indices],
    ] = 1.0
    slots[:, :, PHASE_COUNT] = valid
    return slots.reshape(len(rows), FEATURE_DIMENSION)
