import numpy as np

from bpmr_net.features import encode_b0_features, valid_center_coordinates


def test_masked_feature_encoding():
    labels = np.zeros((15, 15), dtype=np.int16)
    labels[:, 8:] = 2
    support = np.ones_like(labels, dtype=bool)
    support[7, 8] = False
    features = encode_b0_features(labels, support, [7], [7])
    slots = features.reshape(1, 80, 5)
    assert features.shape == (1, 400)
    assert np.all((slots[:, :, :4].sum(axis=2) == 1) | (slots[:, :, 4] == 0))
    assert slots[:, :, 4].sum() == 79


def test_valid_centers_respect_radius():
    support = np.ones((12, 13), dtype=bool)
    rows, columns = valid_center_coordinates(support)
    assert len(rows) == (12 - 8) * (13 - 8)
    assert rows.min() == columns.min() == 4
