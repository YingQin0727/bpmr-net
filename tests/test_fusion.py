import numpy as np

from bpmr_net.fusion import entropy_select, orientation_average, template_average


def test_template_and_orientation_fusion_are_normalized():
    rng = np.random.default_rng(2)
    templates = [rng.random((7, 4)) for _ in range(4)]
    orientations = [template_average(templates) for _ in range(3)]
    result = orientation_average(orientations)
    assert result.shape == (7, 4)
    assert np.allclose(result.sum(axis=1), 1.0)


def test_entropy_selector_returns_boolean_mask():
    b0 = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.999, 0.0003, 0.0003, 0.0004]])
    bpmr = np.asarray([[0.7, 0.1, 0.1, 0.1], [0.4, 0.2, 0.2, 0.2]])
    selected, mask = entropy_select(b0, bpmr)
    assert mask.dtype == bool
    assert mask.tolist() == [True, False]
    assert np.allclose(selected.sum(axis=1), 1.0)
