import numpy as np
import torch

from bpmr_net.models import BPMRNet, build_b0_model, features_to_patch


def test_bpmr_shape_probability_and_parameter_count():
    model = BPMRNet(build_b0_model(1701))
    features = torch.from_numpy(
        np.random.default_rng(1).random((12, 400), dtype=np.float32)
    )
    with torch.no_grad():
        logits, probability, b0_logits, b0_probability = model(features)
    assert logits.shape == probability.shape == b0_logits.shape == b0_probability.shape
    assert probability.shape == (12, 4)
    assert torch.allclose(probability.sum(dim=1), torch.ones(12), atol=1e-6)
    assert torch.allclose(b0_probability.sum(dim=1), torch.ones(12), atol=1e-6)
    assert model.trainable_parameter_count == 383


def test_feature_patch_center_is_empty():
    patch = features_to_patch(torch.ones(3, 400))
    assert patch.shape == (3, 5, 9, 9)
    assert torch.count_nonzero(patch[:, :, 4, 4]) == 0
