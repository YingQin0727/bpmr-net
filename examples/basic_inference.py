from __future__ import annotations

import numpy as np
import torch

from bpmr_net import BPMRNet, build_b0_model, encode_b0_features


labels = np.zeros((20, 20), dtype=np.int16)
labels[:, 10:] = 1
labels[10:, :] += 2
support = np.ones_like(labels, dtype=bool)
rows, columns = np.mgrid[4:16, 4:16]
features = encode_b0_features(labels, support, rows.ravel(), columns.ravel())

model = BPMRNet(build_b0_model(seed=1701))
with torch.no_grad():
    _, probability, _, _ = model(torch.from_numpy(features))

print("features:", features.shape)
print("probabilities:", tuple(probability.shape))
print("trainable refinement parameters:", model.trainable_parameter_count)
