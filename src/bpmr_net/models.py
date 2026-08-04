"""Frozen B0 and BPMR-Net model definitions used by the paper workflow."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from .features import B0_OFFSETS, FEATURE_DIMENSION, PHASE_COUNT


B0_WIDTHS = (FEATURE_DIMENSION, 100, 50, 25, PHASE_COUNT)


class B0Network(nn.Module):
    """Four-layer reference network operating on 400-D neighborhood features."""

    def __init__(self, widths: Sequence[int] = B0_WIDTHS) -> None:
        super().__init__()
        widths = tuple(int(value) for value in widths)
        if widths != B0_WIDTHS:
            raise ValueError(f"B0 widths must be {B0_WIDTHS}, received {widths}")
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(left, right))
            layers.append(nn.Sigmoid() if index == 3 else nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_b0_model(seed: int) -> B0Network:
    """Build the deterministic B0 initialization used in paired experiments."""

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    return B0Network()


def derive_model_seed(
    paired_seed: int,
    condition_id: str,
    orientation: str,
    template_phase: int,
) -> int:
    payload = (
        f"STAGE5B|MODEL|{int(paired_seed)}|{condition_id}|"
        f"{orientation}|{int(template_phase)}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def features_to_patch(features: torch.Tensor) -> torch.Tensor:
    """Restore 80 feature slots into a five-channel 9 x 9 patch."""

    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"expected [N, {FEATURE_DIMENSION}] features")
    slots = features.reshape(features.shape[0], len(B0_OFFSETS), PHASE_COUNT + 1)
    patch = torch.zeros(
        (features.shape[0], PHASE_COUNT + 1, 9, 9),
        dtype=features.dtype,
        device=features.device,
    )
    offsets = torch.as_tensor(np.asarray(B0_OFFSETS), device=features.device)
    rows = offsets[:, 0] + 4
    columns = offsets[:, 1] + 4
    patch[:, :, rows, columns] = slots.permute(0, 2, 1)
    return patch


class BPMRNet(nn.Module):
    """Masked multi-receptive residual refinement over a frozen B0 network.

    The trainable residual branch contains exactly 383 parameters. At zero
    initialization its output is state-equivalent to the supplied B0 model.
    """

    def __init__(self, b0: nn.Module) -> None:
        super().__init__()
        self.b0 = b0
        for parameter in self.b0.parameters():
            parameter.requires_grad = False
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        5,
                        5,
                        kernel_size=3,
                        dilation=dilation,
                        padding=dilation,
                        groups=5,
                        bias=False,
                    ),
                    nn.Conv2d(5, 8, kernel_size=1),
                    nn.ReLU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.residual_head = nn.Linear(24, PHASE_COUNT)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.gate = nn.Linear(3, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.197224577)
        if self.trainable_parameter_count != 383:
            raise RuntimeError(
                f"unexpected BPMR-Net trainable parameter count: "
                f"{self.trainable_parameter_count}"
            )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def b0_probability_and_logits(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            raw = self.b0(features)
            probability = raw / torch.clamp(raw.sum(dim=1, keepdim=True), min=1e-12)
            logits = torch.log(torch.clamp(probability, min=1e-12))
        return probability, logits

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b0_probability, b0_logits = self.b0_probability_and_logits(features)
        patch = features_to_patch(features)
        representation = torch.cat(
            [branch(patch)[:, :, 4, 4] for branch in self.branches], dim=1
        )
        residual = self.residual_head(representation)
        entropy = -(
            b0_probability * torch.log(torch.clamp(b0_probability, min=1e-12))
        ).sum(dim=1, keepdim=True)
        top_two = torch.topk(b0_probability, k=2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        support_ratio = features.reshape(features.shape[0], 80, 5)[
            :, :, 4
        ].mean(dim=1, keepdim=True)
        gate = torch.sigmoid(
            self.gate(torch.cat([entropy, margin, support_ratio], dim=1))
        )
        logits = b0_logits + gate * residual
        probability = torch.softmax(logits, dim=1)
        return logits, probability, b0_logits, b0_probability


# Compatibility alias used in the frozen paper scripts.
R2Lite = BPMRNet
