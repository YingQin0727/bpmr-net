"""Controlled model variants for the MTCOMM major-revision experiments.

These classes intentionally live outside ``src/bpmr_net/models.py`` so the
submitted-paper implementation remains frozen.  The revision variants preserve
BPMR-Net's residual construction while changing one registered factor at a
time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from bpmr_net.features import B0_OFFSETS, PHASE_COUNT
from bpmr_net.models import features_to_patch


DEFAULT_DILATIONS = (1, 2, 4)
MATCHED_SINGLE_SCALE_DILATIONS = (1, 1, 1)


class RevisionBPMRNet(nn.Module):
    """Revision-only BPMR variant with registered gate/RF controls.

    Parameters
    ----------
    b0:
        Source B0 predictor.  It is frozen in all BPMR variants.
    dilations:
        Three branch dilation values.  ``(1, 2, 4)`` reproduces the submitted
        model; ``(1, 1, 1)`` is the matched-parameter single-scale ablation.
    gate_mode:
        ``"adaptive"`` uses entropy, top-two margin and support ratio exactly
        as the submitted model. ``"constant"`` retains the same Linear(3,1)
        module but fixes its three input weights to zero so only the scalar
        bias can modulate residual magnitude.
    """

    def __init__(
        self,
        b0: nn.Module,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        gate_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        dilations = tuple(int(value) for value in dilations)
        if len(dilations) != 3 or any(value < 1 for value in dilations):
            raise ValueError(f"expected three positive dilations, got {dilations}")
        if gate_mode not in {"adaptive", "constant"}:
            raise ValueError(f"unsupported gate_mode: {gate_mode}")

        self.b0 = b0
        for parameter in self.b0.parameters():
            parameter.requires_grad = False

        self.dilations = dilations
        self.gate_mode = gate_mode
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
                for dilation in dilations
            ]
        )
        self.residual_head = nn.Linear(24, PHASE_COUNT)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        # Keep the same module shape as the submitted BPMR-Net.  In constant
        # mode the three weights are frozen at zero and only the bias trains.
        self.gate = nn.Linear(3, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.197224577)
        if gate_mode == "constant":
            self.gate.weight.requires_grad = False

        if self.total_residual_parameter_count != 383:
            raise RuntimeError(
                "revision BPMR residual parameter structure changed: "
                f"{self.total_residual_parameter_count}"
            )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def total_residual_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("b0.")
        )

    def b0_probability_and_logits(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            raw = self.b0(features)
            probability = raw / torch.clamp(raw.sum(dim=1, keepdim=True), min=1e-12)
            logits = torch.log(torch.clamp(probability, min=1e-12))
        return probability, logits

    def gate_inputs(self, features: torch.Tensor, b0_probability: torch.Tensor) -> torch.Tensor:
        entropy = -(
            b0_probability * torch.log(torch.clamp(b0_probability, min=1e-12))
        ).sum(dim=1, keepdim=True)
        top_two = torch.topk(b0_probability, k=2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        support_ratio = features.reshape(features.shape[0], len(B0_OFFSETS), 5)[
            :, :, 4
        ].mean(dim=1, keepdim=True)
        return torch.cat([entropy, margin, support_ratio], dim=1)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        b0_probability, b0_logits = self.b0_probability_and_logits(features)
        patch = features_to_patch(features)
        representation = torch.cat(
            [branch(patch)[:, :, 4, 4] for branch in self.branches], dim=1
        )
        residual = self.residual_head(representation)
        gate_inputs = self.gate_inputs(features, b0_probability)
        gate = torch.sigmoid(self.gate(gate_inputs))
        logits = b0_logits + gate * residual
        probability = torch.softmax(logits, dim=1)
        return (
            logits,
            probability,
            b0_logits,
            b0_probability,
            gate,
            residual,
        )


def build_revision_bpmr(
    b0: nn.Module,
    variant_id: str,
) -> RevisionBPMRNet:
    """Construct one registered BPMR revision variant."""

    if variant_id in {"A2", "A3"}:
        return RevisionBPMRNet(
            b0,
            dilations=DEFAULT_DILATIONS,
            gate_mode="adaptive",
        )
    if variant_id == "A4":
        return RevisionBPMRNet(
            b0,
            dilations=DEFAULT_DILATIONS,
            gate_mode="constant",
        )
    if variant_id == "A5":
        return RevisionBPMRNet(
            b0,
            dilations=MATCHED_SINGLE_SCALE_DILATIONS,
            gate_mode="adaptive",
        )
    raise ValueError(f"variant {variant_id} is not a BPMR residual variant")
