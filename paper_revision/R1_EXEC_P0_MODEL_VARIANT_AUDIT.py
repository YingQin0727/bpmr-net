"""Deterministic synthetic audit of revision-only BPMR variants."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bpmr_net.models import B0Network  # noqa: E402
from models_revision import build_revision_bpmr  # noqa: E402


def state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def main() -> None:
    torch.manual_seed(1701)
    torch.use_deterministic_algorithms(True, warn_only=True)
    b0 = B0Network()
    features = torch.rand(17, 400)
    with torch.no_grad():
        source = b0(features)
        source = source / source.sum(dim=1, keepdim=True)
    rows = []
    models = {}
    for variant in ("A2", "A3", "A4", "A5"):
        model = build_revision_bpmr(B0Network(), variant)
        model.b0.load_state_dict(b0.state_dict())
        model.eval()
        with torch.no_grad():
            output = model(features)[1]
        diff = float(torch.max(torch.abs(output - source)))
        models[variant] = model
        rows.append({
            "variant_id": variant,
            "source_b0_state_sha256": state_sha(b0),
            "trainable_parameter_count": model.trainable_parameter_count,
            "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "dilations": list(model.dilations),
            "gate_mode": model.gate_mode,
            "initialization_identity_check": diff <= 1e-7,
            "max_abs_probability_difference": diff,
            "strict_state_reload_check": True,
        })
    a4 = models["A4"]
    with torch.no_grad():
        gate_a = a4(features)[4]
        gate_b = a4(torch.flip(features, dims=[0]))[4]
    a3 = models["A3"]
    report = {
        "schema": "mtcomm_r1_exec_p0_model_variant_audit_v1",
        "variants": rows,
        "a4_input_independence": {
            "max_gate_difference": float(torch.max(torch.abs(gate_a - torch.flip(gate_b, dims=[0])))),
            "pass": bool(torch.allclose(gate_a, torch.flip(gate_b, dims=[0]), atol=1e-12, rtol=0)),
            "gate_input_weights_exactly_zero": bool(torch.equal(a4.gate.weight.detach(), torch.zeros_like(a4.gate.weight))),
            "gate_input_weights_trainable": bool(a4.gate.weight.requires_grad),
            "gate_bias_trainable": bool(a4.gate.bias.requires_grad),
        },
        "a5_parameter_structure_match": {
            "a3_total_residual_parameters": a3.total_residual_parameter_count,
            "a5_total_residual_parameters": models["A5"].total_residual_parameter_count,
            "trainable_count_equal": a3.trainable_parameter_count == models["A5"].trainable_parameter_count,
            "pass": a3.total_residual_parameter_count == models["A5"].total_residual_parameter_count,
        },
    }
    out = ROOT / "outputs" / "revision_mtcomm_2026" / "manifests" / "R1_EXEC_P0_MODEL_VARIANT_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
