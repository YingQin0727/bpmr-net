# Gate mechanism audit

## Decision

`SUPPORT_RATIO_DEGENERACY_CAUSE=FIXED_FULLY_SUPPORTED_WINDOW`

`GATE_MECHANISM_CLAIM_SUPPORTED=PARTIAL`

The held-out evidence supports descriptive entropy/margin-conditioned gate behavior and an A3/A4 registered contrast. It does **not** support the stronger claim that the gate is support-aware: support ratio is constant at 1.0 for every retained event.

## Why support ratio is constant

1. `paper_revision/training_events.py:18-24` selects centers only after full 9x9 support erosion. The comment and loop require all 81 pixels, including the center, to be supported.
2. `src/bpmr_net/features.py:64-86` encodes the 80 non-center offsets and stores each support flag in feature slot 4.
3. `paper_revision/models_revision.py:122-131` computes support ratio as the mean of those 80 support slots.
4. `paper_revision/heldout_evaluation_r0.py:201-207` applies the same full-support center construction to the held-out events. The R1 protocol therefore evaluates only fully supported neighborhoods.

This is a design consequence of the event selector, not a CSV aggregation error. It is also not a hidden test-label read. The feature constructor can represent missing support, but the frozen train/validation/test event selector removes every event with missing support before the gate sees it.

## Gate path and units

`gate_inputs` returns `[entropy, top-two probability margin, support_ratio]`. `forward` applies `sigmoid(Linear(3,1))` to those inputs and multiplies the residual logits. The evaluator records the gate value, residual magnitude and probability shift separately. Therefore a large gate is not itself a large probability correction; the correction is `gate * residual` followed by softmax.

R1 gate rows: 8024. Support values observed: 1. Mean/min/max learned gate: 0.41677 / 0.266861 / 0.60691.

| Correlation | Defined rows | Mean | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| gate vs entropy | 6956 | -0.626701 | -1 | 1 | Descriptive within event strata; not causal |
| gate vs margin | 6956 | 0.865042 | -1 | 1 | Descriptive within event strata; not causal |
| gate vs support | 0 | NA | NA | NA | Undefined when support has zero variance |

The evaluator's `corr` returns null when either vector has length <=1 or zero standard deviation (`paper_revision/heldout_evaluation_r0.py:227-228`). Empty correctness quadrants are omitted rather than assigned zero. This is appropriate missing-value handling, but it does not create evidence for a support effect.

## What can be claimed

- Safe: A3 has a learned scalar gate whose event-level values vary with the non-constant entropy/margin inputs, and the A3/A4 contrast is a registered descriptive ablation.
- Not safe: “the gate adapts to support sparsity,” “support-aware gating improves reconstruction,” or any causal claim from the correlations.
- Required 3-D follow-up: deliberately vary observation support or plane sparsity while holding clamp semantics and model identity fixed, then report gate and reconstruction/topology outcomes. That experiment is planned, not run here.

Frozen R1 audit status: `PASS`; no original R1 artifact was modified.
