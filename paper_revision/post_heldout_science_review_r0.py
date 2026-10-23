"""Read-only post-held-out science review and minimal 3-D planning.

This script consumes frozen R1 CSV/JSON outputs and source-code text only.  It
does not load checkpoints, labels, support arrays, or test payloads, and it
does not change any frozen evaluation artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "outputs/revision_mtcomm_2026/REVISION_REPLICATION_evaluation/DesignA/HELDOUT_R1"
REGISTRY = ROOT / "outputs/revision_mtcomm_2026/R1_RECONSTITUTION_P1_PLANE_REGISTRY.csv"
REVIEWER = ROOT / "outputs/revision_mtcomm_2026/R1_RECONSTITUTION_P0_REVIEWER_COVERAGE.csv"
OUT = ROOT / "paper_revision/post_heldout_science_review_r0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: str | None) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    return float(value)


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def fmt(value: float | None, digits: int = 6) -> str:
    return "NA" if value is None else f"{value:.{digits}g}"


def write_text(name: str, text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def phase_tables(phase_rows: list[dict[str, str]], paired_rows: list[dict[str, str]]) -> str:
    phase_names = {0: "oil", 1: "brine", 2: "rock", 3: "gas"}
    variants = ("A1", "A3", "A4", "A5")
    lines = [
        "# Phase-resolved science summary",
        "",
        "REVISION_DESIGNA_POST_HELDOUT_SCIENCE_REVIEW_R0 is read-only. The R1 evaluator used fixed four-class semantics: phase 0 oil, 1 brine, 2 rock, 3 gas; 255 is unsupported.",
        "",
        "## Interpretation",
        "",
        "The held-out task is 2-D teacher-forced neighborhood classification. Every test event exposes the true supported 80-neighbor plane features and predicts the center phase. It is not a 3-D rollout or a topology-preserving reconstruction. No phase-extinction rows were found in R1 (positive truth prevalence with zero predicted prevalence: 0). This does not establish 3-D non-extinction.",
        "",
        "Mean phase metrics are averaged over registered conditions within each material and analysis set. They are descriptive condition summaries, not independent-material inference.",
    ]
    for aset in ("PRIMARY", "SENSITIVITY"):
        lines += ["", f"## {aset} phase means", "", "| Material | Phase | A1 recall | A1 F1 | A1 prevalence error | A3 recall | A3 F1 | A3 prevalence error | A4 recall | A4 F1 | A4 prevalence error | A5 recall | A5 F1 | A5 prevalence error |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for material in ("WF2", "GI"):
            for phase in range(4):
                cells = []
                for variant in variants:
                    subset = [
                        row for row in phase_rows
                        if row["analysis_set"] == aset and row["material"] == material
                        and row["variant"] == variant and int(row["phase"]) == phase
                    ]
                    cells += [
                        fmt(mean([number(row[metric]) for row in subset]))
                        for metric in ("recall", "f1", "prevalence_error")
                    ]
                lines.append(f"| {material} | {phase} ({phase_names[phase]}) | " + " | ".join(cells) + " |")

    lines += ["", "## Registered paired phase differences", "", "Differences are right minus left, matching R1. Negative prevalence-error deltas mean a smaller signed predicted-minus-truth error only when interpreted with the baseline sign; absolute prevalence error remains the safer diagnostic.", "", "| Analysis set | Material | Comparison | Phase | Mean recall delta | Mean F1 delta | Mean prevalence-error delta | Negative F1 rows / total |", "|---|---|---|---:|---:|---:|---:|---:|"]
    pair_map = {"A1_vs_A3": ("A1", "A3"), "A3_vs_A4": ("A3", "A4"), "A3_vs_A5": ("A3", "A5")}
    for aset in ("PRIMARY", "SENSITIVITY"):
        for material in ("WF2", "GI"):
            for comparison, (left, right) in pair_map.items():
                for phase in range(4):
                    left_rows = {(row["unit"], int(row["phase"])): row for row in phase_rows if row["analysis_set"] == aset and row["material"] == material and row["variant"] == left and int(row["phase"]) == phase}
                    right_rows = {(row["unit"], int(row["phase"])): row for row in phase_rows if row["analysis_set"] == aset and row["material"] == material and row["variant"] == right and int(row["phase"]) == phase}
                    deltas = {metric: [] for metric in ("recall", "f1", "prevalence_error")}
                    for key in sorted(left_rows.keys() & right_rows.keys()):
                        for metric in deltas:
                            lv = number(left_rows[key][metric]); rv = number(right_rows[key][metric])
                            if lv is not None and rv is not None:
                                deltas[metric].append(rv - lv)
                    neg = sum(value < 0 for value in deltas["f1"])
                    lines.append(f"| {aset} | {material} | {comparison} | {phase} ({phase_names[phase]}) | {fmt(mean(deltas['recall']))} | {fmt(mean(deltas['f1']))} | {fmt(mean(deltas['prevalence_error']))} | {neg}/{len(deltas['f1'])} |")

    lines += [
        "",
        "## WF2 phase 0 focus",
        "",
        "WF2 phase 0 is the registered oil phase. The comparison tables retain it even when prevalence is low; no phase is removed for an unfavorable result. The R1 evidence supports reporting its 2-D recall/F1 and prevalence error, but not a claim that oil remains connected, non-extinct, or correctly represented in 3-D.",
        "",
        f"Source metrics: `{(R1 / 'REVISION_DESIGNA_HELDOUT_R1_PHASE_METRICS.csv').relative_to(ROOT)}` (SHA256 `{sha(R1 / 'REVISION_DESIGNA_HELDOUT_R1_PHASE_METRICS.csv')}`).",
    ]
    return "\n".join(lines)


def seed_audit(seed_rows: list[dict[str, str]]) -> str:
    counts = Counter(int(row["n_training_seeds"]) for row in seed_rows)
    primary = Counter(int(row["n_training_seeds"]) for row in seed_rows if row["analysis_set"] == "PRIMARY")
    sensitivity = Counter(int(row["n_training_seeds"]) for row in seed_rows if row["analysis_set"] == "SENSITIVITY")
    ci_defined = sum(number(row["conditional_seed_ci95_lower"]) is not None for row in seed_rows)
    return f"""# Uncertainty scope audit

The only uncertainty table used here is the frozen post-processing output `{(R1 / 'REVISION_DESIGNA_HELDOUT_R1_SEED_UNCERTAINTY.csv').relative_to(ROOT)}` (SHA256 `{sha(R1 / 'REVISION_DESIGNA_HELDOUT_R1_SEED_UNCERTAINTY.csv')}`). No event bootstrap, event-level p-value, or material-population confidence interval is valid under the frozen protocol.

| Quantity | Value |
|---|---:|
| Seed-conditional rows | {len(seed_rows)} |
| Rows with one registered training seed (CI undefined) | {counts[1]} |
| Rows with three registered training seeds | {counts[3]} |
| Rows with pointwise df=2 t CI | {ci_defined} |
| PRIMARY rows (n=1 / n=3) | {primary[1]} / {primary[3]} |
| SENSITIVITY rows (n=1 / n=3) | {sensitivity[1]} / {sensitivity[3]} |

The n=3 interval is conditional on fixed material, spacing, orientation, template and shared test events; it describes training-seed variation only. The n=1 rows must be reported as CI undefined. The 3 TIER_2 models remain in the pre-frozen SENSITIVITY set and do not become TIER_1 through this review. The two source volumes, correlated ROIs/orientations, and shared events do not support broad material generalization or event-level pseudoreplication.

R1 PRIMARY/SENSITIVITY pair counts remain 714/720. No model was requalified after reading held-out results.
"""


def matrix_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    def add(tier: str, comment: str, material: str, roi: int, model: str, global_um: int, local_um: int, clamp: str, seed: int, order: str, init: str, note: str) -> None:
        index = len(rows) + 1
        rows.append({
            "trajectory_id": f"3D-{index:03d}",
            "tier": tier,
            "reviewer_comments": comment,
            "material": material,
            "roi": str(roi),
            "model": model,
            "global_fit_um": str(global_um),
            "local_observation_um": str(local_um),
            "clamp_mode": clamp,
            "training_seed": str(seed),
            "rollout_seed": str(seed),
            "sweeps_saved": "0;1;3;5;10",
            "update_order": order,
            "initialization": init,
            "required_outputs": "phase_fraction;phase_extinction;components;largest_component;Euler;TPCF;chords;gate;XY_XZ_YZ_slices;runtime",
            "paired_control": "B0_SOURCE_vs_A3_same_condition" if model == "A3" else "paired_with_A3_same_condition",
            "estimated_reference_cpu_hours": "0.71",
            "estimated_optimized_cpu_hours": "0.41_unvalidated",
            "gpu_status": "UNDETERMINED",
            "notes": note,
        })
    # Core: all three registered ROIs, both materials, B0/A3, two pre-registered sparsities.
    for material in ("WF2", "GI"):
        for roi in (0, 1, 2):
            for model in ("B0_SOURCE", "A3"):
                for local in (25, 1000):
                    add("MUST", "R1-C2;R2-C3;R2-C4;R2-C10", material, roi, model, 250, local, "hard_observed", 1701, "forward", "train_prior", "All registered ROIs; no performance-based ROI selection. Save the same five sweep states for truth/B0/A3 and orthogonal slices.")
    # Seed sensitivity: fixed registry-first ROI, both paired models, two additional seeds.
    for material in ("WF2", "GI"):
        for model in ("B0_SOURCE", "A3"):
            for seed in (1702, 1703):
                add("MUST", "R2-C5", material, 0, model, 250, 25, "hard_observed", seed, "forward", "train_prior", "ROI0 is selected by registry order before any 3-D result; seed variation is conditional, not material generalization.")
    # One-factor order and initialization controls.
    for material in ("WF2", "GI"):
        for order, init, label in (("reverse", "train_prior", "update-order"), ("forward", "uniform", "initialization")):
            add("MUST", "R2-C5", material, 0, "A3", 250, 25, "hard_observed", 1701, order, init, f"One-factor {label} control against forward/train_prior; no interaction claim.")
    # Global/local density cross at fixed local observation density.
    for material in ("WF2", "GI"):
        for global_um in (250, 1000):
            add("MUST", "R2-C6", material, 0, "A3", global_um, 25, "hard_observed", 1701, "forward", "train_prior", "Global fit and local observation spacing are recorded separately; same ROI, mask identity and clamp semantics.")
    # Explicit clamp-by-sparsity diagnostic. No-clamp rows are diagnostic, not primary.
    for material in ("WF2", "GI"):
        for local in (25, 1000):
            for clamp in ("hard_observed", "no_clamp_diagnostic"):
                add("MUST", "R1-C2;R2-C5;R2-C6", material, 0, "A3", 250, local, clamp, 1701, "forward", "train_prior", "Separates clamping from plane sparsity; no-clamp is a diagnostic control and cannot replace hard-data results.")
    assert len(rows) == 48
    return rows


def write_matrix(rows: list[dict[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "MINIMAL_3D_EXPERIMENT_MATRIX.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    metrics = read_csv(R1 / "REVISION_DESIGNA_HELDOUT_R1_ALL_METRICS.csv")
    phases = read_csv(R1 / "REVISION_DESIGNA_HELDOUT_R1_PHASE_METRICS.csv")
    paired = read_csv(R1 / "REVISION_DESIGNA_HELDOUT_R1_PAIRED_COMPARISONS.csv")
    gates = read_csv(R1 / "REVISION_DESIGNA_HELDOUT_R1_GATE_DIAGNOSTICS.csv")
    seed_rows = read_csv(R1 / "REVISION_DESIGNA_HELDOUT_R1_SEED_UNCERTAINTY.csv")
    audit = read_json(R1 / "REVISION_DESIGNA_HELDOUT_R1_FINAL_AUDIT.json")
    registry = read_csv(REGISTRY)
    reviewer = read_csv(REVIEWER)

    support_values = sorted({number(row["support_ratio"]) for row in gates if number(row["support_ratio"]) is not None})
    support_counts = Counter(int(row["count"]) for row in gates)
    corr_fields = ("gate_entropy_pearson", "gate_margin_pearson", "gate_support_pearson")
    corr_summary = {}
    for field in corr_fields:
        values = [number(row[field]) for row in gates]
        values = [value for value in values if value is not None]
        corr_summary[field] = dict(defined=len(values), mean=mean(values), minimum=min(values) if values else None, maximum=max(values) if values else None)
    gate_values = [number(row["learned_gate"]) for row in gates]
    gate_values = [value for value in gate_values if value is not None]
    support_audit = f"""# Gate mechanism audit

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

R1 gate rows: {len(gates)}. Support values observed: {", ".join(fmt(value) for value in support_values)}. Mean/min/max learned gate: {fmt(mean(gate_values))} / {fmt(min(gate_values))} / {fmt(max(gate_values))}.

| Correlation | Defined rows | Mean | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|
| gate vs entropy | {corr_summary['gate_entropy_pearson']['defined']} | {fmt(corr_summary['gate_entropy_pearson']['mean'])} | {fmt(corr_summary['gate_entropy_pearson']['minimum'])} | {fmt(corr_summary['gate_entropy_pearson']['maximum'])} | Descriptive within event strata; not causal |
| gate vs margin | {corr_summary['gate_margin_pearson']['defined']} | {fmt(corr_summary['gate_margin_pearson']['mean'])} | {fmt(corr_summary['gate_margin_pearson']['minimum'])} | {fmt(corr_summary['gate_margin_pearson']['maximum'])} | Descriptive within event strata; not causal |
| gate vs support | {corr_summary['gate_support_pearson']['defined']} | {fmt(corr_summary['gate_support_pearson']['mean'])} | {fmt(corr_summary['gate_support_pearson']['minimum'])} | {fmt(corr_summary['gate_support_pearson']['maximum'])} | Undefined when support has zero variance |

The evaluator's `corr` returns null when either vector has length <=1 or zero standard deviation (`paper_revision/heldout_evaluation_r0.py:227-228`). Empty correctness quadrants are omitted rather than assigned zero. This is appropriate missing-value handling, but it does not create evidence for a support effect.

## What can be claimed

- Safe: A3 has a learned scalar gate whose event-level values vary with the non-constant entropy/margin inputs, and the A3/A4 contrast is a registered descriptive ablation.
- Not safe: “the gate adapts to support sparsity,” “support-aware gating improves reconstruction,” or any causal claim from the correlations.
- Required 3-D follow-up: deliberately vary observation support or plane sparsity while holding clamp semantics and model identity fixed, then report gate and reconstruction/topology outcomes. That experiment is planned, not run here.

Frozen R1 audit status: `{audit['status']}`; no original R1 artifact was modified.
"""
    write_text("GATE_MECHANISM_AUDIT.md", support_audit)
    write_text("PHASE_RESOLVED_SCIENCE_SUMMARY.md", phase_tables(phases, paired))
    write_text("UNCERTAINTY_SCOPE_AUDIT.md", seed_audit(seed_rows))
    rows = matrix_rows()
    write_matrix(rows)

    write_text("THREE_D_COST_AND_RISK_PLAN.md", f"""# Minimal 3-D cost and risk plan

`MINIMAL_3D_TRAJECTORY_COUNT={len(rows)}`. The matrix is a plan only; no 3-D trajectory, reconstruction, comparator or new test read was started in this review.

## Cost basis

The frozen P2 budget measured approximately 0.709 CPU hours per reference trajectory and 0.410 hours per optimized trace/vector trajectory. The optimized path is not a validated scientific worker and is therefore not the planning baseline.

| Component | Reference CPU hours | Optimized path (unvalidated) |
|---|---:|---:|
| {len(rows)} planned trajectories | {len(rows) * 0.709:.1f} | {len(rows) * 0.410:.1f} |
| Evaluation, plots, hash/restart reserve | 8.0 | 8.0 |
| Planning estimate | **{len(rows) * 0.709 + 8:.1f}** | **{len(rows) * 0.410 + 8:.1f}** |

`ESTIMATED_CPU_HOURS={len(rows) * 0.709 + 8:.1f}` for a conservative reference implementation. This excludes implementation of the published comparator and any failed-isolation recovery. The existing budget reports no measured compatible GPU implementation; therefore `GPU_REQUIRED=UNDETERMINED`. A local CPU run is plausible for this reduced matrix, but GPU acceleration cannot be claimed.

## Must complete for the core reviewer questions

- Paired B0/A3 on WF2 and GI, all three frozen ROI identities, both registered local observation spacings, fixed hard clamping, and saved sweeps 0, 1, 3, 5 and 10.
- Phase fractions, extinction counts, connected components/largest component, Euler/connectivity, TPCF, chord statistics and fixed orthogonal XY/XZ/YZ slices using the same truth/B0/A3 states.
- One-factor order and initialization controls, with the same observation masks and deterministic seeds.
- Runtime, memory, checkpoint identity and no-test-label training provenance for every trajectory.

## Helpful if cost permits

- The pre-specified three-seed sensitivity rows in the matrix.
- The global-fit/local-observation cross and the clamp-by-sparsity diagnostic rows.
- A separately validated Yeong–Torquato-style hard-data comparator under identical masks and observation exclusion. It is not included in the count because its implementation and cost are not validated.
- A measured GPU pilot after the CPU reference path is correct; no GPU result may replace the reference audit.

## Explicit limitations

The matrix does not establish independent-material generalization, convergence beyond 10 sweeps, effective properties, or a complete six-variant 3-D factorial. The no-clamp rows are diagnostic controls and cannot be reported as the main reconstruction protocol. The current 2-D evidence cannot be used to choose a favorable ROI, seed or checkpoint.
""")

    reviewer_lines = [
        "# Reviewer comment evidence map",
        "",
        "This map combines the frozen reviewer coverage and the new read-only R1 science review. It does not change any reviewer qualification or experiment membership.",
        "",
        "| Comment | Current direct evidence | Status after R1 review | Minimal remaining evidence |",
        "|---|---|---|---|",
    ]
    status = {
        "R1-C1": ("Editorial definition only", "Text-only; no new science needed", "Define WF2/GI on first use"),
        "R1-C2": ("R1 2-D metrics; no 3-D topology", "PARTIAL", "Paired 3-D B0/A3 fidelity, topology and fixed orthogonal slices"),
        "R2-C1": ("840-model frozen B0/A0-A5 held-out comparison", "CONDITIONAL", "Report capacity/provenance limits; no additional 2-D run"),
        "R2-C2": ("Frozen plane registry, phase semantics and event hashes", "SUPPORTED for data description", "Add non-cherry-picked 3-D truth/mask visuals"),
        "R2-C3": ("Three registered ROIs; two source volumes", "PARTIAL", "Use all frozen ROIs; do not claim independent material replication"),
        "R2-C4": ("2-D per-phase metrics; no 3-D connectivity", "PARTIAL", "3-D fractions, extinction, components, Euler/TPCF/chords"),
        "R2-C5": ("No 3-D sweep or order/init result", "OPEN", "Sweeps 0/1/3/5/10 plus one-factor order/init rows"),
        "R2-C6": ("No 3-D global/local cross", "OPEN", "Registered global-fit × local-observation cross"),
        "R2-C7": ("Comparator selected but not implemented/validated", "OPEN", "Validate hard-data comparator and measure cost before comparison"),
        "R2-C8": ("Not established by this held-out model evaluation", "OPEN", "Validation-only selector analysis and fixed test reporting"),
        "R2-C9": ("Gate association measured; support ratio degenerate", "PARTIAL", "3-D sparsity/clamp intervention; no causal gate claim"),
        "R2-C10": ("No 3-D orthogonal slices in R1", "OPEN", "Fixed XY/XZ/YZ slices from shared B0/A3 states"),
    }
    for row in reviewer:
        comment = row.get("comment", "")
        if comment not in status:
            continue
        evidence, state, gap = status[comment]
        reviewer_lines.append(f"| {comment} | {evidence} | {state} | {gap} |")
    reviewer_lines += [
        "",
        "The current R1 result can support narrow 2-D loss/parameterization/ablation statements. It cannot support broad 3-D fidelity, topology, long-horizon stability, effective-property or published-baseline superiority claims. The gate wording must be narrowed because the support feature has no variance in the frozen 2-D event population.",
        "",
        f"Frozen R1 audit: `{(R1 / 'REVISION_DESIGNA_HELDOUT_R1_FINAL_AUDIT.json').relative_to(ROOT)}` SHA256 `{sha(R1 / 'REVISION_DESIGNA_HELDOUT_R1_FINAL_AUDIT.json')}`.",
    ]
    write_text("REVIEWER_COMMENT_EVIDENCE_MAP.md", "\n".join(reviewer_lines))

    metadata = {
        "stage": "REVISION_DESIGNA_POST_HELDOUT_SCIENCE_REVIEW_R0",
        "frozen_r1_audit_sha256": sha(R1 / "REVISION_DESIGNA_HELDOUT_R1_FINAL_AUDIT.json"),
        "frozen_r1_metrics_sha256": sha(R1 / "REVISION_DESIGNA_HELDOUT_R1_ALL_METRICS.csv"),
        "registered_planes": len(registry),
        "reviewer_rows_read": len(reviewer),
        "support_ratio_values": support_values,
        "gate_rows": len(gates),
        "minimal_3d_trajectory_count": len(rows),
        "estimated_reference_cpu_hours": round(len(rows) * 0.709 + 8, 1),
        "gpu_required": "UNDETERMINED",
        "new_scientific_computation_started": False,
        "new_test_labels_read": False,
    }
    (OUT / "REVISION_DESIGNA_POST_HELDOUT_SCIENCE_REVIEW_R0_METADATA.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
