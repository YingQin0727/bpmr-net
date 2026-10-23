# Design A held-out evaluation R1

REVISION_DESIGNA_HELDOUT_EVALUATION_R1=PASS

840/840 models evaluated. PRIMARY=833 models and 714 pairs; SENSITIVITY=840 models and 720 pairs.

The original strict access-evidence rule remains FAIL. TIER_2 models are retained only in the pre-frozen SENSITIVITY set; no eligibility was changed after test reads.

This is a 2D teacher-forced held-out classification evaluation using the frozen event cap, sampling, four-class metrics and seed-conditional uncertainty protocol. Two source volumes do not support broad cross-material generalization claims.

Paired deltas are right minus left; negative NLL/Brier/ECE and positive accuracy/F1 favor the right model.

## PRIMARY
A0_vs_A1: n=119; categorical_nll=-0.00215196; brier=-5.05376e-05; ece=-0.000379957; accuracy=-1.41807e-05; balanced_accuracy=-0.00257874; macro_f1=-0.000403803
A1_vs_A3: n=119; categorical_nll=-0.00346866; brier=-0.000914881; ece=-0.000824001; accuracy=0.000233368; balanced_accuracy=0.0118961; macro_f1=0.00538617
A2_vs_A3: n=119; categorical_nll=-0.00214285; brier=6.63318e-05; ece=-0.00169466; accuracy=-3.9916e-05; balanced_accuracy=0.000614328; macro_f1=-0.000218834
A3_vs_A4: n=119; categorical_nll=0.000325001; brier=9.5608e-05; ece=-4.33603e-05; accuracy=-3.71148e-05; balanced_accuracy=-0.000999349; macro_f1=-0.000844426
A3_vs_A5: n=119; categorical_nll=-0.00037168; brier=-0.000126651; ece=8.11248e-05; accuracy=9.01611e-05; balanced_accuracy=0.001208; macro_f1=0.00121291
B0_SOURCE_vs_A0: n=119; categorical_nll=-0.0135758; brier=-0.000368287; ece=-0.0154228; accuracy=6.96779e-05; balanced_accuracy=-0.00244523; macro_f1=0.00157265

## SENSITIVITY
A0_vs_A1: n=120; categorical_nll=-0.00213441; brier=-4.03924e-05; ece=-0.000370967; accuracy=-2.34375e-05; balanced_accuracy=-0.00296841; macro_f1=-0.000570422
A1_vs_A3: n=120; categorical_nll=-0.00348007; brier=-0.000922418; ece=-0.000828305; accuracy=0.000239757; balanced_accuracy=0.0122589; macro_f1=0.00551045
A2_vs_A3: n=120; categorical_nll=-0.00213096; brier=6.74748e-05; ece=-0.00168428; accuracy=-4.16667e-05; balanced_accuracy=0.000490584; macro_f1=-0.000263236
A3_vs_A4: n=120; categorical_nll=0.000324658; brier=9.53562e-05; ece=-4.32931e-05; accuracy=-3.69792e-05; balanced_accuracy=-0.000977298; macro_f1=-0.000832492
A3_vs_A5: n=120; categorical_nll=-0.000370972; brier=-0.000126081; ece=8.05933e-05; accuracy=8.95833e-05; balanced_accuracy=0.00118553; macro_f1=0.00119448
B0_SOURCE_vs_A0: n=120; categorical_nll=-0.0135824; brier=-0.000373284; ece=-0.015429; accuracy=7.43056e-05; balanced_accuracy=-0.0021099; macro_f1=0.0016657

Seed uncertainty is recorded in `REVISION_DESIGNA_HELDOUT_R1_SEED_UNCERTAINTY.csv`; intervals are conditional on fixed source/spacing/orientation/template and shared events, not material generalization.

Gate diagnostics report entropy, margin, support ratio, learned gate, residual magnitude, probability shift and all four B0/A3 correctness transitions. Correlations are descriptive and not causal.

SCIENTIFIC_TRAINING_COMPLETE=YES
HELDOUT_EVALUATION_COMPLETE=YES
THREE_D_RECONSTRUCTION_COMPLETE=NO
PUBLISHED_BASELINE_COMPLETE=NO
OS_LEVEL_ISOLATION=NOT_ESTABLISHED
