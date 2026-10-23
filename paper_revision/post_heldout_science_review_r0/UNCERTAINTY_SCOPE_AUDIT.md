# Uncertainty scope audit

The only uncertainty table used here is the frozen post-processing output `outputs\revision_mtcomm_2026\REVISION_REPLICATION_evaluation\DesignA\HELDOUT_R1\REVISION_DESIGNA_HELDOUT_R1_SEED_UNCERTAINTY.csv` (SHA256 `063bcba3ef289f2f6c5f46119288c96e47d69c22e0385c2aa8ea03d3acfaa239`). No event bootstrap, event-level p-value, or material-population confidence interval is valid under the frozen protocol.

| Quantity | Value |
|---|---:|
| Seed-conditional rows | 5148 |
| Rows with one registered training seed (CI undefined) | 3420 |
| Rows with three registered training seeds | 1728 |
| Rows with pointwise df=2 t CI | 1728 |
| PRIMARY rows (n=1 / n=3) | 1692 / 864 |
| SENSITIVITY rows (n=1 / n=3) | 1728 / 864 |

The n=3 interval is conditional on fixed material, spacing, orientation, template and shared test events; it describes training-seed variation only. The n=1 rows must be reported as CI undefined. The 3 TIER_2 models remain in the pre-frozen SENSITIVITY set and do not become TIER_1 through this review. The two source volumes, correlated ROIs/orientations, and shared events do not support broad material generalization or event-level pseudoreplication.

R1 PRIMARY/SENSITIVITY pair counts remain 714/720. No model was requalified after reading held-out results.
