# Phase-resolved science summary

REVISION_DESIGNA_POST_HELDOUT_SCIENCE_REVIEW_R0 is read-only. The R1 evaluator used fixed four-class semantics: phase 0 oil, 1 brine, 2 rock, 3 gas; 255 is unsupported.

## Interpretation

The held-out task is 2-D teacher-forced neighborhood classification. Every test event exposes the true supported 80-neighbor plane features and predicts the center phase. It is not a 3-D rollout or a topology-preserving reconstruction. No phase-extinction rows were found in R1 (positive truth prevalence with zero predicted prevalence: 0). This does not establish 3-D non-extinction.

Mean phase metrics are averaged over registered conditions within each material and analysis set. They are descriptive condition summaries, not independent-material inference.

## PRIMARY phase means

| Material | Phase | A1 recall | A1 F1 | A1 prevalence error | A3 recall | A3 F1 | A3 prevalence error | A4 recall | A4 F1 | A4 prevalence error | A5 recall | A5 F1 | A5 prevalence error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF2 | 0 (oil) | 0.881698 | 0.882428 | -2.43056e-06 | 0.913121 | 0.900939 | 1.84028e-05 | 0.910099 | 0.898455 | 2.01389e-05 | 0.917953 | 0.903875 | 2.11806e-05 |
| WF2 | 1 (brine) | 0.943788 | 0.944854 | -4.96528e-05 | 0.953931 | 0.950248 | 0.000145486 | 0.952798 | 0.949418 | 0.000135069 | 0.954842 | 0.951506 | 0.000135417 |
| WF2 | 2 (rock) | 0.998541 | 0.99822 | 0.000613889 | 0.998421 | 0.998279 | 0.000270139 | 0.998435 | 0.998267 | 0.000320833 | 0.998424 | 0.998307 | 0.000222569 |
| WF2 | 3 (gas) | 0.928191 | 0.937329 | -0.000561806 | 0.933363 | 0.940378 | -0.000434028 | 0.932085 | 0.939782 | -0.000476042 | 0.935795 | 0.941909 | -0.000379167 |
| GI | 0 (oil) | 0.947342 | 0.951512 | -7.48588e-05 | 0.965966 | 0.957884 | 0.000161723 | 0.965685 | 0.957272 | 0.000167726 | 0.967693 | 0.960426 | 0.000149718 |
| GI | 1 (brine) | 0.917677 | 0.911567 | 4.97881e-05 | 0.94445 | 0.918223 | 0.000241525 | 0.9432 | 0.916346 | 0.000246469 | 0.941872 | 0.918794 | 0.000212924 |
| GI | 2 (rock) | 0.998611 | 0.998241 | 0.000706568 | 0.998442 | 0.998311 | 0.000251766 | 0.998453 | 0.998302 | 0.000289195 | 0.998449 | 0.998349 | 0.000189619 |
| GI | 3 (gas) | 0.934561 | 0.944546 | -0.000681497 | 0.937899 | 0.947432 | -0.000655014 | 0.936867 | 0.947106 | -0.00070339 | 0.940172 | 0.948215 | -0.00055226 |

## SENSITIVITY phase means

| Material | Phase | A1 recall | A1 F1 | A1 prevalence error | A3 recall | A3 F1 | A3 prevalence error | A4 recall | A4 F1 | A4 prevalence error | A5 recall | A5 F1 | A5 prevalence error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WF2 | 0 (oil) | 0.881698 | 0.882428 | -2.43056e-06 | 0.913121 | 0.900939 | 1.84028e-05 | 0.910099 | 0.898455 | 2.01389e-05 | 0.917953 | 0.903875 | 2.11806e-05 |
| WF2 | 1 (brine) | 0.943788 | 0.944854 | -4.96528e-05 | 0.953931 | 0.950248 | 0.000145486 | 0.952798 | 0.949418 | 0.000135069 | 0.954842 | 0.951506 | 0.000135417 |
| WF2 | 2 (rock) | 0.998541 | 0.99822 | 0.000613889 | 0.998421 | 0.998279 | 0.000270139 | 0.998435 | 0.998267 | 0.000320833 | 0.998424 | 0.998307 | 0.000222569 |
| WF2 | 3 (gas) | 0.928191 | 0.937329 | -0.000561806 | 0.933363 | 0.940378 | -0.000434028 | 0.932085 | 0.939782 | -0.000476042 | 0.935795 | 0.941909 | -0.000379167 |
| GI | 0 (oil) | 0.946789 | 0.951418 | -8.22917e-05 | 0.966086 | 0.957675 | 0.000166667 | 0.965855 | 0.957055 | 0.000173611 | 0.96774 | 0.960255 | 0.000152778 |
| GI | 1 (brine) | 0.914539 | 0.910386 | 3.33333e-05 | 0.943611 | 0.918044 | 0.000235764 | 0.94248 | 0.916252 | 0.000240972 | 0.940978 | 0.918461 | 0.000207986 |
| GI | 2 (rock) | 0.998613 | 0.998236 | 0.000721875 | 0.998448 | 0.99831 | 0.000262847 | 0.998459 | 0.998301 | 0.000300694 | 0.998454 | 0.998349 | 0.000200347 |
| GI | 3 (gas) | 0.934509 | 0.94437 | -0.000672917 | 0.93776 | 0.947451 | -0.000665278 | 0.936711 | 0.947134 | -0.000715278 | 0.940039 | 0.948218 | -0.000561111 |

## Registered paired phase differences

Differences are right minus left, matching R1. Negative prevalence-error deltas mean a smaller signed predicted-minus-truth error only when interpreted with the baseline sign; absolute prevalence error remains the safer diagnostic.

| Analysis set | Material | Comparison | Phase | Mean recall delta | Mean F1 delta | Mean prevalence-error delta | Negative F1 rows / total |
|---|---|---|---:|---:|---:|---:|---:|
| PRIMARY | WF2 | A1_vs_A3 | 0 (oil) | 0.0314225 | 0.0185113 | 2.08333e-05 | 23/60 |
| PRIMARY | WF2 | A1_vs_A3 | 1 (brine) | 0.0101431 | 0.00539359 | 0.000195139 | 10/60 |
| PRIMARY | WF2 | A1_vs_A3 | 2 (rock) | -0.000120407 | 5.93796e-05 | -0.00034375 | 20/60 |
| PRIMARY | WF2 | A1_vs_A3 | 3 (gas) | 0.00517274 | 0.00304822 | 0.000127778 | 15/60 |
| PRIMARY | WF2 | A3_vs_A4 | 0 (oil) | -0.0030216 | -0.00248428 | 1.73611e-06 | 15/60 |
| PRIMARY | WF2 | A3_vs_A4 | 1 (brine) | -0.00113317 | -0.000829717 | -1.04167e-05 | 37/60 |
| PRIMARY | WF2 | A3_vs_A4 | 2 (rock) | 1.45675e-05 | -1.19798e-05 | 5.06944e-05 | 36/60 |
| PRIMARY | WF2 | A3_vs_A4 | 3 (gas) | -0.00127848 | -0.000595754 | -4.20139e-05 | 43/60 |
| PRIMARY | WF2 | A3_vs_A5 | 0 (oil) | 0.00483172 | 0.00293564 | 2.77778e-06 | 10/60 |
| PRIMARY | WF2 | A3_vs_A5 | 1 (brine) | 0.000910879 | 0.00125785 | -1.00694e-05 | 17/60 |
| PRIMARY | WF2 | A3_vs_A5 | 2 (rock) | 3.66778e-06 | 2.85376e-05 | -4.75694e-05 | 16/60 |
| PRIMARY | WF2 | A3_vs_A5 | 3 (gas) | 0.00243173 | 0.00153114 | 5.48611e-05 | 11/60 |
| PRIMARY | GI | A1_vs_A3 | 0 (oil) | 0.018624 | 0.00637285 | 0.000236582 | 17/59 |
| PRIMARY | GI | A1_vs_A3 | 1 (brine) | 0.0267731 | 0.00665615 | 0.000191737 | 22/59 |
| PRIMARY | GI | A1_vs_A3 | 2 (rock) | -0.000168502 | 6.91526e-05 | -0.000454802 | 23/59 |
| PRIMARY | GI | A1_vs_A3 | 3 (gas) | 0.00333889 | 0.00288608 | 2.64831e-05 | 17/59 |
| PRIMARY | GI | A3_vs_A4 | 0 (oil) | -0.000280488 | -0.000612074 | 6.00282e-06 | 30/59 |
| PRIMARY | GI | A3_vs_A4 | 1 (brine) | -0.00124966 | -0.0018775 | 4.9435e-06 | 32/59 |
| PRIMARY | GI | A3_vs_A4 | 2 (rock) | 1.07257e-05 | -8.85874e-06 | 3.74294e-05 | 33/59 |
| PRIMARY | GI | A3_vs_A4 | 3 (gas) | -0.0010326 | -0.000326023 | -4.83757e-05 | 37/59 |
| PRIMARY | GI | A3_vs_A5 | 0 (oil) | 0.00172755 | 0.00254197 | -1.20056e-05 | 9/59 |
| PRIMARY | GI | A3_vs_A5 | 1 (brine) | -0.00257741 | 0.00057089 | -2.86017e-05 | 22/59 |
| PRIMARY | GI | A3_vs_A5 | 2 (rock) | 6.23175e-06 | 3.88316e-05 | -6.21469e-05 | 15/59 |
| PRIMARY | GI | A3_vs_A5 | 3 (gas) | 0.00227289 | 0.000783147 | 0.000102754 | 14/59 |
| SENSITIVITY | WF2 | A1_vs_A3 | 0 (oil) | 0.0314225 | 0.0185113 | 2.08333e-05 | 23/60 |
| SENSITIVITY | WF2 | A1_vs_A3 | 1 (brine) | 0.0101431 | 0.00539359 | 0.000195139 | 10/60 |
| SENSITIVITY | WF2 | A1_vs_A3 | 2 (rock) | -0.000120407 | 5.93796e-05 | -0.00034375 | 20/60 |
| SENSITIVITY | WF2 | A1_vs_A3 | 3 (gas) | 0.00517274 | 0.00304822 | 0.000127778 | 15/60 |
| SENSITIVITY | WF2 | A3_vs_A4 | 0 (oil) | -0.0030216 | -0.00248428 | 1.73611e-06 | 15/60 |
| SENSITIVITY | WF2 | A3_vs_A4 | 1 (brine) | -0.00113317 | -0.000829717 | -1.04167e-05 | 37/60 |
| SENSITIVITY | WF2 | A3_vs_A4 | 2 (rock) | 1.45675e-05 | -1.19798e-05 | 5.06944e-05 | 36/60 |
| SENSITIVITY | WF2 | A3_vs_A4 | 3 (gas) | -0.00127848 | -0.000595754 | -4.20139e-05 | 43/60 |
| SENSITIVITY | WF2 | A3_vs_A5 | 0 (oil) | 0.00483172 | 0.00293564 | 2.77778e-06 | 10/60 |
| SENSITIVITY | WF2 | A3_vs_A5 | 1 (brine) | 0.000910879 | 0.00125785 | -1.00694e-05 | 17/60 |
| SENSITIVITY | WF2 | A3_vs_A5 | 2 (rock) | 3.66778e-06 | 2.85376e-05 | -4.75694e-05 | 16/60 |
| SENSITIVITY | WF2 | A3_vs_A5 | 3 (gas) | 0.00243173 | 0.00153114 | 5.48611e-05 | 11/60 |
| SENSITIVITY | GI | A1_vs_A3 | 0 (oil) | 0.0192966 | 0.0062567 | 0.000248958 | 18/60 |
| SENSITIVITY | GI | A1_vs_A3 | 1 (brine) | 0.029072 | 0.00765829 | 0.000202431 | 22/60 |
| SENSITIVITY | GI | A1_vs_A3 | 2 (rock) | -0.000165331 | 7.45145e-05 | -0.000459028 | 23/60 |
| SENSITIVITY | GI | A1_vs_A3 | 3 (gas) | 0.0032503 | 0.00308167 | 7.63889e-06 | 17/60 |
| SENSITIVITY | GI | A3_vs_A4 | 0 (oil) | -0.00023113 | -0.000619943 | 6.94444e-06 | 31/60 |
| SENSITIVITY | GI | A3_vs_A4 | 1 (brine) | -0.0011308 | -0.00179201 | 5.20833e-06 | 32/60 |
| SENSITIVITY | GI | A3_vs_A4 | 2 (rock) | 1.0547e-05 | -9.25409e-06 | 3.78472e-05 | 34/60 |
| SENSITIVITY | GI | A3_vs_A4 | 3 (gas) | -0.00104832 | -0.000316995 | -5e-05 | 37/60 |
| SENSITIVITY | GI | A3_vs_A5 | 0 (oil) | 0.00165407 | 0.00257992 | -1.38889e-05 | 9/60 |
| SENSITIVITY | GI | A3_vs_A5 | 1 (brine) | -0.00263249 | 0.000417156 | -2.77778e-05 | 23/60 |
| SENSITIVITY | GI | A3_vs_A5 | 2 (rock) | 5.76509e-06 | 3.85458e-05 | -6.25e-05 | 15/60 |
| SENSITIVITY | GI | A3_vs_A5 | 3 (gas) | 0.00227893 | 0.000767095 | 0.000104167 | 15/60 |

## WF2 phase 0 focus

WF2 phase 0 is the registered oil phase. The comparison tables retain it even when prevalence is low; no phase is removed for an unfavorable result. The R1 evidence supports reporting its 2-D recall/F1 and prevalence error, but not a claim that oil remains connected, non-extinct, or correctly represented in 3-D.

Source metrics: `outputs\revision_mtcomm_2026\REVISION_REPLICATION_evaluation\DesignA\HELDOUT_R1\REVISION_DESIGNA_HELDOUT_R1_PHASE_METRICS.csv` (SHA256 `522b4e302f749cc0704ba6193e315fb205257802089487db15ba92b88c77811b`).
