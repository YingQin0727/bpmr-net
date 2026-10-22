# Design A evidence release R2

REVISION_DESIGNA_EVIDENCE_RELEASE_R2=CONDITIONAL
MODEL_INTEGRITY=PASS
TRAINING_PROVENANCE=PARTIAL
ACCESS_EVIDENCE=PARTIAL
DATA_LEAKAGE_EVIDENCE=PARTIAL_NO_DIRECT_VIOLATION_OBSERVED
ORIGINAL_STRICT_RELEASE=FAIL
CONDITIONAL_SCIENTIFIC_RELEASE=YES
HELDOUT_EVALUATION_READY=YES

The original 720/720 task-bound access-evidence rule remains unmet. Available Python evidence contains 720 rows, 718 unique task identities and 622,927 events, with zero observed denials or prohibited reads. Missing A0/A1 streams remain unknown. A3 has three raw rows; two are byte-identical and the final row records the pending-to-epoch50_reexecution.pt commit, but there is no authenticated process attempt ID for complete historical attribution.

Model and training identity evidence remains strong: 120 B0 plus 720 A0-A5 checkpoint identities, hashes, strict reload, finite parameters, fixed 50 epochs, source-B0 hashes and train/validation event hashes are verified. No test payload was read in this stage.

Evidence tiers: TIER_1=837; TIER_2=3; TIER_3=0. TIER_2 consists of target-unit A0, A1 and A3. PRIMARY_SET contains 833 model entries and 714 comparison-unit pairs across 119 complete units. SENSITIVITY_SET contains 840 model entries and 720 comparison-unit pairs, including disclosed TIER_2 models.

The conditional decision permits the next stage to run frozen independent test evaluation under CONDITIONAL_R2_READ_ONLY, with PRIMARY_SET and SENSITIVITY_SET selected before any test payload read. It does not convert the original strict release to PASS and does not prove that missing streams had no unobserved prohibited access. TIER_2 results must remain sensitivity evidence.

UNRESOLVED_ACCESS_RECORDS=4 raw records: two missing task streams plus two exact duplicate A3 rows. The original logs and scientific payloads were not modified. OS_LEVEL_ISOLATION=NOT_ESTABLISHED.

No training, retuning, test evaluation, 3D reconstruction or external baseline was executed.
