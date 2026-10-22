# Design A access evidence recovery R1

REVISION_DESIGNA_ACCESS_EVIDENCE_RECOVERY_R1=PARTIAL
HELDOUT_EVALUATION_READY=NO

- A0: MISSING; A1: MISSING (ORIGINAL_ACCESS_EVIDENCE_UNAVAILABLE).
- Recovered original completion outputs: A0 at 2026-10-19 06:59:09 UTC; A1 initial completion before an interrupted campaign, followed by a successful limit-2 execution at 07:04:34 UTC. These execution outputs do not contain the per-file access stream.
- The early A0 source snapshot had no installed access guard; a later A1 snapshot installed a guard but did not persist its records. Source snapshots cannot prove the exact code loaded in every earlier attempt.
- Available Python log: 720 rows, 718 task identities, 622,927 events; 0 observed policy/path violations. Missing events remain unknown.
- A3 full attempt attribution: FAIL. Rows 2/3 are byte-identical and lack attempt IDs/timestamps; row 4 records pending-to-epoch50_reexecution.pt. The failed orphan is retained, and exactly one formal A3 checkpoint remains in the manifest. Its bytes equal the orphan; that equality alone is not proof of an independent rerun.
- Model integrity: PASS; all 840 frozen checkpoint hashes rechecked. The prior strict-reload audit is retained. No original evidence modified; source hashes stayed stable.

The native stream cannot establish A0/A1 process/time coverage, and the wrapper overwrites its trace destination on each launch. No direct evidence of prohibited reading was found in available streams; absence of a missing stream cannot prove no reading occurred. OS_LEVEL_ISOLATION=NOT_ESTABLISHED.

Keep the original held-out release gate blocked. The minimal alternative is an explicit new acceptance decision that discloses missing A0/A1 original access history and unresolved A3 attempt provenance. Code/input/checkpoint hashes and existing events support limited integrity claims but cannot recreate lost history. No standard was relaxed and no evaluation or retraining was started.
