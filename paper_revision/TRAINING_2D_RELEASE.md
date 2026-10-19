# Revision 2D training access check

This stage replaces the former OS gate only with a scientific access audit. It
does not establish OS isolation. Formal training remains disabled.

Run from the repository root:

```powershell
.\.venv-revision311\Scripts\python.exe -B -m paper_revision.audit_training_2d
```

`audit_training_2d.main` freezes the exact allowed plane/configuration/code paths,
verifies all 4,132 train/validation payload hashes and split bounds, installs
`TrainingAccess.hook`, then imports and calls `train_revision_2d.dry_run`.
The full 2,246-row registry is read as metadata; no test payload is read or hashed.
The hook records path, access mode, phase and allow/deny decision. Python open
calls outside the whitelist, child processes, dynamic ctypes loads, new memory
maps and filesystem link/mutation capabilities are denied. Unregistered bytecode
caches are denied; source imports fall back to the permitted `.py` files.

The dry run uses eight fully supported 9×9 centers per train/validation unit,
applies the frozen plane-axis adapter, and runs one Adam step per B0/A0–A5 model.
Both materials, all three orientations, all three seeds and four template
identities are checked: 72 paired units, 504 save/reload checks. Events are paired
across variants. All adaptations share that unit's newly initialized and updated
dry B0; A2–A5 keep it frozen. No historical checkpoint is loaded. Dry checkpoints
are marked DRY_RUN_ONLY and rejected by the formal Design A metadata predicate.
Template identities follow the existing four-replicate event convention; this
check does not establish new template-specific masking semantics.

This bounded path is not the complete campaign executor. The frozen 100,000-event
cap, 4,096 batch size, 50 source epochs and 50 continuation epochs are checked as
configuration, not exercised by the dry run. Plan C's 120 new B0 / 720 adaptation
budget remains a conditional estimate. No formal epoch50 source exists.

Separate boundaries remain: `p1_dataset.main` prepares data; this module handles
2D training kernels; `p2_runner` is a synthetic reconstruction audit, not a
released scientific reconstructor. A final scientific evaluator is not released.
None of those preparation/reconstruction/evaluation modules is imported by the
new training kernel. Legacy entrypoints are unchanged and are not authorized.

## Remaining blockers and exact locations

- `training_access.py:TrainingAccess.hook` observes CPython audit events, not all
  native I/O. Direct CRT/Win32 reads within loaded PyTorch/NumPy DLLs remain
  unverified. `audit_training_2d.main` records dependency bootstrap separately;
  it cannot deny or exhaustively inspect native import-time I/O. An independent
  native file-access trace covering bootstrap and the same launcher is required
  before claiming no unaudited path. Python monkeypatching cannot repair this
  coverage gap, and no VM/account/ACL change is performed here.
- `train_revision_2d.tiny_events` intentionally selects eight dry-run centers,
  not the full deterministic seeded event cap. Full event manifest generation,
  all density regimes and scheduling must be implemented and verified before
  authorizing the campaign. `require_design_a` is a rejection predicate, not a
  cryptographically authenticated registry of future epoch50 artifacts.

Consequently TRAINING_ACCESS_AUDIT and SCIENTIFIC_DATA_ISOLATION remain FAIL and
TRAINING_READY remains NO. OS_LEVEL_ISOLATION is NOT_ESTABLISHED.

Local evidence lives under
`outputs/revision_mtcomm_2026/TRAINING_2D_ACCESS_RELEASE/`: `release.json` contains
the decision and code/config hashes, `access.json` contains bootstrap and guarded
events, `payloads.json` lists verified hashes, `dry_run.json` records all paired
checkpoint hashes. Checkpoints are test artifacts only, never scientific results.
