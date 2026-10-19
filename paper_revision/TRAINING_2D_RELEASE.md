# Revision 2D training release — R2

`REVISION_TRAINING_2D_RELEASE_R2=PASS`
`TRAINING_READY=YES`
`OS_LEVEL_ISOLATION=NOT_ESTABLISHED`

This decision releases the bound revision-only 2D training kernels, full event
selection and fixed scheduler. It establishes auditable scientific data
separation with documented limitations, not OS access isolation. No formal
training, scientific reconstruction, published-method comparison, original-data
ACL change, VM installation or historical experiment modification occurred.

| Gate | Result |
|---|---|
| TRAINING_INPUT_AUDIT | PASS |
| TRAINING_ACCESS_AUDIT | PASS |
| PROHIBITED_READ_INJECTION_TEST | PASS |
| EVENT_CAP_AUDIT | PASS |
| PAIRED_EVENT_AUDIT | PASS |
| EPOCH50_SCHEDULE_AUDIT | PASS |
| DESIGN_A_COMPATIBILITY | PASS |
| TRAINING_DRY_RUN | PASS |
| SCIENTIFIC_DATA_ISOLATION | PASS_WITH_DOCUMENTED_LIMITATIONS |
| OS_LEVEL_ISOLATION | NOT_ESTABLISHED |

## Evidence and executed call chain

The frozen registry retains 2,246 planes: 1,150 train, 916 validation and 180
hidden test-truth rows. Registry SHA256 is
`42e8cf78339b992ab4e10af1d8c5798230bc71467ce65f69f86c91d995a560bf`.
All 4,132 train/validation label/support payload hashes and spatial split bounds
passed the existing checks. Hidden truth rows are metadata only; their payloads
and original WF2/GI 3D sources/backups were never read, including for hashing.

The actual traced path was `trace_training_r2.main` → suspended underlying
Python 3.11 interpreter with the frozen venv site-packages →
`training_release_r2.main` → `training_events.inventory/event_arrays` →
`training_campaign_2d.run_fixed50` / `optimizer_step` →
`audit_training_2d.main` / `train_revision_2d.dry_run`. This covers NumPy 2.4.6,
PyTorch 2.14.1+cpu, model/features imports, Adam lazy imports and checkpoint IO.
No preparation, reconstruction or final-evaluation module is imported by this
chain. Plane bytes enter through `Path.read_bytes`, are SHA256-verified, and
become arrays via `numpy.frombuffer`; checkpoint bytes use `BytesIO` with
`torch.save/load`, rather than a native filename loader.

Non-elevated Frida 17.24.2 intercepted `NtCreateFile`, `NtOpenFile`, `NtReadFile`,
`NtWriteFile`, `NtCreateSection` and `NtMapViewOfSection`. Child gating attached
to the controlled probe child and a framework bootstrap child before resuming
them: three processes in total, zero instrumentation errors. The trace contains
368,496 records, reaches the completion marker, and has continuous per-process
sequence numbers. Successful forbidden reads, unclassified content access and
unresolved successful handles are all **zero**. Runtime Python/DLL/code reads,
KnownDlls mappings, directory metadata, pipes/devices, two exclusively newly
created runtime scratch files, registered inputs and isolated test outputs were
classified separately. DOS-device aliases were resolved; no blanket disk-device
whitelist was used. No writes to registered inputs were observed.

The full-event Python whitelist recorded 23,898 allowed accesses. The existing
41 prohibited-access injections all passed. Additional tests denied six Win32
opens (two truth paths, two original raw sources and their two discovered
backups) and two NumPy native `fromfile` / `memmap` attempts. Allowed native
Win32 read, NumPy file/mapping read and child read used only a synthetic canary.
The native hooks redirect prohibited opens to an invalid object namespace
before calling the original API and return access denied. No truth bytes are
needed to demonstrate rejection. Repeated denied metadata probes are retained
in the native evidence and are not successful reads.

## Complete events and actual fixed scheduling

The frozen Plan C matrix gives 120 paired units and 240 complete train/validation
streams. Events use sorted registered plane indices then row/column ordinals,
full 9×9 support erosion, no periodic wrap, deterministic sampling without
replacement capped at 100,000, and sorted selected ordinals. All streams were
built from real legal data and independently reselected. Scalar feature checks
at both ends of every selected plane verify the 400 feature slots and targets.
The JSON records population, count and SHA256 for event identity, selected
ordinals, features and targets for every stream. Counts are 224 × 100,000,
8 × 83,892 and 8 × 41,946. Train counts are 112 × 100,000 and 8 × 83,892.

All seven variants consume the same per-unit event hashes. The **same actual
batch iterator used by the formal executor** traversed every train unit for
B0_SOURCE and A0–A5, without updates: 120 × 7 × 50 epoch records. Every epoch
checks its real contiguous batch indices, total count, batch count and order
hash. All paired orders match. Batch size is 4,096, shuffle is false; seeds are
1701/1702/1703 according to the frozen matrix. Initialization seed is the first
eight hexadecimal digits of SHA256(`material/Gspacing/axis/seed/template`),
mechanically extending the existing G250 convention; validation selection uses
that value + 1. This adds no new data-selection or template-masking rule.

Adam is fixed at lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.
The existing 504 real-data single-step/save/reload checks passed again. Seven
additional real-data tests execute the actual last batch of epoch 1 (1,696)
and first batch of epoch 2 (4,096), save/reload model + Adam + RNG, and compare
resumed parameters and optimizer tensors exactly to uninterrupted execution.
All pass. These bounded tests do not claim convergence or scientific results.

Every test checkpoint is marked `NON_SCIENTIFIC_TEST` and is rejected by the
production Design A predicate. Primary output is hard-fixed to epoch 50;
`primary_checkpoint` rejects epochs 1, 2 and 49. The shared scheduler exposes no
early-stopping or best-validation replacement. New B0 initializes from scratch;
it never loads a historical B0. A0/A1 continue the same common B0 and A2–A5 freeze
it; prior paired source-hash and initialization checks remain passing.

## Scope and limitations

Frida observes user-mode ntdll calls, not kernel ETW. Direct syscalls,
pre-attachment loader activity and arbitrary future native extensions are not
exhaustively covered. Early inherited unnamed handles used in this run were
identified as pipes/character devices, not silently accepted unknown files.
Python runtime directories are trusted code dependencies; malicious environment
replacement is not isolated. The observed NumPy/PyTorch call chain has no
unauthorized data dependency, but this is not a claim that all third-party
native code is incapable of reading other files. No administrator privilege was
used or required for this evidence. Kernel-level tracing/OS strong isolation
remains unestablished and is not a remaining scientific release gate.

The release binds code, registry, frozen configuration and venv configuration
hashes. Changed bindings stop the next launcher. Re-audit is required after
changes affecting this chain. Legacy launchers, 3D reconstruction and final
scientific evaluation remain outside this release.

## Reproduce R2 only

From the repository root (private Frida tooling already exists in the ignored
`outputs/revision_mtcomm_2026/TRAINING_2D_ACCESS_RELEASE/trace_tools`):

```powershell
D:/python.exe -B -m paper_revision.trace_training_r2
D:/python.exe -B -m paper_revision.finalize_training_r2
```

Local detailed evidence stays under
`outputs/revision_mtcomm_2026/TRAINING_2D_ACCESS_RELEASE/`: `r2_native.jsonl`,
`r2_trace_controller.json`, `r2_execution.json`, and reused `access.json`,
`release.json`, `payloads.json`, `dry_run.json`. Their SHA256 values are in the
single final machine-readable `paper_revision/TRAINING_2D_RELEASE_R2.json`.
Old R1 result files are historical snapshots, superseded by this R2 decision.

## Next stage only — not run here

Exact common-B0 command from the repository root:

```powershell
D:/python.exe -B -m paper_revision.trace_training_r2 --source-b0
```

This explicitly launches `training_campaign_2d.train_source_b0` through the same
native trace controller, installs the exact Python whitelist, checks release
bindings and event hashes, and starts **120 sequential new B0 tasks**, each for
50 epochs (148,400 optimizer steps in total). It refuses existing scientific
checkpoint targets. It does not start the 720 adaptations.

Checkpoint pattern:
`outputs/revision_mtcomm_2026/REVISION_REPLICATION_training/DesignA/{material}/G{spacing}/{axis}/seed{seed}/template{template}/epoch50.pt`.
Scientific outputs are separate from all dry checkpoints. Native campaign logs
use `source_b0_native.jsonl` and `source_b0_trace_controller.json`, preserving R2
evidence. A source completion manifest records saved checkpoint hashes.

A three-step real 4,096-event source benchmark on this CPU with one PyTorch
thread took 15.3–17.3 ms/step under tracing: approximately **0.64 CPU hours** of
optimizer work for the 120 B0 tasks. Budget **about 1–2 local elapsed hours**
including event construction, hashing, IO and tracing; this is an estimate from
a short non-scientific benchmark, not a measured full-campaign runtime. No
formal source checkpoint was produced during R2.
