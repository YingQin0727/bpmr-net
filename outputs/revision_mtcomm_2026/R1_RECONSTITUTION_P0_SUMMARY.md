# REVISION_REPLICATION — P0 feasibility audit

REVISION_R1_RECONSTITUTION_P0=PARTIAL

Material Passport: manuscript MTCOMM-D-26-05155; stage P0; evidence: byte hashes, official project metadata, frozen ROI registry, reviewer letter and synthetic CPU tests. No new scientific payloads or results. Verification is limited to the checks explicitly recorded.

WF2 and GI are available locally under `D:\Paper2\paper2_stage02a_source_recovery\official_download`. Both 3,246,233,490-byte files match the authoritative frozen ROI source hashes. Official project 167 (DOI 10.17612/P7HT11, ODC-BY 1.0) documents 5 um voxels, seeded watershed and raw phases 1=oil, 2=brine, 3=rock, 4=gas. Canonical conversion is raw minus one. GI hash in repository data/README.md is inconsistent; the historical file was preserved. Full label-domain scanning remains a preparation gate.

Design A uses a new common B0 fitted on revision train only. A0–A5 branch from identical source hashes; fixed epoch50 primary endpoints avoid test-driven selection. Historical B0 is ineligible because original fit exposure cannot be excluded. Architecture comparisons still differ in trainable freedom (46,529 versus 383/380); equal steps are not equal capacity.

Proposed spatial z slabs separate train, validation and test with 128-voxel guards and paired split coordinates across both states. Historical ROI unions plus guards are excluded. Additional 48^3 ROIs follow a deterministic geometry/hash rule independent of results. Same-rock robustness cannot establish material generalization. New plane manifests must include source, coordinate, phase and payload hashes. No plane was generated.

The role-aware artifact helper passes synthetic denial tests. It does not replace OS mount/account separation. No scientific worker is released until raw/truth exclusion is verified end to end; this is a hard campaign gate, not a claim that leakage has already been ruled out.

Reviewer letter: D:/Downloads/论文1修改comments.txt. Coverage CSV contains all 2 Reviewer 1 concerns and 10 Reviewer 2 comments. Shared trajectories answer fidelity, phase structure, sections, gate diagnostics and selector use. Longer trajectories and the global/local cross use shared fitted models. Dominant-phase baseline uses train-only priors. An established comparator remains to be specified; no claim that all reviewer requests are resolved.

Python 3.11.17 environment: `.venv-revision311`; repository dev/paper dependencies and CPU torch installed. A0–A5 finite-loss forward/backward and strict state reload passed. Synthetic process sampled RSS peaked at 417.5 MiB. Serial voxel proxy: B0 2.36 ms, BPMR 12.97 ms; one 48^3 BPMR trajectory at 10 sweeps is approximately 3.98 hours. Illustrative fitting matrix is 24.9 hours; primary fully crossed rollout proxy is 586.8 serial days. Reduced one-global/one-local schedule still estimates 65.2 days. These exclude scheduler/I/O/metrics and are conditional estimates, not scientific runtimes.

LOCAL_CPU_NOT_FEASIBLE applies to the full candidate serial matrix within the 19-day revision window. Local fitting and reduced pilots remain practical. A server is unnecessary for P0; campaign hardware is undecided until a sufficient reduced schedule or validated CPU optimization is established. No server rental or GPU execution occurred.

Remaining gates:
- Full raw phase-domain validation pending preparation (semantics and source identity established).
- No registered new plane payloads by P0 instruction; auditable manifest release pending.
- Role helper tested but OS-level worker isolation and end-to-end leakage proof pending.
- Historical B0 exposure cannot be cleared; Design B ineligible.
- Established comparator implementation not selected/verified.
- Full candidate reconstruction matrix exceeds local budget; Windows scientific runner and reduced scientifically sufficient schedule not validated.

STOP P0. No campaign launch.
