# Minimal Design A 3-D B0/A3 comparison

Status: PASS for the four pre-registered trajectories only.

The fixed condition was ROI0, 48^3, global fit spacing 250 um, local observation spacing 25 um, hard observed clamping, train-prior initialization, forward sequential Gibbs updates, rollout seed 1701, and saved sweeps 0, 3, 5, and 10. ROI and model identities were fixed before reconstruction quality was inspected.

WF2 and GI used their registered 25 um sparse observations. The ROI origin is ZYX [2300, 800, 200], and the registered test slab is separated from the train and validation z slabs by the frozen split and guard-band rule. The observation mask hash is shared by geometry; material observation file hashes are distinct.

The four trajectories completed with the reference sequential, immediate-writeback Gibbs implementation. B0 and A3 used the same initial state, observed mask, update order, and NumPy rollout stream within each material. The A3 trajectory is slower because it evaluates the frozen residual model at every unobserved voxel.

## Runtime

| Material | Variant | Wall time (s) | Peak RSS (MB) |
|---|---:|---:|---:|
| GI | A3 | 5328.1 | 252.2 |
| GI | B0_SOURCE | 917.8 | 252.4 |
| WF2 | A3 | 5363.5 | 247.9 |
| WF2 | B0_SOURCE | 918.0 | 246.0 |
| **Total** | 4 trajectories | **12527.3** | serial CPU |

## Scientific reading

Phase fractions, extinction flags, six-connected component counts and largest-component sizes for every saved sweep are in `MINIMAL_3D_B0_A3_RESULTS.csv`. The sweep-10 orthogonal slices are in `MINIMAL_3D_B0_A3_SLICES.png` (with editable SVG/PDF copies). These four trajectories provide a direct B0/A3 comparison for WF2 and GI under one fixed registered condition; they do not establish broad cross-material or general 3-D superiority.

The observed outputs retain all four phases in these four trajectories at sweeps 0, 3, 5 and 10; no phase-extinction flag was produced. This is a statement about these registered ROIs and this fixed condition only. It does not convert the 2-D minority-phase findings into a general 3-D claim.

Reviewer issues partially addressed: Design A B0 versus A3 can now be compared in 3-D with hard observation clamping, fixed sparse observations, and explicit runtime/topology outputs. Remaining gaps are additional ROI/spacing/sweep sensitivity, initialization/update-order sensitivity, longer-sweep behavior beyond 10, and external comparator baselines.

Summary SHA256: `67f9ee1d68ce16e95543d0d32164025040dbda36136b4df7b3c5289f16b23bb1`
