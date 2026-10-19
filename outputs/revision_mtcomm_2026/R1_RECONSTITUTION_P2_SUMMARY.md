# REVISION_REPLICATION P2

REVISION_R1_RECONSTITUTION_P2=PARTIAL

SCIENTIFIC_CAMPAIGN_READY=NO

Material Passport: manuscript MTCOMM-D-26-05155; stage P2. Evidence consists of frozen P1 artifacts, native access tests, synthetic sequential-runner comparisons, non-test baseline prototypes and CPU measurements. No model fitting or scientific reconstruction was performed.

G1 isolation: FAIL. Current Python runs under the medium-integrity admin identity. It can read permitted train/validation/observation files, but can also read prohibited source/test truth and two raw copies in the recycle bin. Original ACLs were recorded and remain unchanged. No dedicated execution identity or isolated VM was provisioned. Role-specific write permissions, evaluation write denial and alternate-path bypass denial remain untested. These require an actual enforceable OS boundary, not a loader wrapper.

Registry integrity: PASS. All 2,246 plane identities and payload paths are unique; label/support hashes and registered bounds agree. Each of the 18 observation files was compared with its corresponding truth: observed labels agree, and hidden positions contain sentinel 255. Train/validation arrays contain only their declared spatial regions. Guards are 128 voxels; feature radius is 4. Full-square support erosion is required before events are built. No fit events were generated. Raw label 0 stays outside defined four-phase support.

Trajectory manifest: 284 unique neural trajectories. The core has 252 = 6 ROIs × 2 distinct local masks × 7 model configurations × 3 paired seeds. There are 16 additional global-density comparisons and 16 initialization/order comparisons. G250 core cases are reused. Local 250/1000 um masks are identical and are not run twice. Saved sweeps continue with RNG state rather than restarting. Three training seeds and three rollout seeds are paired, not fully crossed. Checkpoints do not yet exist. ROIs are correlated subvolumes of one Ketton specimen.

G2 production equivalence: FAIL, with 96 fixture comparisons passing. Tests cover two flow-state-labelled synthetic families, two mask densities, two seeds, three update orders, two phase compositions, invalid support, minority phases, entropy extremes and actual disk resume. Probabilities, saved states, RNG identities, order and clamping matched in the tested fixtures. Reference functions were extracted without editing their bodies. The optimized path traces the 12-model bank and retains reference encoding; no dependent voxel updates are batched. Tests use 4³ synthetic states under the current identity, not the intended isolated full scientific worker. They do not prove all-input or all-checkpoint equivalence.

Measured update times: B0 reference 1.337 ms, trace 0.771 ms; BPMR reference 8.294 ms, trace 4.792 ms. Independent 1/2/4-worker forward-bank throughput and sampled RAM were measured. Those short kernels do not establish complete-pipeline scaling; the budget uses one worker conservatively.

G3 fair baseline: FAIL pending production validation. A new Apache-2.0 repository-native Yeong–Torquato-style four-phase hard-clamp prototype completed eight 8³/16³, 64-move runs. Only two registered G250 training planes supplied statistics; 3D inputs were synthetic. Clamps and phase counts were preserved. No third-party implementation was copied. Publication metadata and abstract were verified at https://doi.org/10.1103/PhysRevE.58.224. This is explicitly an adaptation, not the original binary implementation. Pooling the exact allowed training information, 48³ cost, mixing and a fair stopping budget remain unvalidated.

Plan C budget: neural reconstruction 201.5 reference CPU hours or 116.4 trace CPU hours; prior synthetic training estimate 13.8 hours. Comparator estimate 1.18 hours uses explicit volume scaling, not measured 48³ performance. The evaluation proxy is 0.02 hours and excludes several metrics and figures. Combined measured/assumed work is 131.4 CPU hours. Adding 24 hours for remaining evaluation/visualization and 30% verification/restart reserve gives 202.0 serial elapsed hours. These are conditional estimates, not a completion guarantee.

LOCAL_CPU_CONDITIONALLY_FEASIBLE. No cloud rental is justified by the current Plan C measurements. Hardware choice does not resolve isolation or scientific comparator validation. Reviewer C3, C5, C6 and C7 retain the documented material-scope, replication-breadth and comparator gaps.

Remaining release gates:
- G1: no actual restricted scientific identity configured/tested; native probes read source/truth and2alternativecopies.
- G2:96synthetic cases pass but complete original runner/intended isolated executable and scientific checkpoint semantics unverified.
- G3:8non-test prototype runs establish clamp-preserving mechanics only; allowed-information parity,48cube runtime/mixing/stopbudget unvalidated.
- Training contract specified, event construction/runner not implemented; source MSE probability normalization is explicit revision choice, requires executable check.
- Reviewer C3,C5,C6,C7 remain partial; same Ketton specimen, restricted longhorizon/cross replication and unvalidated comparator.

STOP P2. No scientific campaign is authorized or started.
