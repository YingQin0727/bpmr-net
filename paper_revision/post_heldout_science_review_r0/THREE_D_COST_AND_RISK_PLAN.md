# Minimal 3-D cost and risk plan

`MINIMAL_3D_TRAJECTORY_COUNT=48`. The matrix is a plan only; no 3-D trajectory, reconstruction, comparator or new test read was started in this review.

## Cost basis

The frozen P2 budget measured approximately 0.709 CPU hours per reference trajectory and 0.410 hours per optimized trace/vector trajectory. The optimized path is not a validated scientific worker and is therefore not the planning baseline.

| Component | Reference CPU hours | Optimized path (unvalidated) |
|---|---:|---:|
| 48 planned trajectories | 34.0 | 19.7 |
| Evaluation, plots, hash/restart reserve | 8.0 | 8.0 |
| Planning estimate | **42.0** | **27.7** |

`ESTIMATED_CPU_HOURS=42.0` for a conservative reference implementation. This excludes implementation of the published comparator and any failed-isolation recovery. The existing budget reports no measured compatible GPU implementation; therefore `GPU_REQUIRED=UNDETERMINED`. A local CPU run is plausible for this reduced matrix, but GPU acceleration cannot be claimed.

## Must complete for the core reviewer questions

- Paired B0/A3 on WF2 and GI, all three frozen ROI identities, both registered local observation spacings, fixed hard clamping, and saved sweeps 0, 1, 3, 5 and 10.
- Phase fractions, extinction counts, connected components/largest component, Euler/connectivity, TPCF, chord statistics and fixed orthogonal XY/XZ/YZ slices using the same truth/B0/A3 states.
- One-factor order and initialization controls, with the same observation masks and deterministic seeds.
- Runtime, memory, checkpoint identity and no-test-label training provenance for every trajectory.

## Helpful if cost permits

- The pre-specified three-seed sensitivity rows in the matrix.
- The global-fit/local-observation cross and the clamp-by-sparsity diagnostic rows.
- A separately validated Yeong–Torquato-style hard-data comparator under identical masks and observation exclusion. It is not included in the count because its implementation and cost are not validated.
- A measured GPU pilot after the CPU reference path is correct; no GPU result may replace the reference audit.

## Explicit limitations

The matrix does not establish independent-material generalization, convergence beyond 10 sweeps, effective properties, or a complete six-variant 3-D factorial. The no-clamp rows are diagnostic controls and cannot be reported as the main reconstruction protocol. The current 2-D evidence cannot be used to choose a favorable ROI, seed or checkpoint.
