# Mask-aware 3-D paper comparison

Source trajectories are frozen from commit `fa72c0c`; no Gibbs updates were rerun.

## Why the former center slices were identical

The measured observation mask coverage at XY z=24, XZ y=24 and YZ x=24 is 100% for both materials. These planes are fully observed, and the reconstruction uses hard observed clamping after every sequential Gibbs update. The B0 and A3 slices therefore remain identical by construction. The observed values on these center planes also match the registered truth values exactly in this ROI.

## Mask-only slice selection

The comparison figures use index 23 in each direction. This is the minimum-coverage plane (33.984375% observed, 66.015625% unobserved); ties were resolved by nearest distance to the center index 24 and then the lower index. No model or truth values entered this choice. The mask is geometrically identical for WF2 and GI, so the selected indices are identical, while the material observations and reconstructed phase fields remain distinct.

## Topology interpretation

At sweep 10, A3 reduces the large B0 component-count excess for both materials and is closer to the truth in the sum of absolute component-count differences, both overall and in the unobserved region. This is evidence of reduced B0 fragmentation under this fixed condition. It is not proof of topology preservation: A3 still has more components than truth for several phases, and the dominant-component fraction for GI phase 0 in the unobserved region is higher than truth, consistent with possible over-connection. Therefore the defensible statement is conditional improvement in fragmentation, with residual smoothing/over-connection risk.

All phase fractions, absolute errors, six-connected component counts and largest-component fractions are in `MASK_AWARE_3D_METRICS.csv`. `Truth` rows are reported once per region; B0 and A3 rows cover sweeps 3, 5 and 10.

## Scope

The results support a fixed-ROI, fixed-mask B0/A3 3-D comparison for WF2 and GI. They do not establish broad 3-D generalization, superiority across ROI or spacing choices, or equivalence of topology to the truth. The figures are descriptive and do not replace the unresolved sensitivity matrix or external baseline comparisons.
