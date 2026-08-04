# Data access and integrity

Raw research volumes are not included in this repository. Obtain them from the
original data provider and comply with the dataset-specific license and
citation requirements.

The paper workflow expects the following dense label volumes:

| Material | Filename | Shape/type | Required SHA-256 |
|---|---|---|---|
| WF2 | `WF2_5um_1030x1033x3051_8b_segmented_corrected.raw` | `1030 x 1033 x 3051`, uint8 | `932853c36761504216c3a734a852c08f6bf451ba6a5c4da9367a2ec8b724d22a` |
| GI | `GI_5um_1030x1033x3051_8b_segmented.raw` | `1030 x 1033 x 3051`, uint8 | `35f336f0a4afea34110fe1f5cd4d1cca70ef128c7e06dad41370d652b97f8c94` |

Recommended local layout:

```text
data/
  raw/
    WF2_5um_1030x1033x3051_8b_segmented_corrected.raw
    GI_5um_1030x1033x3051_8b_segmented.raw
  processed/
  manifests/
```

Verify the files before preprocessing:

```bash
python scripts/verify_dense_sources.py --data-root data/raw
```

Do not silently substitute a file with a different checksum. Record any
legitimate correction as a new immutable data version.
