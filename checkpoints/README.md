# Checkpoint releases

Checkpoints are intentionally excluded from ordinary Git history.

A versioned checkpoint release should contain:

- one directory per material, spacing, orientation, paired training seed, and
  phase template;
- `best_checkpoint.pt` and its completion metadata;
- a CSV/JSON manifest containing relative paths, byte sizes, SHA-256 values,
  model seed, method ID, best epoch, and software release tag;
- the exact configuration and data-manifest checksums;
- a short license/provenance statement.

Recommended layout:

```text
checkpoints/
  WF2/spacing_25um/x/seed_1701/template_phase_0/B0/best_checkpoint.pt
  WF2/spacing_25um/x/seed_1701/template_phase_0/ORIGINAL_R2/best_checkpoint.pt
  ...
  checkpoint_manifest.csv
```

Use a GitHub Release, Zenodo, or another research-data repository and replace
this document with the final DOI/download instructions before publication.
