# Release checklist

- [x] Confirm code ownership with all authors and the institution.
- [x] Add the repository URL to `pyproject.toml` and `CITATION.cff`.
- [x] Add the confirmed paper title and author list.
- [ ] Add the final journal metadata and DOI after publication.
- [x] Confirm Apache-2.0 is approved by all rights holders.
- [ ] Publish checkpoint/archive DOI and update download instructions.
- [ ] Confirm dataset-specific redistribution and citation terms.
- [ ] Run `pytest` and `ruff check src tests scripts examples`.
- [x] Run a clean-environment installation test.
- [x] Scan working tree and complete Git history for secrets.
- [x] Verify no raw volumes, private checkpoints, or absolute user/server paths.
- [x] Create versioned tag `v0.1.0`.
- [ ] Create GitHub Release with source archive and SHA-256 manifest.
- [ ] Enable branch protection, dependency updates, secret scanning, and private
  vulnerability reporting where supported.
