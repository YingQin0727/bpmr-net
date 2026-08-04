# Public-release scope

## Included

- Exact public definitions of the B0 network and 383-parameter BPMR-Net branch.
- The 400-D masked neighborhood encoder.
- Template/orientation fusion and ES-BPMR selection.
- Synthetic smoke tests and portable packaging metadata.
- Frozen paper workflow scripts for traceability.
- Dataset/checkpoint filenames and integrity expectations.

## Excluded

- Server credentials and IP-specific launch helpers.
- Machine-specific Python environments.
- Raw WF2/GI volumes and private plane stores.
- Training checkpoints and reconstruction states.
- Internal review bundles, monitoring logs, temporary jobs, and abandoned
  experimental branches.
- Manuscript correspondence and personal metadata.

## Relationship to the research archive

The public repository is an executable software release, not a byte-for-byte
copy of the full research workspace. The research archive retains the complete
audit trail. Public release artifacts should refer to immutable archive or DOI
records by version and SHA-256.
