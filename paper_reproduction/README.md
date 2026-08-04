# Paper reproduction entry points

This directory preserves the main scripts used to execute and evaluate the
factorial paper workflow. Machine-specific roots have been replaced with
environment variables, and the public B0/BPMR-Net definitions are imported
from `bpmr_net`.

## Reproduction levels

### Level 1: public software smoke test

Fully supported by this Git repository:

```bash
pip install -e ".[dev,paper]"
bpmr-net-demo
pytest
```

### Level 2: inspect released prediction arrays

Download the held-out probability-map release and run:

```bash
python scripts/inspect_probability_package.py \
  path/to/heldout_pixel_probabilities.npz
```

### Level 3: full factorial training and reconstruction

This requires the separately released research archive:

- registered sparse-plane manifests and plane files;
- the frozen final-v1 registries;
- B0/BPMR-Net checkpoints or the authorized training inputs;
- the legacy Gibbs/proxy runtime referenced by the reconstruction script;
- WF2/GI dense truth only for final evaluation, never for training or
  reconstruction.

The archive must be identified by a DOI/release version and SHA-256 manifest.
It is not appropriate to silently reconstruct missing manifests from dense
truth because doing so can violate the frozen evaluation protocol.

## Environment variables

| Variable | Meaning | Portable default |
|---|---|---|
| `BPMR_PROJECT_ROOT` | cloned public repository | repository root |
| `BPMR_OUTPUT_ROOT` | factorial output directory | `outputs/factorial_v2` |
| `BPMR_FINAL_V1_ROOT` | frozen final-v1 registries | `data/final_v1` |
| `BPMR_DEVELOPMENT_ROOT` | frozen manifests | `data/manifests` |
| `BPMR_BASELINE_ROOT` | released baseline archive | `data/baseline` |
| `BPMR_PLANE_STORE_ROOT` | registered sparse-plane store | `data/plane_store` |
| `BPMR_ROSTER_PATH` | development-data roster | under `data/manifests` |
| `BPMR_GATEB_ROOT` | archived Gate-B runtime | `external/gateb` |
| `BPMR_LEGACY_R2_ROOT` | archived 3-D runtime | `external/legacy_r2` |
| `BPMR_HELDOUT_OUTPUT_ROOT` | qualitative map output | under `outputs` |

## Main scripts

- `factorial_protocol_and_training_v2.py`: protocol audit, paired training,
  checkpoint freeze, 2-D evaluation, and observation-package creation.
- `factorial_multiseed_reconstruction_v2.py`: 3-D multiseed reconstruction.
- `factorial_evaluation_and_package_v2.py`: 3-D metrics, statistics, figures,
  tables, and integrity manifests.
- `generate_heldout_qualitative_maps_v1.py`: pixel-level held-out probability
  maps for truth, B0, BPMR-Net, and log-probability improvement.

## Formal safeguards

- `withheld` sections are used only for final 2-D evaluation.
- Dense truth is not read during training or reconstruction.
- Methods share event identity and stochastic reconstruction streams.
- Template and orientation outputs use equal-weight normalized averaging.
- The public names map as `ORIGINAL_R2 -> BPMR-Net` and
  `SELECTIVE_V2 -> ES-BPMR`.

The scripts are retained for scientific traceability. Level-3 execution should
be considered unsupported until the external research archive DOI and exact
download instructions replace the placeholders in this repository.
