# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
python -m pip install -e ".[dev,paper]"
pytest
ruff check .
```

## Pull requests

1. Keep scientific claims separate from software-behavior claims.
2. Add or update tests for code changes.
3. Do not change frozen thresholds, phase ordering, feature offsets, or fusion
   rules without documenting the scientific compatibility impact.
4. Do not commit raw research volumes, private checkpoints, credentials,
   reviewer correspondence, or machine-specific absolute paths.
5. Record user-facing changes in the pull-request description.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
