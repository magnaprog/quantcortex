# Repository guide

## Layout

Importable code lives under `quantcortex`. Use absolute imports and the existing
`data`, `alpha`, `portfolio`, `timing`, `risk`, `backtest`, `execution`, and
`strategies` layers. Tests use deterministic synthetic data. Research notebooks
are stored without outputs. See [architecture](docs/architecture.md) and the
[contributor guide](docs/onboarding.md) for contracts and examples.

## Checks

Run from the repository root after installing `requirements/dev.lock` and the
package with `pip install --no-deps -e .`:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

Target Python 3.11 through 3.14. Use four-space indentation, `snake_case`
functions and modules, and `PascalCase` classes. Keep comments focused on units,
assumptions, invariants, and non-obvious decisions. Use ASCII in code and prose;
preserve proper-name diacritics in author and bibliography metadata.

## Implementation constraints

- Preserve the strict allocation contract and the bounded exposure contract in
  `quantcortex/portfolio/base.py`. Exposure controls can leave residual cash.
- Keep features and training labels causal. Both engines execute decisions at a
  later observed bar. Date-only fundamentals become available strictly after
  their declared timestamp by default.
- Every backtest has a `TransactionCostModel`. Nonzero cash income uses a complete
  aligned return series; missing observations stop the run.
- Keep optional provider, broker, storage, and machine-learning imports lazy.
- Preserve order intent before broker submission, same-symbol retry guards,
  pre-trade risk checks, and paper-trading defaults.
- Distinguish accounting quantities from broker shares, one-way turnover from
  purchases plus sales, and arithmetic attribution from compound return.
- Add focused regression tests for changes to accounting, timing, allocation,
  execution state, or error behavior. Keep refactors within the requested scope.

## Data and generated files

Keep credentials, account data, provider matrices, model state, executed notebook
outputs, and private review notes out of Git. `local_data/`, `reports/`, and
`output/` are for ignored inputs and working files. A failed data request must
not silently substitute generated performance data.

Publish reports from reviewed source and permitted inputs, with the permission
basis, input digest, source hashes, settings, and output hashes recorded in the
manifest. Follow [PERFORMANCE.md](PERFORMANCE.md) for report generation and
review the [data-source record](docs/data-source-due-diligence.md) before sharing
results. Changing source or inputs calls for regeneration and review of the
resulting figures and metrics. Do not hand-edit generated values or checksums.
Approval to rebuild a report does not imply permission to retrieve new data.

## Review

Use concise Conventional Commit messages. Describe the resulting behavior,
validation, and material data or API assumptions. Preserve technical meaning
when editing prose, including quantities, signs, aggregation, and limitations.
Tests and saved records establish their stated scope; they do not establish
broker connectivity, provider permissions, or future trading performance.
