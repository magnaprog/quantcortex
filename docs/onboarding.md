# Contributor guide

QuantCortex supports quantitative research and guarded paper trading. Strategies
produce portfolio weights; separate layers validate, execute, and account for
those decisions. The repository does not establish future profitability or
readiness for real-money deployment.

## Start here

1. Set up the development environment below and run the checks.
2. Read [AGENTS.md](../AGENTS.md), the repository entry guide, and
   [CONTRIBUTING.md](../CONTRIBUTING.md) for change and review requirements.
3. Read the [architecture](architecture.md) before changing a pipeline layer.
   Start with [weight contracts](architecture.md#weight-contracts) and
   [timing and accounting](architecture.md#timing-and-accounting).
4. Pick the relevant path:

| Your change | Read next |
|---|---|
| Provider, factor, allocator, overlay, or strategy | [Architecture and extension rules](architecture.md#extension-rules) |
| Backtest or comparative evaluation | [Evaluation contracts](evaluation-contracts.md) |
| Orders, broker adapters, or persistence | [Execution state](architecture.md#execution-state) and [paper operations](production-readiness.md#paper-cycle) |
| Research report or notebook | [Local data](../local_data/README.md), [performance reporting](../PERFORMANCE.md#generate-a-report), and [data due diligence](data-source-due-diligence.md) |

Use the report manifest to check the input, source, settings, and output hashes
for published results. Check the CI run for the exact commit under review.

## Set up and check

Run from the repository root with Python 3.11 through 3.14. For example:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/ruff check .
.venv/bin/python -m pytest tests/ -q --cov=quantcortex --cov-fail-under=60
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

For a first focused test:

```bash
.venv/bin/python -m pytest \
  tests/test_weight_interface.py::test_equal_weight_sums_to_one -v
```

`requirements/dev.lock` includes the scientific core, test, lint, and notebook
tools. It does not install broker/provider SDKs, storage clients, optional ML,
RL, NLP or regime backends, Poetry, or package-build tooling.

| Environment | Install source |
|---|---|
| Scientific core | `requirements/core.lock` |
| Core, tests, and lint | `requirements/test.lock` |
| Core and notebooks | `requirements/notebooks.lock` |
| Broker SDKs | `requirements/brokers.lock` |
| Providers, brokers, storage, and regime support | `requirements/runtime.lock` |
| Package-build tools | `requirements/build.lock` |
| Individual optional capabilities | `ml`, `rl`, `regime`, `nlp`, `brokers`, `providers`, or `storage` extras in [pyproject.toml](../pyproject.toml) |

To install all extras with Poetry:

```bash
poetry install --with test,dev -E all
```

Keep optional imports inside the methods that need them. Installing an extra
must not become a prerequisite for importing the core package.
Use [.env.example](../.env.example) for configuration names and paper defaults.
Docker Compose reads `.env` for interpolation; direct Python commands do not.
Export required values in the shell or explicitly load a trusted file.

## Trace one decision

```text
data -> point-in-time features -> select -> allocate -> timing -> risk
                                                        -> target weights
                                                           -> backtest
                                                           -> paper trading
```

The pipeline is `w_t = R_t(T_t(A_t(S_t(X_<=t))))`. Read
[portfolio/base.py](../quantcortex/portfolio/base.py) and
[base_strategy.py](../quantcortex/strategies/base_strategy.py) as you trace a selection
through `rebalance()` and `generate_weights()`.

Preserve these contracts:

- Allocators produce finite `float64` vectors. Long-only allocations sum to one;
  market-neutral allocations sum to zero within configured bounds.
- Timing and risk overlays use the relaxed exposure contract. It has a gross
  cap and no sum constraint; its default bounds permit shorts. Long-only callers
  must enforce a zero lower bound. Do not renormalize away intended cash.
- A close-derived target executes on the first strictly later price bar. A
  target with no later bar remains unexecuted.
- Every backtest requires a `TransactionCostModel`. Only `cash_returns=None`
  selects implicit zero-return cash; any supplied series, even all zeros, must
  cover every price bar with finite returns.
- Features and membership must be available by the decision timestamp. A
  look-ahead diagnostic or unused purge/embargo helper provides no general
  guarantee against leakage.
- Target weights differ from executed holdings. The event engine holds units
  and lets weights drift; the vectorized engine re-pegs weights between
  rebalances without charging for those implied trades.
- Paper-trading orders require pre-trade risk checks, durable intent, and broker
  reconciliation after an uncertain outcome. Direct adapter calls do not supply
  those orchestration controls.

To see the local execution lifecycle without a network call:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py --offline
```

This dry run uses deterministic synthetic prices and supplies no performance
or authenticated-broker evidence. Live-price preview, paper-trading orders, and
recovery instructions are in [Production readiness](production-readiness.md#paper-cycle).

## Find the owning layer

All importable modules live under `quantcortex`; use absolute imports.

| Path | Responsibility |
|---|---|
| `quantcortex/data/` | Providers, CSV inputs, processors, universes, storage |
| `quantcortex/alpha/` | Factors, features, ML/NLP models, factor validation |
| `quantcortex/portfolio/` | Allocation and exposure contracts, optimizers |
| `quantcortex/timing/`, `quantcortex/risk/` | Exposure overlays and risk estimates |
| `quantcortex/strategies/` | Selection, allocation, overlays, target generation |
| `quantcortex/backtest/` | Allocation records, engines, fills, costs, metrics, validation |
| `quantcortex/execution/` | Orders, positions, pre-trade risk, brokers, persistence |
| `quantcortex/research/` | Strategy rules, causal model fitting, and comparative evaluation |
| `tests/`, `schemas/` | Regression and conformance checks, machine interfaces |
| `scripts/`, `research/` | Operational commands and unexecuted notebooks |
| `docs/`, `requirements/` | Documentation and dependency locks |
| `local_data/`, `reports/` | Ignored input and report directories; only the local-data README is tracked |

Before editing, state the input, output, timing, and failure contracts for the behavior
change. Add a focused regression for accounting, causality, risk, or execution state,
using a small hand-checkable result and relevant failure cases. See [extension
rules](architecture.md#extension-rules) for each component type.

## Use research inputs explicitly

Use a permitted owner-supplied CSV or an explicit live-provider opt-in. A failed
real-data fetch must stop; it must never become a synthetic performance report.
Keep raw and cached observations, account state, and executed notebook output
out of Git.

With a permitted local matrix containing the required symbols and warm-up history:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_report.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --cash-proxy-symbol SHV
```

The public checkout does not supply that matrix. The
[local-data guide](../local_data/README.md) includes wide adjusted-close and
single-symbol OHLCV examples. Reports and notebooks use bounded forward fill.
For external review, provide the source, permission basis, retrieval and
adjustment fields, and `--manifest-out` described in
[PERFORMANCE.md](../PERFORMANCE.md#generate-a-report). Record the actual trial
history; the report's default trial count is not evidence of how many variants
were tried. SHV is a cash proxy and reporting benchmark, not a claim of a
risk-free asset.

For notebooks with local inputs:

```bash
export QUANTCORTEX_PRICES_CSV="$PWD/local_data/prices.csv"
export QUANTCORTEX_OHLCV_CSV="$PWD/local_data/aapl_ohlcv.csv"
.venv/bin/jupyter lab research/
```

Choose exactly one adjusted-close source. Alternatively, after installing
provider dependencies and reviewing the data terms, use
`QUANTCORTEX_LIVE_YFINANCE=1` with `QUANTCORTEX_PRICES_CSV` unset.
`QUANTCORTEX_OHLCV_CSV` supplements the local source for the Alpha158 notebook.
Notebooks 01 through 05 cover data quality, factor research, portfolio
construction, backtest analysis, and the trading bridge. CI executes them on
synthetic test fixtures; those results establish software behavior only.

Explicit live diagnostics, with provider dependencies installed:

```bash
PYTHONPATH=. .venv/bin/python scripts/validate_performance.py --live-yfinance
PYTHONPATH=. .venv/bin/python scripts/survivorship_demo.py --live-yfinance
```

These diagnostics do not supply delisted-security coverage. Research use,
derived publication, reviewer access, and raw redistribution are separate
permission questions. Owner approval and hashes do not resolve provider terms.

## Prepare a contribution

Run the focused regression, full Ruff checks, and the full test suite. Use the
[contribution guide](../CONTRIBUTING.md) for broker SDK, dependency, packaging,
container, figure, and release checks. The [CI workflow](../.github/workflows/ci.yml)
defines the broader checks; identify the exact commit checked when reporting a result.

Before review:

- Keep the change in its owning layer and avoid unrelated refactoring.
- State behavior, data/API assumptions, verification results, and remaining
  limitations. Distinguish skipped, failed, simulated, paper, and live checks.
- Exclude secrets, raw data, local state, and notebook outputs.
- Regenerate reports from reviewed source and permitted inputs, following
  [PERFORMANCE.md](../PERFORMANCE.md). Inspect the figures and verify the
  manifest; never hand-edit generated values or checksums.
- Report unfavorable results and post-hoc changes honestly. Open schemas and
  fixtures do not make the unavailable historical matrices reproducible.

Report security-sensitive failures through [SECURITY.md](../SECURITY.md).

## Troubleshoot

| Symptom | Check |
|---|---|
| A script cannot import `quantcortex` | Run from the root; install editable with `--no-deps -e .` or use `PYTHONPATH=.`. |
| Optional import fails or SDK/ML tests skip | Install the relevant lock or extra. Do not move an optional import to module scope. |
| Matplotlib cache is unwritable | Set `MPLCONFIGDIR` to a writable directory, for example `/tmp/quantcortex-mpl`. |
| LightGBM/XGBoost loading fails on macOS | The libraries may require `brew install libomp`. |
| Report test finds a source or artifact hash mismatch | Identify what changed, then regenerate and review the report as described in [PERFORMANCE.md](../PERFORMANCE.md). Never repair a checksum by hand. |
| Input digest differs | Recover the authorized input or review a new release; do not attach old provenance to replacement data. |
| Paper cycle rejects a stale bar or uncertain order | Follow [paper-cycle recovery](production-readiness.md#paper-cycle); do not bypass freshness or retry guards. |
