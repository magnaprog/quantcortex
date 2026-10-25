# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository. It is the authoritative agent guide; `AGENTS.md` is a
short orientation for other agents and must stay consistent with this file.

## Commands

A `.venv` is the reviewed local environment. `requirements/dev.lock` installs
the scientific core plus test and notebook tooling; optional integrations use
Poetry extras or the dedicated exported locks under `requirements/`. Install
the development lock with `pip install -r requirements/dev.lock`, then run
`pip install --no-deps -e .`. Run everything from the repo root.

- Tests: `.venv/bin/python -m pytest tests/ -q` (pytest config sets
  `pythonpath = ["."]`, so the `quantcortex` package resolves from the root).
- Single test: `.venv/bin/python -m pytest tests/test_weight_interface.py::test_equal_weight_sums_to_one -v`
- Lint: `.venv/bin/ruff check .`; CI requires this check to pass. Auto-fix: `ruff check . --fix`.
- Dependency changes: edit `pyproject.toml`, then run
  `scripts/update_dependency_locks.sh`; commit `poetry.lock` and every changed
  export under `requirements/`. Because `pyproject.toml` and `poetry.lock` are
  release-critical fingerprinted inputs, a dependency change also requires
  regenerating both artifact releases from the new clean source commit,
  with the expansion wrapper first and the paper wrapper second. Dependabot
  lock-only PRs therefore cannot pass CI and are handled through this coordinated flow.
- Operational scripts import `quantcortex.*`, so run them with the root on the
  path: `PYTHONPATH=. .venv/bin/python scripts/<name>.py`
  (validate_performance, generate_report, survivorship_demo, verify_brokers,
  paper_trade_cycle, run_paper_experiments). Performance commands require an explicit source:
  `generate_report.py --prices-csv local_data/published_rotation_prices.csv --cash-proxy-symbol SHV` or
  `generate_report.py --live-yfinance`; `validate_performance.py` requires
  `--live-yfinance`; `survivorship_demo.py` requires `--live-yfinance`;
  `paper_trade_cycle.py` requires either `--offline` or `--live-yfinance`.
- Expansion release: freeze and commit the protocol before provider requests;
  keep both matrices under ignored `local_data/expansion/`. Source commits require
  explicit user approval. After approval, commit release-critical source. From
  the repo root, set `export PYTHON_BIN="$PWD/.venv/bin/python"` once for the
  detached worktrees, then run
  `scripts/release_expansion_artifacts.sh local_data/expansion` first.
  A first expansion release needs a fixed UTC `QUANTCORTEX_EXPANSION_GENERATED_AT`.
  Changed release-critical source or inputs require explicit, fixed UTC values
  for both `QUANTCORTEX_EXPANSION_GENERATED_AT` and `QUANTCORTEX_GENERATED_AT`,
  even if only paper source changed. Input-digest and provenance overrides are
  only for separately authorized new inputs. Leave them unset for source-only
  releases; these instructions do not authorize refetching data. Never edit
  aggregate tables, generated LaTeX, manifests, or figures by hand.
- Paper release: run
  `scripts/release_paper_artifacts.sh local_data/published_rotation_prices.csv`
  second from the same source commit, with no intervening commit. Follow the
  [coordinated release flow](paper/README.md#coordinated-release), including its
  unchanged-source/input replay instructions. The wrapper regenerates `docs/img/`
  and the paper from a detached clean worktree, then builds with pinned Tectonic.
  Generated tables and figures must match `paper/results/manifest.json`; visually
  review every public and anonymous PDF page before commit. Direct
  `scripts/build_paper.sh` runs overwrite tracked build provenance. Use the
  unchanged-release rebuild instructions in `paper/README.md`: verify the source
  manifest and take `PAPER_SOURCE_COMMIT` and `SOURCE_DATE_EPOCH` from the recorded
  build manifest. Changed bound source requires coordinated release wrappers.
- CI (`.github/workflows/ci.yml`) defines ruff + pytest with a 60% coverage floor
  on Python 3.11-3.14 from exported locks. Separate jobs execute notebooks on
  deterministic test-only fixtures, check real broker SDK request classes,
  build/smoke-install the wheel, reject dependency-lock drift, and build plus
  smoke-test the read-only container.

## Public source exports

Use a vetted tracked-file snapshot with a fresh public source root P. Do not
copy private Git history, raw inputs or untracked files. Existing owner
permission for derived aggregates remains as recorded in the manifests;
provider terms are a separate, independently unverified matter. For a new public
root, use fixed UTC generation timestamps and the coordinated expansion-then-paper
flow from P, then commit the outputs as artifact child A with source_commit=P.
Unchanged replay requires that recorded source commit to be available locally.
The old protocol-freeze identifier records private history, not public access or
independent proof of pre-data timing. Keep the protocol bytes/hash and preserve
all source/artifact binding checks. Test receipts must identify the candidate
commit they checked; a workflow file does not establish current public CI status.

## Working norms

Treat every change as unaudited until the relevant tests and invariants have
been checked directly.
- Required checks: `ruff check .` and `pytest tests/ -q` must pass before a
  change is done; CI enforces both. When fixing a bug, add a regression test
  under `tests/` (see `tests/test_regression_guards.py` for the style: assert a
  hand-derived/canonical value, not a snapshot of current behavior).
- Make surgical changes: touch only what the task needs; do not reformat,
  re-sort imports, or "improve" unrelated code. The regressions found here
  (regime non-determinism, the pre-trade contract, factor-cap ordering, the
  engine's executed-vs-target accounting) came from subtle interactions, so
  incidental edits to the money path are high-risk.
- Do not add configuration, abstractions, or flexibility beyond the requested
  change.
- Confirm changes with broad effects before doing them: anything that rewrites
  every import, moves the package layout, or touches the money path belongs in
  its own reviewed change, not bundled into unrelated work.

## The weight contract (keystone: quantcortex/portfolio/base.py)

Every component that emits weights validates them through one of two functions;
which one depends on where it sits in the pipeline:

- `enforce_weight_contract` (STRICT): float64 `(n_assets,)`; long-only weights
  are in `[0, 1]` and sum to `1.0`, while market-neutral weights are in
  `[-1, 1]` and sum to `0.0`. This is the **allocation**
  layer contract; `PortfolioOptimizer.optimize` applies it automatically, so
  subclasses only implement `_compute_weights` returning a raw vector that
  already satisfies the contract for the configured mode.
- `enforce_exposure_contract` (RELAXED): nonempty, finite, one-dimensional
  float64 weights within caller-supplied `lower` and `upper` bounds, defaulting
  to `[-1, 1]`, with gross exposure (`sum |w|`) `<= max_gross`. It enforces no
  allocation mode or sum constraint. To enforce promised long-only output,
  callers must pass `lower=0`; the default bounds permit shorts. This is the
  contract for **timing/risk overlays and the post-overlay strategy output**,
  where reduced long-only exposure leaves the remainder in cash. A flat book
  sums to 0; a long-only book at half exposure sums to 0.5.

Books with regime gates or volatility scaling need not sum to 1. Violations of
the applicable contract raise `WeightContractViolationError` immediately.

## The pipeline (quantcortex/strategies/base_strategy.py)

`w_t = R_t( T_t( A_t( S_t( X<=t ) ) ) )`: Select -> Allocate -> Timing -> Risk.
- Subclass `Strategy` and implement `select(ctx) -> pd.Series` (alpha scores
  indexed by the chosen symbols). Override `allocate` for score-driven weights;
  the default delegates to the injected `PortfolioOptimizer`.
- Timing and risk overlays are `callable(weights, ctx) -> weights`; concrete
  strategies wire components (HMMRegime, VIXScaler, CircuitBreaker, ...) with
  small lambdas that pull the right data out of the `StrategyContext`.
- `generate_weights(prices, rebalance_dates)` produces the date-by-symbol target
  panel that the backtest engines consume.

## Conventions that are load-bearing (violating them breaks the build)

- **Lazy imports.** Every heavy/optional dependency (torch, lightgbm/xgboost/
  catboost, hmmlearn, transformers, stable-baselines3, gymnasium, alpaca-py,
  ib_async, ccxt, yfinance, fredapi, polygon, redis, sqlalchemy, lxml) is
  imported *inside* the method that uses it, with an offline fallback or a clear
  ImportError. Modules must import with only the scientific core. Do not add a
  heavy import at module top level.
- **Mandatory costs.** Backtest engines (`quantcortex/backtest/engines/`) raise
  if constructed without a `TransactionCostModel`. Slippage in that model is a
  flat per-trade rate; size/impact-aware costs live in
  `quantcortex/backtest/execution_models/market_impact.py`.
- **Cash accounting.** Residual cash is `1 - sum(risky weights)` for the
  published long-only strategy. Supply an aligned per-period cash-return series
  when cash earns a return; missing bars fail closed. Sharpe and benchmark
  comparisons must use the same explicit cash or risk-free series.
- **Causality / PIT controls.** Research code must use only information
  available at the decision time. `pit_enforcer.py` validates fundamentals
  against `announcement_date`; matching is strict-before by default. Callers
  must opt into exact matches only with observed intraday timestamps; the
  opt-in itself does not establish when the data became available. The look-ahead
  tools are diagnostics, not proof that an arbitrary pipeline is leakage-free;
  walk-forward purge/embargo controls only apply when the caller uses that engine.
- **Determinism.** `quantcortex/timing/hmm_regime.py` uses threadpoolctl, when
  available, to limit supported BLAS/OpenMP pools to one thread during HMM or
  GMM fitting and the prediction used to label states within `fit()`.
  Subsequent calls through `predict_regime()` do not set a thread limit.
  This reduces sensitivity to floating-point reduction order during fitting;
  it does not guarantee identical results across environments. Keep model fits
  seeded and verify repeatability with fixed inputs and the recorded software
  environment.
- **Explicit research data.** Do not commit market-data snapshots or executed
  notebook outputs. Reports and notebooks must use either an owner-supplied
  permitted CSV or an explicit live-provider opt-in; they must never silently
  substitute generated prices after a fetch failure. Derived performance charts
  may be published only with explicit owner approval, adjacent provenance, an
  input digest, and artifact hashes. The same rule applies to reviewed paper
  aggregates and figures under `paper/`; raw provider matrices remain local.
  Repository freezing reduces post-data discretion but is not external
  preregistration or a temporal holdout, and documentation must not imply it is.
  Synthetic fixtures remain appropriate for tests and the clearly labeled
  `paper_trade_cycle.py --offline` dry run.

## Layer ABCs to subclass

`DataProvider` (quantcortex/data/providers/base.py): `fetch_ohlcv/
fetch_fundamentals/fetch_macro`, canonical UTC-naive OHLCV schema. `Universe`
(quantcortex/data/universe/base.py): point-in-time membership;
`SP500Universe.from_wikipedia()` reconstructs historical constituents from the
current Wikipedia change table, rejects dates before source coverage, and is
still only an approximation. Named index classes do not silently select their
survivorship-biased demo subsets. `Broker` (quantcortex/execution/brokers/
base.py): `submit_order/get_positions/get_account`, adapters lazy-load their SDK.
`OrderManager`: a validated NEW -> SUBMITTED -> PARTIALLY_FILLED -> FILLED
lifecycle, with cancellation and rejection terminal paths. Submission-intent
state is persisted separately before any broker call.

The Alpaca and IB adapters target `alpaca-py` and `ib_async`. SDK-shaped mocks
and real SDK model-construction tests do not establish authenticated transport,
account permissions, reconnect behavior, or venue-side idempotency. Treat those
paper-account checks as release requirements.

## Packaging

All importable code lives under one top-level package, `quantcortex` (e.g.
`from quantcortex.portfolio.base import enforce_weight_contract`). `tests/`,
`scripts/`, `research/`, `paper/`, and `docs/` sit at the repo root, outside the package.
Keep new modules inside `quantcortex/` and import them absolutely as
`quantcortex.<subpkg>...`; there are no relative imports and no top-level
package squatting. Docker commands follow the same namespace rule. Keep
secrets, local data, reports, research material, paper artifacts, tests, and
local build output excluded from the runtime image via `.dockerignore`.

## Honesty norms

Do not tune toward a single backtest or use an arbitrary metric threshold as a
substitute for evidence. Preserve the full configuration or hypothesis set for
DSR and BHY analysis, and report unfavorable results as-is.
Every published run needs source, permission, date-window, adjustment, and input
digest metadata. See PERFORMANCE.md. Use ASCII punctuation in source and docs;
established proper-name diacritics in author and bibliography metadata are the
only intentional exceptions.
