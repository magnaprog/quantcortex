# QuantCortex developer and researcher onboarding

Editorial and source cross-check: 2026-10-09. Released artifact provenance and
historical validation remain tied to the revisions recorded in their manifests.

This guide explains QuantCortex's architecture, research design, and operational
workflow, with the accounting, causality, provenance, and execution contracts
that changes must preserve.

## Contents

1. [Project purpose and boundaries](#1-project-purpose-and-boundaries)
2. [Current checkout status](#2-current-checkout-status)
3. [Recommended onboarding path](#3-recommended-onboarding-path)
4. [Environment setup](#4-environment-setup)
5. [Core mental model](#5-core-mental-model)
6. [Load-bearing contracts](#6-load-bearing-contracts)
7. [Repository map](#7-repository-map)
8. [Data layer](#8-data-layer)
9. [Alpha and feature research](#9-alpha-and-feature-research)
10. [Portfolio construction](#10-portfolio-construction)
11. [Timing and risk overlays](#11-timing-and-risk-overlays)
12. [Strategy pipeline](#12-strategy-pipeline)
13. [Backtesting and evaluation](#13-backtesting-and-evaluation)
14. [Execution, broker adapters, and state](#14-execution-broker-adapters-and-state)
15. [Research program and paper methodology](#15-research-program-and-paper-methodology)
16. [Data, artifacts, and provenance](#16-data-artifacts-and-provenance)
17. [Paper build and release workflow](#17-paper-build-and-release-workflow)
18. [Developer workflows](#18-developer-workflows)
19. [Researcher workflows](#19-researcher-workflows)
20. [Tests and CI](#20-tests-and-ci)
21. [Production-readiness boundaries](#21-production-readiness-boundaries)
22. [Common mistakes](#22-common-mistakes)
23. [Troubleshooting](#23-troubleshooting)
24. [Glossary](#24-glossary)
25. [Sources of truth](#25-sources-of-truth)
26. [First-contribution checklist](#26-first-contribution-checklist)

## 1. Project purpose and boundaries

QuantCortex is a modular platform for quantitative research and guarded paper
execution, organized around portfolio weights. Importable code lives in the
`quantcortex` package.
Research notebooks, operational scripts, tests, paper material, and generated
artifacts live at the repository root.

The platform is designed to make these boundaries explicit:

- point-in-time data and decision timestamps;
- target generation versus portfolio accounting;
- strict allocator output versus exposure-reducing overlays;
- residual risky exposure versus residual cash;
- target weights versus executed holdings;
- event-driven versus vectorized accounting;
- exact arithmetic identities versus causal claims;
- software conformance versus empirical reproducibility;
- offline broker-model checks versus authenticated venue behavior;
- source code versus generated evidence;
- data provenance versus data redistribution rights.

The repository supports:

- deterministic factor, portfolio, strategy, and backtest research;
- explicit transaction-cost and residual-cash accounting;
- canonical allocation records and machine-readable evaluation contracts;
- vectorized and event-driven backtest engines;
- paper-only execution workflows with fail-closed state handling;
- provider and broker adapters behind optional, lazy imports;
- reproducible aggregate reports and paper builds from reviewed inputs;
- a NeurIPS 2026-format research paper and a TAE workshop camera-ready mode.

The repository does **not** establish:

- future profitability, alpha, investment skill, or investment advice;
- a population estimate of backtest error;
- a general benchmark for all financial ML systems;
- causal effects from the reported protocol contrasts;
- provider-data research-use rights, provider permission to publish derived
  aggregates, reviewer-access rights, or permission to redistribute provider
  observations; recorded repository-owner authorization is a separate matter;
- authenticated broker connectivity, account permissions, or venue idempotency;
- calibrated spread, impact, capacity, or order-size models;
- production monitoring, incident response, disaster recovery, or regulatory approval;
- readiness for real-money deployment.

Passing tests support only the behavior they exercise; platform certification
requires broader evidence.

## 2. Current checkout status

Read the [retrospective manifest](../paper/results/manifest.json),
[expansion manifest](../paper/expansion/results/manifest.json), and
[PDF build manifest](../paper/build_manifest.json) for the released source
revision and generation timestamps. Compare those records with the current
checkout before treating the PDFs as current. They establish local artifact
provenance, not current-checkout validation or remote publication status:

- The coordinated retrospective and expansion releases bind the same committed
  source revision. Generated CSV, JSON, LaTeX, metadata for allocation records, `docs/img/`,
  figures, manifests, and PDFs were regenerated together.
- The public PDF and TAE camera-ready have identified authors. All three builds,
  including the anonymous derivative, have a 10-page body limit. References and
  appendices are excluded. The live call is <https://tai-eval.github.io/cfp/>.
  The [official accepted-paper list](https://tai-eval.github.io/papers/) confirms
  Submission 423. This checkout does not establish which PDF is currently
  uploaded to OpenReview.
- The current manuscript has 13 figures, including two descriptive analyses
  added during revision: annual excess returns over SHV and convention effects
  for each gradient boosting seed. They reuse the recorded paths and preserve the frozen
  primary protocol.
- The three owner-authorized market-data matrices and both expansion
  `.metadata.json` sidecars are not committed or redistributed. Exact-digest
  regeneration requires permitted local copies under ignored `local_data/`
  paths; they are not supplied by a public checkout.
- Exact-digest refresh mode requires fixed UTC generation timestamps, both
  expansion digests, and complete retrospective provenance. Default replay still
  rejects any input or provenance change from the reviewed manifest.
- The manifests record owner authorization on 2026-08-29 for the live-data run
  and publication of derived aggregates. Provider terms remain independently
  unverified; this establishes neither provider permission nor a prohibition.
  The recorded owner authorization does not establish additional rights for an
  OpenReview release or CC BY 4.0 grant. The retained `paper/checklist.tex`
  answers `No` to Code of Ethics conformity and proper respect for terms of
  existing assets, reflecting that evidence limit.

A new public export must bind regenerated artifacts to its own source root P
and artifact child A, as described in Section 17.3. Historical private manifests
and workflow definitions do not establish that a public checkout has been built,
validated or published. Check the exact candidate's receipts before claiming a
passing CI result.

Current matrix digests are:

| Input | Required SHA-256 |
|---|---|
| Retrospective seven-symbol matrix | `9c499ec55c630b596aa724a78de36ee5d98a2edf453b534080e1ad105b46a9c4` |
| U.S. sector ETF matrix | `e3c7338257d3ea35ccc1a94a5d296c58fd1a752cf64c3d5d04c91d5036fc9a8c` |
| Country equity ETF matrix | `aeb1cad415f5edf72385e557e5426d993fcdd87a6fb1c178ff533a2641a768ac` |

## 3. Recommended onboarding path

### First hour

1. Read Sections 1, 2, and 6 for the project scope, current status, and contracts.
2. Read `CLAUDE.md`; it is the authoritative implementation guide.
3. Read `docs/architecture.md` and `docs/evaluation-contracts.md`.
4. Create the reviewed development environment.
5. Run Ruff and the focused contract tests.
6. Read `quantcortex/portfolio/base.py` and
   `quantcortex/strategies/base_strategy.py` before modifying a strategy.

Then follow the path closest to your work:

- core developer: Sections 5 through 14, 18, 20, 22, and 26;
- researcher: Sections 8 through 13, 15 through 17, and 19;
- execution developer: Sections 6, 12 through 14, 18.7, and 21;
- paper or release maintainer: Sections 15 through 17 and 20;
- data/provider maintainer: Sections 6.6, 8, 16, 18.2, and 21.

### First day

1. Trace one target from `Strategy.select()` through allocation, overlays,
   `generate_weights()`, serialization of canonical allocation records, and both engines.
2. Read `quantcortex/backtest/costs/transaction_costs.py` and
   `quantcortex/backtest/engines/cash.py`.
3. Read `quantcortex/execution/order_manager.py`,
   `quantcortex/execution/state_persistence.py`, and
   `quantcortex/execution/pre_trade_risk.py`.
4. Execute the offline paper-trading cycle.
5. Walk through the research notebooks using deterministic fixtures or an
   explicitly selected local data source.
6. Read `PERFORMANCE.md`, `paper/preregistration.md`, `paper/README.md`, and
   `docs/data-source-due-diligence.md` before interpreting a reported metric or
   changing paper or release artifacts.
7. Read `CONTRIBUTING.md` before preparing a change for review. Use `.env.example`
   for configuration names; direct Python commands do not load `.env` automatically.

### Before the first change

1. Identify the owning layer.
2. Write down the input and output contract.
3. Identify whether the change touches money, causality, order state, data rights,
   generated evidence, or optional dependencies.
4. Add the narrowest regression test that locks the intended behavior.
5. Run the relevant focused test, full Ruff, and the broadest feasible test suite.
6. Do not refactor adjacent code unless the requested change requires it.

## 4. Environment setup

### Supported Python versions

`pyproject.toml` supports Python 3.11 through 3.14:

```text
>=3.11,<3.15
```

CI tests and package builds run on all four versions. Python 3.12 is the reviewed
local environment in this checkout, but any supported version is valid.

### Reviewed local setup

Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.lock
.venv/bin/python -m pip install --no-deps -e .
```

The editable install makes operational scripts import `quantcortex` directly.
Without an editable install, invoke scripts with `PYTHONPATH=.`.

### Core dependencies

The scientific core depends on:

- NumPy;
- pandas;
- SciPy;
- scikit-learn;
- Matplotlib;
- PyArrow.

### Optional dependency groups

| Extra | Purpose | Representative packages |
|---|---|---|
| `ml` | Gradient boosting models | XGBoost, LightGBM, CatBoost |
| `rl` | Reinforcement-learning allocators | Stable Baselines3, Gymnasium, PyTorch |
| `regime` | HMM regime model | hmmlearn |
| `nlp` | Financial language models | Transformers, PyTorch |
| `brokers` | Broker adapters | alpaca-py, ib_async, CCXT |
| `providers` | External data providers | yfinance, Polygon, FRED, lxml |
| `storage` | Persistent stores | Redis, SQLAlchemy, psycopg2 |
| `all` | All optional capabilities | All groups above |

A full Poetry environment can be installed with:

```bash
poetry install --with test,dev -E all
```

Heavy and credentialed integrations must remain lazy imports. Importing the core
package must not require broker SDKs, PyTorch, boosting libraries, provider
clients, databases, or credentials.

### Lock files

| File | Use |
|---|---|
| `poetry.lock` | Authoritative Poetry resolution |
| `requirements/core.lock` | Scientific core |
| `requirements/test.lock` | Core plus test and lint tools |
| `requirements/notebooks.lock` | Core plus notebook tools |
| `requirements/dev.lock` | Core plus test, lint, and notebook tools |
| `requirements/brokers.lock` | Core plus broker SDKs |
| `requirements/runtime.lock` | Core plus providers, brokers, storage, and regime support |
| `requirements/build.lock` | Package-build tooling |

`requirements/dev.lock` does not install Poetry, package-build tooling, broker or
provider SDKs, storage clients, or optional ML, RL, NLP, and regime backends.
Install the lock or Poetry extra for the workflow being run.

Lock regeneration requires Poetry 2.4.1 and `poetry-plugin-export` 1.10.0. Install
or select those versions, or set `POETRY_BIN`, before running:

```bash
scripts/update_dependency_locks.sh
```

Commit `pyproject.toml`, `poetry.lock`, and every changed exported lock together.
Dependency changes also invalidate release source fingerprints and require a
coordinated empirical artifact release.

Configuration names and paper-mode defaults are documented in `.env.example`.
Docker Compose reads `.env` for interpolation. Direct Python commands do not;
export values in the current shell or load a trusted file explicitly.

### Basic checks

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

The repository CI command includes coverage:

```bash
.venv/bin/python -m pytest tests/ -q \
  --cov=quantcortex --cov-fail-under=60
```

## 5. Core mental model

A QuantCortex strategy emits target risky weights from information available by
a declared decision time. The execution and accounting layers turn those targets
into trades and returns.

The core strategy equation is:

```text
w_t = R_t(T_t(A_t(S_t(X_<=t))))
```

Where:

- `S` selects symbols and produces alpha scores;
- `A` allocates selected symbols into a fully budgeted portfolio;
- `T` applies timing overlays that may reduce exposure;
- `R` applies risk overlays that may reduce or reshape exposure;
- `w_t` is the final target-weight vector;
- residual long-only capital is cash.

The end-to-end path is:

```text
data
  -> point-in-time features
  -> strategy selection
  -> strict portfolio allocation
  -> timing overlays
  -> risk overlays
  -> relaxed target exposure
  -> canonical target tape
      -> backtest engine
      -> target validation
         -> PositionManager order translation
         -> combined target, order, and post-trade risk validation
         -> durable submission intent
         -> broker adapter
```

Target generation, serialization of allocation records, cash, costs, comparators, and engine
behavior have separate contracts because the same target decision can produce
different return paths under different execution and accounting semantics.

## 6. Load-bearing contracts

### 6.1 Strict allocation contract

`quantcortex.portfolio.base.enforce_weight_contract` validates allocator output.

Long-only allocation must:

- be compatible with a nonempty one-dimensional numeric vector;
- contain no booleans or nonfinite values;
- keep every weight in `[0, 1]`;
- sum to `1.0`.

Market-neutral allocation must:

- be compatible with a nonempty, finite one-dimensional numeric vector;
- contain no booleans;
- remain within configured per-asset bounds;
- sum to `0.0`.

The function returns a contiguous `float64` array. Market-neutral mode imposes no
minimum gross or separate long/short budget, so an all-zero vector passes unless
the caller adds another exposure requirement. Violations raise
`WeightContractViolationError`.

`PortfolioOptimizer.optimize()` applies this contract automatically. A subclass
implements `_compute_weights()` and must already return a feasible raw vector.

### 6.2 Relaxed exposure contract

Timing and risk overlays can intentionally move capital to cash.
`enforce_exposure_contract` checks:

- a nonempty, finite, one-dimensional numeric vector;
- caller-supplied lower and upper bounds, which default to `[-1, 1]`;
- gross exposure no greater than `max_gross`;
- no target sum.

This helper is mode-agnostic. `Strategy.rebalance()` runs all overlays, checks
aggregate shape, then applies one final relaxed check with `lower=0.0` in
long-only mode and `lower=-1.0` otherwise. It does not validate each custom
overlay separately or enforce a market-neutral sum. Custom overlays must
preserve and test portfolio-mode semantics explicitly. Version-1 canonical
allocation records are long-only and reject negative target weights later.

For a correctly preserved long-only book:

- risky weights summing to `1.0` mean fully invested;
- risky weights summing to `0.5` mean half risky and half cash;
- all zeros mean fully cash.

Do not run overlay output through the strict allocation contract. That would
incorrectly force reduced exposure back to full investment.

### 6.3 Decision and execution timing

A target computed from the close at time `t` executes on the first strictly later
available price bar. It does not earn the return ending at the close that created
the signal.

Both engines enforce this rule. A target without a later bar does not execute.
The paper's retroactive signal-close assignment is deliberately infeasible and
exists only as a sensitivity diagnostic.

### 6.4 Cash contract

For the published long-only workflows:

```text
residual cash weight = 1 - sum(risky weights)
```

Only `cash_returns=None` selects an implicit zero-return cash account. Every
supplied Series, including an all-zero Series, must contain a finite return for
every price bar. Missing observations fail the run instead of silently becoming
zero.

Keep two concepts separate:

- `r_t^c`: return credited to residual cash in portfolio accounting;
- `r_t^f`: reporting benchmark subtracted for excess Sharpe.

The paper baseline uses SHV for both. Its zero-cash diagnostic sets `r_t^c = 0`
while retaining SHV as `r_t^f`.

### 6.5 Transaction-cost contract

Every backtest engine requires a `TransactionCostModel`. Passing `None` is an
error.

The general model supports:

- commission;
- slippage;
- sell-side tax;
- optional ADV liquidity caps;
- feasible position reductions before exposure-increasing legs;
- scaling or blocking exposure increases when a gross cap binds;
- executed rather than merely desired weights.

If a pre-trade book is already over its gross cap and ADV limits prevent enough
liquidation, the model executes every feasible reduction but can remain over the
cap. Using both a fill model that adjusts fill prices and a positive
`TransactionCostModel.slippage` value double-counts slippage; the event-driven
engine warns about this combination.

The paper's illustrative rate is 13 basis points per dollar bought or sold:

```text
modeled cost = 0.0013 * (B_t + S_t)
```

Where `B_t` and `S_t` are gross buy and sell notionals. This is gross two-sided
traded notional. It is not one-way turnover `max(B_t, S_t)`. In engine results,
`turnover` reports that one-way measure, while `traded_notional` reports
`B_t + S_t` and is the proportional-cost reconciliation base.

The model does not estimate spread, impact, venue effects, volatility dependence,
capacity, or order-size effects. `AlmgrenChriss` exists as a separate execution
model; it does not make the paper's flat cost assumption a capacity model.

### 6.6 Point-in-time contract

Features and membership must use only information available at the decision time.

- Fundamentals become available at `announcement_date`, not fiscal period end.
- Date-only records use strict-before matching.
- Same-timestamp use requires verified intraday timestamps and explicit opt-in.
- Historical index membership and historical prices are separate data needs.
- A current constituent list applied historically is survivorship biased.
- Purge and embargo protect only callers that use the walk-forward splitter.
- A look-ahead diagnostic cannot prove that an arbitrary pipeline is leakage-free.

### 6.7 Determinism contract

Seeds, data ordering, tie-breaking, and model fits must be deterministic. The HMM
fit pins thread pools because multithreaded floating-point ordering can flip a
near-boundary regime assignment.

Expansion score ties use ascending symbol order. Seeds 11, 29, 47, 71, and 97
apply only to frozen expansion gradient boosting repetitions. Other learned components have
their own defaults. Automatic backend selection can change the model family when
installed packages change; exact reproduction requires an explicit backend and
pinned library version.

### 6.8 Fail-closed contract

At trust boundaries, invalid or uncertain state stops the workflow:

- malformed allocation records are rejected;
- missing cash or required data stops the run;
- unsupported contract versions are rejected;
- stale revisions are rejected by versioned execution-snapshot writes;
- Alpaca paper-cycle orchestration persists submission intent before its broker
  call and blocks automatic retry after an uncertain outcome;
- release wrappers reject dirty or untracked release-critical source;
- input and source hash mismatches stop artifact publication;
- PDF validation scanner failures stop publication.

### 6.9 Generated-evidence contract

Never hand-edit generated tables, figures, manifests, checksums, files containing
hashes of allocation records, protocol output, or empirical CSVs. Change the
source generator, test it,
regenerate through the release wrapper, inspect every diff, and rebuild the paper.

## 7. Repository map

| Path | Responsibility |
|---|---|
| `quantcortex/data/` | Providers, CSV validation, processors, universes, storage |
| `quantcortex/alpha/` | Factors, feature engineering, ML/NLP models, factor validation |
| `quantcortex/portfolio/` | Weight contracts and allocators |
| `quantcortex/timing/` | Regime, trend, and volatility exposure overlays |
| `quantcortex/risk/` | Drawdown, VaR/CVaR, factor, volatility, and Kelly controls |
| `quantcortex/strategies/` | End-to-end select, allocate, time, and risk pipelines |
| `quantcortex/backtest/` | Allocation record conformance, engines, costs, fills, metrics, validation |
| `quantcortex/execution/` | Pre-trade risk, orders, positions, brokers, state persistence |
| `quantcortex/research/` | Shared expansion experiment primitives |
| `scripts/` | Reports, experiments, releases, paper builds, operational checks |
| `research/` | Unexecuted tutorial and research notebooks |
| `tests/` | Deterministic unit, regression, integration-shape, and artifact tests |
| `schemas/` | JSON Schemas for allocation records and evaluation contracts |
| `docs/` | Architecture, evidence, data-rights, readiness, and onboarding docs |
| `paper/` | Manuscript, protocol, generated aggregates, figures, and PDFs |
| `local_data/` | Ignored owner-supplied data; only its README is tracked |
| `reports/` | Ignored local report output |
| `requirements/` | Exported dependency locks |
| `.github/workflows/ci.yml` | CI job definitions |

## 8. Data layer

### 8.1 Provider interface

`quantcortex.data.providers.base.DataProvider` defines the provider boundary.
Providers share an interface for:

- OHLCV data;
- fundamentals;
- macro data;
- derived price and return convenience paths.

Canonical market-data frames use validated symbol fields and timestamp indexes.
Timezone handling normalizes timestamps to UTC-naive internal indexes where the
code contract requires it.

Provider capabilities differ:

| Provider | OHLCV | Fundamentals | Macro |
|---|---:|---:|---:|
| `YFinanceProvider` | Yes | Yes, with heuristic availability dates | No |
| `PolygonProvider` | Yes | Yes | No |
| `AlpacaProvider` | Yes | No | No |
| `CCXTProvider` | Yes | No | No |
| `FMPProvider` | Yes | Yes | Yes |
| `FREDProvider` | No | No | Yes |

Unsupported methods raise `NotImplementedError`. Canonical provider OHLCV indexes
are normalized to UTC and then made timezone-naive. Provider clients are optional and lazily imported. Network access must be
explicit. After a real-data fetch fails, the workflow must not substitute
synthetic data and report performance from it.

`CCXTProvider` advances through successive candle pages until it reaches the
requested end or receives an empty page. This includes exchanges that cap each
page below the requested limit. A repeated, nonadvancing page fails the fetch.

Two adapters require extra PIT caution. YFinance fundamentals synthesize
`announcement_date` as `period_end + 45 calendar days`; this is a heuristic, not
an observed publication timestamp. Standard FRED retrieval returns latest revised
observations, not ALFRED vintages. Do not call either source point-in-time safe
without verified publication or vintage data.

### 8.2 Local CSV inputs

`quantcortex.data.local_csv` provides:

- `load_price_matrix`;
- `load_ohlcv_csv`;
- `sha256_file`;
- `LocalDataError`.

Wide adjusted-close input:

```csv
date,QQQ,VGT,GLD,TLT,SPY,VIG,SHV
2024-01-02,100.0,101.0,102.0,103.0,104.0,105.0,106.0
```

Requirements:

- unique parseable dates;
- numeric, finite, strictly positive observed symbol values;
- complete required symbol set;
- explicit cash-proxy column when cash earns a return.

`load_price_matrix()` permits missing observations, forward-fills up to five
consecutive missing rows per symbol by default, then drops rows that remain
incomplete. `require_complete=True`
checks completeness after any configured fill. Reports and notebooks use that
bounded-fill policy. Fixed paper experiments disable forward fill and require
complete rows.

Single-symbol OHLCV input:

```csv
date,open,high,low,close,adj_close,volume
2024-01-02,100.0,102.0,99.0,101.0,101.0,1500000
```

All price fields must be positive, and volume must be nonnegative. For each row,
`high` must be at least the maximum of `open`, `low`, and `close`, while `low`
must be at most the minimum of `open`, `high`, and `close`.

### 8.3 Processors

| Module | Purpose |
|---|---|
| `adjustments.py` | Validate adjustment relationships and corporate-action consistency |
| `calendar.py` | Trading calendars and weekly/monthly decision schedules |
| `lookahead_detector.py` | Detect future-index and feature-alignment violations |
| `pit_enforcer.py` | Strict point-in-time merge and availability rules |

### 8.4 Universes

`PITMembership` stores validated half-open membership intervals
`[start_date, end_date)`; a recorded `end_date` is excluded. `Universe` provides
the abstract constituent interface.

Available implementations include S&P 500 and Nasdaq-100 universes. The
Wikipedia S&P 500 reconstruction is approximate and valid only within its source
coverage. It does not supply delisted-security prices and does not make a study
fully survivorship safe.

### 8.5 Storage

Optional storage modules include:

- `ParquetStore` for local columnar data;
- `RedisCache` for cache/state integration;
- `TimescaleStore` for SQL time-series persistence.

Storage clients remain optional. Do not add a top-level import that makes a core
module require Redis, SQLAlchemy, or psycopg2.

### 8.6 Data rights and provenance

Raw provider matrices belong under ignored `local_data/`. Record separately:

- provider and adapter version;
- permission or license basis;
- retrieval timestamp;
- request settings;
- symbol mapping;
- adjustment method;
- timezone;
- cleaning and missing-data rules;
- input SHA-256;
- row and date coverage.

A hash identifies a matching input; it does not establish permission.
Repository-owner approval does not settle provider terms. Raw provider
observations stay out of Git, and any redistribution requires an explicit basis.

## 9. Alpha and feature research

### 9.1 Classical factors

| Class | Module | Main concern |
|---|---|---|
| `MomentumFactor` | `quantcortex/alpha/factors/classical/momentum.py` | Causal return windows and cross-sectional ranking |
| `LowVolFactor` | `quantcortex/alpha/factors/classical/low_vol.py` | Volatility/beta estimation and cross-sectional normalization |
| `ValueFactor` | `quantcortex/alpha/factors/classical/value.py` | PIT fundamentals, market-cap derivation, adjusted-price caveats |
| `QualityFactor` | `quantcortex/alpha/factors/classical/quality.py` | PIT accounting fields, TTM values, safe ratios |

Shared helpers validate panels, compute returns/covariances, and normalize
cross-sectional scores. Fundamental factors must use availability timestamps,
not report periods alone.
Quality ratios require matching fiscal quarters across their components. When
a new quarter is available for only some components, the ratio remains missing
until the required fields arrive.

### 9.2 Feature engineering

- `Alpha158` implements the Qlib feature layout without a Qlib dependency.
  Its rolling features require full windows, and missing VWAP is replaced by
  `(high + low + close) / 3`. Matching the 158-column default layout does not
  establish numerical parity with Qlib. Close-derived features are available
  only after that bar closes.
- `MacroFeatures` shifts monthly series by configurable approximate publication
  lags before forward-filling to a business-day index. Defaults are 35 calendar
  days for UNRATE, PMI, and NAPM. These are assumptions, not verified release
  times; callers still align features to actual decision timestamps.

### 9.3 Machine-learning and NLP factors

- `GBDTFactor` supports LightGBM, XGBoost, CatBoost, and scikit-learn backends.
  `model="auto"` uses the first installed backend in that order. Reproducible
  work must select a backend explicitly, pin its version, and record parameters.
- `NeuralFactor` defaults to a scikit-learn backend. Explicit `backend="auto"`
  prefers Torch when available, so environment changes can change the model.
- `NewsScorer` defaults to deterministic lexicon scoring. Daily aggregation
  discards intraday times, so either filter news at the exact decision cutoff or
  delay a close-of-day signal until the next close; otherwise, post-close news is
  included in that calendar date's score.
- `FinBERTSentiment(backend="auto")` falls back to its lexicon only when the
  optional package or model files are unavailable. Inference, malformed-output,
  and label-schema failures surface instead of changing models silently.

XGBoost, LightGBM, CatBoost, PyTorch, and Transformers are optional. Import them
inside the method that needs them. A missing optional dependency should produce a
clear error or documented offline fallback, not break import of the package.

### 9.4 Factor validation

- `AlphalensReport` computes information coefficient, quantile returns, and
  turnover diagnostics.
- `FactorDecay` evaluates signal decay.
- Validation code does not repair point-in-time defects in factor construction.
- Multiple variants require an explicit trial universe and appropriate
  multiplicity handling. A favorable IC or backtest is not sufficient evidence
  of a robust factor.

## 10. Portfolio construction

Read `quantcortex.portfolio.base`, which defines allocation and exposure
contracts, before editing any allocator or strategy.

Available allocators:

| Class | Modes | Purpose and boundary |
|---|---|---|
| `EqualWeight` | Long-only | `1/N`; natural contract sanity check |
| `MeanVariance` | Long-only, market-neutral | Sensitive to expected-return and covariance estimates |
| `MinimumVariance` | Long-only | Rejects market-neutral mode because zero-net minimum variance is trivially flat without another exposure constraint |
| `RiskParity` | Long-only | Depends on covariance quality |
| `HierarchicalRiskParity` | Long-only | Deterministic ordering and distance validity matter |
| `BlackLitterman` | Long-only, market-neutral | View and confidence assumptions must be explicit |
| `DRLAllocator` | Long-only | Heuristic inference needs only core dependencies; PPO training needs RL extras |

For `MeanVariance`, a Series of expected returns must contain each asset label
exactly once. The allocator aligns it to the return columns. Arrays and lists
use column order.

An untrained `DRLAllocator` fails closed by default. Deterministic heuristic
fallback requires explicit `untrained_policy="heuristic"`. Core CI checks this
contract and fallback; it does not train PPO.

Useful base helpers:

- `normalize_long_only`;
- `normalize_market_neutral`;
- `project_bounded_sum`;
- `validate_return_panel`;
- `prepare_return_panel`;
- `enforce_weight_contract`;
- `enforce_exposure_contract`.

To add an allocator:

1. subclass `PortfolioOptimizer`;
2. implement `_compute_weights()`;
3. return a vector already feasible for the configured mode;
4. let `optimize()` enforce the contract;
5. test finite values, bounds, budget, deterministic ordering, degenerate inputs,
   and failure behavior.

## 11. Timing and risk overlays

Overlays are callables with this interface:

```text
weights, context -> weights
```

A strategy binds component-specific inputs through a small lambda or partial.
Overlays must preserve vector shape and return output valid under the relaxed
exposure contract.

### Timing components

| Class | Role |
|---|---|
| `HMMRegime` | HMM/GMM-style regime exposure gate with deterministic fit controls |
| `TSMomentum` | Time-series momentum exposure overlay |
| `KAMA` | Kaufman adaptive moving-average timing |
| `VIXScaler` | Inverse-volatility/VIX exposure scaling with explicit floor and cap |

`HMMRegime` accepts `hmm`, `gmm`, or `auto`. `auto` selects HMM when importable
and falls back to GMM only when HMM import or loading fails; fit failures surface.
Repository reference strategies pin GMM. Standalone timing methods generally do
not know a decision timestamp, so callers must restrict inputs to observations
available at that time.

### Risk components

| Class | Role |
|---|---|
| `CircuitBreaker` | Drawdown-based exposure reduction; latches flat until `reset()` unless `auto_reset=True` |
| `VolTargeting` | Realized-volatility exposure scaling |
| `VaRCVaR` | Positive-loss VaR/CVaR estimates; does not enforce a limit or apply an overlay |
| `FactorExposureLimiter` | Factor-exposure constraints; caller must supply PIT loadings |
| `KellyCriterion` | Kelly-style sizing |

An overlay is not an allocator. It may reduce exposure, but it must not silently
renormalize the portfolio back to full risky investment. Risk classes do not
establish provenance or timing of caller-supplied estimates.

## 12. Strategy pipeline

`StrategyContext` contains:

- `as_of` decision timestamp;
- price observations no later than `as_of`;
- return observations no later than `as_of`;
- optional universe;
- an `extra` dictionary for component-specific inputs.

It rejects future top-level price or return observations, duplicate timestamps,
duplicate symbols, infinite values, and invalid prices. `extra` is checked only
to be a dictionary. It has no generic schema, symbol, timestamp, completeness,
freshness, or PIT validation. Each producer and consumer validates its nested
payload before use.

`Strategy` defines:

1. `select(ctx) -> pd.Series`: required alpha scores indexed by selected symbols;
2. `allocate(scores, ctx) -> np.ndarray`: defaults to the configured optimizer;
3. `apply_overlays(weights, ctx)`: timing overlays followed by risk overlays;
4. `rebalance(ctx) -> RebalanceResult`: validates selection, allocation, aggregate
   overlay shape, and final relaxed exposure;
5. `generate_weights(prices, rebalance_dates) -> pd.DataFrame`: emits the target
   panel consumed by engines.

`generate_weights()` skips a requested date when no prior return history exists.
After selection begins, an empty selection creates an explicit all-cash target so
a stale invested target is not carried forward. Its
`extra["current_weights"]` value is the previous generated target, not
marked-to-market or executed holdings. The method has no general per-decision
`extra` input. Strategies needing actual positions, fill state, or dated auxiliary
features must construct `StrategyContext` explicitly and call `rebalance()`, or
use a custom driver.

Concrete strategies:

| Strategy | Main use |
|---|---|
| `MultiAssetRotation` | Weekly multi-asset rotation reference strategy |
| `MomentumMLStrategy` | Momentum plus optional ML selection |
| `MacroTimingStrategy` | Macro-conditioned timing |
| `SentimentNLPStrategy` | NLP sentiment selection |
| `DRLPortfolioStrategy` | Optional reinforcement-learning strategy |

To add a strategy:

1. subclass `Strategy`;
2. implement `select()` using only data available by `ctx.as_of`;
3. override `allocate()` only if scores directly determine weights;
4. wire overlays without importing optional dependencies at module import time;
5. use `generate_weights()` for the engine handoff;
6. test empty selections, unknown symbols, missing history, causality, allocation
   mode, overlay shape, residual cash, and determinism.

## 13. Backtesting and evaluation

### 13.1 Canonical allocation records

Canonical allocation records separate strategy decisions from engine accounting.
Their long form is:

```text
decision_timestamp,symbol,target_weight
2024-01-01T00:00:00Z,A,1.0
2024-01-01T00:00:00Z,B,0.0
```

`quantcortex.backtest.conformance` provides:

- `validate_target_tape`;
- `weights_to_target_tape`;
- `target_tape_to_weights`;
- `target_tape_to_payload`;
- `target_tape_from_payload`.

The version-1 payload declares:

- `schema_version`;
- complete symbol set;
- maximum gross exposure;
- decision records.

JSON Schema checks envelope and record shape, schema version, unique declared
symbols, timestamp syntax, primitive types, and scalar weight bounds. It does not
check record membership in the declared universe, complete books, unique
timestamp-symbol pairs, or aggregate gross exposure.

`target_tape_from_payload()` and `validate_target_tape()` add exact-key and JSON
number checks, binary64 finiteness, UTC normalization, symbol consistency,
duplicate rejection, a complete symbol book at every decision, long-only weights,
per-decision gross limits, and unsupported-version rejection.

Allocation records contain decisions, not prices, fills, orders, or realized
returns. A payload hash identifies the records but cannot reconstruct provider data or empirical
returns.

### 13.2 Evaluation contract

`schemas/evaluation_contract.schema.json` and
`paper/results/evaluation_contract.json` declare:

- allocation record version and limits;
- selection, allocation, timing, and risk semantics;
- execution timing;
- cash treatment;
- cost model;
- comparator roles;
- order-state behavior;
- evidence classes.

The evaluation contract records declared semantics. Schema validation checks
shape, constants, and nonempty prose. It leaves the truth of those declarations
and their connection to a particular engine invocation unverified. For reproduction,
persist the allocation records and the details the contract does not fully identify in a
run manifest: concrete selector and allocator classes, model backend and library
version, parameters, input digests, and the specific engine call. Named runtime
checks, schemas, property tests, and failure behavior supply the enforcement
evidence.

### 13.3 Vectorized engine

`VectorizedBacktest` supports approximate evaluation and parameter sweeps. It:

- requires a `TransactionCostModel`;
- snaps decisions to the first strictly later bar;
- computes returns from executed weights;
- supports residual cash and optional ADV caps;
- implicitly re-pegs weights every bar between explicit rebalances at zero cost;
- does not model holdings drift from relative asset returns.

That daily re-pegging convention is the central approximation.

### 13.4 Event-driven engine

`EventDrivenBacktest` is the reference accounting path. It:

- holds explicit units against the supplied price matrix;
- lets risky weights drift between rebalances;
- sizes new targets against post-cost NAV;
- applies explicit fill semantics;
- tracks cash and trade accounting bar by bar;
- solves the cost-aware rebalance sizing problem.

The generic engine does not verify that its input is adjusted close. In the
reviewed paper workflow, pseudo-shares are total-return accounting units valued
at adjusted closing prices. They are not nominal broker shares and do not
reproduce actual broker lots or fill uncertainty.

Agreement between engines in one run is a diagnostic, not proof that they are
interchangeable. Their difference bundles re-pegging, drift, trade accounting,
and post-cost sizing semantics.

### 13.5 Walk-forward validation

`WalkForwardOptimizer` supports expanding or rolling splits, purges the last
`label_horizon` training samples, and leaves `embargo_gap` bars before each test
window. Protection applies only when callers use the splitter and set the true
forward-label horizon. It does not make arbitrary model training point-in-time.
Other helpers with a separate purge count must likewise cover every training
label that can cross a decision boundary.

### 13.6 Fill and impact models

- `ExecutionModel` defines the fill interface.
- `IdealFill` is an optimistic benchmark.
- `VWAPFill` models slippage around a bar's VWAP or a stated price proxy. It
  rejects adjustments that produce a nonpositive or nonfinite fill price.
- `AlmgrenChriss` provides a simplified impact fill-price model and a separate
  linear-impact liquidation-schedule helper. The fill model changes the current
  fill price, not future market prices; the schedule helper is not applied
  automatically by the engine.

Do not combine slippage in both a fill model and `TransactionCostModel` without
checking for double counting.

### 13.7 Metrics and statistical diagnostics

`BacktestResult` stores returns, engine-effective weights, turnover, costs,
`equity_curve`, cash weights, and metadata. Its weights are post-trade positions
on the price grid, not a dedicated copy of original target decisions. Persist the
canonical allocation records separately for target-versus-executed analysis. `Tearsheet`
computes common performance diagnostics.

Research validation includes:

- conventional and probabilistic Sharpe utilities;
- Deflated Sharpe Ratio helpers;
- Bonferroni, Benjamini-Hochberg, and BHY procedures;
- look-ahead audits;
- survivorship checks.

These tools require defensible inputs. A guessed trial count does not identify a
valid Deflated Sharpe Ratio. Multiple-testing procedures need the actual candidate
set and aligned evidence. Diagnostics do not repair missing trial history.
The probabilistic and deflated Sharpe helpers return `nan` when the estimated
Sharpe variance is nonpositive or nonfinite. A short sample can produce an
invalid moment approximation even when its returns are finite and variable.

### 13.8 Minimal engine example

```python
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.vectorized import VectorizedBacktest

cost_model = TransactionCostModel()
engine = VectorizedBacktest(cost_model, capital=1_000_000.0)
weights = strategy.generate_weights(prices, rebalance_dates)
result = engine.run(weights, prices, cash_returns=cash_returns)
```

Use a complete aligned `cash_returns` series when residual cash earns a nonzero
return. Use the event-driven engine when holdings drift and post-cost sizing are
part of the evaluated construct.

## 14. Execution, broker adapters, and state

### 14.1 Broker interface and adapter boundaries

`quantcortex.execution.brokers.base.Broker` defines order submission, position
retrieval, account retrieval, and validated `Position` and `AccountInfo` models.
SDK imports are lazy.

| Adapter | Paper/live boundary | Extra capability |
|---|---|---|
| `AlpacaBroker` | Requires exact official paper or live HTTPS host | Deterministic client-order-ID lookup and open-order reconciliation used by paper cycle |
| `IBBroker` | Rejects known paper/live port mismatches | Cannot classify an arbitrary host and port as paper or live |
| `CCXTBroker` | Paper mode requires advertised sandbox support and calls `set_sandbox_mode(True)` | Exchange behavior varies |

The Alpaca adapter keeps `done_for_day` and `stopped` orders unresolved, including
any reported partial fill. A `suspended` or unknown status stops parsing and
requires reconciliation before the caller can proceed.

The IB adapter includes an explicitly selected account on submitted orders.
Position retrieval accepts USD stock contracts with a valid contract ID.
Other instruments require an adapter that preserves their contract details;
the stock adapter rejects them before translating positions into orders.
An empty position response for that account is accepted after checking the
session's managed accounts. Cancellation requires an unambiguous API order
identifier for the selected account and client; a permanent identifier alone
requires reconciliation.

Direct `submit_order()` calls validate an individual request and send it. Broker
adapters do not invoke `PreTradeRiskCheck`, require `OrderManager`, persist a
submission intent, or block a retry after uncertain transport. Caller
orchestration must add those controls.

### 14.2 Position translation and pre-trade risk

`PositionManager.target_weights_to_orders()` converts target weights and current
signed quantities into order intents. It requires a finite positive price for
every targeted or held symbol. Whole-share mode rounds targets toward zero;
fractional books require `allow_fractional=True`.

The guarded submission sequence is:

```text
target validation
  -> PositionManager order translation
  -> PreTradeRiskCheck.assert_safe(targets, complete order batch,
                                   prices, capital, current positions)
  -> durable submission intent
  -> broker call
```

Translation precedes the final assertion because `PreTradeRiskCheck` validates
targets, individual order notional, and reconstructed post-trade exposure. It
validates supplied values; it does not query broker buying power or determine
market-data freshness. The paper cycle performs freshness checks separately.
Do not bypass this final portfolio-level gate.

### 14.3 Order lifecycle

`OrderManager` permits these paths:

```text
NEW -> SUBMITTED -> FILLED
NEW -> SUBMITTED -> PARTIALLY_FILLED -> FILLED
NEW or SUBMITTED -> CANCELLED or REJECTED
PARTIALLY_FILLED -> CANCELLED
```

Repeated partial fills are valid. Filled, canceled, and rejected orders are
terminal. Illegal transitions, duplicate orders, and unknown orders raise explicit
errors.

Incremental fills must consistently include prices or leave them unknown.
Mixing the two raises before quantity or history changes, because an average
fill price cannot be recovered from incomplete price information.
Replaying a cumulative fill with the same quantity and a different average
price also requires reconciliation. The position manager rejects that price
correction rather than silently changing the basis of intervening trades.

Order lifecycle and broker-submission intent are distinct. A local order can have
a valid lifecycle while external submission remains uncertain.

### 14.4 State persistence

`ExecutionSnapshot` stores positions, known orders, submission intents, and
reconciliation metadata as one versioned snapshot.

Typed loaders distinguish a missing record from malformed stored data. Invalid
containers, duplicate position keys after normalization, and disagreement
between a position key and its symbol stop recovery.

File and Redis state writes are atomic per individual operation. Only
`save_execution_snapshot()` couples all execution fields atomically and enforces
`expected_revision`; generic `save_state()`, `save_positions()`, and
`save_orders()` do not provide optimistic revision checking or a coherent
multi-call transaction.

Persist-before-submit, deterministic client-ID reconciliation, and uncertain
retry blocking are guarantees of Alpaca paper-cycle orchestration, not direct
broker-adapter calls. Other integrations must implement equivalent durable intent,
broker reconciliation, and retry blocking.

### 14.5 Paper-trading cycle

Offline, with deterministic generated prices and no network call:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py --offline
```

Explicit live-yfinance path:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py --live-yfinance
```

Paper submission:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py \
  --live-yfinance --submit
```

Live-yfinance and Alpaca dependencies are absent from `requirements/dev.lock`;
install `requirements/runtime.lock` or the required provider and broker extras.
The script uses the fixed risky universe `QQQ,VGT,GLD,TLT,SPY,VIG` and refuses
to use a real-money Alpaca endpoint.

Without both Alpaca credentials, `--live-yfinance` uses live prices but a local
$100,000 empty-account simulation. With credentials, it reads paper-account
state, but any unresolved open order blocks preview and submission. Whole-share
translation means fractional holdings, or held symbols absent from the price
panel, can stop order construction. If sells and buys are both needed, one
invocation submits only sells; rerun after fills and refreshed account state to
submit buys.

On a submit invocation, the cycle first refreshes persisted nonterminal order
IDs against Alpaca. When it reconciles an order, it saves the order state and
submission ledger in the execution snapshot and exits without submitting new
orders. Re-run after unresolved orders finish to obtain a fresh account snapshot
before the next submission. Missing or mismatched orders remain blocked.

Before each submission, the cycle queries Alpaca by deterministic client order
ID. If that lookup finds no broker order corresponding to a prior local attempt, the
attempt remains uncertain and is not retried automatically. Use `--retry-confirmed-missing` only after Alpaca dashboard
or support confirms that prior intent was not routed.

Operational market orders can fill in the current venue session when it is open.
Orders submitted after the finalized close wait for the next eligible session.
They are not performance-equivalent to the paper experiment's modeled next-close
fills.

## 15. Research program and paper methodology

### 15.1 Evaluated construct

The paper evaluates historical simulated performance under declared:

- information timing;
- target-generation rules;
- execution timing;
- residual-cash treatment;
- modeled transaction costs;
- comparator semantics;
- engine semantics;
- provenance and evidence checks.

It does not evaluate prediction accuracy, future profitability, alpha, investment
skill, causal treatment effects, deployment safety, or a population benchmark.

### 15.2 Retrospective worked audit

The first study is a fixed six-ETF weekly rotation strategy evaluated from
2018-01-02 through 2025-12-31.

The risky universe is QQQ, VGT, GLD, TLT, SPY, and VIG. SHV is the residual-cash
proxy and Sharpe reporting benchmark.

The strategy combines group ranking, residual momentum, a Gaussian-mixture
exposure gate, and volatility scaling. Its recorded specification is:

- owner-authorized Yahoo Finance adjusted closes retrieved through yfinance 1.5.1
  with `auto_adjust=False` on 2026-08-29;
- source window 2016-01-04 through 2025-12-31, with 503 pre-evaluation sessions
  available and 274 required;
- groups `(QQQ, VGT)`, `(GLD, TLT)`, and `(SPY, VIG)`;
- first available session of each week as decision schedule;
- top two groups by a 126-session information ratio relative to QQQ;
- within selected groups, a 126-session one-factor regression against QQQ,
  followed by a separate 126-session residual-return formation window with a
  21-session gap;
- proportional positive-score weights with a 0.60 maximum per ETF;
- three-component full-covariance GMM, at most 100 iterations, seed 42,
  covariance regularization `1e-5`, and features consisting of QQQ returns plus
  two exact rescalings of one 20-session realized-volatility series;
- GMM component exposure multipliers `0.0`, `0.5`, and `1.0` after deterministic
  regime relabeling;
- separate 21-session realized-volatility proxy targeting 20 volatility points,
  with multiplier clipped to `[0.3, 1.0]`;
- plain-momentum fallback when benchmark, group signal, or regression estimate is
  unavailable;
- cash when all mature selected-group residual scores are nonpositive.

Known construct problems are retained rather than repaired after observing the
result:

- QQQ is both a group member and ranking benchmark. Its regression against
  itself should have zero residual, but positive floating-point error can pass
  the score filter and become a material allocation after normalization. The
  released results retain this implementation defect;
- two regime features are exact volatility rescalings;
- early regime windows are immature;
- one seed controls the mixture fit;
- earlier trial count is unknown.

Therefore this study is a worked accounting audit, not confirmatory evidence.

Primary accounting:

- event-driven pseudo-shares;
- weekly targets;
- execution on the first strictly later bar;
- SHV residual cash;
- 13 bp per dollar bought or sold;
- no forward fill in the fixed paper matrix.

Reported descriptive values include:

| Metric | Value |
|---|---:|
| Net CAGR | 2.05% |
| Gross CAGR before modeled cost | 3.86% |
| SHV CAGR | 2.50% |
| Net Sharpe relative to SHV | -0.04 |
| Gross Sharpe relative to SHV | 0.25 |
| Net maximum drawdown | -9.25% |
| Mean risky exposure | 30.97% |
| Fully cash sessions | 46.89% |
| Comparison portfolio following target exposure: CAGR after costs | 6.10% |
| Comparison portfolio following target exposure: Sharpe after costs | 0.67 |

These values belong to one historical path under the recorded assumptions.

### 15.3 Exact arithmetic decomposition

Let:

- `r_t^gross` be gross strategy return;
- `r_t^net` be net strategy return;
- `r_t^c` be residual-cash return;
- `x_t` be realized risky exposure;
- `b_t` be the fixed equal-initial-weight buy-and-hold basket return;
- `x_bar` be full-sample mean realized risky exposure;
- `m_t = x_t b_t + (1 - x_t) r_t^c`;
- `k_t = x_bar b_t + (1 - x_bar) r_t^c`.

Then:

```text
r_t^net - r_t^c
  = (r_t^gross - m_t)     active risky allocation
  + (m_t - k_t)           dynamic exposure timing
  + (k_t - r_t^c)         passive risky exposure
  + (r_t^net - r_t^gross) modeled implementation cost
```

Annualized arithmetic means add exactly because the identity holds daily.
Compounded wealth and CAGR do not decompose this way.

Reported annualized components:

| Component | Mean |
|---|---:|
| Active risky allocation | -3.05% |
| Dynamic exposure timing | +0.54% |
| Passive risky exposure | +4.00% |
| Modeled implementation cost | -1.76% |
| Net excess over SHV | -0.26% |

Interpretation limits:

- active risky allocation combines group selection, within-risky weighting, and
  event-engine drift;
- it is not pure security selection;
- timing is relative to full-sample mean exposure and is ex post;
- components are reference-dependent, noncausal, and not separately tradable;
- the basket at actual exposure is before its own trading costs.

### 15.4 Comparator roles

Do not merge these roles:

1. **Basket at actual exposure, before costs**
   - exact ex-post arithmetic attribution;
   - matches realized risky exposure;
   - before its own trading costs;
   - not tradable or causal.

2. **Retrospective comparison portfolio following target exposure**
   - follows the strategy's declared target risky exposure;
   - invests risky exposure in a fixed equal-initial-weight buy-and-hold basket;
   - uses next-bar event accounting;
   - credits residual SHV;
   - pays its own modeled costs.

3. **Expansion comparison portfolio following target exposure**
   - follows the strategy's declared target risky exposure;
   - re-equalizes it across risky ETF columns only, excluding SHV, at each
     decision;
   - uses next-bar event accounting, SHV cash, and its own costs.

Both comparison portfolios are conditional on the strategy target path. Neither is
an independent benchmark or causal control.

### 15.5 Repository-frozen expansion

The expansion protocol is repository-frozen but not externally registered. Its
recorded commit timestamp precedes recorded provider-request timestamps, but all
are author-controlled and do not independently prove wall-clock ordering or
public availability. The 2018 to 2025 interval is not a temporal holdout.
The freeze identifier below comes from private development history. Retaining
its value and the protocol byte hash in a public snapshot does not make that
history publicly accessible or prove a public pre-data freeze.

Frozen provenance record:

- protocol commit `4018f4063f46889f41d6981db5a71079e1dbd713`;
- commit timestamp `2026-06-18T15:25:58-07:00`, equivalent to
  `2026-06-18T22:25:58Z`;
- protocol SHA-256
  `e49e41a12a19fa5404a573ba5e21eb8a2888e616985f8c610d9652866923315c`;
- current provider retrieval timestamp `2026-08-29T15:00:36Z` for both panels.

Data range:

- adjusted closes from 2014-01-02 through 2025-12-31;
- evaluation from 2018-01-02 through 2025-12-31;
- 2,011 evaluation sessions;
- complete rows only;
- no forward fill or symbol substitution.

Each panel records 3,018 complete rows, zero dropped incomplete rows, 1,007
pre-evaluation sessions across 48 pre-evaluation months, and 2,011 evaluation
sessions. Retrieval used yfinance 1.5.1, selected `Adj Close`, used an exclusive
request end of 2026-01-01, and set `actions`, `auto_adjust`, `repair`, and
`threads` to `false`.

U.S. sector panel:

```text
XLB XLE XLF XLI XLK XLP XLU XLV XLY SHV
```

Country equity panel:

```text
EWA EWC EWG EWH EWJ EWL EWP EWQ EWS EWU SHV
```

In both panels, SHV is only the cash-return proxy and Sharpe benchmark. It is
excluded from risky signal generation, regression training rows, risky allocation records,
and comparator risky equalization.

These are fixed surviving ETF panels with overlapping broad market exposures,
not independent replications or point-in-time constituent populations.

Monthly decisions occur on the first complete session of each month and execute
on the first strictly later panel close.

Frozen families:

| Family | Rule |
|---|---|
| Time-series momentum | Allocate equally among assets with positive 252-session returns; if none qualify, set risky exposure to zero so residual cash earns the SHV return |
| Cross-sectional momentum | Hold top three returns from `t-252` through `t-21`, irrespective of sign, at one-third each |
| Short-term reversal | Hold up to three lowest negative 5-session returns at one-third each without renormalizing unused exposure |
| Gradient boosting | Predict 21-session forward log return; hold up to three highest strictly positive predictions at one-third each without renormalizing unused exposure; use five fixed seeds |

Gradient boosting details:

- pooled asset-month rows without symbol identity;
- log-return features over 5, 21, 63, 126, and 252 sessions;
- sample standard deviations of daily log returns over 21 and 63 sessions;
- current price divided by trailing 252-session high minus one;
- ascending ordinal ranks of 21- and 252-session log returns, normalized to
  `[0, 1]` with ascending-symbol tie breaks;
- no standardization or winsorization;
- 21-session forward log-return labels entering training only when their endpoint
  is observable on or before current decision;
- rolling cap of 60 mature decision months;
- minimum 24 mature decision months;
- `sklearn.ensemble.GradientBoostingRegressor` with 100 trees, maximum depth 2,
  learning rate 0.03, minimum leaf size 10, and subsample 0.8;
- seeds 11, 29, 47, 71, and 97;
- ascending-symbol resolution for every score and rank tie.

Each seed defines a separate walk-forward run, with a new model fit at every
monthly decision, its own allocation records, and its own backtest. The family
metric is the arithmetic mean of individual seed metrics. The reported Sharpe is
the mean of the individual seed Sharpes. No ensemble return stream is traded.
The five seeds are fixed, not sampled from a population. Bootstrap resampling does not refit models.

Recorded descriptive baselines:

| Panel | Family | Annualized arithmetic return | Sharpe relative to SHV | CAGR | Maximum drawdown |
|---|---|---:|---:|---:|---:|
| U.S. sectors | Time-series momentum | 11.44% | 0.49 | 10.29% | -34.16% |
| U.S. sectors | Cross-sectional momentum | 14.19% | 0.60 | 13.04% | -32.06% |
| U.S. sectors | Short-term reversal | 11.02% | 0.44 | 9.54% | -45.08% |
| U.S. sectors | Gradient boosting | 13.66% | 0.53 | 12.10% | -39.96% |
| Country equities | Time-series momentum | 2.18% | -0.02 | 0.88% | -40.45% |
| Country equities | Cross-sectional momentum | 8.84% | 0.33 | 7.21% | -40.73% |
| Country equities | Short-term reversal | 6.03% | 0.20 | 4.48% | -43.91% |
| Country equities | Gradient boosting | 7.87% | 0.31 | 6.60% | -33.56% |

Seven of eight sample Sharpes are positive, and every sector result exceeds its
same-family country result. These are dependent historical descriptions, not
population or alpha estimates. Individual gradient boosting seed Sharpes span
0.47 to 0.60 in sectors and 0.29 to 0.36 in countries.

Appendix Table 7 reports interval counts for block lengths of 5, 21, and 63
sessions. The following summary retains the 21-session intervals across eight
family-panel cells; category counts are
`below zero / overlaps zero / above zero`:

| Contrast | Annualized-return interval counts | Sharpe interval counts |
|---|---:|---:|
| Retroactive signal-close minus next-bar | 3 / 5 / 0 | 3 / 5 / 0 |
| Zero cash minus SHV cash | 5 / 3 / 0 | 5 / 3 / 0 |
| Zero cost minus 13 bp cost | 0 / 0 / 8 | 0 / 0 / 8 |
| Strategy minus comparison portfolio after costs | 1 / 7 / 0 | 1 / 7 / 0 |
| Vectorized minus event-driven | 0 / 4 / 4 | 0 / 4 / 4 |

Removing modeled cost raises annualized arithmetic return by 0.57 to 1.81
percentage points. Vectorized-minus-event-driven annualized-return differences
range from 0.02 to 0.19 percentage points. Across all deterministic and
gradient boosting seed paths, maximum absolute one-day difference is 93.21 bp, and largest
absolute terminal-wealth gap is 3.89%. Sector comparator Sharpes for time-series
and cross-sectional momentum both serialize as `0.547788592112` at the published
12-significant-digit precision. The rank artifact flags both rows as tied and uses
the strategy key only to make their displayed positions deterministic. Do not
reduce comparator rank changes to one unique count.

### 15.6 Five paired contrasts

The expansion reports five finite differences:

1. retroactive signal-close assignment minus audited next-bar execution;
2. zero residual-cash return minus SHV cash;
3. zero modeled cost minus the 13 bp charge;
4. net strategy minus its comparison portfolio following target exposure, after costs;
5. vectorized minus event-driven accounting.

In the zero-cash contrast, only the residual-cash return becomes zero; SHV remains
the fixed Sharpe reporting benchmark.

The first three are baseline-conditional convention diagnostics that hold allocation
records, input matrices, and primary event engine fixed. The comparator contrast
changes allocation, turnover, and costs. The engine contrast bundles daily weight resets,
drift, trade accounting, and post-cost sizing.

The five differences are not additive. Interactions are not estimated. The
closed-interval variant assigns a target to a return interval that has already
ended. It could not have been executed, and it is not a candidate strategy.

### 15.7 Uncertainty

Primary outcomes:

- paired annualized arithmetic-return difference;
- paired difference in conventional Sharpe relative to SHV.

Bootstrap design:

- circular-block bootstrap;
- 5,000 draws;
- two-sided, unstudentized, pointwise 95% percentile intervals;
- primary block length 21 sessions;
- sensitivity block lengths 5 and 63 sessions;
- retrospective seed 42, with every decomposition component jointly resampled by
  one common temporal-index draw so daily identity is preserved;
- expansion seed 20260618, with one common temporal-index draw within each
  family-panel contrast applied to both contrasted paths, SHV, and every fixed
  gradient boosting seed path.

The bootstrap does not:

- refit models;
- resample panels, families, allocation records, or design choices;
- treat fixed seeds as a population;
- estimate interaction uncertainty;
- adjust for multiplicity;
- provide simultaneous coverage across eight dependent cells.

A positive-draw fraction is a resampling diagnostic, not a posterior probability
or adjusted p-value. Stability across block lengths is sensitivity evidence, not
proof of stationarity or future performance.

### 15.8 Claims that must not be made

Do not describe the paper as showing:

- prediction accuracy;
- deployable alpha;
- future performance;
- investment skill;
- causal effects of protocol choices;
- an independent causal comparator;
- a temporal out-of-sample test;
- a general benchmark;
- valid population inference;
- production readiness;
- data redistribution rights;
- full independent empirical reproducibility.

Open schemas and synthetic fixtures reproduce software semantics. Historical
return reproduction still requires matching private matrices, source,
configuration, environment, and hashes of reconstructed allocation records.

## 16. Data, artifacts, and provenance

Empirical release inputs remain local, ignored, and untracked:

- `local_data/published_rotation_prices.csv`;
- `local_data/expansion/us_sector_etfs.csv`;
- `local_data/expansion/us_sector_etfs.metadata.json`;
- `local_data/expansion/country_equity_etfs.csv`;
- `local_data/expansion/country_equity_etfs.metadata.json`.

Raw matrices and payloads of empirical allocation records are not published. A
payload hash can verify a reconstructed candidate but cannot reconstruct the payload. Exact
historical reproduction requires matching matrices, source, configuration,
environment, and hashes of reconstructed allocation records.

Lawful research use, publication of derived results, reviewer access, and raw
redistribution are separate permission questions. Repository-owner assertions
and content hashes do not establish provider authority or license compatibility.
Apply `docs/data-source-due-diligence.md`.

### 16.1 Files that may be edited directly

- Python source;
- shell wrappers;
- LaTeX manuscript source;
- bibliography;
- documentation;
- schemas;
- tests.

`paper/expansion/protocol.json` is not directly editable. It is byte-for-byte
frozen at commit `4018f4063f46889f41d6981db5a71079e1dbd713` and SHA-256
`e49e41a12a19fa5404a573ba5e21eb8a2888e616985f8c610d9652866923315c`.
Preserve frozen values, design, and historical names. Append dated post-freeze
clarifications only to `paper/preregistration.md`; that file is release-critical
and a clarification still requires coordinated regeneration.

### 16.2 Files that must be generated

Do not hand-edit:

- `paper/results/*`;
- `paper/figures/*`;
- `paper/expansion/results/*`;
- `paper/expansion/figures/*`;
- `docs/img/*`;
- `paper/quantcortex_audit_neurips2026.pdf`;
- `paper/quantcortex_audit_anonymous.pdf`;
- `paper/quantcortex_audit_neurips2026.sha256`;
- `paper/quantcortex_audit_anonymous.sha256`;
- `paper/quantcortex_audit_neurips2026.sources.sha256`;
- `paper/build_manifest.json`;
- generated report manifests and figures;
- any PDF under `output/pdf/`.

`output/pdf/quantcortex_tae_camera_ready.pdf` is ignored and must not be committed.

Historical machine strings are provenance, not preferred current terminology.
Preserve `cost_per_one_way_gross_notional`, comparator type
`causal target-exposure-matched equal weight`, and switch
`causal_exposure_matched_comparator` byte-for-byte in the frozen protocol. Despite
the legacy cost key's name, the charge is `0.0013 * (B_t + S_t)`, not one-way
turnover.

Keep `economic_counterfactual` as a version-1 evaluation-contract schema key
unless performing a versioned schema migration. Current prose uses
`paired convention switch` and comparison portfolio following target exposure.

### 16.3 Manifest model

Retrospective and expansion empirical result manifests bind:

- input digests and source metadata;
- source commit, source-tree digest, and per-file hashes;
- dependency lock and configuration;
- environment and thread-pool metadata;
- hashes of canonical allocation records and metadata;
- generated tables, figures, and artifact hashes;
- generation timestamp.

`paper/build_manifest.json` has a narrower role. It records paper source commit
and source-date epoch, Tectonic version and bundle digest, public, anonymous, and
TAE PDF digests, and source-manifest digest.
`paper/quantcortex_audit_neurips2026.sources.sha256` contains per-file hashes for
exact snapshotted manuscript source, generated values and manifests, figures,
build script, and bundle publisher.

Hashes establish identity and internal provenance. They do not establish provider
permission, external-data correctness, scientific validity, future behavior, or
complete independent reproducibility when raw inputs are unavailable.

### 16.4 Evidence classes

Use the narrowest accurate evidence label:

| Evidence class | What it can support |
|---|---|
| Exact identity | Algebraic equality on recorded series |
| Property test | Invariants across generated/synthetic cases |
| Paired protocol contrast | Difference between two declared evaluation paths |
| Fault injection | Failure response under an injected condition |
| SDK conformance | Request/response compatibility with installed SDK models |
| Untested live behavior | Explicitly no supporting live evidence |

A unit test cannot establish authenticated connectivity, market capacity, data
rights, or future profitability.

## 17. Paper build and release workflow

### 17.1 Paper source modes

| Entry point | Mode |
|---|---|
| `paper/main.tex` | Identified TAE final paper |
| `paper/anonymous.tex` | Generic anonymous review mode |
| `paper/submission.tex` | Identified TAE camera-ready (`dblblindworkshop, final`) |

Review modes suppress author, affiliation, repository, system name, and exact
fingerprint identifiers. The identified public PDF names the authors and uses the TAE final footer.

### 17.2 Direct build

A direct `scripts/build_paper.sh` run overwrites tracked PDFs, checksums, and build
provenance. Existing aggregates alone do not make it an unchanged release replay.
From the repository root, first verify every entry with
`sha256sum --check paper/quantcortex_audit_neurips2026.sources.sha256`.
Only if that check passes, set `PAPER_SOURCE_COMMIT` and `SOURCE_DATE_EPOCH` from
`source_commit` and `source_date_epoch` in the recorded `paper/build_manifest.json`
before invoking the build. Changed bound sources require the coordinated release
in Section 17.3, not a direct build labeled with old release metadata.

Requirements:

- Tectonic 0.16.9;
- `default_bundle_v33.tar` with SHA-256
  `6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c`;
- `pdftotext`;
- `pdfinfo`;
- `pdfdetach`;
- `pdffonts`;
- `strings`;
- `flock`;
- Python.

The build:

- snapshots every bound source file;
- compiles all modes from immutable snapshot bytes;
- checks source stability;
- verifies a ten-page body limit for the public, TAE camera-ready, and
  anonymous builds;
- checks review text, metadata, URLs, attachments, raw strings, and fingerprints;
- checks font embedding and rejects Type 3 fonts;
- rejects overfull boxes and scanner failures;
- publishes the complete bundle with rollback on partial failure;
- writes repository-root-relative source and build manifests.

Outputs:

- tracked public PDF under `paper/`;
- tracked anonymous PDF under `paper/`;
- untracked TAE camera-ready PDF under `output/pdf/`;
- checksums, `paper/quantcortex_audit_neurips2026.sources.sha256`, and
  `paper/build_manifest.json`.

Visual review of every page is still required. Automated geometry and text checks
do not prove that a figure is readable or semantically correct.

### 17.3 Coordinated empirical release order

For a new public export, use vetted tracked files to create a fresh source root
P without private Git history. Apply the sequence below with fixed UTC
timestamps, using the already authorized local inputs, and commit regenerated
outputs in child A. Both releases must name P; retain the ancestry, source-byte,
input and artifact checks. An imported manifest naming an unavailable private
source commit cannot support unchanged replay. The old protocol-freeze ID is
historical metadata; P-to-A lineage proves only the new artifact-source binding.

1. Finish and review release-critical source changes.
2. Run focused source tests, full Ruff, and the full test suite. Artifact-binding
   failures can occur after bound source changes and before coordinated regeneration;
   record them as failures until a reviewed release resolves the mismatch.
3. After explicit user approval, commit release-critical source.
4. Verify the authorized retrospective matrix, both expansion matrices, and both
   expansion metadata sidecars against the intended SHA-256 values.
5. Changed release-critical source or inputs require explicit, fixed UTC
   generation timestamps for both wrappers, even if only paper source changed.
   Replace the placeholders below with those timestamps. Only for unchanged
   reviewed source and inputs, omit both generation-timestamp overrides to reuse
   the recorded source commits and timestamps. Leave input-refresh digest and
   provenance overrides unset unless new inputs are separately authorized; this
   workflow does not authorize refetching data. From the repository root, set an
   absolute `PYTHON_BIN` once for the detached worktrees and run expansion first:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
QUANTCORTEX_EXPANSION_GENERATED_AT="YYYY-MM-DDTHH:MM:SSZ" \
  scripts/release_expansion_artifacts.sh local_data/expansion
```

6. After expansion succeeds, run the paper wrapper second from the same source
   commit, with no intervening commit:

```bash
QUANTCORTEX_GENERATED_AT="YYYY-MM-DDTHH:MM:SSZ" \
  scripts/release_paper_artifacts.sh \
  local_data/published_rotation_prices.csv
```

Only for a separately authorized panel refresh, also set both
`QUANTCORTEX_EXPANSION_US_SECTOR_INPUT_SHA256` and
`QUANTCORTEX_EXPANSION_COUNTRY_INPUT_SHA256`. A separately authorized retrospective
refresh also requires `QUANTCORTEX_EXPECTED_INPUT_SHA256` and all four provenance
variables: `QUANTCORTEX_DATA_PROVIDER`, `QUANTCORTEX_PERMISSION_BASIS`,
`QUANTCORTEX_RETRIEVED_AT`, and `QUANTCORTEX_ADJUSTMENT_METHOD`.

7. Compare every changed numerical CSV value and qualitative claim with the prior
   release. For replay, require unchanged hashes for inputs and allocation records.
   For an authorized refresh, investigate and document every change to results,
   allocation records, ranks, provenance fields, and identity hashes before
   publication.
8. Inspect every generated diff and manifest.
9. Run Ruff, focused artifact tests, and the full suite again; require no failures.
10. Visually inspect every page of the public, anonymous, and TAE camera-ready
    PDFs. Do not run `scripts/build_paper.sh` after the paper wrapper; that wrapper
    already invokes it with release-specific source epoch, and a direct build can
    overwrite tracked PDFs and manifests with different bytes.
11. Commit reviewed tracked generated artifacts in a separate artifact commit. Do
    not commit `output/pdf/quantcortex_tae_camera_ready.pdf`.
12. Submit to a venue only after every remaining rights, venue, CI, and post-upload
    verification gate is clear.

### 17.4 Release-wrapper guarantees

Both wrappers:

- require committed, tracked, clean release-critical source;
- acquire one repository publication lock;
- use detached worktrees at the recorded source commit;
- snapshot inputs outside the worktree;
- reject input, protocol, source, and provenance mismatches;
- use the publisher from the reviewed source revision;
- publish a complete artifact bundle or restore prior destinations;
- reject tracked raw market-data inputs.

The paper wrapper also regenerates the reviewed `docs/img/` performance gallery
and requires expansion artifacts to name the same source commit.

### 17.5 TAE camera-ready gates

The 2026-08-29 submission deadline has passed. The
[official accepted-paper list](https://tai-eval.github.io/papers/) confirms
Submission 423 at the non-archival TAE workshop. This is not acceptance in the
NeurIPS main proceedings. The [camera-ready instructions](https://tai-eval.github.io/),
checked on 2026-10-09, set the upload deadline to October 31, 2026 (AoE).
Use the submission's Camera Ready Revision button on OpenReview. The site lists
the in-person Sydney workshop for December 12, 2026, 08:00-17:00 AEDT.

The call specifies:

- use `\usepackage[dblblindworkshop, final]{neurips_2026}`;
- set `\workshoptitle{TAE (Trust-AI-Eval): Can We Trust AI Evaluation?}`;
- upload one PDF containing the paper, references, and appendices;
- end every camera-ready body paragraph, float, and caption by page 10. References
  must start no later than page 11;
- references and appendices do not count toward the ten-page camera-ready
  body limit;
- present accepted papers at the in-person poster session.

The repository retains the answered template checklist for the anonymous
review copy. The identified camera-ready and arXiv versions omit it at the
authors' request. The camera-ready instructions do not request a checklist.
The identified final versions also omit the acknowledgments section at the
authors' request. References start on a new page after the conclusion.
Repository build checks retain the separate
anonymous build, apply anonymity checks to that build, and require embedded
fonts without Type 3 fonts. The current OpenReview form was not rechecked for
this guide; consult it before supplying an external code or artifact link.

For a future OpenReview upload, check the applicable requirements below and
carry forward existing recorded authorizations. These are preparation checks,
not evidence that a remote upload has occurred:

- publication and provider-rights basis covering derived results, public
  OpenReview release, and CC BY 4.0;
- exact authorized matrices and expansion sidecars;
- author list and verified OpenReview profiles;
- author email-sharing, public-release, and CC BY 4.0 consent;
- in-person Sydney poster presenter, registration, and travel feasibility;
- recheck the published workshop day before making travel arrangements;
- current call, template, checklist, form, and any concurrent-submission rule;
- references checked against primary or official records;
- revalidation of unversioned 2026 preprints;
- geometry, font, and visual scans of every page, plus an anonymity scan of the
  separate anonymous build and an author and workshop-title scan of the
  camera-ready;
- the same text, metadata, URL, attachment, font, and visual checks on any PDF
  downloaded from OpenReview after an authorized upload.

Non-archival workshop status does not resolve data rights.

## 18. Developer workflows

### 18.1 Run one test

```bash
.venv/bin/python -m pytest \
  tests/test_weight_interface.py::test_equal_weight_sums_to_one -v
```

### 18.2 Add a provider

1. Subclass `DataProvider`.
2. Keep provider SDK import inside the method that uses it.
3. Normalize output to canonical schemas and timestamps.
4. Make network access and credentials explicit.
5. Reject malformed, incomplete, or ambiguous data.
6. Add mocked behavior tests and real SDK-shape tests where applicable.
7. Document rights, adjustment, calendar, timezone, and missing-data assumptions.

### 18.3 Add a factor

1. Put it under the correct `alpha` subpackage.
2. Define its availability timestamp and warm-up window.
3. Use PIT fundamental merging when needed.
4. Make cross-sectional universe assumptions explicit.
5. Keep tie-breaking deterministic.
6. Test shifted inputs, missing data, early windows, and future-data rejection.
7. Add validation diagnostics without treating them as proof of alpha.

### 18.4 Add an allocator

1. Subclass `PortfolioOptimizer`.
2. Implement `_compute_weights()` only.
3. Preserve `PortfolioMode` semantics.
4. Return a feasible raw vector.
5. Test budget, bounds, finite values, degenerate covariance, and deterministic
   asset ordering.

### 18.5 Add an overlay

1. Implement `callable(weights, StrategyContext) -> weights`.
2. Preserve vector shape.
3. Use only context data available by `as_of`.
4. Return relaxed exposure, not a forced full-risk budget.
5. Test all-cash, partial-exposure, cap, and failure cases.

### 18.6 Add a strategy

1. Subclass `Strategy`.
2. Implement `select()`.
3. Reuse an existing optimizer or override `allocate()` for score-driven weights.
4. Bind overlays through context-aware callables.
5. Generate target panels through `generate_weights()`.
6. Validate through serialization of canonical allocation records before comparative engine
   research.

### 18.7 Change backtest or money-path code

Money-path code includes:

- weight validation;
- target timing;
- cash accounting;
- transaction costs;
- execution fills;
- post-cost sizing;
- turnover;
- NAV;
- order translation;
- order state;
- broker submission intent;
- persistence and reconciliation.

For any such change:

1. derive a small hand-checkable expected result;
2. add a focused regression test against that result;
3. test invalid and partial-failure paths;
4. compare both engines only when semantics are intended to match;
5. run the full suite;
6. regenerate published artifacts only through reviewed inputs and wrappers.

### 18.8 Change a dependency

Lock generation requires Poetry 2.4.1 and `poetry-plugin-export` 1.10.0:

```bash
# Edit pyproject.toml first.
scripts/update_dependency_locks.sh
.venv/bin/python -m pytest tests/test_dependency_locks.py -q
```

Set `POETRY_BIN` if those versions are installed outside `PATH`. Review every
changed lock. A lock change is also a release source change.

### 18.9 Package and container checks

```bash
.venv/bin/python -m pip install -r requirements/build.lock
.venv/bin/python -m build
POSTGRES_PASSWORD=local-only docker compose config --quiet
docker build --tag quantcortex:local .
docker run --rm --read-only --tmpfs /tmp quantcortex:local
```

Runtime image uses a non-root user. `.dockerignore` excludes listed tests, local
data, reports, paper artifacts, `.env` patterns, and build output; it is not a
secret scanner. Keep credentials outside build context and explicitly exclude any
repository-specific secret path.

### 18.10 Commit and review discipline

- Use short conventional commits such as `fix: reconcile cash returns`.
- Keep source and coordinated generated artifacts in logically separate commits.
- Do not commit credentials, `.env`, local execution state, account identifiers,
  raw market data, cached provider data, or executed notebook outputs.
- PRs must state behavior, data/API assumptions, verification, and remaining
  limitations.
- Security-sensitive failures belong in a private security advisory, not a public
  issue.

## 19. Researcher workflows

### 19.1 Local performance report

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_report.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --cash-proxy-symbol SHV
```

For externally reviewed output, also provide explicit provider, permission,
retrieval, adjustment, and manifest fields. Those fields record assertions; they
do not independently establish rights.

### 19.2 Live-data diagnostics

Provider dependencies are absent from `requirements/dev.lock`. Install
`requirements/runtime.lock` or relevant provider extra before running:

```bash
.venv/bin/python scripts/validate_performance.py --live-yfinance
.venv/bin/python scripts/survivorship_demo.py --live-yfinance
```

These commands require an explicit live-data opt-in. Review provider terms first.
They exit on failed real-data retrieval rather than substitute synthetic prices.
The yfinance diagnostics do not provide delisted-security coverage.

### 19.3 Research notebooks

Notebooks are committed in an unexecuted state and run in CI on deterministic
synthetic fixtures.

| Notebook | Topics |
|---|---|
| `01_data_quality.ipynb` | Price loading, PIT enforcement, look-ahead scan, data-quality plots |
| `02_factor_research.ipynb` | Alpha158, classical factors, IC/ICIR, quantiles, turnover, decay |
| `03_portfolio_construction.ipynb` | Optimizer fitting and mandatory-cost allocation backtests |
| `04_backtest_analysis.ipynb` | Weekly targets, backtest, tearsheet, DSR, BHY correction |
| `05_live_trading_bridge.ipynb` | Pre-trade risk, order translation, lifecycle, persistence, paper connection |

Choose exactly one adjusted-close source:

```bash
export QUANTCORTEX_PRICES_CSV="$PWD/local_data/prices.csv"
export QUANTCORTEX_OHLCV_CSV="$PWD/local_data/aapl_ohlcv.csv"
.venv/bin/jupyter lab research/
```

Or opt into live yfinance:

```bash
export QUANTCORTEX_LIVE_YFINANCE=1
.venv/bin/jupyter lab research/
```

Do not set both `QUANTCORTEX_PRICES_CSV` and
`QUANTCORTEX_LIVE_YFINANCE=1`. `QUANTCORTEX_OHLCV_CSV` supplements the local
adjusted-close source for the Alpha158 notebook.

### 19.4 Research design checklist

Before inspecting outcomes, record:

- research question;
- universe and point-in-time membership policy;
- input source and rights basis;
- adjustment and missing-data policy;
- decision and execution timestamps;
- target-generation rule;
- cash proxy and cash accounting;
- transaction-cost base and rate;
- comparator role;
- engine semantics;
- candidate set and trial history;
- fixed seeds and tie-breaking;
- uncertainty procedure;
- exclusion and deviation policy;
- artifact and source hashes.

After results exist, do not silently change these choices. Label post-hoc work as
exploratory and retain unfavorable cases.

## 20. Tests and CI

### 20.1 Local test categories

| Test area | Representative files |
|---|---|
| Weight and exposure contracts | `test_weight_interface.py`, `test_fail_closed_invariants.py` |
| Cost and accounting | `test_transaction_costs.py`, `test_regression_guards.py`, `test_fail_closed_invariants.py` |
| Causality and PIT | `test_pit_enforcer.py`, `test_lookahead_detector.py`, `test_data_integrity.py` |
| Execution safety | `test_execution_safety.py`, `test_order_manager.py` |
| Broker SDK shape | `test_broker_sdk_conformance.py` |
| Research statistics | `test_research_validation.py`, `test_research_expansion.py` |
| Artifact provenance | `test_paper_artifacts.py`, `test_expansion_artifacts.py` |
| Packaging and policy | `test_import_surface.py`, `test_repository_data_policy.py`, `test_dependency_locks.py` |
| Figures | `test_plotting_style.py`, deterministic paper figure tests |

Tests should use deterministic synthetic data unless they explicitly validate a
reviewed local artifact. Test fixtures establish software behavior, not historical
performance.

### 20.2 CI jobs

These are workflow definitions. A passing result must be tied to the exact
candidate commit; a private validation receipt does not establish public CI status.

| Job | What it checks when passing |
|---|---|
| Lint and tests, Python 3.11-3.14 | Core imports without optional dependencies, Ruff, pytest, 60% coverage floor |
| Notebook smoke, Python 3.11 | All notebooks execute on deterministic fixtures and render expected figures |
| Broker SDK conformance, Python 3.12 | Installed Alpaca/IB SDK models match adapter construction and parsing |
| Package build/install, Python 3.11-3.14 | sdist/wheel build, clean install, public imports, no restricted market-data snapshot |
| Dependency lock drift | Poetry exports exactly match every committed requirement lock |
| Deterministic paper build | Pinned Tectonic rebuild is byte-identical to tracked artifacts |
| Container smoke | Compose parses, image builds, read-only default and offline cycle work |

What CI does not prove:

- live provider correctness or rights;
- authenticated broker behavior;
- production database/Redis recovery;
- venue-side duplicate prevention;
- fill quality;
- PPO training or broad LightGBM, XGBoost, CatBoost, Torch, and FinBERT
  training/inference compatibility;
- economic validity or future returns.

## 21. Production-readiness boundaries

Before real-money use, independently complete at least these controls:

### Broker certification

- authenticated paper tests for permissions, reconnects, rejects, partial fills,
  cancellations, and duplicate submissions;
- venue-side idempotency tests;
- position, cash, order, and fill reconciliation before every cycle.

### State and recovery

- Redis integration tests against a real service;
- backup restoration and corruption drills;
- concurrent-writer and disaster-recovery exercises;
- immutable audit records.

### Data and market model

- licensed point-in-time data with delisted-security coverage;
- authoritative venue completion signals instead of the conservative 17:00 New
  York daily-bar cutoff;
- verified corporate actions, calendars, stale-price policy, and symbol mapping;
- calibrated spread, volatility, size, impact, and capacity models.

### Operations

- managed secrets and least privilege;
- structured logs, metrics, and alerts;
- kill switches and independent exposure limits;
- incident ownership and rollback procedures;
- staging environment matching production;
- independent release approval and regulatory review.

These controls remain required after CI passes.

## 22. Common mistakes

1. **Forcing post-overlay weights to sum to one.** This removes intended cash and
   violates the relaxed exposure contract.
2. **Applying a close-derived target to the return ending at the same close.** This
   is look-ahead, not same-close execution.
3. **Omitting a cash series while claiming SHV cash.** Nonzero cash returns require
   complete aligned observations.
4. **Calling SHV risk free.** It is a cash proxy and reporting benchmark.
5. **Using one-way turnover as the paper cost base.** Paper costs use
   `0.0013 * (buys + sells)`.
6. **Treating vectorized and event-driven engines as equivalent.** Daily re-pegging
   and fixed pseudo-share quantities, whose weights can drift, are different constructs.
7. **Calling a comparison portfolio independent or causal.** Its target exposure
   path comes from the strategy.
8. **Calling the exact decomposition tradable or causal.** It is an ex-post
   arithmetic identity.
9. **Interpreting a hash of allocation records as reproducible data.** It cannot
   reconstruct prices, fills, or returns.
10. **Using current constituents historically without disclosure.** This introduces
    survivorship bias.
11. **Treating a purge/embargo utility as global leakage protection.** It protects
    only the split that uses it.
12. **Importing optional SDKs or PyTorch at module scope.** Core imports must remain
    lightweight.
13. **Assuming relaxed exposure validation preserves portfolio mode.** Default
    bounds permit shorts; custom overlays must preserve long-only or neutrality
    explicitly.
14. **Retrying an uncertain broker submission.** Reconcile first. Alpaca paper
    cycle blocks automatic retry, but direct adapter calls do not.
15. **Hand-editing generated evidence.** Fix the generator and rerun the release.
16. **Changing frozen protocol fields to improve labels or outcomes.** Never edit
    `protocol.json`; append a dated clarification to `preregistration.md` without
    rewriting the historical record.
17. **Ignoring exact metric ties.** State the tie and ranking convention; do not
    publish a convention-dependent count as unique.
18. **Claiming provider permission from repository ownership.** Rights require a
    separate basis.
19. **Publishing a direct build as a coordinated empirical release.** Direct builds
    reuse existing generated aggregates and do not regenerate them.
20. **Assuming broker mocks establish live readiness.** They test mapping, not
    transport, account permissions, or venue behavior.
21. **Using a favorable backtest threshold as proof.** Report outcomes without
    tuning to pass an arbitrary target.

## 23. Troubleshooting

### Import errors for optional packages

Confirm the relevant extra or lock is installed. Do not solve the problem by
moving the optional import to module scope.

```bash
.venv/bin/python -m pip install -r requirements/brokers.lock
```

### Scripts cannot import `quantcortex`

Install the package in editable mode or add repository root to `PYTHONPATH`:

```bash
.venv/bin/python -m pip install --no-deps -e .
# or
PYTHONPATH=. .venv/bin/python scripts/<script>.py
```

### Matplotlib cache is not writable

```bash
export MPLCONFIGDIR=/tmp/quantcortex-mpl
```

### Broker tests are skipped

Core tests intentionally omit broker SDKs. Install `requirements/brokers.lock`
and run:

```bash
.venv/bin/python -m pytest tests/test_broker_sdk_conformance.py -q
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

### Torch or optional ML tests are skipped

Install the relevant extra. A skip in the core environment is expected if the
optional dependency is absent.

### Paper build fails immediately

Check:

- Tectonic is exactly 0.16.9;
- Poppler tools, `strings`, and `flock` are installed;
- pinned bundle digest is available;
- bound source files have not changed during the build;
- no build runs concurrently outside the shared wrapper lock.

### Paper build reports anonymity or metadata text

Do not weaken the scanner. Locate the author, institution, URL, repository,
system name, or exact fingerprint in source, metadata, figure payload, or embedded
PDF content. Rebuild after fixing the source.

### Artifact tests report source-tree or script hash drift

If experiment source changed, generated artifacts are stale. Do not update the
hash manually. Follow Section 17.3: obtain explicit user approval for the source
commit, recover the exact authorized inputs, and run expansion then paper release
with fixed UTC generation timestamps.

### Release wrapper rejects dirty source

After explicit user approval, commit reviewed release-critical source first.
Do not bypass the guard or use an untracked substitute script.

### Input digest mismatch

Stop. The file is not the reviewed matrix. Do not fetch a replacement and attach
old provenance. Recover the exact authorized input or define and review a new
release.

### Live paper cycle refuses a bar

The cycle requires a complete panel ending at the latest finalized NYSE daily
session and uses a conservative 17:00 America/New_York cutoff. It does not
forward-fill stale symbols.

### State says broker outcome is uncertain

Do not retry based on transport result alone. Paper cycle queries Alpaca by
deterministic client order ID. If lookup still finds no corresponding order, use
`--retry-confirmed-missing` only after Alpaca dashboard or support confirms prior
intent was not routed.

### macOS boosting-library error

LightGBM and XGBoost may require:

```bash
brew install libomp
```

## 24. Glossary

**Adjusted-close pseudo-share**  
A total-return accounting unit valued at adjusted closing prices in the paper
workflow. The generic event-driven engine does not verify whether input prices
are adjusted. This unit is not a nominal broker share.

**Allocation weight**  
Strict allocator output before timing and risk overlays. Long-only output sums to
one; market-neutral output sums to zero and can be flat.

**Baseline-conditional diagnostic**  
A comparison that keeps allocation records, input matrix, and primary engine fixed while
changing one declared convention.

**Canonical allocation records**

Versioned long-form decision records with `decision_timestamp`, `symbol`, and
`target_weight`, plus symbol and gross-exposure declarations.

**Cash return, `r_t^c`**  
Return credited to residual portfolio cash.

**Comparison portfolio following target exposure**

A portfolio whose risky exposure follows a strategy target path, with next-bar
accounting, its own allocation, and its own modeled costs.

**Decision timestamp**  
Time by which all signal inputs must be observable. It is not the execution bar.

**Evaluation contract**  
Machine-readable declaration of target, timing, cash, cost, comparator, engine,
order-state, and evidence semantics.

**Event-driven engine**  
Reference engine that holds pseudo-share quantities fixed between trades while
allowing weights to drift. It applies explicit cash, fill, trade, and post-cost
sizing semantics.

**Executable check**  
A named runtime invariant or failure response enforced by code. A descriptive
JSON string is not executable by itself.

**Gross exposure**  
Sum of absolute risky weights.

**Gross two-sided traded notional**  
Total bought plus total sold. This is the paper's proportional cost base.

**One-way turnover**  
`max(total buys, total sells)`. Useful for reporting, but distinct from two-sided
cost notional.

**Point in time (PIT)**  
Data and membership as known by the decision timestamp, including publication or
announcement delay.

**Basket at actual exposure, before costs**

Ex-post gross reference used in the exact arithmetic decomposition.

**Reporting benchmark, `r_t^f`**  
Return subtracted from strategy return when computing excess Sharpe.

**Repository-frozen**  
Recorded in a repository before recorded data request. This does not imply
external preregistration, a public timestamp, or temporal out-of-sample
evaluation.

**Residual cash**  
Long-only capital not assigned to risky weights.

**Target weight**  
Desired risky allocation after allocation, timing, and risk overlays. It is not a
fill, actual holding, or realized return.

**Vectorized engine**  
Engine that implicitly re-pegs weights every bar between explicit
rebalances.

## 25. Sources of truth

When documents disagree, verify current code and use this hierarchy:

| Topic | Primary source |
|---|---|
| Agent and repository implementation rules | `CLAUDE.md` |
| Package versions and dependency groups | `pyproject.toml` and `poetry.lock` |
| Exported environments | `requirements/*.lock` |
| Weight and exposure behavior | `quantcortex/portfolio/base.py` |
| Strategy pipeline | `quantcortex/strategies/base_strategy.py` |
| Allocation record runtime behavior | `quantcortex/backtest/conformance.py` |
| Serialized allocation record shape | `schemas/canonical_target_tape.schema.json` |
| Evaluation-contract shape | `schemas/evaluation_contract.schema.json` |
| Engine behavior | `quantcortex/backtest/engines/` |
| Cost behavior | `quantcortex/backtest/costs/transaction_costs.py` |
| Cash behavior | `quantcortex/backtest/engines/cash.py` |
| Order translation | `quantcortex/execution/position_manager.py` |
| Execution risk gate | `quantcortex/execution/pre_trade_risk.py` |
| Execution lifecycle | `quantcortex/execution/order_manager.py` |
| Snapshot persistence | `quantcortex/execution/state_persistence.py` |
| Durable Alpaca paper orchestration | `scripts/paper_trade_cycle.py` |
| Current tests and CI | `tests/` and `.github/workflows/ci.yml` |
| Expansion design | `paper/expansion/protocol.json` |
| Historical protocol narrative | `paper/preregistration.md` |
| Retrospective generated evidence | `paper/results/manifest.json` |
| Expansion generated evidence | `paper/expansion/results/manifest.json` |
| Performance terminology | `PERFORMANCE.md` |
| Paper build and release | `paper/README.md` and release scripts |
| TAE camera-ready requirements and release limitations | `paper/README.md` and Section 17.5 |
| Data acceptance | `local_data/README.md` and `docs/data-source-due-diligence.md` |
| Production gaps | `docs/production-readiness.md` |
| Security reporting | `SECURITY.md` |

Documentation can become stale. Before changing behavior, inspect the current
implementation and tests at the cited path.

## 26. First-contribution checklist

### Understanding

- [ ] Read `CLAUDE.md`.
- [ ] Understand strict allocation versus relaxed exposure.
- [ ] Understand close decision versus first-later-bar execution.
- [ ] Understand residual cash and reporting benchmark distinction.
- [ ] Understand vectorized versus event-driven semantics.
- [ ] Understand allocation records, evaluation contracts, and evidence classes.
- [ ] Understand generated-artifact and raw-data restrictions.

### Change design

- [ ] Identify the owning package layer.
- [ ] State input, output, timestamp, and failure contracts.
- [ ] Identify optional dependency and credential boundaries.
- [ ] Identify whether the change touches money, causality, execution state, data,
      or published evidence.
- [ ] Keep scope minimal and avoid unrelated refactoring.

### Verification

- [ ] Add a focused regression test.
- [ ] Test invalid and partial-failure paths when relevant.
- [ ] Run the focused test.
- [ ] Run `.venv/bin/ruff check .`.
- [ ] Run `.venv/bin/python -m pytest tests/ -q` or document the known temporary
      artifact failures separately.
- [ ] Run broker SDK, notebook, package, container, or paper checks when touched.
- [ ] Inspect generated diffs and every PDF page when artifacts change.

### Safety and provenance

- [ ] Do not commit secrets, account data, local state, raw provider matrices, or
      executed notebook outputs.
- [ ] Do not bypass pre-trade risk, source cleanliness, input hashes, or PDF scans.
- [ ] Do not hand-edit generated evidence.
- [ ] Record data source, rights basis, settings, timestamps, and hashes.
- [ ] Report limitations and unfavorable results without tuning them away.
- [ ] Keep live, paper, simulated, and untested behavior clearly separated.
