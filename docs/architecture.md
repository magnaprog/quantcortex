# Architecture

quantcortex is a platform for research and guarded paper execution built around
portfolio weights.
All importable code lives under `quantcortex`; scripts, notebooks, tests, paper
artifacts, and operational documentation remain at the repository root.

## Pipeline Boundaries

```text
data -> alpha -> strategy selection -> portfolio allocation
                                      -> timing overlays
                                      -> risk overlays
                                      -> target weights
                                         |-> backtest
                                         `-> pre-trade risk -> broker adapter
```

| Package | Primary modules |
|---|---|
| `quantcortex.data` | Provider adapters, `local_csv.py`, PIT processors, universes, storage |
| `quantcortex.alpha` | Classical factors, Alpha158-style features, GBDT/NLP baselines, validation |
| `quantcortex.portfolio` | Weight contracts, equal weight, minimum variance, HRP, risk parity, RL |
| `quantcortex.timing` | GMM/HMM regimes, time-series momentum, KAMA, VIX scaling |
| `quantcortex.risk` | Circuit breakers, VaR/CVaR, factor exposure, volatility targeting, Kelly sizing |
| `quantcortex.backtest` | Vectorized and event-driven engines, fills, costs, metrics, validation |
| `quantcortex.execution` | Pre-trade risk, order/position management, brokers, state persistence |
| `quantcortex.strategies` | End-to-end select, allocate, overlay, and rebalance workflows |

## Weight Contracts

Portfolio allocators return finite one-dimensional `float64` arrays. Long-only
allocators must be bounded in `[0, 1]` and sum to one; market-neutral allocators
must sum to zero and satisfy their configured per-asset bounds. Violations raise
`WeightContractViolationError`.

Timing and risk overlays may intentionally reduce exposure. Their output uses
the relaxed exposure contract: finite bounded weights with gross exposure no
greater than the configured cap and no sum constraint. Its default `[-1, 1]`
bounds permit shorts; callers must pass `lower=0` or separately enforce a
long-only policy. In a long-only book, capital outside risky positions is cash.

Reuse the weight-validation helpers in `quantcortex.portfolio.base` when writing
a strategy.

## Timing and Accounting

A target computed from close data at time `t` executes on the first strictly
later available bar and starts earning returns after that execution. Both
backtest engines enforce this decision/holding distinction.

Every backtest requires a `TransactionCostModel`. Risky returns, residual-cash
returns, turnover, and costs use one capital clock. A nonzero cash return must
be supplied explicitly. If a cash-return series is supplied, missing observations
fail the run; omitting the series selects zero-return cash.

The event-driven engine is the reference accounting path: it holds pseudo-share
quantities fixed between trades while weights drift with prices and NAV, sizes
targets against post-cost NAV, and applies fill semantics. Those pseudo-shares
are total-return accounting units valued at adjusted closing prices, not
nominal broker shares. The vectorized engine re-pegs target weights between explicit
rebalances and is intended for approximations and sweeps. Agreement in one
experiment is a diagnostic, not proof that the engines are interchangeable.

`quantcortex.backtest.conformance` defines the canonical long-form allocation records
used to compare engines without coupling them to a strategy implementation.
Record validation checks complete timestamp-symbol decisions, finite long-only
weights, and gross exposure. See `docs/evaluation-contracts.md`.

The paper distinguishes a basket at actual exposure before costs from comparison
portfolios that follow target exposure and pay their own trading costs:

- The basket at actual exposure before costs is the reference for exact ex-post
  arithmetic attribution.
- The retrospective comparison portfolio invests the strategy's target risky
  exposure in a fixed equal-initial-weight buy-and-hold basket. It uses next-bar
  accounting, residual SHV, and its own modeled costs.
- The expansion comparison portfolio re-equalizes that target exposure across panel
  assets at each decision. It also uses next-bar accounting, residual SHV, and
  its own modeled costs.

Both comparison portfolios are conditional on the strategy target path. They are not
independent benchmarks or causal controls.

## Point-in-Time Data

Fundamental records become available according to announcement timestamps, not
period ends. Date-only inputs use strict-before matching unless a source with
verified intraday release times explicitly opts into same-timestamp use.

Historical index membership and historical prices are separate requirements.
The Wikipedia S&P 500 reconstruction provides approximate membership within its
coverage window; it does not provide delisted-security prices.

## Execution State

Paper execution persists positions, orders, submission intents, and metadata as
one versioned snapshot. Writes are atomic and use optimistic concurrency. The
paper workflow stores an intent as `ATTEMPTING` before broker submission and
blocks automatic retry after an uncertain outcome until broker reconciliation.
Intent state is distinct from the order lifecycle, which separately tracks
`NEW`, `SUBMITTED`, partial or complete fills, cancellation, and rejection.

Broker SDK imports remain lazy. Offline conformance and SDK model-construction
tests validate request and response mapping, not authenticated transport or
venue behavior.

## Extension Rules

1. Add importable modules inside the appropriate `quantcortex` package.
2. Preserve absolute imports and existing layer ownership.
3. Keep provider, broker, and heavyweight ML dependencies lazy.
4. Make timestamp, adjustment, universe, and cash assumptions explicit.
5. Add focused regression tests for accounting, causality, weight, or execution-state changes.
6. Treat generated reports and paper artifacts as outputs of reviewed scripts, never hand-edited evidence.
