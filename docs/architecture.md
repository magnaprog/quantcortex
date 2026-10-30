# Architecture

QuantCortex separates strategy decisions from their execution and accounting.
All importable code lives under `quantcortex`; scripts, notebooks, tests, and
documentation sit at the repository root. For setup, use the
[contributor guide](onboarding.md).

## Pipeline boundaries

```text
data -> point-in-time features -> selection -> allocation -> timing -> risk
                                                           -> target weights
                                                              -> backtest
                                                              -> paper trading
```

| Package | Responsibility and entry points |
|---|---|
| [data](../quantcortex/data/) | `DataProvider`, local CSV validation, PIT processors, universes, optional storage |
| [alpha](../quantcortex/alpha/) | Classical factors, Alpha158-style features, ML/NLP factors, validation |
| [portfolio](../quantcortex/portfolio/) | Weight contracts, equal weight, mean/minimum variance, HRP, risk parity, Black-Litterman, RL |
| [timing](../quantcortex/timing/) | HMM/GMM regimes, time-series momentum, KAMA, VIX scaling |
| [risk](../quantcortex/risk/) | Circuit breakers, VaR/CVaR estimates, factor limits, volatility targeting, Kelly sizing |
| [strategies](../quantcortex/strategies/) | Selection, allocation, overlays, rebalance, target panels |
| [backtest](../quantcortex/backtest/) | Canonical allocation records, engines, fills, costs, metrics, validation |
| [execution](../quantcortex/execution/) | Pre-trade risk, orders, positions, brokers, state persistence |
| [research](../quantcortex/research/) | Strategy rules, causal model fitting, and comparative evaluation |

Optional provider, broker, storage, and heavyweight ML packages are imported
inside the methods that use them. Missing dependencies need a clear error or
documented fallback. The scientific core must import without them or credentials.

## Weight contracts

Both helpers in [portfolio/base.py](../quantcortex/portfolio/base.py) return
contiguous `float64` arrays and reject empty, nonfinite, boolean, or
non-one-dimensional weights. Bounds and sums use the configured tolerance;
violations raise `WeightContractViolationError`.

| Boundary | Helper | Required behavior |
|---|---|---|
| Allocation | `enforce_weight_contract` | Long-only weights sum to `1.0` and use nonnegative bounds, normally `[0, 1]`. Market-neutral weights sum to `0.0` within configured per-asset bounds, defaulting to `[-1, 1]`. |
| Timing/risk output | `enforce_exposure_contract` | Weights stay within supplied bounds and `sum(abs(weights)) <= max_gross`; there is no sum constraint. Default bounds are `[-1, 1]`. |

`PortfolioOptimizer.optimize()` enforces allocation validity. Subclasses
implement `_compute_weights()` and return an already feasible vector.
Market-neutral allocation has no minimum gross or separate long/short budget;
a flat vector passes unless the caller adds a requirement.

Use `lower=0` to enforce long-only exposure. A long-only book summing to `0.5`
holds the remainder in cash; an all-zero book is fully cash. Running that output
through the strict allocation contract rejects the intended exposure reduction.
Renormalizing it removes the cash allocation.

`Strategy.rebalance()` checks the aggregate overlay shape and final relaxed exposure,
using `lower=0.0` in long-only mode and `lower=-1.0` otherwise. It does not validate
each custom overlay separately or enforce a market-neutral sum after overlays. Test
long-only and market-neutral behavior explicitly. Version-1 canonical allocation records
support long-only targets; a market-neutral strategy is not automatically compatible
with that serialization boundary.

## Strategy context and targets

[StrategyContext](../quantcortex/strategies/base_strategy.py) holds `as_of`,
prices, returns, an optional universe, and an `extra` dictionary. It rejects
future top-level price/return observations, duplicate timestamps or symbols,
infinities, and invalid observed prices. Validation of `extra` only checks that
it is a dictionary: callers must validate nested symbols, timestamps,
completeness, freshness, and point-in-time availability.

`Strategy` implements the following sequence:

1. `select(ctx) -> pd.Series` returns finite scores indexed by selected symbols.
2. `allocate(scores, ctx)` delegates to the injected optimizer by default. Override it
   when scores determine the weights.
3. `apply_overlays(weights, ctx)` runs timing overlays, then risk overlays.
4. `rebalance(ctx)` validates selection, allocation, aggregate shape, and final
   exposure and returns a `RebalanceResult`.
5. `generate_weights(prices, rebalance_dates)` produces the date-by-symbol target
   panel for the engines.

`generate_weights()` skips dates without prior return history. Once selection
can run, an empty selection produces an explicit all-cash target, preventing
carry-forward of a stale invested book. Its `extra["current_weights"]` is the
previous generated target, not executed or marked-to-market holdings. There is
no general per-decision `extra` input. For actual positions, fill state, or dated
auxiliary features, construct `StrategyContext` and call `rebalance()` or use a
custom driver.

## Timing and accounting

A target computed from close data at `t` executes on the first strictly later
available price bar. It starts earning risky returns after execution, not in the
interval ending at the signal close. A target without a later bar does not
execute. Both engines enforce this rule.

Every engine requires a
[TransactionCostModel](../quantcortex/backtest/costs/transaction_costs.py).
Passing `None` is an error. Risky returns, cash returns, trades, and costs share
the price-bar timeline.

For the long-only workflows, residual cash is `1 - sum(risky weights)`.
Only `cash_returns=None` selects implicit zero-return cash. A supplied series,
even all zeros, must contain a finite return for every price bar; missing values
fail the run. See [cash.py](../quantcortex/backtest/engines/cash.py).
Cash credited to the portfolio (`r_t^c`) and the benchmark subtracted for excess
Sharpe (`r_t^f`) are separate. Changing the cash income assumption does not by
itself change the benchmark used to calculate excess returns.

| Engine | Accounting and limits |
|---|---|
| [EventDrivenBacktest](../quantcortex/backtest/engines/event_driven.py) | Holds units fixed between trades, lets weights drift with prices and NAV, sizes targets against post-cost NAV, and applies explicit fill semantics. This is the reference accounting path. |
| [VectorizedBacktest](../quantcortex/backtest/engines/vectorized.py) | Uses executed weights at explicit trades but implicitly re-pegs them each intervening bar at zero cost. Use for approximations and sweeps. It does not model holdings drift. |

The generic event engine does not verify adjusted-close inputs. When adjusted
closing prices are used, pseudo-shares are total-return accounting units, not
nominal broker shares or evidence of executable fill prices.
Agreement between engines in one experiment does not make them interchangeable.
Differences combine re-pegging, drift, trade accounting, and post-cost sizing.

The cost model supports commission, flat slippage, sell-side tax, and optional
average daily volume (ADV) caps. It executes feasible reductions before
exposure-increasing legs and scales or blocks increases when the gross cap
binds. If an already over-cap book cannot liquidate enough under ADV limits,
it executes feasible reductions but may remain over cap. Report executed
holdings, not merely desired targets.

Fill-price slippage plus positive `TransactionCostModel.slippage` can double-count
slippage; the event engine warns about that combination. `IdealFill` is an optimistic
benchmark. `VWAPFill` rejects nonpositive or nonfinite adjusted fill prices.
`AlmgrenChriss` supplies a simplified impact fill-price model and a separate helper for
generating a liquidation schedule; it does not change future market prices or
automatically apply that schedule. A flat cost rate does not establish trading
capacity. See the [execution models](../quantcortex/backtest/execution_models/)
and [cost-base definition](evaluation-contracts.md#cash-costs-and-comparators).

`BacktestResult` contains returns, equity, engine-effective post-trade weights,
cash weights, turnover, costs, and metadata. Persist canonical allocation records
separately for target-versus-executed analysis. See
[Evaluation contracts](evaluation-contracts.md) for the wire format, engine
example, comparator roles, and evidence limits.

## Point-in-time data

[DataProvider](../quantcortex/data/providers/base.py) defines `fetch_ohlcv`,
`fetch_fundamentals`, and `fetch_macro`. Unsupported capabilities raise
`NotImplementedError`. Canonical provider OHLCV indexes normalize to UTC and
then become timezone-naive. Preserve schemas and make network access explicit;
do not substitute synthetic prices after a real-data fetch fails.

Use [local_csv.py](../quantcortex/data/local_csv.py) for validated local inputs
and [local_data/README.md](../local_data/README.md) for CSV examples. Dates must
be unique and parseable, observed prices finite and positive, required symbols
present, and cash-proxy data explicit. OHLCV rows require nonnegative volume and
consistent highs/lows. `load_price_matrix()` forward-fills at most five missing
rows per symbol by default, then drops incomplete rows. `require_complete=True`
checks after configured filling. Set `max_ffill=0` to disable filling.

Fundamental availability follows `announcement_date`, not period end.
Date-only records use strict-before matching. Same-timestamp use requires
verified intraday timestamps and explicit opt-in; the opt-in does not prove
availability. Quality-ratio components must cover the same fiscal quarters and
remain missing until every required component arrives.

[PITMembership](../quantcortex/data/universe/base.py) uses half-open intervals
`[start_date, end_date)`. Historical membership and delisted-security prices
are separate requirements. The Wikipedia S&P 500 reconstruction is approximate,
rejects dates before its source coverage, and supplies no delisted prices. Named
index classes do not silently select their survivorship-biased demo subsets.

Provider-specific limits:

- YFinance fundamentals use `period_end + 45 calendar days` as a heuristic
  announcement date, not an observed release timestamp.
- Standard FRED retrieval supplies latest revised observations, not ALFRED
  vintages. Neither source is automatically point-in-time safe.
- CCXT candle pagination continues through capped pages and rejects a repeated,
  nonadvancing page.

Record provider/adapter version, rights basis, retrieval timestamp and settings,
symbol mapping, adjustments, timezone, cleaning, row/date coverage, and input
SHA-256. Keep observations local. See
[data due diligence](data-source-due-diligence.md); identity hashes do not prove
permission.

## Component-specific limits

Account for these limits when extending the modules; inventories and signatures remain
in the linked source packages.

| Component | Behavior to preserve or account for |
|---|---|
| `Alpha158` | Qlib-style 158-column layout without a Qlib dependency; full rolling windows; missing VWAP uses `(high + low + close) / 3`. Layout does not establish numerical parity. Close features become available after the close. |
| `MacroFeatures` | Approximate monthly release lags default to 35 calendar days for UNRATE, PMI, NAPM before business-day forward fill. Callers still align to actual decisions. |
| `GBDTFactor` | `model="auto"` tries LightGBM, XGBoost, CatBoost, then scikit-learn. Explicit backend, version, and parameters are needed for reproduction. |
| `NeuralFactor` | Defaults to scikit-learn; explicit `backend="auto"` prefers Torch when available. Environment changes can change the model. |
| `NewsScorer` | Defaults to deterministic lexicon scores. Daily aggregation discards intraday times: filter at the decision cutoff or delay the signal until the next close. |
| `FinBERTSentiment` | `backend="auto"` falls back only when optional packages/model files are unavailable; inference, malformed-output, and label-schema failures surface. |
| `MeanVariance` | Expected-return Series must name each asset exactly once and are aligned to return columns; arrays/lists follow column order. |
| `MinimumVariance` | Rejects market-neutral mode: a zero-net minimum-variance problem without another exposure constraint is trivially flat. |
| `DRLAllocator` | Untrained inference fails closed unless `untrained_policy="heuristic"` is explicit. Heuristic inference uses core dependencies; PPO training needs RL extras and is not exercised by core CI. |
| `HMMRegime` | `auto` chooses HMM if importable, falls back to GMM only on import/loading failure, and surfaces fit failures. Reference strategies pin GMM. |
| `CircuitBreaker` | Latches flat until `reset()` unless `auto_reset=True`. |
| `VaRCVaR` | Estimates positive-loss VaR/CVaR; it does not itself enforce a limit or apply an overlay. |
| `FactorExposureLimiter` | Requires caller-supplied PIT loadings; risk classes do not establish provenance or timing of estimates. |

Keep seeds, ordering, and tie-breaking fixed. `HMMRegime` uses threadpoolctl when
available to limit supported BLAS/OpenMP pools to one thread during fit and the
prediction used to label states. Later `predict_regime()` calls do not set that
limit; cross-environment identity is not guaranteed. Standalone timing methods
generally do not know the decision timestamp, so callers must restrict their
inputs. Record model settings, seeds, and tie rules with each evaluation; see
[performance reporting](../PERFORMANCE.md).

`AlphalensReport` and `FactorDecay` supply diagnostics; they cannot repair
point-in-time defects. `WalkForwardOptimizer` purges the last `label_horizon`
training samples and leaves `embargo_gap` bars before each test window. Protection
applies only when the splitter is used with the true forward-label horizon.
Other purge controls must likewise cover labels crossing the decision boundary.
For trial-history and Sharpe-estimation limits, see
[research evidence](evaluation-contracts.md#evidence-classes).

## Execution state

The guarded sequence is target validation, order translation, final portfolio
risk validation, durable submission intent, then the broker call.
[PositionManager](../quantcortex/execution/position_manager.py) needs finite
positive prices for every targeted or held symbol. Whole-share sizing rounds
toward zero; fractional books require `allow_fractional=True`.
[PreTradeRiskCheck](../quantcortex/execution/pre_trade_risk.py) validates targets,
the complete order batch, and reconstructed post-trade exposure against supplied
positions, prices, and capital. It does not query buying power or establish
price freshness. Translation must precede this final portfolio-level assertion.

[OrderManager](../quantcortex/execution/order_manager.py) permits:

```text
NEW -> SUBMITTED -> FILLED
NEW -> SUBMITTED -> PARTIALLY_FILLED -> FILLED
NEW or SUBMITTED -> CANCELLED or REJECTED
PARTIALLY_FILLED -> CANCELLED
```

Repeated partial fills are valid; filled, cancelled, and rejected orders are
terminal. Unknown/duplicate orders and illegal transitions fail. Incremental
fills must consistently include prices or leave them unknown; mixing the two
fails before mutation. A repeated cumulative fill with changed average price
requires reconciliation rather than silently changing the position basis.

[ExecutionSnapshot](../quantcortex/execution/state_persistence.py) combines
positions, orders, intents, and reconciliation metadata. Only
`save_execution_snapshot(..., expected_revision=...)` provides an atomic,
versioned update across these fields. Generic `save_state()`, `save_positions()`,
and `save_orders()` are individually atomic but have no optimistic revision
check or multi-call transaction. Typed loaders distinguish absent records from
malformed data and reject duplicate normalized positions or key/symbol mismatch.

Alpaca paper-cycle orchestration persists intent as `ATTEMPTING` before
submission and blocks uncertain retries until reconciliation. Intent state and
order lifecycle are distinct. Direct `Broker.submit_order()` validates and sends
an individual request; it does not invoke portfolio risk, require `OrderManager`,
persist intent, or prevent retries. Other integrations must supply those controls.
See [Production readiness](production-readiness.md) for adapter boundaries and
paper-cycle operation.

## Extension rules

| Change | Implementation and regression focus |
|---|---|
| Provider | Subclass `DataProvider`; preserve canonical schema/timezone, explicit network access, lazy SDK imports, and documented rights/adjustment/calendar/missing-data assumptions. Test malformed and ambiguous data with mocks and SDK-shape checks where applicable. |
| Factor | Define availability, warm-up, universe, and deterministic ties. Test shifted inputs, incomplete early windows, missing data, and future-data rejection. |
| Allocator | Subclass `PortfolioOptimizer`, implement `_compute_weights()`, return a feasible raw vector. Test budget, bounds, finiteness, ordering, degenerate covariance, and failure behavior. |
| Overlay | Implement `callable(weights, StrategyContext) -> weights`. Preserve shape, time cutoff, and portfolio mode; test cash, partial exposure, caps, and failures without forced renormalization. |
| Strategy | Subclass `Strategy`, implement `select()`, override allocation only for score-driven weights, bind overlays, and serialize targets for comparative research. Test empty/unknown selections, missing history, mode, shape, cash, and determinism. |
| Backtest or execution | Derive a small expected result by hand. Test timing, cash, costs, fills, NAV, risk, state, and partial failures as applicable. Compare engines only when semantics should agree. |

Run the relevant regression and full repository checks. Use the
[contribution guide](../CONTRIBUTING.md) for dependency and artifact changes.
Generated evidence must come from reviewed generators and inputs, never manual
edits to output files.
