# Performance Evaluation

This repository publishes derived results from an owner-authorized reference
audit and an expansion across two ETF sets under a recorded protocol. Raw market data
and executed notebook outputs remain uncommitted. Other generated reports
remain ignored unless publication has explicit owner approval, complete
provenance, and artifact hashes.

## Published Reference Run

The README charts use adjusted closes for QQQ, VGT, GLD, TLT, SPY, VIG, and SHV
retrieved through yfinance 1.5.1 on August 29, 2026. For the current chart-generation
timestamp, see `generated_at` in `docs/img/performance_manifest.json`.
The source spans January 4, 2016 to December 31, 2025; evaluation spans
January 2, 2018 to December 31, 2025 after 503 warm-up sessions. SHV is the
residual-cash return proxy. The
owner authorizes publication of the derived charts, but the project does not
independently establish that the provider's terms permit publication. When the
mature selected-group residual-momentum signal has no positive member, the
strategy holds cash.

| Metric | Value |
|---|---:|
| Net nominal CAGR | +2.05% |
| Gross nominal CAGR before modeled costs | +3.86% |
| Annualized volatility | 6.00% |
| Net conventional sample Sharpe relative to SHV | -0.04 |
| Gross conventional sample Sharpe relative to SHV before modeled costs | +0.25 |
| Maximum drawdown | -9.25% |
| Annualized one-way turnover | 10.38x |
| Annualized gross traded notional | 13.51x |
| Arithmetic sum of modeled transaction-cost return drag | 14.02% |
| Mean risky exposure | 30.97% |
| Fraction of sessions entirely in cash | 46.89% |
| Cash proxy CAGR | +2.50% |
| Basket at actual exposure: Sharpe relative to SHV before costs | +0.85 |
| Comparison portfolio following target exposure: Sharpe relative to SHV, after costs | +0.67 |

Strategy returns are net of the platform default: 3 bps commission plus 10 bps
flat slippage per executed buy and sell dollar. The paper experiment encodes the
same 13 bps as one all-in symmetric charge; the two representations are
numerically identical under this linear model. General-report market benchmarks
are gross, and the ADV cap is inactive.

One-way turnover is useful for portfolio-change reporting. The 13 bps charge is
applied to gross two-sided traded notional measured against pre-trade NAV. The
14.02% aggregate is instead the arithmetic sum of daily return drag against
prior-close NAV, so the two quantities differ slightly on rebalance days with
nonzero returns.

The strategy has positive nominal growth but negative return in excess of SHV
after modeled costs. It trails both the basket at actual daily exposure before
costs and the comparison portfolio following the strategy's target exposure
with next-bar accounting and its own costs.

The exploratory DSR of 0.045 assumes 10 trials and uses a simplifying
single-series variance estimate. The historical trial record and dependent
cross-trial Sharpe dispersion are unavailable, so this is not a validated
multiple-testing correction. Exact source and artifact hashes are in
`docs/img/performance_manifest.json`.

### Exact return attribution

The paper experiment decomposes each daily net return in excess of SHV into an
active risky allocation, dynamic exposure timing around the full-sample mean,
passive risky exposure, and modeled cost. The identity holds on every date;
annualized arithmetic means therefore add exactly before rounding. The active
allocation component combines group selection, within-risky weighting, and the
event engine's between-rebalance weight drift as prices and NAV change while
pseudo-share quantities stay fixed between trades. It is not a pure
security-selection estimate, and no component is causally identified.

| Component | Annualized mean | 21-session bootstrap interval |
|---|---:|---:|
| Active risky allocation | -3.05% | [-5.52%, -0.57%] |
| Dynamic exposure timing | +0.54% | [-2.35%, +3.45%] |
| Passive risky exposure | +4.00% | [+0.85%, +7.04%] |
| Modeled implementation cost | -1.76% | [-2.12%, -1.41%] |
| Net excess over SHV | -0.26% | [-4.29%, +3.82%] |

The constant-exposure series is an ex-post diagnostic, not a tradable
benchmark, and the intervals are unstudentized percentile intervals
conditional on this historical path. `paper/results/return_decomposition.csv`
contains 5-, 21-, and 63-session results.

### Comparison portfolio and Sharpe uncertainty

The basket at actual exposure supplies the reference for exact arithmetic
attribution. It is before its own trading costs and is not tradable. The separate
comparison portfolio following target exposure invests the strategy's declared
gross exposure on the next bar in a fixed
equal-initial-weight buy-and-hold basket, assigns the remainder to SHV, and pays
the same 13 bps proportional cost. Its net CAGR is
6.10% and its conventional sample Sharpe relative to SHV is 0.67.

The strategy's annualized arithmetic return minus that comparison portfolio,
both after costs, is -3.86%, with a primary 21-session circular-block interval of
[-6.40%, -1.35%]. Its conventional active Sharpe is -0.86 with a directly
resampled interval of [-1.40, -0.31]. The interval for the strategy's own
conventional Sharpe relative to SHV is [-0.69, +0.67]. These intervals describe this
historical path under the declared block-bootstrap assumptions; they are not
iid normal confidence intervals or evidence of future performance. See
`paper/results/comparator_diagnostics.csv` and
`paper/results/sharpe_uncertainty.csv`.

### Protocol diagnostics

These diagnostics hold target weights, the input matrix, and the primary event
engine fixed while changing one convention.
The audited Sharpe relative to SHV is -0.04. Setting modeled costs to zero raises it
to +0.25. Assigning a target formed at the decision close to the return interval
that ends at that close raises it to +0.12. The close is observable when the
decision is formed, but that assignment could not have been executed. Setting
residual-cash return to zero lowers Sharpe to -0.31 while retaining SHV as the
reporting benchmark. See
`paper/results/protocol_switches.csv`.

## Repository-Frozen Expansion

The recorded timestamp of expansion-protocol commit
`4018f4063f46889f41d6981db5a71079e1dbd713` precedes the recorded timestamps of
the two provider requests. These records are author-controlled and do not prove
wall-clock ordering or when the commit became public. The protocol was not
externally registered, and the 2018-2025 evaluation interval had already
occurred. The design records declared choices but is not a temporal
out-of-sample test.

The evaluation covers time-series momentum, cross-sectional momentum,
short-term reversal, and a walk-forward gradient boosting model across U.S.
sector and country-equity ETF panels. Each family uses monthly mature signals,
next-bar event accounting, SHV residual cash, and the same 13 bps proportional
cost. Gradient boosting reports all five frozen seeds; its Sharpe is the mean
of the individual seed Sharpes.

| Panel | Strategy family | Arithmetic return | Sharpe relative to SHV |
|---|---|---:|---:|
| U.S. sectors | Time-series momentum | 11.44% | 0.49 |
| U.S. sectors | Cross-sectional momentum | 14.19% | 0.60 |
| U.S. sectors | Short-term reversal | 11.02% | 0.44 |
| U.S. sectors | Gradient boosting | 13.66% | 0.53 |
| Country equities | Time-series momentum | 2.18% | -0.02 |
| Country equities | Cross-sectional momentum | 8.84% | 0.33 |
| Country equities | Short-term reversal | 6.03% | 0.20 |
| Country equities | Gradient boosting | 7.87% | 0.31 |

These baselines are inputs to the audit, not discoveries. Under the primary
21-session circular-block analysis, removing modeled cost raises annualized
return by 0.57-1.81 percentage points and the interval lies entirely above zero in all
eight family-panel cells. The closed-interval assignment has an interval entirely
below zero in three cells and an interval that includes zero in five. Zero
residual-cash return has an interval entirely below zero in five cells and an
interval that includes zero in three. These counts are not counts of negative
point estimates. The maximum one-day
vectorized-minus-event-driven difference is 93.21 bps and the largest absolute
terminal-wealth gap is 3.89%. The engine contrast bundles daily re-pegging,
weight drift while pseudo-share quantities stay fixed, trade accounting, and
post-cost target sizing. It is not an
external-engine conformance claim.

All intervals are unstudentized, pointwise, and conditional on the selected
historical paths. They are not multiplicity-adjusted evidence of future alpha.
See `paper/expansion/results/`, the plots under `paper/expansion/figures/`, and
the frozen design in `paper/preregistration.md`.

## Generate a Report

Use a licensed or otherwise permitted wide adjusted-close CSV:

```bash
PYTHONPATH=. python scripts/generate_report.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --cash-proxy-symbol SHV \
  --start 2018 --end 2025 --n-trials 10 \
  --manifest-out reports/performance_manifest.json \
  --data-provider "$DATA_PROVIDER" \
  --permission-basis "$DATA_PERMISSION_BASIS" \
  --retrieved-at "$DATA_RETRIEVED_AT" \
  --adjustment-method "$DATA_ADJUSTMENT_METHOD"
```

The required columns are documented in
[local_data/README.md](local_data/README.md). The command
writes charts to ignored `reports/img/` and prints markdown tables containing
the local file path, SHA-256 digest, and observed date window. By default it
loads two years before `--start` to warm the signals, carries that strategy
state into the requested evaluation window, and excludes the pre-roll returns
from reported metrics. The source must contain at least 274 pre-evaluation
sessions. Use `--warmup-years 0` only when a deliberately cold-started report
is appropriate; that override is disclosed in the generated settings.
The provenance options record owner-supplied facts and assertions; they do not
constitute independent verification that publication or redistribution is
permitted. Missing fields are labeled incomplete in the report. The requested
manifest records the input, source tree, settings, and artifact hashes.

The fixed paper experiment is stricter than the general report: it accepts no
missing price rows, performs no forward fill, and requires at least 274 signal
warm-up sessions before the evaluation window.

`scripts/release_paper_artifacts.sh` is intentionally bound to the reviewed
input digest. When release-critical paths are unchanged, it regenerates from the
recorded source commit and timestamp. Changed experiment, report, or paper source
requires an explicit `QUANTCORTEX_GENERATED_AT` and regenerates from current
`HEAD`. The wrapper fails rather than labeling a different matrix with the
reviewed provider provenance. A new panel requires a new, explicitly reviewed
experiment configuration and source record.

For an explicitly requested live download:

```bash
PYTHONPATH=. python scripts/generate_report.py --live-yfinance
PYTHONPATH=. python scripts/validate_performance.py --live-yfinance --pit
```

Review Yahoo's terms and the
[yfinance legal disclaimer](https://ranaroussi.github.io/yfinance/) before use.
Live historical data may be revised, so preserve your own permitted input if
exact reproduction matters.

## Generated Artifacts

The command writes `reports/report.md` plus these plots under `reports/img/`:

- `report_overview.png`: equity, drawdown, allocation, turnover, and costs in a
  compact review image.
- `equity_vs_benchmarks.png`: strategy net of modeled costs versus gross SPY
  and an equal-initial-weight buy-and-hold basket.
- `performance_attribution.png`: strategy before and after modeled costs,
  the basket at actual exposure before costs, and the cash proxy on one capital clock.
- `drawdown.png`: the strategy underwater curve.
- `rolling_sharpe.png`: trailing 126-session Sharpe relative to SHV.
- `rolling_risk.png`: trailing 126-session annualized volatility and beta to
  SPY.
- `allocation_and_exposure.png`: realized asset weights, invested gross
  exposure, and cash under the share-based event engine.
- `turnover_and_costs.png`: executed one-way turnover, gross traded notional,
  and cumulative modeled transaction-cost return drag.
- `monthly_returns.png`: monthly net-return heatmap.
- `return_distribution.png`: log-count daily net-return histogram, historical
  tail markers, and a normal Q-Q diagnostic. Its horizontal axis uses theoretical
  normal quantiles and its vertical axis uses standardized observed returns;
  the axis correction leaves returns and performance metrics unchanged.

The Markdown report links every plot and includes performance metrics,
evaluation settings, data provenance, and the monthly-return table. Outputs
remain ignored by default. Publish them only with explicit owner approval,
complete adjacent provenance, an input digest, and hashes for every artifact.

The current command must not fabricate diagnostics it cannot support. Add
walk-forward or live-start boundaries only when the run records those regimes;
capacity and slippage curves only with spread, volume, and order-size inputs;
factor attribution only with validated factor returns/exposures; and fill
quality only from authenticated order and execution records.

The expansion release is separate from the general report. From a clean source
commit, `scripts/release_expansion_artifacts.sh local_data/expansion` regenerates
eight CSV tables, allocation record and data-provenance JSON, generated LaTeX,
and seven figures in a detached worktree. The two descriptive analyses added
during revision use the recorded paths. `yearly_performance.csv` has 128
run-year rows, and `seed_contract_effects.csv` has 50 paired Sharpe differences.
Appendix Table 7 reports interval counts for block lengths of 5, 21, and 63
sessions. These displays do not change the frozen primary protocol. The manifest
binds that protocol,
both input hashes, source tree, environment, and every output hash. The paper
release requires expansion artifacts from the same clean source commit.

## Reporting Requirements

- Report the data provider, license or permission basis, retrieval date, date
  window, symbols, adjustment method, and input-file digest.
- Set `--n-trials` to the actual number of configurations evaluated. The
  default `10` is a convenience, not a factual statement about a research run.
  Dependent trials require both defensible effective multiplicity and measured
  cross-trial dispersion; a raw configuration count alone does not make DSR
  valid. Pass the measured cross-trial variance of per-observation Sharpes with
  `--sr-variance`. Without it, the scripts disclose that they use a simplifying
  single-series variance estimate, so the calculation is exploratory.
- Compare against relevant buy-and-hold, equal-weight, cash, and
  exposure-matched references. Distinguish ex-post attribution identities from
  comparison portfolios that follow strategy target paths and pay their own costs.
- Label whether benchmarks are gross or cost-adjusted and state the risk-free
  or cash-return series used for Sharpe. The basket at actual exposure is before
  costs; the comparison portfolio following target exposure pays its own costs.
- Report costs, turnover, and maximum drawdown. Report the Deflated Sharpe
  Ratio only with defensible effective multiplicity, dependent cross-trial
  Sharpe dispersion, and the selected return series' sample length, skewness,
  and kurtosis. Otherwise omit it or label the calculation exploratory with its
  simplifying assumptions.
- Keep failed variants in the trial count; do not tune until a target is met.

## Known Limitations

The default transaction-cost model uses flat commission and slippage rates.
`quantcortex/backtest/execution_models/market_impact.py` provides a simplified
fill-price model driven by volatility and order size relative to average daily
volume (ADV). It does not model bid-ask spread or change future market prices,
and it is not wired into the reference report. The report supplies no ADV series,
so its configured volume cap is also inactive. The reference report uses the
event-driven engine, which holds pseudo-shares and sizes targets against
post-cost NAV. These are total-return accounting units valued at adjusted
closing prices, not nominal broker shares. The
vectorized engine remains available for approximations and sweeps but holds
target weights constant between explicit rebalances without charging for the
implied re-pegging trades.
Residual cash must be supplied explicitly when it earns a nonzero return; the
engines reject missing cash-proxy bars rather than filling them silently.
Single-name tests remain survivorship-biased unless the price feed includes
delisted securities and the universe is point-in-time.
`SP500Universe.from_wikipedia()` supplies approximate, coverage-limited
historical membership, not delisted price history. yfinance fundamentals use a
documented 45-day filing-date proxy rather than exact announcement timestamps;
date-only records become available strictly after that timestamp by default.

The released rotation strategy regresses QQQ against itself. Its theoretical
residual is zero, but positive floating-point error can pass the strict
positive-score filter and become a material allocation after normalization.
The published audit retains that implementation defect. Fixing the signal
would change the target path and requires a separate reviewed experiment.

Close-derived target weights execute at the next available bar's close in both
backtest engines. Weekly schedules use the first observed session of each week,
so a Monday exchange holiday moves the decision to Tuesday rather than skipping
the week. Monthly validation decisions use the last observed session of each
month. Report any separate signal warm-up period; using the evaluation window
itself for model warm-up can materially bias comparisons.

No metric threshold in this repository is evidence that a strategy is
profitable or deployable. Interpret every result against its data vintage,
comparator, costs, exposure, uncertainty, and recorded research trial history.
