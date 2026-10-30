# Performance evaluation

The report examples use an owner-authorized historical study. Raw market data
and executed notebook outputs remain outside Git. Other generated reports remain
ignored unless publication has explicit owner approval, source records, and
artifact hashes.

## Current strategy report

The example charts use adjusted closes for QQQ, VGT, GLD, TLT, SPY, VIG, and SHV
retrieved through yfinance 1.5.1 on August 29, 2026. For the current chart-generation
timestamp, see `generated_at` in `docs/img/performance_manifest.json`.
The source spans January 4, 2016 to December 31, 2025; evaluation spans
January 2, 2018 to December 31, 2025 after 503 warm-up sessions. SHV is the
residual-cash return proxy. The
owner authorizes publication of the derived charts, but the project does not
independently establish that the provider's terms permit publication. When no
asset in the selected groups has a positive mature residual-momentum score,
the strategy holds cash.

These charts evaluate the current library strategy after the correction to
residual momentum. The paper retains the original QQQ calculation for its
historical audit and evaluates the correction separately.

| Metric | Value |
|---|---:|
| Net nominal CAGR | +2.53% |
| Gross nominal CAGR before modeled costs | +4.01% |
| Annualized volatility | 5.48% |
| Net conventional sample Sharpe relative to SHV | +0.03 |
| Gross conventional sample Sharpe relative to SHV before modeled costs | +0.30 |
| Maximum drawdown | -9.61% |
| Annualized one-way turnover | 8.83x |
| Annualized gross traded notional | 11.04x |
| Arithmetic sum of modeled transaction-cost return drag | 11.46% |
| Mean risky exposure | 27.91% |
| Fraction of sessions entirely in cash | 50.77% |
| Cash proxy CAGR | +2.50% |
| Basket at actual exposure: Sharpe relative to SHV before costs | +0.93 |

Strategy returns are net of the platform default: 3 bps commission plus 10 bps
flat slippage per executed buy and sell dollar. Market benchmarks are gross,
and the ADV cap is inactive.

Costs apply to purchases plus sales measured against portfolio value before
trading. The table sums their daily effect on returns, measured against the
previous close's portfolio value. These denominators differ when prices move
before a trade.

The strategy spends about half the sessions in cash. After trading costs, its
nominal growth is close to SHV and its sample Sharpe relative to SHV is near
zero. Its gross sample Sharpe also falls below the basket at the same daily
exposure. The basket provides a reference for return attribution before costs.

The exploratory deflated Sharpe ratio is 0.069, assuming 10 trials. It uses a
variance estimate from the reported return series. The historical trial count
and dispersion of Sharpe estimates across dependent trials are unknown, so
this value does not provide a validated adjustment for selecting among
strategies. Exact source and artifact hashes are in
`docs/img/performance_manifest.json`.

Further attribution and evaluation comparisons are described in the
[workshop paper](https://openreview.net/forum?id=Yjm9wkbzMb).

## Generate a report

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

For an explicitly requested live download:

```bash
PYTHONPATH=. python scripts/generate_report.py --live-yfinance
PYTHONPATH=. python scripts/validate_performance.py --live-yfinance --pit
```

Review Yahoo's terms and the
[yfinance legal disclaimer](https://ranaroussi.github.io/yfinance/) before use.
Live historical data may be revised, so preserve your own permitted input if
exact reproduction matters.

## Generated figures

The command writes `reports/report.md` plus these plots under `reports/img/`:

- [report_overview.png](docs/img/report_overview.png): equity, drawdown, allocation, turnover, and costs in a
  compact review image.
- [equity_vs_benchmarks.png](docs/img/equity_vs_benchmarks.png): strategy net of modeled costs versus gross SPY
  and an equal-initial-weight buy-and-hold basket.
- [performance_attribution.png](docs/img/performance_attribution.png): strategy before and after modeled costs,
  the basket at actual exposure before costs, and the cash proxy on one capital clock.
- [drawdown.png](docs/img/drawdown.png): the strategy drawdown curve.
- [rolling_sharpe.png](docs/img/rolling_sharpe.png): trailing 126-session Sharpe relative to SHV.
- [rolling_risk.png](docs/img/rolling_risk.png): trailing 126-session annualized volatility and beta to
  SPY.
- [allocation_and_exposure.png](docs/img/allocation_and_exposure.png): realized asset weights, invested gross
  exposure, and cash under the share-based event engine.
- [turnover_and_costs.png](docs/img/turnover_and_costs.png): executed one-way turnover, gross traded notional,
  and cumulative modeled transaction-cost return drag.
- [monthly_returns.png](docs/img/monthly_returns.png): monthly net-return heatmap.
- [return_distribution.png](docs/img/return_distribution.png): log-count daily net-return histogram, historical
  tail markers, and a normal Q-Q diagnostic. Its horizontal axis uses theoretical
  normal quantiles and its vertical axis uses standardized observed returns;
  both axes describe the same observed return sample.

The Markdown report links every plot and includes performance metrics,
evaluation settings, data provenance, and the monthly-return table. Outputs
remain ignored by default. Publish them only with explicit owner approval,
complete adjacent provenance, an input digest, and hashes for every artifact.

The current command must not fabricate diagnostics it cannot support. Add
walk-forward or live-start boundaries only when the run records those regimes;
capacity and slippage curves only with spread, volume, and order-size inputs;
factor attribution only with validated factor returns/exposures; and fill
quality only from authenticated order and execution records.

## Reporting results

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

## Interpretation

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

The residual momentum score is zero when an asset's prices match its
regression benchmark and the regression is valid. This prevents numerical
roundoff from passing the positive-score filter. The strategy uses ordinary
momentum when the history or benchmark variance cannot support the regression.

The reference evaluation executes close-derived target weights at the next
available bar's close. Weekly schedules use the first observed session of each week,
so a Monday exchange holiday moves the decision to Tuesday rather than skipping
the week. Monthly validation decisions use the last observed session of each
month. Report any separate signal warm-up period; using the evaluation window
itself for model warm-up can materially bias comparisons.

No metric threshold in this repository is evidence that a strategy is
profitable or deployable. Interpret every result against its data vintage,
comparator, costs, exposure, uncertainty, and recorded research trial history.
