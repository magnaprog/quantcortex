# Market data inputs

Place permitted market data in this directory. Everything except this file
is gitignored. Keep raw and cached provider observations out of Git. Permission
to use data or publish derived results does not change that repository rule.

Record the provider, permission or license basis, retrieval timestamp, symbol
mapping, corporate-action adjustment method, timezone, and any cleaning steps
outside the CSV. Use the [data-source checklist](../docs/data-source-due-diligence.md)
to review the permitted uses before sharing results.

## Price matrix

`scripts/generate_report.py` and the research notebooks accept a date-by-symbol
matrix. For example:

```csv
date,QQQ,VGT,GLD,TLT,SPY,VIG,SHV
2024-01-02,100.0,101.0,102.0,103.0,104.0,105.0,106.0
```

Dates must be unique and parseable. Symbol columns must be numeric and strictly
positive. Reports and notebooks forward-fill missing prices for at most five
consecutive rows per symbol, then drop incomplete rows. Notebook universes require
all their symbol columns.

Set `QUANTCORTEX_PRICES_CSV` to the absolute path for notebooks, or pass
`--prices-csv local_data/<file>.csv` to `scripts/generate_report.py`.
Notebook runs must choose exactly one adjusted-close source: this variable or
`QUANTCORTEX_LIVE_YFINANCE=1`. `QUANTCORTEX_OHLCV_CSV` is a supplemental local
input for the Alpha158 notebook, not a second adjusted-close source.
Include the configured cash-proxy column, such as `SHV`, whenever residual
cash should earn a nonzero return.

For a performance report, include at least the requested pre-evaluation warm-up
history, which defaults to two calendar years. The loader cannot reconstruct
signal history that is absent from the file.

For a report intended for external review, also pass `--data-provider`,
`--permission-basis`, `--retrieved-at`, and `--adjustment-method`. These fields
record the owner's provenance assertions but do not independently establish
that redistribution is permitted. Pass `--manifest-out` as well so the input,
source tree, settings, and generated artifacts are recorded with their hashes.
See [PERFORMANCE.md](../PERFORMANCE.md) for report commands and review guidance.

## OHLCV data for one symbol

Notebook 02 also requires actual OHLCV data when using local files:

```csv
date,open,high,low,close,adj_close,volume
2024-01-02,100.0,102.0,99.0,101.0,101.0,1500000
```

All six fields are required. Prices must be positive, volume non-negative, and
high/low internally consistent. Set `QUANTCORTEX_OHLCV_CSV` to this file.
