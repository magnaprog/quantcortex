# Repository Guidelines

## Project Structure & Module Organization

All importable code lives under `quantcortex`; use absolute imports such as
`from quantcortex.portfolio.base import enforce_weight_contract`. Pipeline
layers are `data`, `alpha`, `portfolio`, `timing`, `risk`, `backtest`,
`execution`, and `strategies`. Root directories include `tests/`, `research/`,
`scripts/`, `paper/`, ignored `local_data/`, and ignored `reports/`. See
`CLAUDE.md` for layer details.

## Build, Test, and Development Commands

Run from the repo root:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check .
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
PYTHONPATH=. .venv/bin/python scripts/generate_report.py \
  --prices-csv local_data/published_rotation_prices.csv \
  --cash-proxy-symbol SHV
```

For paper artifacts, follow the [coordinated release flow](paper/README.md#coordinated-release)
and `CLAUDE.md`: obtain explicit user approval for the source commit, then run
expansion first and paper second from that same commit. The flow covers the
absolute Python path, fixed timestamps for changed source or inputs, and the
checks required before an unchanged direct-build replay.

A new public export uses vetted tracked files and a fresh source root P, without
private Git history or untracked material. Regenerate expansion and paper from P
with fixed UTC timestamps, then record outputs in artifact child A. Retain old
protocol-freeze identifiers as private historical metadata; they do not prove a
public pre-data freeze. See the coordinated release flow for source and artifact
binding checks. Workflow definitions do not establish a passing public CI run.

Install from `requirements/dev.lock`. Scripts need `PYTHONPATH=.` unless
installed editable. Set `MPLCONFIGDIR` to a writable directory when required.

## Coding Style & Naming Conventions

Target Python 3.11-3.14. Use 4-space indentation, `snake_case`
modules/functions, `PascalCase` classes, and ASCII source and documentation.
Preserve proper-name diacritics in author and bibliography metadata. Keep
comments concise and only for non-obvious logic.
Prefer existing layer boundaries and `quantcortex.portfolio.base` contracts.

## Testing Guidelines

Tests use pytest and deterministic synthetic data. Add focused regression tests
for contract, accounting, causality, and execution-state changes. Keep optional
integrations lazy and mock external brokers/providers. Use `test_*.py` and
`test_*`.

## Load-Bearing Constraints

Do not weaken the strict allocation contract or relaxed post-overlay exposure
contract. Backtests must use a `TransactionCostModel`; nonzero residual-cash
returns require a complete aligned series. Factors and strategy features must
stay point-in-time and causal. Optional integrations stay lazy imports.

## Commit & Pull Request Guidelines

Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `ci:`); keep
messages short and imperative. PRs should explain behavior changes, list
verification commands, call out data or API assumptions, and link issues.
Include screenshots only for generated chart or notebook/report changes.

## Agent-Specific Instructions

`CLAUDE.md` is the authoritative agent guide; this file is the short
orientation and must stay consistent with it. Read `CLAUDE.md` for depth. Work
from the requested change and verify it directly. State assumptions when
ambiguity affects correctness. Keep edits surgical, avoid speculative
abstractions, and do not refactor unrelated code. For money-path code, add or
update regression tests.

## Security & Configuration Tips

Never commit `.env`, credentials, broker account data, local state, market-data
snapshots, or executed notebook outputs. Published performance and paper
artifacts require owner approval, adjacent provenance, an input digest, and
artifact hashes; release them from a clean commit with the release wrappers.
Ordinary reports remain ignored. Use `.env.example` for configuration names.
Synthetic data is limited to tests and clearly labeled dry runs. Preserve
pre-trade risk checks, paper-mode defaults, and point-in-time data discipline.
