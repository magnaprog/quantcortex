# Evaluation contracts

A strategy's decision stream and the engine that turns it into returns have
separate contracts. The [evaluation schema](../schemas/evaluation_contract.schema.json)
defines the format for declaring those semantics. For allocation and strategy APIs, see
[Architecture](architecture.md).

## Canonical allocation records

The code calls these engine-neutral decision records a "target tape". Preserve
that term in function and schema names. Its long-form rows contain a decision
timestamp, symbol, and target weight:

```text
decision_timestamp,symbol,target_weight
2024-01-01T00:00:00Z,A,1.0
2024-01-01T00:00:00Z,B,0.0
```

The version-1 JSON envelope declares the complete symbol universe and gross
limit. This synthetic example illustrates the format only:

```json
{
  "schema_version": 1,
  "symbols": ["A", "B"],
  "max_gross": 1.0,
  "records": [
    {
      "decision_timestamp": "2024-01-01T00:00:00Z",
      "symbol": "A",
      "target_weight": 1.0
    },
    {
      "decision_timestamp": "2024-01-01T00:00:00Z",
      "symbol": "B",
      "target_weight": 0.0
    }
  ]
}
```

[conformance.py](../quantcortex/backtest/conformance.py) supplies
`validate_target_tape`, `weights_to_target_tape`, `target_tape_to_weights`,
`target_tape_to_payload`, and `target_tape_from_payload`.

The [canonical schema](../schemas/canonical_target_tape.schema.json) checks envelope and
record structure, supported version, unique declared symbols, timestamp syntax,
primitive types, and scalar bounds. Use runtime validation too: the schema does not
express complete books, record membership in the declared universe, unique
timestamp-symbol pairs, or aggregate gross limits.

`target_tape_from_payload()` checks exact keys, JSON numbers rather than booleans
or numeric strings, finite binary64 values, offset-bearing timestamps, symbols,
and version. `validate_target_tape()` normalizes timestamps to UTC-naive internal
values, rejects duplicates and incomplete books, and enforces long-only weights
and the per-decision gross cap. Every decision includes every declared symbol,
including zero-weight positions. Empty records and unsupported versions fail.

These are close-of-bar decisions. The reference engines execute on the first
strictly later price bar; a target without a later bar remains unexecuted.
Record validation does not establish feature availability or execution timing.
Targets contain no prices, orders, fills, or realized returns.

Keep the complete allocation records alongside the input and run metadata.
A hash can verify a reconstructed copy but cannot recreate its contents or
historical returns. The [synthetic conformance fixtures](../tests/fixtures/conformance/)
exercise software semantics without market data.

## Declared semantics and runtime checks

The evaluation contract records selection, allocation, overlays, execution,
cash, costs, comparator roles, order-state policy, and evidence classes. Schema
validation checks shape, constants, and nonempty prose; it does not verify the
truth of declarations or bind them to a particular engine invocation.

For reproduction, retain allocation records and a run manifest identifying
concrete selector and allocator classes, backend/library versions, parameters,
input digests, environment, and the engine call. Named runtime checks, property
tests, and failure tests supply enforcement evidence. The paper-trading policy
requires persisted intent before submission, uncertain-retry blocking, and stale
revision rejection. Fault-injection tests support those claims; historical
returns supply no order-state evidence. Direct broker calls have a narrower
contract; see [execution state](architecture.md#execution-state).

## Run an engine

Given an initialized `strategy`, a price matrix, decision dates, and complete
aligned cash returns, the existing vectorized example is:

```python
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.vectorized import VectorizedBacktest

cost_model = TransactionCostModel()
engine = VectorizedBacktest(cost_model, capital=1_000_000.0)
weights = strategy.generate_weights(prices, rebalance_dates)
result = engine.run(weights, prices, cash_returns=cash_returns)
```

This is an API fragment, not a standalone data download or performance result.
The vectorized engine implicitly re-pegs weights between rebalances at zero
cost. Use `EventDrivenBacktest` when the evaluation requires holdings drift,
fill semantics, and post-cost sizing. Persist the original target records;
`result.weights` represents engine-effective positions. See
[timing and accounting](architecture.md#timing-and-accounting).

## Cash, costs, and comparators

Only `cash_returns=None` selects implicit zero-return cash. Any supplied series,
even all zeros, must cover every price bar with finite observations. Cash return
and the excess-Sharpe benchmark are separate choices. For example, cash can earn
zero while performance is measured relative to SHV.

With a proportional charge of 13 basis points per dollar bought or sold:

```text
modeled cost = 0.0013 * (B_t + S_t)
one-way turnover = max(B_t, S_t)
```

`B_t + S_t` is gross two-sided traded notional, reported as `traded_notional`;
`turnover` is the one-way measure. Use consistent units: when buys and sells
are fractions of pre-trade NAV, the cost is a return drag. The flat charge is
an assumption, not a calibrated spread, impact, order-size, or capacity model.

### Comparator contracts

| Reference | Exposure and allocation | Accounting and role |
|---|---|---|
| Basket at actual exposure, before costs | Realized risky exposure applied to a fixed equal-initial-weight buy-and-hold basket | Ex-post arithmetic attribution; before its own trading costs and not presented as tradable |
| Retrospective comparison portfolio | Strategy target risky exposure invested in that fixed basket | Next-bar event accounting, residual SHV, and its own modeled costs |
| Expansion comparison portfolio | Strategy target risky exposure re-equalized across risky panel assets at every decision; SHV excluded | Next-bar event accounting, residual SHV, and its own modeled costs |

Both comparison portfolios depend on the strategy target path. They are not
independent benchmarks or causal controls. The exact daily attribution identity
adds in annualized arithmetic means, not CAGR or compounded wealth; its timing
term is relative to full-sample mean realized exposure. Components are
reference-dependent, noncausal, and not separately tradable. The active risky
term includes group selection, weighting, and event-engine drift, not pure
security selection. Definitions and results remain in
the [workshop paper](https://openreview.net/forum?id=Yjm9wkbzMb).

Preserve the version-1 schema key `economic_counterfactual` unless performing
a versioned migration. Describe the actual comparison and its assumptions in
the run record.

## Evidence classes

| Claim type | Supporting evidence and scope |
|---|---|
| Exact identity | Algebraic equality on recorded series |
| Property test | Invariants across the tested generated/synthetic cases |
| Paired protocol contrast | Difference between two declared evaluation paths |
| Fault injection | Response to an injected failure |
| SDK conformance | Request/response compatibility with installed SDK models |
| Untested live behavior | Explicit absence of supporting live evidence |

A unit test cannot establish connectivity, data rights, capacity, future
performance, or production readiness. The paper's historical contrasts are
conditional diagnostics, not causal effects, an independent benchmark, or a
population estimate of backtest error. Repository freezing is not external
preregistration, independent chronology, or a temporal holdout.

Record the candidate set and trial history before interpreting multiple variants.
Deflated Sharpe Ratio and multiple-testing procedures need defensible inputs;
a guessed trial count or missing variants cannot support those calculations.
Probabilistic and deflated Sharpe helpers return `nan` when their estimated
Sharpe variance is nonpositive or nonfinite, which can occur even with finite,
variable returns in a short sample. See
[reporting requirements](../PERFORMANCE.md#reporting-results).

Historical reproduction depends on matching permitted inputs, source,
configuration, environment, and allocation records. For external operational
checks, see [Production readiness](production-readiness.md).
