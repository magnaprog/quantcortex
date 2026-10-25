# Evaluation Contracts

QuantCortex separates a strategy's decision stream from the engine that turns
those decisions into returns. The machine-readable contract for the reference
study is generated at `paper/results/evaluation_contract.json` and validated
against `schemas/evaluation_contract.schema.json`.

## Canonical Allocation Records

Each allocation record contains a decision timestamp, symbol, and target weight.
The code calls the collection a "target tape"; function and schema names retain
that term. Its engine-neutral rows have this form:

```text
decision_timestamp,symbol,target_weight
2024-01-01T00:00:00Z,A,1.0
2024-01-01T00:00:00Z,B,0.0
```

`quantcortex.backtest.conformance.validate_target_tape` requires unique
timestamp-symbol rows, a complete symbol set at every decision, finite
long-only weights, and gross exposure no greater than the declared limit.
Timestamps identify close-of-bar decisions; the reference engines execute them
on the first strictly later price bar. The schema-tagged version-1 JSON envelope
also declares the complete symbol set and gross limit:

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

`target_tape_to_payload` and `target_tape_from_payload` implement that envelope.
`schemas/canonical_target_tape.schema.json` specifies its serialized structure
and primitive constraints for other engines.

Runtime validation in `validate_target_tape` additionally enforces cross-record
portfolio invariants that the schema does not express: no duplicate
timestamp-symbol rows, the declared symbol universe at every decision, and the
per-decision gross limit.

The retrospective runner reconstructs each variant through this boundary before
backtesting. Expansion target generators return weights reconstructed from the
same boundary; the experiment runner revalidates, serializes, and hashes each
payload before evaluation. Both studies publish canonical payload hashes in
`paper/results/target_tape_hashes.json` and
`paper/expansion/results/target_tape_hashes.json`.

The evaluation contract also records post-overlay exposure rules and the
paper-trading order-state policy: persist intent before submission, block
automatic retry after an uncertain outcome, and reject stale state revisions.
Fault-injection tests support those operational claims; the historical return
panel supplies no evidence about order-state behavior.

The fixtures under `tests/fixtures/conformance/` are deterministic and
synthetic by design. They test software semantics only and are never used for
performance claims.

## Comparator Contracts

The paper distinguishes a basket at actual exposure before costs from comparison
portfolios that follow target exposure and pay their own trading costs:

- The basket at actual exposure before costs supplies the reference for an exact
  ex-post arithmetic identity and is not presented as tradable.
- The retrospective comparison portfolio invests the strategy's declared target
  risky exposure in a fixed equal-initial-weight buy-and-hold basket.
- The expansion comparison portfolio re-equalizes the declared target risky
  exposure across panel assets at each decision.

Both comparison portfolios use next-bar event accounting, residual SHV, and their
own proportional costs. They are conditional on strategy target paths, not
independent benchmarks or causal controls. Keep them distinct from the arithmetic
identity using the basket at actual exposure.

## Evidence Classes

Claims must identify their evidence type: exact identity, property test, paired
protocol contrast, fault injection, SDK conformance, or untested live behavior.
A passing unit test cannot establish broker connectivity, data rights, market
capacity, or future performance. See the paper appendix for the
claim-to-evidence table and `docs/production-readiness.md` for external gates.
