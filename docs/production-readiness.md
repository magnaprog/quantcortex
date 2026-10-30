# Production readiness

QuantCortex supports research and guarded paper trading. Passing tests or CI
does not certify it for real-money deployment. The implemented controls below
have specific boundaries; authenticated venue behavior and production operations
require separate evidence and independent review.

## Broker certification

Adapters validate individual requests and lazy-load their SDKs. Direct calls do
not enforce portfolio risk, persist submission intent, or prevent uncertain
retries; those controls belong to orchestration.

| Adapter | Implemented boundary and limits |
|---|---|
| `AlpacaBroker` (`alpaca-py`) | Requires the exact official paper/live HTTPS host. Client-order-ID lookup and open-order reconciliation support the paper cycle. `done_for_day` and `stopped` remain unresolved, including partial fills; `suspended` or unknown status requires reconciliation. |
| `IBBroker` (`ib_async`) | Rejects known paper/live port mismatches but cannot classify an arbitrary host/port. Orders include the selected account. Stock positions require USD contracts with a valid contract ID; other instruments need an adapter preserving their contract details. A flat account is accepted only after managed-account validation. Cancellation needs an unambiguous API order ID for that account/client; a permanent ID alone requires reconciliation. |
| `CCXTBroker` | Paper mode requires advertised sandbox support and calls `set_sandbox_mode(True)`. Venue behavior varies. |

Offline mocks and installed SDK model tests check request/response mapping.
Before deployment, run authenticated paper tests for permissions, reconnects,
rejects, partial fills, cancellations, duplicate submissions, and venue-side
idempotency. Reconcile positions, cash, orders, and fills before every cycle.

## Paper cycle

Run from the repository root. The local dry run uses deterministic generated
prices and makes no broker or provider calls:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py --offline
```

For an explicitly chosen live-price workflow, install
`requirements/runtime.lock` or the required provider/broker extras first and
review the data terms. Preview:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py --live-yfinance
```

Paper-trading orders use a separate command:

```bash
PYTHONPATH=. .venv/bin/python scripts/paper_trade_cycle.py \
  --live-yfinance --submit
```

The [script](../scripts/paper_trade_cycle.py) uses `QQQ,VGT,GLD,TLT,SPY,VIG` and refuses
a real-money Alpaca endpoint. If either Alpaca credential is missing, `--live-yfinance`
uses live prices with a local $100,000 empty-account simulation. With credentials it
reads the paper account; unresolved open orders block preview and submission.
Configuration names are in [.env.example](../.env.example); direct Python commands do
not load `.env` automatically.

The panel must be complete and end at the latest finalized NYSE daily session,
using a conservative 17:00 America/New_York cutoff. Stale symbols are not
forward-filled. Whole-share translation can stop on fractional holdings or held
symbols absent from the price panel. If both sells and buys are needed, one
invocation submits only sells; rerun after fills with refreshed account state
for buys.

On submission, the cycle first refreshes persisted nonterminal order IDs. If it
reconciles an order, it saves the snapshot and exits without new submissions.
Rerun after unresolved orders finish to obtain fresh account state. Missing or
mismatched orders remain blocked.

Before each submission, the cycle queries Alpaca by deterministic client order
ID. A missing lookup result after a prior local attempt remains uncertain; it
is not permission to retry. Use `--retry-confirmed-missing` only after the
Alpaca dashboard or support confirms that the prior intent was not routed.
Do not bypass this check by inventing a new identifier.

Market orders may fill in the current session when the venue is open. Orders
submitted after the finalized close wait for the next eligible session. This
differs from the reference backtests' next-bar-close fill convention and supplies
no evidence of equivalent fill quality or performance.

## State and recovery

The paper cycle validates targets, translates orders, checks the complete batch
and reconstructed post-trade exposure, then durably records `ATTEMPTING` before
calling the broker. An uncertain outcome blocks automatic retry until broker
reconciliation. See [Execution state](architecture.md#execution-state) for the
order lifecycle, fill-price rules, and API boundaries.

`ExecutionSnapshot` combines positions, known orders, submission intents, and
reconciliation metadata in one atomic, versioned write. Stale revisions fail.
Generic state/position/order writes do not provide the same coherent transaction
or revision check. Typed recovery rejects malformed stored data rather than
treating it as missing.

Before production use, exercise backup restoration, corruption, concurrent
writes, and disaster recovery in both file and Redis backends. File concurrency
has test coverage; Redis transactions still need integration testing with a
real service. Retain immutable audit records and verify recovery across
concurrent clients.

## Research and data

- Conduct [data-source due diligence](data-source-due-diligence.md), retaining
  provenance and a documented permission basis. Research use, derived publication,
  reviewer access, and raw redistribution are separate questions.
- Obtain exact filing timestamps, point-in-time membership, and delisted-security
  prices for production single-name research. Approximate membership alone is
  insufficient.
- Validate corporate actions, calendars, symbol mapping, and stale-price policy
  against intended venues. Replace the paper CLI's 17:00 cutoff with an
  authoritative venue or vendor completion signal before production use.
- Calibrate spread, volatility, size, impact, and capacity assumptions when order
  size makes impact material. Flat slippage and adjusted-close pseudo-shares
  do not establish actual fill quality.

## Deployment controls

Committed dependency locks and digest-pinned base images constrain parts of the build.
Dockerfile apt packages are not version-pinned, so the whole image is not thereby
reproducible. CI defines dependency-export checks, container smoke tests, and SDK model
tests; inspect receipts for the exact candidate commit to see which checks passed.

The application container runs as a non-root user with a read-only root filesystem.
Before deployment:

- Run container, database, Redis, and broker integration tests in staging that
  matches production.
- Add managed secrets, least privilege, structured logs, metrics, alerting, and
  immutable audit records.
- Define kill switches, exposure limits, incident ownership, rollback procedures,
  independent release approval, and applicable regulatory review.

Core CI does not establish optional PPO training or broad boosting, Torch, and
FinBERT training/inference compatibility. Select and test the actual backend
and numerical environment used in the deployment.

## Release evidence

For published reports, follow [PERFORMANCE.md](../PERFORMANCE.md) and the
[data-source record](data-source-due-diligence.md). Generate from reviewed source
and permitted inputs. Archive the exact source revision, environment, dependency
lock, configuration, input digest, generation timestamp, output hashes, trial
history, and validation results. For an operational release, also
retain paper-account certification evidence and independent sign-off, and resolve
the applicable controls above before deployment.

A source hash or local test receipt does not establish remote publication,
provider permission, authenticated broker behavior, or production readiness.
