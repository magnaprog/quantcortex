# Contributing to QuantCortex

Start with [AGENTS.md](AGENTS.md), the repository entry guide, and the
[contributor guide](docs/onboarding.md). Keep changes within the existing
pipeline boundaries and verify the behavior you change.

## Development setup

Use Python 3.11 through 3.14 from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/dev.lock
.venv/bin/python -m pip install --no-deps -e .
```

Run before opening a pull request:

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest tests/ -q --cov=quantcortex --cov-fail-under=60
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

The development lock omits optional SDKs and ML backends. Install the relevant
[lock or extra](docs/onboarding.md#set-up-and-check) for the work being checked.
A skip does not verify that integration.

## Change discipline

Use four-space indentation, `snake_case` functions and modules, `PascalCase`
classes, and absolute `quantcortex.*` imports. Keep source and documentation
ASCII, preserving proper-name diacritics in author and bibliography metadata.
Keep optional dependencies lazy.

Preserve [allocation and exposure contracts](docs/architecture.md#weight-contracts),
point-in-time data handling, explicit transaction costs, paper-mode defaults,
and fail-closed execution. Changes to accounting, risk checks, order state,
broker adapters, or causality require focused regression tests with
hand-checkable expected results and relevant invalid or partial-failure cases.
Avoid unrelated refactoring.

Do not commit credentials, `.env`, account data, raw or cached market data,
executed notebook outputs, local state, or ordinary generated reports.
Published derived figures require owner approval, adjacent provenance, an input
digest, artifact hashes, and regeneration from permitted inputs. Owner approval
does not establish provider rights; use the
[data-source checklist](docs/data-source-due-diligence.md).
Report security-sensitive failures through [SECURITY.md](SECURITY.md).

## Integration and evidence changes

Broker-adapter changes require real SDK model tests as well as mocked behavior:

```bash
.venv/bin/python -m pip install -r requirements/brokers.lock
.venv/bin/python -m pytest tests/test_broker_sdk_conformance.py -q
PYTHONPATH=. .venv/bin/python scripts/verify_brokers.py
```

These checks do not establish authenticated transport, account permissions,
reconnect behavior, or venue-side idempotency. Keep those limits explicit in
review; see [Production readiness](docs/production-readiness.md).

When packaging or container behavior changes:

```bash
.venv/bin/python -m pip install -r requirements/build.lock
.venv/bin/python -m build
POSTGRES_PASSWORD=local-only docker compose config --quiet
docker build --tag quantcortex:local .
docker run --rm --read-only --tmpfs /tmp quantcortex:local
```

`.dockerignore` excludes known private paths; it is not a secret scanner.
Keep credentials outside the build context. The
[CI workflow](.github/workflows/ci.yml) also defines clean wheel-install and
notebook checks.

For report changes, follow [PERFORMANCE.md](PERFORMANCE.md). Before publishing
derived results, obtain maintainer approval for the source and data, and check
the [permission basis](docs/data-source-due-diligence.md). Generate from a clean
source commit with an explicit generation timestamp and a manifest recording
the input digest, source hashes, settings, and output hashes. Permission to
regenerate a report does not authorize a new provider request.

Run the report and data-policy checks, inspect the figures at their intended
display size, and review changes in the metrics and source records before
publication. Do not hand-edit generated tables, figures, manifests, or checksums.

Use [quantcortex.backtest.metrics.plotting](quantcortex/backtest/metrics/plotting.py)
for repository figures. Pair important series with distinct line styles,
markers, or hatching rather than relying on color alone. Check that labels and
legends remain readable at the intended display size.

## Dependencies

Edit `pyproject.toml`, then regenerate all locks using Poetry 2.4.1 and
`poetry-plugin-export` 1.10.0. Set `POETRY_BIN` if needed:

```bash
scripts/update_dependency_locks.sh
.venv/bin/python -m pytest tests/test_dependency_locks.py -q
```

Commit `pyproject.toml`, `poetry.lock`, and every changed export under
`requirements/` together. CI rejects lock drift. Dependency changes also
invalidate report source fingerprints. Regenerate affected published reports
with the new dependencies and review their results as described above.

## Pull requests

Use short conventional commits such as `fix: reconcile cash returns`. Describe
the behavior change, data and API assumptions, verification commands and results,
and remaining limitations. Link relevant issues. Include updated figures only
when report output changes.

Tie validation receipts to the exact candidate commit. Record any temporary
source or artifact mismatches as failures until regeneration and verification
resolve them. A workflow definition does not establish a passing CI run.
