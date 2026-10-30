"""Reject malformed legacy submission ledgers before snapshot migration."""

from unittest.mock import Mock

import pytest

from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)
from scripts import paper_trade_cycle


@pytest.fixture
def store(tmp_path):
    return StatePersistence(file_path=tmp_path / "state.json")


def test_missing_submission_ledger_uses_default_without_writing(store, tmp_path):
    default = {}

    assert store.load_state(
        paper_trade_cycle.SUBMISSION_LEDGER_KEY, default=default
    ) is default
    assert paper_trade_cycle._load_submission_ledger(store) == {}
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("payload", [None, [], False, 0, ""])
def test_persisted_falsey_submission_ledger_is_not_a_missing_key(store, payload):
    store.save_state(paper_trade_cycle.SUBMISSION_LEDGER_KEY, payload)
    loaded = store.load_state(paper_trade_cycle.SUBMISSION_LEDGER_KEY, default={})

    assert type(loaded) is type(payload)
    assert loaded == payload
    with pytest.raises(StatePersistenceError, match="must be a JSON object"):
        paper_trade_cycle._load_submission_ledger(store)


@pytest.mark.parametrize("payload", [None, [], False, 0, ""])
def test_malformed_submission_ledger_stops_before_snapshot_write(
    store, tmp_path, monkeypatch, payload
):
    store.save_state(paper_trade_cycle.SUBMISSION_LEDGER_KEY, payload)
    before = (tmp_path / "state.json").read_bytes()
    save_snapshot = Mock(wraps=store.save_execution_snapshot)
    monkeypatch.setattr(store, "save_execution_snapshot", save_snapshot)

    with pytest.raises(StatePersistenceError, match="must be a JSON object"):
        paper_trade_cycle._initialise_submission_state(
            store, positions={}, orders=[], metadata={"broker": "alpaca-paper"}
        )

    save_snapshot.assert_not_called()
    assert (tmp_path / "state.json").read_bytes() == before
    assert store.load_execution_snapshot() is None


@pytest.mark.parametrize("persisted", [False, True], ids=["missing", "empty"])
def test_missing_or_empty_submission_ledger_initializes_empty_snapshot(store, persisted):
    if persisted:
        store.save_state(paper_trade_cycle.SUBMISSION_LEDGER_KEY, {})

    ledger, context = paper_trade_cycle._initialise_submission_state(
        store, positions={}, orders=[], metadata={"broker": "alpaca-paper"}
    )

    snapshot = store.load_execution_snapshot()
    assert ledger == {}
    assert context["revision"] == 1
    assert snapshot.intents == {}
    assert snapshot.metadata["migrated_legacy_submission_ledger"] is False


def test_valid_submission_ledger_is_preserved_during_migration(store):
    expected = {
        "qc-existing": {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
            "state": "UNKNOWN",
        }
    }
    store.save_state(paper_trade_cycle.SUBMISSION_LEDGER_KEY, expected)

    assert paper_trade_cycle._load_submission_ledger(store) == expected
    ledger, context = paper_trade_cycle._initialise_submission_state(
        store, positions={}, orders=[], metadata={"broker": "alpaca-paper"}
    )

    snapshot = store.load_execution_snapshot()
    assert ledger == expected
    assert context["revision"] == 1
    assert snapshot.intents == expected
    assert snapshot.metadata["migrated_legacy_submission_ledger"] is True
    assert store.load_state(paper_trade_cycle.SUBMISSION_LEDGER_KEY) == expected
