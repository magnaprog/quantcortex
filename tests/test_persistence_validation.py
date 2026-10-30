"""Persisted execution payloads must satisfy the typed state contracts."""

import pytest

from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)


@pytest.fixture
def store(tmp_path):
    return StatePersistence(file_path=tmp_path / "state.json")


@pytest.fixture
def snapshot_payload():
    return {
        "schema_version": 1,
        "revision": 1,
        "updated_at": "2024-01-02T00:00:00+00:00",
        "positions": {},
        "orders": [],
    }


@pytest.mark.parametrize("via_snapshot", [False, True])
@pytest.mark.parametrize(
    "field,payload",
    [
        ("positions", None),
        ("positions", False),
        ("positions", 0),
        ("positions", ""),
        ("positions", []),
        ("orders", None),
        ("orders", False),
        ("orders", 0),
        ("orders", ""),
        ("orders", {}),
    ],
)
def test_loaders_reject_present_malformed_falsey_payloads(
    store, snapshot_payload, via_snapshot, field, payload
):
    if via_snapshot:
        snapshot_payload[field] = payload
        store.save_state(store.EXECUTION_SNAPSHOT_KEY, snapshot_payload)
        load = store.load_execution_snapshot
    else:
        store.save_state(field, payload)
        load = store.load_positions if field == "positions" else store.load_orders

    with pytest.raises(StatePersistenceError, match=f"persisted {field}"):
        load()


@pytest.mark.parametrize("payload", [None, False, 0, "", [], {}])
def test_snapshot_loader_rejects_present_malformed_falsey_snapshot(store, payload):
    store.save_state(store.EXECUTION_SNAPSHOT_KEY, payload)

    with pytest.raises(StatePersistenceError, match="execution snapshot"):
        store.load_execution_snapshot()


def test_missing_state_and_valid_empty_payloads_retain_defaults(store, snapshot_payload):
    assert store.load_positions() == {}
    assert store.load_orders() == []
    assert store.load_execution_snapshot() is None

    store.save_state(store.POSITIONS_KEY, {})
    store.save_state(store.ORDERS_KEY, [])
    assert store.load_positions() == {}
    assert store.load_orders() == []

    for field in ("positions", "orders"):
        del snapshot_payload[field]
    store.save_state(store.EXECUTION_SNAPSHOT_KEY, snapshot_payload)
    snapshot = store.load_execution_snapshot()
    assert snapshot.positions == {}
    assert snapshot.orders == []
    assert snapshot.revision == 1


@pytest.mark.parametrize("via_snapshot", [False, True])
@pytest.mark.parametrize(
    "positions,message",
    [
        ({"AAPL": {"symbol": "MSFT", "quantity": 1.0}}, "does not match"),
        ({"AAPL": 1.0, " AAPL ": 2.0}, "duplicate position symbol"),
        ({" AAPL ": 1.0, "AAPL": 2.0}, "duplicate position symbol"),
        ({" ": 1.0}, "persisted positions"),
    ],
)
def test_position_loaders_reject_inconsistent_symbols(
    store, snapshot_payload, via_snapshot, positions, message
):
    if via_snapshot:
        snapshot_payload["positions"] = positions
        store.save_state(store.EXECUTION_SNAPSHOT_KEY, snapshot_payload)
        load = store.load_execution_snapshot
    else:
        store.save_state(store.POSITIONS_KEY, positions)
        load = store.load_positions

    with pytest.raises(StatePersistenceError, match=message):
        load()


@pytest.mark.parametrize("via_snapshot", [False, True])
def test_position_loaders_normalize_keys_and_preserve_quantities(
    store, snapshot_payload, via_snapshot
):
    positions = {
        " AAPL ": {"symbol": " AAPL ", "quantity": 2.0, "avg_price": 100.0},
        " MSFT ": -3.0,
        " QQQ ": {"quantity": 0.0},
    }
    if via_snapshot:
        snapshot_payload["positions"] = positions
        store.save_state(store.EXECUTION_SNAPSHOT_KEY, snapshot_payload)
        loaded = store.load_execution_snapshot().positions
    else:
        store.save_state(store.POSITIONS_KEY, positions)
        loaded = store.load_positions()

    assert set(loaded) == {"AAPL", "MSFT", "QQQ"}
    assert all(position.symbol == key for key, position in loaded.items())
    assert {key: position.quantity for key, position in loaded.items()} == {
        "AAPL": 2.0, "MSFT": -3.0, "QQQ": 0.0,
    }
    assert loaded["AAPL"].avg_price == 100.0
