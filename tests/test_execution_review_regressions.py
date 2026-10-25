"""Execution state must retain instrument identity and reject updates atomically."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from quantcortex.execution.brokers.base import BrokerError, Position
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import (
    Order,
    OrderError,
    OrderManager,
    OrderStatus,
)
from quantcortex.execution.position_manager import PositionManager
from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)


@pytest.mark.parametrize(
    "changed",
    [
        {"secType": "OPT", "multiplier": "100"},
        {"secType": "FUT"},
        {"currency": "EUR"},
        {"secType": None},
        {"currency": None},
        {"conId": None},
        {"conId": 0},
        {"conId": True},
        {"symbol": None},
    ],
)
def test_ib_rejects_positions_without_supported_stock_identity(changed):
    fields = {"symbol": "AAPL", "secType": "STK", "currency": "USD", "conId": 123}
    fields.update(changed)
    broker = IBBroker(account="DU_REVIEW")
    broker._ib = SimpleNamespace(
        positions=lambda: [SimpleNamespace(
            account="DU_REVIEW", contract=SimpleNamespace(**fields),
            position=1.0, avgCost=500.0,
        )],
    )

    with pytest.raises(BrokerError, match="contract"):
        broker.positions_as_dict()


def test_ib_preserves_supported_usd_stock_position():
    broker = IBBroker(account="DU_REVIEW")
    broker._ib = SimpleNamespace(
        positions=lambda: [SimpleNamespace(
            account="DU_REVIEW",
            contract=SimpleNamespace(symbol="AAPL", secType="STK", currency="USD", conId=123),
            position=-2.0, avgCost=150.0,
        )],
    )
    position, = broker.get_positions()
    assert (position.symbol, position.quantity, position.avg_price) == ("AAPL", -2.0, 150.0)


@pytest.mark.parametrize("field,value", [
    ("avg_price", -1.0), ("market_price", -1.0), ("quantity", float("inf")),
    ("quantity", True), ("quantity", "invalid"),
])
@pytest.mark.parametrize("snapshot", [False, True])
def test_invalid_mutable_position_preserves_persisted_state(tmp_path, field, value, snapshot):
    path = tmp_path / "state.json"
    store = StatePersistence(file_path=path)
    position = Position("AAA", 2.0, avg_price=100.0, market_price=101.0)
    if snapshot:
        initial = store.save_execution_snapshot(positions={"AAA": position}, orders=[])
    else:
        store.save_positions({"AAA": position})
    before = path.read_bytes()
    setattr(position, field, value)

    with pytest.raises(StatePersistenceError, match="invalid"):
        if snapshot:
            store.save_execution_snapshot(
                positions={"AAA": position}, orders=[], expected_revision=initial.revision,
            )
        else:
            store.save_positions({"AAA": position})

    assert path.read_bytes() == before
    if snapshot:
        recovered = store.load_execution_snapshot()
        assert recovered.revision == initial.revision
        positions = recovered.positions
    else:
        positions = store.load_positions()
    assert positions["AAA"] == Position("AAA", 2.0, 100.0, 101.0)


@pytest.mark.parametrize("initial_price", [None, 100.0])
def test_price_only_fill_correction_requires_reconciliation_without_mutation(initial_price):
    manager = PositionManager()
    order = Order(
        "o1", "AAA", "BUY", 10.0, order_type="LIMIT", limit_price=100.0,
        status="FILLED", filled_quantity=10.0, avg_fill_price=initial_price,
    )
    manager.update_fill(order)
    # Intervening activity means a basis change cannot be reconstructed locally.
    manager.update("AAA", -5.0, 105.0)
    positions_before = deepcopy(manager.positions)
    ledger_before = deepcopy(manager._applied_fills)
    order.avg_fill_price = 99.0

    with pytest.raises(ValueError, match="reconcil"):
        manager.update_fill(order)

    assert manager.positions == positions_before
    assert manager._applied_fills == ledger_before
    order.avg_fill_price = 100.0
    manager.update_fill(order)
    assert manager.positions == positions_before


def test_unfilled_snapshot_does_not_require_a_price():
    manager = PositionManager()
    order = Order("o1", "AAA", "BUY", 10.0, status="SUBMITTED")
    assert manager.update_fill(order).quantity == 0.0
    assert manager.positions == {}


@pytest.mark.parametrize("first,second,expected", [
    (0.3, 0.699999999, OrderStatus.PARTIALLY_FILLED),
    (0.9, 0.09999999899999996, OrderStatus.FILLED),
])
def test_split_fill_status_matches_stored_cumulative_quantity(first, second, expected):
    manager = OrderManager()
    order = manager.create_order("split", "AAA", "BUY", 1.0)
    manager.submit("split")
    manager.fill("split", first, 100.0)
    manager.fill("split", second, 100.0)

    assert order.status is expected
    order.validate()
    if expected is OrderStatus.PARTIALLY_FILLED:
        manager.fill("split", fill_price=100.0)
        assert order.status is OrderStatus.FILLED
        assert order.filled_quantity == 1.0


def test_fill_notional_overflow_rejects_before_mutation():
    manager = OrderManager()
    order = manager.create_order("large", "AAA", "BUY", 2.0)
    manager.submit("large")
    before = deepcopy(order)

    with pytest.raises(OrderError, match="finite"):
        manager.fill("large", fill_price=1e308)

    assert manager.get("large") is order
    assert order == before
    manager.fill("large", fill_price=100.0)
    assert order.status is OrderStatus.FILLED
