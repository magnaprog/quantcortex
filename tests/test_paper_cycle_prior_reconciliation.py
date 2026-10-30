"""Prior broker fills must become authoritative before a replacement trade."""

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quantcortex.execution.brokers.alpaca_broker import AlpacaBroker
from quantcortex.execution.brokers.base import AccountInfo, Position
from quantcortex.execution.order_manager import Order, OrderSide, OrderStatus
from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)
from scripts import paper_trade_cycle as cycle


class PaperBroker:
    def __init__(self):
        self.orders = {}
        self.positions = []
        self.lookups = []
        self.submissions = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_open_orders(self):
        return []

    def get_account(self):
        return AccountInfo(cash=90000, equity=100000, buying_power=90000)

    def get_positions(self):
        return list(self.positions)

    def find_order_by_client_order_id(self, client_id):
        self.lookups.append(client_id)
        return self.orders.get(client_id)

    def submit_order(self, symbol, side, quantity, *, client_order_id):
        order = Order(
            f"broker-{len(self.submissions) + 1}", symbol, side, quantity,
            status=OrderStatus.SUBMITTED,
        )
        self.orders[client_order_id] = order
        self.submissions.append(client_order_id)
        return order


@pytest.fixture
def submitted_cycle(tmp_path, monkeypatch):
    broker = PaperBroker()
    store = StatePersistence(file_path=tmp_path / "state.json")
    monkeypatch.setattr(
        "quantcortex.execution.brokers.alpaca_broker.AlpacaBroker",
        lambda **kwargs: broker,
    )
    monkeypatch.setattr(cycle, "StatePersistence", lambda **kwargs: store)
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    prices = pd.DataFrame(
        100.0, index=pd.DatetimeIndex(["2026-10-22"]), columns=cycle.UNIVERSE
    )
    assert cycle.run_paper(prices, pd.Series({"QQQ": .1}), submit=True) == 0
    client_id = broker.submissions[0]
    broker.positions = [Position("QQQ", 100, 100, 100)]
    prices.index = pd.DatetimeIndex(["2026-10-23"])
    broker.lookups.clear()
    return broker, store, prices, client_id


def filled(order, **changes):
    return replace(
        order, status=OrderStatus.FILLED, filled_quantity=order.quantity,
        avg_fill_price=100.0, history=[*order.history, OrderStatus.FILLED],
        **changes,
    )


def test_completed_prior_order_reconciles_then_allows_fresh_cycle(submitted_cycle):
    broker, store, prices, client_id = submitted_cycle
    broker.orders[client_id] = filled(broker.orders[client_id])
    flat = pd.Series(dtype=float)
    assert cycle.run_paper(prices, flat, submit=True) == 1
    snapshot = store.load_execution_snapshot()
    assert snapshot.intents[client_id]["state"] == "FILLED"
    assert snapshot.orders[0].status is OrderStatus.FILLED
    assert snapshot.positions["QQQ"].quantity == 100
    assert broker.lookups == [client_id]
    assert broker.submissions == [client_id]
    assert cycle.run_paper(prices, flat, submit=True) == 0
    assert len(broker.submissions) == 2
    assert broker.orders[broker.submissions[-1]].side is OrderSide.SELL
    # The next lookup still reports the sell as unresolved: never resend it.
    assert cycle.run_paper(prices, flat, submit=True) == 1
    assert len(broker.submissions) == 2


@pytest.mark.parametrize("change", ["symbol", "side", "quantity", "broker_id", "client_id"])
def test_mismatched_prior_order_stops_reconciliation(submitted_cycle, change):
    broker, store, prices, client_id = submitted_cycle
    old = broker.orders[client_id]
    changed = {
        "symbol": {"symbol": "TLT"},
        "side": {"side": OrderSide.SELL},
        "quantity": {"quantity": 101, "filled_quantity": 101},
        "broker_id": {"order_id": "different-broker-id"},
    }
    returned = filled(old)
    if change == "client_id":
        # The adapter verifies the response's client ID before returning Order.
        adapter = AlpacaBroker()
        adapter._api = SimpleNamespace(
            get_order_by_client_id=lambda _: SimpleNamespace(client_order_id="other")
        )
        broker.find_order_by_client_order_id = adapter.find_order_by_client_order_id
    else:
        broker.orders[client_id] = replace(returned, **changed[change])
    original = store.load_execution_snapshot().intents
    assert cycle.run_paper(prices, pd.Series(dtype=float), submit=True) == 1
    assert store.load_execution_snapshot().intents == original
    assert broker.submissions == [client_id]


@pytest.mark.parametrize("state", [None, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED])
def test_absent_or_unresolved_prior_order_cannot_release_replacement(
    submitted_cycle, state
):
    broker, store, prices, client_id = submitted_cycle
    old = broker.orders[client_id]
    if state is None:
        del broker.orders[client_id]
    elif state is OrderStatus.PARTIALLY_FILLED:
        broker.orders[client_id] = replace(
            old, status=state, filled_quantity=50, avg_fill_price=100,
            history=[*old.history, state],
        )
    assert cycle.run_paper(prices, pd.Series(dtype=float), submit=True) == 1
    assert client_id in broker.lookups
    assert store.load_execution_snapshot().intents[client_id]["state"] in {
        "SUBMITTED", "PARTIALLY_FILLED"
    }
    assert broker.submissions == [client_id]


def test_snapshot_failure_cannot_release_replacement(submitted_cycle, monkeypatch):
    broker, store, prices, client_id = submitted_cycle
    broker.orders[client_id] = filled(broker.orders[client_id])
    save = store.save_execution_snapshot

    def fail_reconciliation(**kwargs):
        if kwargs["intents"][client_id]["state"] == "FILLED":
            raise StatePersistenceError("injected snapshot write failure")
        return save(**kwargs)

    monkeypatch.setattr(store, "save_execution_snapshot", fail_reconciliation)
    assert cycle.run_paper(prices, pd.Series(dtype=float), submit=True) == 1
    assert store.load_execution_snapshot().intents[client_id]["state"] == "SUBMITTED"
    assert broker.submissions == [client_id]
