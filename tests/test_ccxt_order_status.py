"""Translate exchange order status and cumulative fills without venue access."""

from types import SimpleNamespace

import pytest

from quantcortex.execution.brokers.ccxt_broker import CCXTBroker
from quantcortex.execution.order_manager import OrderError, OrderSide, OrderStatus


def submit_response(status, filled):
    broker = CCXTBroker()
    broker._exchange = SimpleNamespace(
        create_order=lambda *args: {
            "id": "status-example",
            "status": status,
            "filled": filled,
            "average": 100.0 if filled else None,
        }
    )
    return broker.submit_order("AAA/BBB", OrderSide.BUY, 5.0)


@pytest.mark.parametrize(
    ("filled", "expected"),
    [
        (None, OrderStatus.SUBMITTED),
        (0.0, OrderStatus.SUBMITTED),
        (2.0, OrderStatus.PARTIALLY_FILLED),
        (5.0 - 2e-9, OrderStatus.PARTIALLY_FILLED),
        (5.0 - 5e-10, OrderStatus.FILLED),
        (5.0, OrderStatus.FILLED),
    ],
)
def test_open_order_status_reflects_cumulative_fill(filled, expected):
    order = submit_response("open", filled)

    assert order.status is expected
    assert order.filled_quantity == (filled or 0.0)
    assert order.remaining_quantity == pytest.approx(5.0 - (filled or 0.0))
    assert order.avg_fill_price == (100.0 if filled else None)
    assert order.history == (
        [OrderStatus.SUBMITTED]
        if expected is OrderStatus.SUBMITTED
        else [OrderStatus.SUBMITTED, expected]
    )
    order.validate()


@pytest.mark.parametrize("filled", [0.0, 2.0])
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("canceled", OrderStatus.CANCELLED),
        ("cancelled", OrderStatus.CANCELLED),
        ("expired", OrderStatus.CANCELLED),
        ("rejected", OrderStatus.REJECTED),
    ],
)
def test_terminal_status_is_preserved_with_partial_fill(status, expected, filled):
    order = submit_response(status, filled)

    assert order.status is expected
    assert order.filled_quantity == filled
    assert order.history == [OrderStatus.SUBMITTED, expected]
    order.validate()


def test_closed_order_requires_complete_fill():
    assert submit_response("closed", 5.0).status is OrderStatus.FILLED
    with pytest.raises(OrderError, match="full filled quantity"):
        submit_response("closed", 2.0)


@pytest.mark.parametrize("filled", [-1.0, 5.0 + 2e-9])
def test_open_order_rejects_invalid_fill_quantity(filled):
    with pytest.raises(OrderError, match="within order quantity"):
        submit_response("open", filled)
