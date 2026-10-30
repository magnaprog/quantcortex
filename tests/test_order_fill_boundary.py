"""A fill near the completion tolerance must have a valid lifecycle status."""

import math
from types import SimpleNamespace

import pytest

from quantcortex.execution.brokers.ccxt_broker import CCXTBroker
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import (
    Order,
    OrderError,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
)


@pytest.mark.parametrize(
    ("filled", "expected"),
    [
        (math.nextafter(5.0 - 1e-9, 0.0), OrderStatus.PARTIALLY_FILLED),
        (5.0 - 1e-9, OrderStatus.PARTIALLY_FILLED),
        (math.nextafter(5.0 - 1e-9, 5.0), OrderStatus.FILLED),
    ],
)
@pytest.mark.parametrize("route", ["ccxt", "ib", "manager", "split_fill"])
def test_fill_status_at_tolerance_boundary(filled, expected, route):
    if route == "ccxt":
        order = CCXTBroker()._to_order(
            {"id": "boundary", "status": "open", "filled": filled, "average": 100.0},
            "AAA/BBB", OrderSide.BUY, 5.0, OrderType.MARKET, None,
        )
    elif route == "ib":
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=1),
            orderStatus=SimpleNamespace(
                status="Submitted", filled=filled, avgFillPrice=100.0,
            ),
        )
        order = IBBroker()._to_order(
            trade, "AAA", OrderSide.BUY, 5.0, OrderType.MARKET, None,
        )
    else:
        manager = OrderManager()
        order = Order("boundary", "AAA", OrderSide.BUY, 5.0)
        manager.register(order)
        manager.submit(order.order_id)
        first_fill = 0.1 if route == "split_fill" else 0.0
        if first_fill:
            manager.fill(order.order_id, first_fill, 100.0)
        order = manager.fill(order.order_id, filled - first_fill, 100.0)

    assert order.status is expected
    assert order.filled_quantity == filled
    assert order.avg_fill_price == pytest.approx(100.0)
    order.validate()
    opposite = (
        OrderStatus.FILLED
        if expected is OrderStatus.PARTIALLY_FILLED
        else OrderStatus.PARTIALLY_FILLED
    )
    with pytest.raises(OrderError, match="must report"):
        Order("wrong-status", "AAA", OrderSide.BUY, 5.0,
              status=opposite, filled_quantity=filled, avg_fill_price=100.0)
