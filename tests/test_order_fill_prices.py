"""Incremental fills must not invent prices for unknown execution history."""

from copy import deepcopy

import pytest

from quantcortex.execution.order_manager import (
    Order,
    OrderError,
    OrderManager,
    OrderSide,
    OrderStatus,
)


@pytest.mark.parametrize("first_price, second_price", [(None, 100.0), (100.0, None)])
@pytest.mark.parametrize("increment", [1.0, None])
def test_mixed_fill_prices_reject_before_mutating_order(
    first_price, second_price, increment
):
    manager = OrderManager()
    order = manager.create_order("o1", "SYNTH", OrderSide.BUY, 3.0)
    manager.submit("o1")
    manager.fill("o1", 1.0, fill_price=first_price)
    before = deepcopy(order)

    with pytest.raises(OrderError, match="priced and unpriced"):
        manager.fill("o1", increment, fill_price=second_price)

    assert manager.get("o1") is order
    assert order == before
    manager.fill("o1", fill_price=first_price)
    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == 3.0
    assert order.avg_fill_price == first_price


@pytest.mark.parametrize("average, next_price", [(None, 100.0), (100.0, None)])
def test_registered_partial_fill_rejects_mixed_price_history(average, next_price):
    manager = OrderManager()
    order = manager.register(
        Order(
            order_id="o1",
            symbol="SYNTH",
            side=OrderSide.BUY,
            quantity=2.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=1.0,
            avg_fill_price=average,
        )
    )
    before = deepcopy(order)

    with pytest.raises(OrderError, match="priced and unpriced"):
        manager.fill("o1", fill_price=next_price)

    assert order == before


def test_unpriced_fills_keep_cumulative_average_unknown():
    manager = OrderManager()
    order = manager.create_order("o1", "SYNTH", OrderSide.BUY, 3.0)
    manager.submit("o1")

    manager.fill("o1", 1.0)
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price is None
    assert order.status is OrderStatus.PARTIALLY_FILLED
    manager.fill("o1")
    assert order.filled_quantity == 3.0
    assert order.avg_fill_price is None
    assert order.history == [
        OrderStatus.NEW,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    ]


def test_priced_fills_use_quantity_weighted_cumulative_average():
    manager = OrderManager()
    order = manager.create_order("o1", "SYNTH", OrderSide.BUY, 3.0)
    manager.submit("o1")

    manager.fill("o1", 1.0, fill_price=100.0)
    assert order.filled_quantity == 1.0
    assert order.avg_fill_price == 100.0
    manager.fill("o1", fill_price=130.0)
    assert order.filled_quantity == 3.0
    assert order.avg_fill_price == pytest.approx(120.0)
    assert order.status is OrderStatus.FILLED


def test_first_unpriced_fill_ignores_price_on_zero_filled_snapshot():
    manager = OrderManager()
    order = manager.register(
        Order(
            order_id="o1",
            symbol="SYNTH",
            side=OrderSide.BUY,
            quantity=2.0,
            status=OrderStatus.SUBMITTED,
            avg_fill_price=100.0,
        )
    )

    manager.fill("o1", 1.0)
    assert order.avg_fill_price is None
    manager.fill("o1")
    assert order.filled_quantity == 2.0
    assert order.avg_fill_price is None
    assert order.status is OrderStatus.FILLED
