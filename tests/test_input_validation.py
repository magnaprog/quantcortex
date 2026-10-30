"""Input validation for brokers, portfolio views, risk overlays, and order sizing."""

from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from quantcortex.execution.brokers.alpaca_broker import AlpacaBroker
from quantcortex.execution.brokers.base import BrokerError
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import OrderSide, OrderStatus
from quantcortex.execution.position_manager import PositionManager
from quantcortex.portfolio.black_litterman import BlackLitterman
from quantcortex.risk.vol_targeting import VolTargeting


def _alpaca_lookup(id_fields):
    raw = SimpleNamespace(
        **id_fields, symbol="AAPL", side="buy", qty="2", type="market",
        limit_price=None, status="partially_filled", filled_qty="0.5",
        filled_avg_price="100", client_order_id="synthetic-client-id",
    )
    broker = AlpacaBroker(base_url="https://paper-api.alpaca.markets")
    broker._api = SimpleNamespace(get_order_by_client_id=lambda client_order_id: raw)
    return broker.find_order_by_client_order_id("synthetic-client-id")


@pytest.mark.parametrize("id_fields", [
    {}, {"id": None}, {"id": False}, {"id": True}, {"id": ""}, {"id": "  "},
])
def test_alpaca_lookup_rejects_missing_or_boolean_order_ids(id_fields):
    with pytest.raises(BrokerError, match="broker order id"):
        _alpaca_lookup(id_fields)


@pytest.mark.parametrize("order_id", [
    " a1 ", UUID("12345678-1234-5678-1234-567812345678"),
])
def test_alpaca_lookup_preserves_valid_ids_and_partial_fill(order_id):
    order = _alpaca_lookup({"id": order_id})
    assert order.order_id == str(order_id).strip()
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.symbol == "AAPL"
    assert order.side is OrderSide.BUY
    assert order.quantity == 2.0
    assert order.filled_quantity == 0.5
    assert order.remaining_quantity == 1.5
    assert order.avg_fill_price == 100.0


def _account_rows(currency="USD", *, optional=True):
    values = {"NetLiquidation": "1000", "TotalCashValue": "100", "BuyingPower": "200"}
    if optional:
        values.update(AvailableFunds="150", GrossPositionValue="900")
    return [
        SimpleNamespace(account="DU_TEST", tag=tag, value=value,
                        currency=currency, modelCode="")
        for tag, value in values.items()
    ]


def _read_account(rows):
    broker = IBBroker(host="127.0.0.1", port=7497, client_id=1, account="DU_TEST")
    broker._ib = SimpleNamespace(accountSummary=lambda: rows)
    return broker.get_account()


@pytest.mark.parametrize("currency", ["USD", "EUR", "BASE"])
@pytest.mark.parametrize("optional", [True, False])
def test_ib_account_keeps_consistent_currency_values(currency, optional):
    rows = _account_rows(currency, optional=optional)
    rows.append(SimpleNamespace(account="DU_TEST", tag="AccountType",
                                value="INDIVIDUAL", currency="", modelCode=""))
    account = _read_account(rows)
    assert account.currency == currency
    assert (account.equity, account.cash, account.buying_power) == (1000, 100, 200)
    assert account.extra == {
        "account": "DU_TEST",
        "available_funds": 150.0 if optional else 0.0,
        "gross_position_value": 900.0 if optional else 0.0,
    }


@pytest.mark.parametrize("tag", [
    "NetLiquidation", "TotalCashValue", "BuyingPower", "AvailableFunds",
    "GrossPositionValue",
])
@pytest.mark.parametrize(("currency", "other"), [
    ("USD", "EUR"), ("USD", "BASE"), ("BASE", "USD"),
])
def test_ib_account_rejects_mixed_selected_currencies(tag, currency, other):
    rows = _account_rows(currency)
    next(row for row in rows if row.tag == tag).currency = other
    with pytest.raises(BrokerError, match="currenc"):
        _read_account(rows)


@pytest.mark.parametrize(("currency", "value"), [
    ("EUR", "100"), ("EUR", "101"), ("USD", "101"),
])
def test_ib_account_rejects_ambiguous_duplicate_tag(currency, value):
    rows = _account_rows()
    rows.append(SimpleNamespace(account="DU_TEST", tag="TotalCashValue",
                                value=value, currency=currency, modelCode=""))
    with pytest.raises(BrokerError, match="currenc"):
        _read_account(rows)


@pytest.mark.parametrize("currency", [None, "", "   "])
def test_ib_account_rejects_missing_monetary_currency(currency):
    rows = _account_rows()
    rows[1].currency = currency
    with pytest.raises(BrokerError, match="currenc"):
        _read_account(rows)


def test_ib_account_keeps_base_preference_without_inferring_an_iso_currency():
    rows = _account_rows("BASE")
    rows.extend([
        SimpleNamespace(account="DU_TEST", tag="TotalCashValue", value="70",
                        currency="USD", modelCode=""),
        SimpleNamespace(account="DU_TEST", tag="TotalCashValue", value="30",
                        currency="EUR", modelCode=""),
    ])
    account = _read_account(rows)
    assert account.currency == "BASE"
    assert account.cash == 100.0


def test_ib_account_accepts_identical_duplicate_and_normalized_currency():
    rows = _account_rows(" usd ")
    rows.append(SimpleNamespace(account="DU_TEST", tag="TotalCashValue",
                                value="100", currency="USD", modelCode=""))
    account = _read_account(rows)
    assert account.currency == "USD"
    assert account.cash == 100.0


@pytest.mark.parametrize("missing", ["NetLiquidation", "TotalCashValue", "BuyingPower"])
def test_ib_account_still_requires_each_primary_tag(missing):
    rows = [row for row in _account_rows() if row.tag != missing]
    with pytest.raises(BrokerError, match="missing required tag"):
        _read_account(rows)


@pytest.mark.parametrize("missing", [np.nan, None])
@pytest.mark.parametrize("dataframe", [True, False])
def test_black_litterman_rejects_explicit_missing_view_coefficients(missing, dataframe):
    views = [[1.0, missing]]
    if dataframe:
        views = pd.DataFrame(views, columns=["A", "B"])
    with pytest.raises(ValueError, match="finite"):
        BlackLitterman._resolve_views(views, None, [0.03], ["A", "B"], np.eye(2))


@pytest.mark.parametrize("views", [
    pd.DataFrame({"A": [1.0]}),
    pd.DataFrame({"B": [0.0], "A": [1.0]}),
    np.array([[1.0, 0.0]]),
])
def test_black_litterman_omitted_asset_and_explicit_zero_have_same_view(views):
    p, q, omega = BlackLitterman._resolve_views(
        views, None, [0.03], ["A", "B"], np.diag([0.04, 0.09]),
    )
    np.testing.assert_array_equal(p, [[1.0, 0.0]])
    np.testing.assert_array_equal(q, [0.03])
    np.testing.assert_allclose(omega, [[0.04]])


def test_black_litterman_finite_relative_view_keeps_labels_and_uncertainty():
    p, q, omega = BlackLitterman._resolve_views(
        pd.DataFrame({"B": [-1.0], "A": [1.0]}), [0.5], [0.03],
        ["A", "B"], np.diag([0.04, 0.09]),
    )
    np.testing.assert_array_equal(p, [[1.0, -1.0]])
    np.testing.assert_array_equal(q, [0.03])
    np.testing.assert_allclose(omega, [[0.13]])


@pytest.mark.parametrize("views", [pd.DataFrame({"A": [1.0]}), [[1.0, 0.0]]])
def test_black_litterman_valid_view_keeps_allocation_with_fixed_covariance(monkeypatch, views):
    # No estimator is fit: the identity covariance gives posterior [1.875, 1.25].
    optimizer = BlackLitterman()
    monkeypatch.setattr(optimizer, "_ledoit_wolf_cov", lambda returns: np.eye(2))
    returns = pd.DataFrame({"A": [0.01, -0.01], "B": [-0.01, 0.01]})
    weights = optimizer.optimize(returns, views=views, q=[2.5])
    np.testing.assert_allclose(weights, [0.6, 0.4])
    np.testing.assert_allclose(optimizer.optimize(returns), [0.5, 0.5])


@pytest.mark.parametrize("realized_vol", [0.0, 0.5e-6, 1e-6])
@pytest.mark.parametrize(("cap", "expected", "scale"), [
    (0.5, [0.1, -0.05], 0.5),
    (1.0, [0.2, -0.1], 1.0),
    (2.0, [0.2, -0.1], 1.0),
])
def test_vol_targeting_low_vol_respects_scalar_cap_and_no_increase(realized_vol, cap, expected, scale):
    overlay = VolTargeting(max_leverage=cap)
    result = overlay.apply([0.2, -0.1], realized_vol=realized_vol)
    np.testing.assert_allclose(result, expected)
    assert overlay.last_scale == scale
    assert overlay.last_realized_vol == realized_vol


@pytest.mark.parametrize(("weights", "cap", "vol", "expected", "scale"), [
    ([0.2, -0.1], 0.5, 1.000001e-6, [0.1, -0.05], 0.5),
    ([0.8, 0.4], 0.5, 0.0, [1 / 3, 1 / 6], 5 / 12),
    ([2.0, 1.0], 2.0, 0.0, [1.0, 0.5], 0.5),
])
def test_vol_targeting_retains_above_floor_gross_and_asset_caps(weights, cap, vol, expected, scale):
    overlay = VolTargeting(max_leverage=cap)
    np.testing.assert_allclose(overlay.apply(weights, realized_vol=vol), expected)
    assert overlay.last_scale == pytest.approx(scale)


@pytest.mark.parametrize("quantity", [
    1_000_000.0005, -1_000_000.0005, 1_000_000_000.5, -1_000_000_000.5,
])
def test_whole_share_mode_rejects_large_fractional_positions(quantity):
    positions = {"A": quantity}
    with pytest.raises(ValueError, match="allow_fractional=True"):
        PositionManager().target_weights_to_orders(
            {"A": 0.0}, {"A": 1.0}, 100.0, current_positions=positions,
        )
    assert positions == {"A": quantity}


@pytest.mark.parametrize(("quantity", "fractional", "expected"), [
    (1_000_000.0, False, 1_000_000.0),
    (-1_000_000.0, False, 1_000_000.0),
    (1.0 + 0.5e-9, False, 1.0),
    (1_000_000_000.5, True, 1_000_000_000.5),
    (-1_000_000_000.5, True, 1_000_000_000.5),
])
def test_order_sizing_keeps_whole_share_and_opted_in_fractional_quantities(quantity, fractional, expected):
    orders = PositionManager().target_weights_to_orders(
        {"A": 0.0}, {"A": 1.0}, 100.0, current_positions={"A": quantity},
        allow_fractional=fractional,
    )
    assert orders == [{
        "symbol": "A", "quantity": expected,
        "side": OrderSide.SELL if quantity > 0 else OrderSide.BUY,
    }]


def test_whole_share_absolute_tolerance_still_rejects_small_fraction():
    with pytest.raises(ValueError, match="allow_fractional=True"):
        PositionManager().target_weights_to_orders(
            {"A": 0.0}, {"A": 1.0}, 100.0, current_positions={"A": 1.0 + 2e-9},
        )
