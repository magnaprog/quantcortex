"""Metric boundaries, broker identifiers, and persisted submission intents."""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.backtest.engines.vectorized import BacktestResult
from quantcortex.backtest.metrics.tearsheet import Tearsheet
from quantcortex.backtest.validation.deflated_sharpe import sharpe_ratio
from quantcortex.backtest.validation.multiple_testing import MultipleTestingReport
from quantcortex.execution.brokers.base import BrokerError
from quantcortex.execution.brokers.ccxt_broker import CCXTBroker
from quantcortex.execution.brokers.ib_broker import IBBroker
from quantcortex.execution.order_manager import Order, OrderSide, OrderStatus, OrderType
from quantcortex.execution.state_persistence import (
    StatePersistence,
    StatePersistenceError,
)
from scripts.paper_trade_cycle import (
    SUBMISSION_LEDGER_KEY,
    _initialise_submission_state,
    _reconcile_or_submit,
    make_client_order_id,
)


def _result(returns, periods_per_year):
    returns = pd.Series(returns, dtype=float)
    return BacktestResult(
        returns=returns,
        equity_curve=(1.0 + returns).cumprod(),
        weights=pd.DataFrame({"A": np.ones(len(returns))}),
        gross_returns=returns,
        costs=returns * 0.0,
        turnover=returns * 0.0,
        metadata={"periods_per_year": periods_per_year},
    )


@pytest.mark.parametrize("returns", [[-1.0], [0.1, -1.0], [-1.0, 0.0]])
@pytest.mark.parametrize("periods_per_year", [1, 252])
def test_zero_terminal_wealth_has_minus_one_cagr(returns, periods_per_year):
    result = _result(returns, periods_per_year)
    assert result.total_return == -1.0
    assert result.summary()["cagr"] == -1.0
    assert (
        Tearsheet(result.returns, periods_per_year=periods_per_year).compute()["cagr"]
        == -1.0
    )


def test_positive_wealth_and_empty_metric_conventions_remain():
    result = _result([0.1, -0.1], 2)
    assert result.summary()["cagr"] == pytest.approx(-0.01)
    assert Tearsheet(result.returns, periods_per_year=2).compute()[
        "cagr"
    ] == pytest.approx(-0.01)
    empty = _result([], 252)
    assert empty.summary()["cagr"] == 0.0
    assert np.isnan(Tearsheet(empty.returns).compute()["cagr"])


@pytest.mark.parametrize(
    "returns,periods_per_year,expected_cagr",
    [
        pytest.param([-0.5] * 60, 1, -0.5, id="annual-halves"),
        pytest.param(
            [-0.01] * 4000, 252, 0.99**252 - 1.0, id="daily-one-percent-loss"
        ),
    ],
)
def test_cagr_retains_small_positive_compound_wealth(
    returns, periods_per_year, expected_cagr
):
    result = _result(returns, periods_per_year)
    assert result.equity_curve.iloc[-1] > 0.0
    assert result.total_return == -1.0  # Subtraction loses the positive growth.
    # Constant periodic returns give an oracle without the compound-growth product.
    assert result.summary()["cagr"] == pytest.approx(expected_cagr)
    assert Tearsheet(result.returns, periods_per_year=periods_per_year).compute()[
        "cagr"
    ] == pytest.approx(expected_cagr)


@pytest.mark.parametrize(
    "returns,expected_cagr",
    [
        pytest.param([-0.5] * 2000, -0.5, id="compound-product-underflow"),
        pytest.param([1e-20] * 100, 1e-20, id="growth-factor-rounding"),
    ],
)
def test_cagr_annualizes_returns_without_rounding_the_wealth_product(
    returns, expected_cagr
):
    result = _result(returns, periods_per_year=1)
    # For constant annual returns, CAGR equals that return at every horizon.
    assert result.summary()["cagr"] == pytest.approx(expected_cagr, rel=1e-12, abs=0)
    assert Tearsheet(result.returns, periods_per_year=1).compute()[
        "cagr"
    ] == pytest.approx(expected_cagr, rel=1e-12, abs=0)


def test_summary_cagr_accepts_numeric_object_series():
    result = _result([0.1, -0.1], periods_per_year=2)
    result.returns = result.returns.astype(object)
    assert result.summary()["cagr"] == pytest.approx(-0.01)


@pytest.mark.parametrize("observations", [10, 252])
@pytest.mark.parametrize("level", [0.001, -0.001])
@pytest.mark.parametrize("variation", [0.0, 1e-14])
@pytest.mark.parametrize("metric", ["helper", "annualized_helper", "tearsheet", "summary"])
def test_generic_sharpe_constant_and_tiny_variation(
    observations, level, variation, metric,
):
    returns = pd.Series(level, index=pd.RangeIndex(observations))
    returns.iloc[-1] += variation
    if metric == "helper":
        actual = sharpe_ratio(returns)
    elif metric == "annualized_helper":
        actual = sharpe_ratio(returns, periods_per_year=252)
    elif metric == "tearsheet":
        actual = Tearsheet(returns).compute()["sharpe"]
    else:
        summary = _result(returns, 252).summary()
        actual = summary["sharpe"]

    if variation == 0.0:
        if metric == "summary":
            assert actual == 0.0
            assert summary["ann_vol"] == 0.0
        else:
            assert np.isnan(actual)
    else:
        # One perturbed observation has sample sd |delta| / sqrt(n).
        delta = returns.iloc[-1] - level
        expected_sd = abs(delta) / np.sqrt(observations)
        expected = (level + delta / observations) / expected_sd
        if metric != "helper":
            expected *= np.sqrt(252.0)
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected, rel=1e-6)
        if metric == "summary":
            assert summary["ann_vol"] == pytest.approx(
                expected_sd * np.sqrt(252.0), rel=1e-6, abs=0,
            )


def test_tearsheet_constancy_is_checked_on_excess_returns():
    cash = pd.Series(np.arange(10) / 128.0)
    assert np.isnan(Tearsheet(cash + 0.125, risk_free=cash).compute()["sharpe"])
    # Constant raw returns still have a defined Sharpe against varying cash.
    returns = pd.Series(0.125, index=cash.index)
    excess = returns - cash
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252.0)
    assert Tearsheet(returns, risk_free=cash).compute()["sharpe"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad_id", [None, "", "   ", True, False, np.nan, np.inf, 1.5, {}, []]
)
def test_ccxt_response_without_usable_identifier_is_rejected(bad_id):
    broker = CCXTBroker()
    with pytest.raises(BrokerError, match="order id"):
        broker._to_order(
            {"id": bad_id, "status": "open", "filled": 0.0},
            "BTC/USDT",
            OrderSide.BUY,
            1.0,
            OrderType.MARKET,
            None,
        )


@pytest.mark.parametrize("venue_id", ["venue-123", "0", 0, 2**63 - 1])
def test_ccxt_venue_identifier_is_stable_and_exact(venue_id):
    broker = CCXTBroker()
    raw = {"id": venue_id, "status": "open", "filled": 0.0}
    orders = [
        broker._to_order(raw, "BTC/USDT", OrderSide.BUY, 1.0, OrderType.MARKET, None)
        for _ in range(2)
    ]
    assert [order.order_id for order in orders] == [str(venue_id)] * 2


def _ib_trade(order_id=None, perm_id=None):
    return SimpleNamespace(
        order=SimpleNamespace(orderId=order_id, permId=perm_id),
        orderStatus=SimpleNamespace(status="Submitted", filled=0.0, avgFillPrice=0.0),
    )


@pytest.mark.parametrize(
    "order_id,perm_id",
    [
        (None, None),
        (True, 0),
        (np.nan, 0),
        (1.5, 0),
        ("", 0),
        (None, True),
        (None, np.inf),
        (None, -5),
    ],
)
def test_ib_response_without_usable_identifier_is_rejected(order_id, perm_id):
    with pytest.raises(BrokerError, match="order id"):
        IBBroker()._to_order(
            _ib_trade(order_id, perm_id),
            "AAPL",
            OrderSide.BUY,
            1.0,
            OrderType.MARKET,
            None,
        )


@pytest.mark.parametrize(
    "order_id,perm_id,expected",
    [
        (7, 0, "7"),
        (0, 0, "0"),
        (-7, 0, "-7"),
        (0, 12345, "12345"),
        (None, 12345, "12345"),
        (7, 12345, "7"),
        (0, 2**63 - 1, str(2**63 - 1)),
    ],
)
def test_ib_preserves_api_and_permanent_identifier_semantics(
    order_id, perm_id, expected
):
    trade = _ib_trade(order_id, perm_id)
    broker = IBBroker()
    orders = [
        broker._to_order(trade, "AAPL", OrderSide.BUY, 1.0, OrderType.MARKET, None)
        for _ in range(2)
    ]
    assert [order.order_id for order in orders] == [expected, expected]


def test_ib_submission_preserves_zero_allocated_by_client():
    class Client:
        def qualifyContracts(self, contract):
            return [contract]

        def placeOrder(self, contract, order):
            # ib_async assigns getReqId() to the returned Trade's Order and
            # initially reports PendingSubmit, before a permId is available.
            order.orderId = 0
            order.clientId = 1
            self.sent = order
            return SimpleNamespace(
                order=order,
                orderStatus=SimpleNamespace(
                    status="PendingSubmit",
                    filled=0.0,
                    avgFillPrice=0.0,
                ),
            )

    broker = IBBroker()
    broker._ib = Client()
    broker._sdk = {
        "Stock": lambda symbol, exchange, currency: SimpleNamespace(symbol=symbol),
        "MarketOrder": lambda action, quantity: SimpleNamespace(
            orderId=0,
            permId=0,
            action=action,
            totalQuantity=quantity,
        ),
    }
    returned = broker.submit_order("AAPL", OrderSide.BUY, 1.0)
    assert broker._ib.sent.orderId == 0
    assert returned.order_id == "0"
    assert returned.status is OrderStatus.SUBMITTED


class _PaperBroker:
    def __init__(self, existing=None):
        self.existing = existing
        self.submissions = []

    def find_order_by_client_order_id(self, client_id):
        return self.existing

    def submit_order(self, symbol, side, quantity, **kwargs):
        self.submissions.append(kwargs["client_order_id"])
        return Order(
            order_id="broker-1",
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=OrderStatus.SUBMITTED,
        )


@pytest.mark.parametrize(
    "state", ["UNKNOWN", "ATTEMPTING", "NEW", "SUBMITTED", "PARTIALLY_FILLED"]
)
@pytest.mark.parametrize("change", ["quantity", "date", "side", "symbol_case"])
@pytest.mark.parametrize("retry_confirmed_missing", [False, True])
def test_other_unresolved_identifier_blocks_replacement_after_restart(
    tmp_path, state, change, retry_confirmed_missing
):
    path = tmp_path / "state.json"
    original = {"symbol": "QQQ", "side": "BUY", "quantity": 5.0}
    original_date = pd.Timestamp("2024-01-02")
    original_id = make_client_order_id(original_date, original)
    ledger = {original_id: {**original, "state": state}}
    StatePersistence(file_path=path).save_state(SUBMISSION_LEDGER_KEY, ledger)
    store = StatePersistence(file_path=path)
    loaded = store.load_state(SUBMISSION_LEDGER_KEY)
    intent = dict(original)
    as_of = original_date
    if change == "quantity":
        intent["quantity"] = 6.0
    elif change == "date":
        as_of += pd.Timedelta(days=1)
    elif change == "side":
        intent["side"] = "SELL"
    else:
        intent["symbol"] = "qqq"
    assert make_client_order_id(as_of, intent) != original_id
    broker = _PaperBroker()
    with pytest.raises(BrokerError, match="Unresolved.*QQQ"):
        _reconcile_or_submit(
            broker,
            store,
            loaded,
            as_of,
            intent,
            retry_confirmed_missing=retry_confirmed_missing,
        )
    assert broker.submissions == []
    assert loaded == ledger
    assert store.load_state(SUBMISSION_LEDGER_KEY) == ledger


@pytest.mark.parametrize("state", ["FILLED", "CANCELLED", "REJECTED"])
def test_terminal_prior_intent_does_not_block_new_submission(tmp_path, state):
    store = StatePersistence(file_path=tmp_path / "state.json")
    ledger = {"old-id": {"symbol": "QQQ", "state": state}}
    broker = _PaperBroker()
    _, submitted, client_id = _reconcile_or_submit(
        broker,
        store,
        ledger,
        "2024-01-03",
        {"symbol": "QQQ", "side": "BUY", "quantity": 1.0},
    )
    assert submitted
    assert broker.submissions == [client_id]
    assert ledger["old-id"]["state"] == state


def test_unresolved_other_symbol_does_not_block_submission(tmp_path):
    store = StatePersistence(file_path=tmp_path / "state.json")
    ledger = {"old-id": {"symbol": "TLT", "state": "UNKNOWN"}}
    broker = _PaperBroker()
    _, submitted, _ = _reconcile_or_submit(
        broker,
        store,
        ledger,
        "2024-01-03",
        {"symbol": "QQQ", "side": "BUY", "quantity": 1.0},
    )
    assert submitted
    assert ledger["old-id"]["state"] == "UNKNOWN"


def test_existing_client_id_can_reconcile_without_new_submission(tmp_path):
    store = StatePersistence(file_path=tmp_path / "state.json")
    existing = Order(
        order_id="broker-1",
        symbol="QQQ",
        side="BUY",
        quantity=1.0,
        status=OrderStatus.CANCELLED,
    )
    broker = _PaperBroker(existing=existing)
    ledger = {"other-id": {"symbol": "QQQ", "state": "UNKNOWN"}}
    order, submitted, client_id = _reconcile_or_submit(
        broker,
        store,
        ledger,
        "2024-01-03",
        {"symbol": "QQQ", "side": "BUY", "quantity": 1.0},
    )
    assert order is existing
    assert not submitted
    assert broker.submissions == []
    assert store.load_state(SUBMISSION_LEDGER_KEY)[client_id]["state"] == "CANCELLED"
    assert ledger["other-id"]["state"] == "UNKNOWN"


def test_replacement_block_preserves_coherent_snapshot_revision(tmp_path):
    path = tmp_path / "state.json"
    store = StatePersistence(file_path=path)
    store.save_execution_snapshot(
        positions={},
        orders=[],
        intents={"old-id": {"symbol": "QQQ", "state": "UNKNOWN"}},
        metadata={},
    )
    restarted = StatePersistence(file_path=path)
    ledger, context = _initialise_submission_state(
        restarted,
        positions={},
        orders=[],
        metadata={},
    )
    revision = context["revision"]
    before = copy.deepcopy(ledger)
    broker = _PaperBroker()
    with pytest.raises(BrokerError, match="Unresolved.*QQQ"):
        _reconcile_or_submit(
            broker,
            restarted,
            ledger,
            "2024-01-03",
            {"symbol": "QQQ", "side": "BUY", "quantity": 1.0},
            snapshot_context=context,
        )
    after = restarted.load_execution_snapshot()
    assert broker.submissions == []
    assert after.revision == revision
    assert context["revision"] == revision
    assert after.intents == before == ledger


@pytest.mark.parametrize(
    "record", [None, {"state": "UNKNOWN"}, {"symbol": "", "state": "UNKNOWN"}]
)
def test_unclassifiable_prior_intent_fails_before_submission(tmp_path, record):
    store = StatePersistence(file_path=tmp_path / "state.json")
    broker = _PaperBroker()
    ledger = {"other-id": record}
    before = copy.deepcopy(ledger)
    with pytest.raises(StatePersistenceError, match="ledger"):
        _reconcile_or_submit(
            broker,
            store,
            ledger,
            "2024-01-03",
            {"symbol": "QQQ", "side": "BUY", "quantity": 1.0},
        )
    assert broker.submissions == []
    assert ledger == before


def test_multiple_testing_summary_names_by_and_states_dependence_limits():
    report = MultipleTestingReport([0.001, 0.02, 0.9])
    text = report.summary()
    assert "Benjamini-Yekutieli (FDR)" in text
    assert "Benjamini-Hochberg-Yekutieli" not in text
    assert "PRDS" in text
    assert "arbitrary dependence" in text
    assert "valid p-values" in text
