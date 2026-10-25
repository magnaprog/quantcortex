"""Regression checks for exposure, causality, factor data, and local caching."""

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.alpha.factors.classical._fundamentals import (
    pit_panel,
    validate_fundamentals,
)
from quantcortex.alpha.factors.nlp.news_scorer import NewsScorer
from quantcortex.alpha.validation.alphalens_report import (
    compute_information_coefficient,
)
from quantcortex.data.local_csv import LocalDataError, load_price_matrix
from quantcortex.data.providers.fred_provider import FREDProvider
from quantcortex.data.storage.redis_cache import RedisCache
from quantcortex.portfolio.base import PortfolioMode, WeightContractViolationError
from quantcortex.portfolio.equal_weight import EqualWeight
from quantcortex.strategies.base_strategy import Strategy, StrategyContext
from quantcortex.strategies.drl_portfolio import DRLPortfolioStrategy
from quantcortex.strategies.momentum_ml import MomentumMLStrategy


def _context():
    dates = pd.bdate_range(end="2024-06-14", periods=360)
    t = np.arange(len(dates))
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(0.0005 * t + 0.001 * np.sin(t)),
            "B": 80 * np.exp(0.0002 * t + 0.003 * np.cos(t)),
        },
        index=dates,
    )
    return StrategyContext(
        dates[-1], prices, prices.pct_change(fill_method=None).fillna(0.0)
    )


class _FixedStrategy(Strategy):
    def select(self, ctx):
        return pd.Series({"A": 1.0, "B": 0.5})

    def allocate(self, scores, ctx):
        if self.mode is PortfolioMode.MARKET_NEUTRAL:
            return np.array([0.5, -0.5])
        return np.array([0.6, 0.4])


@pytest.mark.parametrize("overlay_kind", ["timing_overlays", "risk_overlays"])
def test_long_only_rejects_final_short_exposure(overlay_kind):
    strategy = _FixedStrategy(
        EqualWeight(), **{overlay_kind: [lambda weights, ctx: -weights]}
    )

    with pytest.raises(WeightContractViolationError):
        strategy.rebalance(_context())


@pytest.mark.parametrize("scale", [0.0, 0.4, 1.0])
def test_long_only_valid_final_weights_remain_unchanged(scale):
    strategy = _FixedStrategy(
        EqualWeight(), timing_overlays=[lambda weights, ctx: weights * scale]
    )

    result = strategy.rebalance(_context())

    np.testing.assert_array_equal(result.allocation_weights, [0.6, 0.4])
    np.testing.assert_array_equal(result.target_weights, np.array([0.6, 0.4]) * scale)


def test_market_neutral_keeps_relaxed_signed_exposure_contract():
    expected = np.array([-0.2, 0.4])
    strategy = _FixedStrategy(
        SimpleNamespace(mode=PortfolioMode.MARKET_NEUTRAL),
        mode=PortfolioMode.MARKET_NEUTRAL,
        risk_overlays=[lambda weights, ctx: expected.copy()],
    )

    np.testing.assert_array_equal(strategy.rebalance(_context()).target_weights, expected)


def _cached_strategy(kind, fitted_at, monkeypatch):
    calls = []
    if kind == "ml":
        strategy = MomentumMLStrategy()

        def predict(features):
            calls.append("predict")
            return np.ones(len(features))

        strategy._model = SimpleNamespace(predict=predict)
        strategy._feature_cols = ["mom_21"]
        strategy._last_refit = fitted_at
    else:
        strategy = DRLPortfolioStrategy()
        strategy.optimizer.model = object()
        strategy.optimizer._asset_names = ("A", "B")
        strategy._trained = True
        strategy._last_train = fitted_at

        def optimize(returns, **kwargs):
            calls.append("optimize")
            return np.array([0.5, 0.5])

        def unexpected_train(returns):
            raise AssertionError("existing same-period cache should not retrain")

        monkeypatch.setattr(strategy.optimizer, "optimize", optimize)
        monkeypatch.setattr(strategy.optimizer, "train", unexpected_train)
    return strategy, calls


def _evaluate_cached_strategy(strategy, ctx):
    if isinstance(strategy, MomentumMLStrategy):
        return strategy.select(ctx)
    ctx.extra["current_weights"] = pd.Series({"A": 0.5, "B": 0.5})
    return strategy.rebalance(ctx)


@pytest.mark.parametrize("kind", ["ml", "drl"])
@pytest.mark.parametrize("future_days", [1, 400])
def test_cached_model_rejects_decision_before_fit(kind, future_days, monkeypatch):
    ctx = _context()
    strategy, calls = _cached_strategy(
        kind, ctx.as_of + pd.Timedelta(days=future_days), monkeypatch
    )

    with pytest.raises(ValueError, match="as_of.*before.*fit"):
        _evaluate_cached_strategy(strategy, ctx)
    assert calls == []


@pytest.mark.parametrize("kind", ["ml", "drl"])
@pytest.mark.parametrize("days_since_fit", [0, 1])
def test_cached_model_keeps_valid_same_period_decisions(kind, days_since_fit, monkeypatch):
    ctx = _context()
    strategy, calls = _cached_strategy(
        kind, ctx.as_of - pd.Timedelta(days=days_since_fit), monkeypatch
    )

    _evaluate_cached_strategy(strategy, ctx)

    assert calls == ["predict" if kind == "ml" else "optimize"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("time_decay", [False, True])
def test_news_deduplication_keeps_earliest_available_copy(reverse, time_decay):
    scorer = NewsScorer(
        sentiment=SimpleNamespace(score=lambda texts: np.ones(len(texts)))
    )
    news = pd.DataFrame(
        {
            "date": ["2024-01-05T18:00:00Z", "2024-01-05T09:00:00Z"],
            "symbol": ["A", "A"],
            "headline": ["same headline", "same headline"],
        }
    )
    if reverse:
        news = news.iloc[::-1]

    result = scorer.aggregate_daily(
        news, as_of="2024-01-05T12:00:00Z", time_decay=time_decay
    )
    before_publication = scorer.aggregate_daily(
        news, as_of="2024-01-05T08:00:00Z", time_decay=time_decay
    )

    assert result.iloc[0, 0] == 1.0
    assert pd.isna(before_publication.iloc[0, 0])


@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("changed_side", ["factor", "returns"])
@pytest.mark.parametrize("scale,offset", [(1e-12, 0.0), (1.0, 1e6)])
def test_ic_does_not_treat_rescaled_nonconstant_values_as_constant(
    method, changed_side, scale, offset
):
    index = pd.DatetimeIndex(["2024-01-05"])
    factor = pd.DataFrame([[1.0, 2.0, 3.0]], index=index)
    returns = pd.DataFrame([[3.0, 2.0, 1.0]], index=index)
    if changed_side == "factor":
        factor = factor * scale + offset
    else:
        returns = returns * scale + offset

    result = compute_information_coefficient(factor, returns, method=method)

    assert result.iloc[0] == pytest.approx(-1.0)


@pytest.mark.parametrize("method", ["spearman", "pearson"])
@pytest.mark.parametrize("constant_side", ["factor", "returns"])
def test_ic_constant_input_remains_undefined(method, constant_side):
    index = pd.DatetimeIndex(["2024-01-05"])
    values = pd.DataFrame([[1.0, 2.0, 3.0]], index=index)
    constant = values * 0.0 + 1e-12
    factor, returns = (constant, values) if constant_side == "factor" else (values, constant)

    assert pd.isna(compute_information_coefficient(factor, returns, method=method).iloc[0])


@pytest.mark.parametrize("symbols", [["A", " A"], [" A", "A "]])
def test_requested_symbols_must_be_unique_after_trimming(tmp_path, symbols):
    path = tmp_path / "prices.csv"
    path.write_text("date,A\n2024-01-01,100\n2024-01-02,101\n")

    with pytest.raises(LocalDataError, match="unique"):
        load_price_matrix(path, symbols=symbols)


def test_single_trimmed_symbol_still_loads(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,A\n2024-01-01,100\n2024-01-02,101\n")

    result = load_price_matrix(path, symbols=[" A "])

    assert result.columns.tolist() == ["A"]
    assert result.A.tolist() == [100.0, 101.0]


def test_local_cache_get_is_atomic_with_concurrent_delete():
    cache = RedisCache()
    cache.set("a", 1)
    released = threading.Event()
    deleted = threading.Event()
    reader = threading.get_ident()

    class ReleaseGate:
        """Schedule deletion at the reader's first fully released critical section."""

        def __init__(self):
            self.lock = threading.RLock()
            self.depth = threading.local()

        def __enter__(self):
            self.lock.acquire()
            self.depth.value = getattr(self.depth, "value", 0) + 1
            return self

        def __exit__(self, *args):
            self.depth.value -= 1
            self.lock.release()
            if (
                threading.get_ident() == reader
                and self.depth.value == 0
                and not released.is_set()
            ):
                released.set()
                assert deleted.wait(3), "concurrent deletion did not finish"

    cache._lock = ReleaseGate()

    def delete():
        if released.wait(3):
            cache.delete("a")
            deleted.set()

    worker = threading.Thread(target=delete)
    worker.start()
    try:
        result = cache.get("a")
    finally:
        worker.join(3)

    assert not worker.is_alive()
    assert deleted.is_set()
    assert result == 1
    assert cache.get("a") is None


def test_local_cache_expiry_and_nonexpiring_entries_remain_supported(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(
        "quantcortex.data.storage.redis_cache.time.monotonic", lambda: now[0]
    )
    cache = RedisCache()
    cache.set("expiring", {"a": 1}, ttl=1)
    cache.set("persistent", b"data", ttl=0)

    assert cache.get("expiring") == {"a": 1}
    now[0] = 101.0
    assert cache.get("expiring") is None
    assert cache.get("persistent") == b"data"


def test_fred_fill_preserves_weekend_observation_and_surrounding_history(monkeypatch):
    source = {
        "UNRATE": pd.Series(
            [3.5, 3.6, 3.7],
            index=pd.to_datetime(["2022-04-29", "2022-05-01", "2022-06-01"]),
        ),
        "VIXCLS": pd.Series(
            [20.0, 21.0], index=pd.to_datetime(["2022-05-02", "2022-05-31"])
        ),
    }
    client = SimpleNamespace(get_series=lambda sid, **kwargs: source[sid])
    provider = FREDProvider()
    monkeypatch.setattr(provider, "_client", lambda: client)

    raw = provider.fetch_macro(list(source), ffill=False)
    filled = provider.fetch_macro(list(source), ffill=True)

    pd.testing.assert_frame_equal(raw, pd.DataFrame(source).sort_index().rename_axis("date"))
    assert filled.index.equals(pd.bdate_range("2022-04-29", "2022-06-01"))
    assert filled.loc["2022-04-29", "UNRATE"] == 3.5
    assert filled.loc["2022-05-02":"2022-05-31", "UNRATE"].eq(3.6).all()
    assert filled.loc["2022-06-01", "UNRATE"] == 3.7
    assert pd.isna(filled.loc["2022-04-29", "VIXCLS"])
    assert filled.loc["2022-05-02", "VIXCLS"] == 20.0
    assert filled.loc["2022-06-01", "VIXCLS"] == 21.0


@pytest.mark.parametrize("allow_exact_matches", [False, True])
def test_ttm_invalid_event_stays_missing_until_valid_reports_arrive(allow_exact_matches):
    records = []
    periods = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"])
    for symbol, value in [("A", 10.0), ("B", 50.0)]:
        for period in periods:
            records.append((symbol, period, period + pd.Timedelta(days=30), value))
    records += [
        ("A", "2024-09-30", "2024-10-30", 10.0),
        ("A", "2024-03-31", "2024-11-15", 11.0),
        ("A", "2024-06-30", "2024-11-20", 12.0),
    ]
    frame = pd.DataFrame(records, columns=["symbol", "period_end", "announcement_date", "value"])
    frame["field"] = "net_income"
    frame = validate_fundamentals(frame)
    index = pd.to_datetime(
        ["2024-01-30", "2024-01-31", "2024-10-30", "2024-10-31",
         "2024-11-15", "2024-11-16", "2024-11-20", "2024-11-21"]
    )

    result = pit_panel(
        frame, "net_income", index, pd.Index(["A", "B", "C"]),
        mode="ttm", allow_exact_matches=allow_exact_matches,
    )

    expected_a = (
        [40.0, 40.0, np.nan, np.nan, np.nan, np.nan, 43.0, 43.0]
        if allow_exact_matches
        else [np.nan, 40.0, 40.0, np.nan, np.nan, np.nan, np.nan, 43.0]
    )
    expected_b = [200.0] * len(index)
    if not allow_exact_matches:
        expected_b[0] = np.nan
    expected = pd.DataFrame({"A": expected_a, "B": expected_b, "C": np.nan}, index=index)
    pd.testing.assert_frame_equal(result, expected)


def test_average_balance_invalid_event_does_not_revive_old_value():
    frame = pd.DataFrame(
        {
            "symbol": "A",
            "field": "total_assets",
            "period_end": ["2023-03-31", "2024-03-31", "2025-09-30", "2024-09-30"],
            "announcement_date": ["2023-04-30", "2024-04-30", "2025-10-30", "2025-11-05"],
            "value": [100.0, 120.0, 160.0, 140.0],
        }
    )
    frame = validate_fundamentals(frame)
    index = pd.to_datetime(["2024-05-01", "2025-10-31", "2025-11-04", "2025-11-06"])

    result = pit_panel(frame, "total_assets", index, pd.Index(["A"]), mode="average_balance")

    expected = pd.DataFrame({"A": [110.0, np.nan, np.nan, 150.0]}, index=index)
    pd.testing.assert_frame_equal(result, expected)
