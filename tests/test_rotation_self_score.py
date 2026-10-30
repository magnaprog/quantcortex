"""A benchmark cannot acquire a residual-momentum position against itself."""

import numpy as np
import pandas as pd
import pytest

from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation


def _benchmark(seed=0, periods=274):
    returns = np.random.default_rng(seed).normal(0.0002, 0.01, periods)
    return pd.Series(
        100.0 * np.cumprod(1.0 + returns),
        index=pd.bdate_range("2020-01-01", periods=periods),
        name="QQQ",
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 4, 5])
@pytest.mark.parametrize("member", ["QQQ", "same_prices"])
def test_identical_prices_have_exactly_zero_residual_score(seed, member):
    benchmark = _benchmark(seed)
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    scores = strategy._residual_momentum(
        benchmark.to_frame(name=member), benchmark
    )

    assert scores is not None
    assert scores[member] == 0.0


def test_self_score_produces_an_explicit_cash_target():
    prices = _benchmark().to_frame()
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    weights = strategy.generate_weights(prices, [prices.index[-1]])

    assert weights.index.tolist() == [prices.index[-1]]
    assert weights.columns.tolist() == ["QQQ"]
    assert weights.iloc[0, 0] == 0.0


@pytest.mark.parametrize("member", ["QQQ", "VGT"])
def test_distinct_prices_keep_a_tiny_positive_residual_score(member):
    returns = np.random.default_rng(12).normal(0.0002, 0.01, 274)
    asset_returns = returns.copy()
    # The estimation window is 1:127; formation is 127:253, before the gap.
    asset_returns[127:253] += 1e-14
    benchmark = _benchmark(12)
    prices = pd.DataFrame(
        {member: 100.0 * np.cumprod(1.0 + asset_returns)},
        index=benchmark.index,
    )
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    scores = strategy._residual_momentum(prices, benchmark)

    assert scores is not None
    assert 0.0 < scores[member] < 1e-10
    assert scores[member] == pytest.approx(126e-14, rel=0.01, abs=1e-15)


def test_short_history_keeps_plain_momentum_fallback():
    prices = pd.DataFrame(
        {"QQQ": np.linspace(100.0, 130.0, 273)},
        index=pd.bdate_range("2020-01-01", periods=273),
    )
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    assert strategy._residual_momentum(prices, prices.QQQ) is None
    weights = strategy.generate_weights(prices, [prices.index[-1]])
    assert weights.iloc[0, 0] == pytest.approx(0.6)


def test_constant_benchmark_keeps_plain_momentum_fallback():
    prices = pd.DataFrame(
        {"QQQ": np.full(274, 100.0), "VGT": np.linspace(100.0, 130.0, 274)},
        index=pd.bdate_range("2020-01-01", periods=274),
    )
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    assert strategy._residual_momentum(prices, prices.QQQ) is None
    weights = strategy.generate_weights(prices, [prices.index[-1]])
    assert weights.iloc[0].to_dict() == pytest.approx({"QQQ": 0.0, "VGT": 0.6})


@pytest.mark.parametrize("missing_position", [50, 180])
def test_incomplete_regression_window_remains_unavailable(missing_position):
    benchmark = _benchmark()
    benchmark.iloc[missing_position] = np.nan
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    assert strategy._residual_momentum(benchmark.to_frame(), benchmark) is None


def test_self_score_uses_the_aligned_benchmark():
    benchmark = _benchmark()
    strategy = MultiAssetRotation(regime=False, vix_scale=False)

    scores = strategy._residual_momentum(
        benchmark.to_frame(), benchmark.iloc[::-1]
    )

    assert scores is not None
    assert scores.QQQ == 0.0
