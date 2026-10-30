from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.research.expansion import (
    _sample_sharpe,
    bootstrap_metric_difference,
    cross_sectional_momentum_targets,
    exposure_matched_comparator_targets,
    invalid_same_close_result,
    learned_gbrt_targets,
    monthly_decision_dates,
    performance_metrics,
    run_engine,
    short_term_reversal_targets,
    time_series_momentum_targets,
    validate_price_panel,
)


def _trend_panel(periods: int = 340) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=periods)
    time = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "A": np.exp(0.001 * time),
            "B": np.exp(-0.001 * time),
            "C": np.ones(periods),
            "D": np.exp(0.001 * time),
            "CASH": np.exp(0.0001 * time),
        },
        index=index,
    )


def _learned_config() -> dict[str, object]:
    return {
        "label_horizon_sessions": 21,
        "feature_return_windows": [5, 21, 63, 126, 252],
        "feature_volatility_windows": [21, 63],
        "training_decision_months": 60,
        "minimum_training_decision_months": 24,
        "n_estimators": 10,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "subsample": 0.8,
        "selection_count": 3,
    }


def test_monthly_schedule_and_price_validation_are_strict():
    prices = _trend_panel()
    validated = validate_price_panel(
        prices,
        risky_symbols=["A", "B", "C", "D"],
        cash_symbol="CASH",
    )
    decisions = monthly_decision_dates(validated.index, start="2020-03-10")

    assert decisions[0] == pd.Timestamp("2020-03-10")
    assert decisions.to_period("M").is_unique
    assert list(validated.columns) == ["A", "B", "C", "D", "CASH"]


def test_rule_targets_follow_frozen_selection_and_tie_breaks():
    prices = _trend_panel()
    decision = pd.DatetimeIndex([prices.index[300]])
    symbols = ["A", "B", "C", "D"]

    time_series = time_series_momentum_targets(
        prices,
        symbols=symbols,
        decisions=decision,
    )
    cross_sectional = cross_sectional_momentum_targets(
        prices,
        symbols=symbols,
        decisions=decision,
        selection_count=1,
    )

    assert time_series.iloc[0].to_dict() == {
        "A": 0.5,
        "B": 0.0,
        "C": 0.0,
        "D": 0.5,
    }
    assert cross_sectional.iloc[0].to_dict() == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
        "D": 0.0,
    }


def test_reversal_does_not_renormalize_unused_exposure():
    index = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {
            "A": [100, 101, 102, 103, 104, 105, 106, 107],
            "B": [100, 100, 100, 100, 100, 90, 90, 90],
            "C": [100] * 8,
            "D": [100, 101, 101, 101, 101, 101, 101, 101],
        },
        index=index,
    )
    targets = short_term_reversal_targets(
        prices,
        symbols=["A", "B", "C", "D"],
        decisions=pd.DatetimeIndex([index[-1]]),
        selection_count=3,
    )

    assert targets.iloc[0].to_dict() == {
        "A": 0.0,
        "B": 1.0 / 3.0,
        "C": 0.0,
        "D": 0.0,
    }
    assert np.isclose(targets.iloc[0].sum(), 1.0 / 3.0)


def test_learned_targets_are_deterministic_and_ignore_future_prices():
    index = pd.bdate_range("2014-01-02", periods=1_500)
    rng = np.random.default_rng(7)
    innovations = rng.normal(
        loc=np.array([0.0003, 0.0001, -0.0001, 0.0002]),
        scale=np.array([0.008, 0.009, 0.007, 0.01]),
        size=(len(index), 4),
    )
    prices = pd.DataFrame(
        np.exp(np.cumsum(innovations, axis=0)) * 100.0,
        index=index,
        columns=["A", "B", "C", "D"],
    )
    all_decisions = monthly_decision_dates(index)
    evaluation = all_decisions[-12:]
    cutoff = evaluation[3]

    first = learned_gbrt_targets(
        prices,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation,
        config=_learned_config(),
        seed=11,
    )
    repeated = learned_gbrt_targets(
        prices,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation,
        config=_learned_config(),
        seed=11,
    )
    perturbed = prices.copy()
    perturbed.loc[perturbed.index > cutoff, "A"] *= 4.0
    future_changed = learned_gbrt_targets(
        perturbed,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation[:4],
        config=_learned_config(),
        seed=11,
    )

    pd.testing.assert_frame_equal(first.weights, repeated.weights)
    pd.testing.assert_series_equal(first.weights.loc[cutoff], future_changed.weights.loc[cutoff])
    assert (first.training_months >= 24).all()
    assert (first.weights.sum(axis=1) <= 1.0 + 1e-12).all()
    assert (first.weights >= 0.0).all(axis=None)


def test_same_close_diagnostic_moves_the_first_earned_return_backward():
    index = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=index)
    cash = pd.Series(0.0, index=index)
    weights = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([index[1]]))

    causal = run_engine(
        weights,
        prices,
        cash,
        cost_rate=0.0,
        engine="event_driven",
    )
    invalid = invalid_same_close_result(
        weights,
        prices,
        cash,
        cost_rate=0.0,
    )

    assert np.isclose(causal.returns.loc[index[1]], 0.0)
    assert np.isclose(causal.returns.loc[index[2]], 0.0)
    assert np.isclose(causal.returns.loc[index[3]], 0.10)
    assert np.isclose(invalid.returns.loc[index[1]], 0.10)


@pytest.mark.parametrize(
    ("choose_winner", "asset", "retroactive_return"),
    [(True, "A", 0.10), (False, "B", -0.10)],
)
def test_timing_example_can_invent_a_gain_or_loss(
    choose_winner, asset, retroactive_return
):
    index = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 110.0, 110.0], "B": [100.0, 90.0, 90.0, 90.0]},
        index=index,
    )
    observed = prices.iloc[1] / prices.iloc[0] - 1.0
    selected = observed.idxmax() if choose_winner else observed.idxmin()
    assert selected == asset
    weights = pd.DataFrame(0.0, index=index[1:2], columns=prices.columns)
    weights.loc[index[1], selected] = 1.0
    cash = pd.Series(0.0, index=index)

    feasible = run_engine(
        weights, prices, cash, cost_rate=0.0, engine="event_driven"
    )
    retroactive = invalid_same_close_result(weights, prices, cash, cost_rate=0.0)

    np.testing.assert_allclose(feasible.returns, 0.0, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        feasible.weights[asset], [0.0, 0.0, 1.0, 1.0], atol=1e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        retroactive.returns,
        [0.0, retroactive_return, 0.0, 0.0],
        atol=1e-14,
        rtol=0.0,
    )
    assert feasible.equity_curve.iloc[-1] == pytest.approx(1.0)
    assert retroactive.equity_curve.iloc[-1] == pytest.approx(
        1.0 + retroactive_return
    )


def test_exposure_matched_comparator_preserves_residual_cash():
    index = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
    weights = pd.DataFrame(
        [[0.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=index,
        columns=["A", "B", "C"],
    )

    comparator = exposure_matched_comparator_targets(weights)

    np.testing.assert_allclose(comparator.iloc[0], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(comparator.iloc[1], [0.0, 0.0, 0.0])


def test_exposure_matched_comparator_rejects_short_targets():
    weights = pd.DataFrame(
        [[0.6, -0.1, 0.0]],
        index=pd.DatetimeIndex(["2024-01-02"]),
        columns=["A", "B", "C"],
    )

    with pytest.raises(ValueError, match="long-only"):
        exposure_matched_comparator_targets(weights)


def test_bootstrap_family_effect_is_reproducible_and_matches_point_estimates():
    index = pd.bdate_range("2024-01-02", periods=12)
    cash = pd.Series(np.linspace(0.0001, 0.0002, len(index)), index=index)
    lhs = [
        pd.Series(np.linspace(-0.01, 0.02, len(index)), index=index),
        pd.Series(np.linspace(-0.008, 0.018, len(index)), index=index),
    ]
    rhs = [
        pd.Series(np.linspace(-0.006, 0.011, len(index)), index=index),
        pd.Series(np.linspace(-0.004, 0.010, len(index)), index=index),
    ]

    first = bootstrap_metric_difference(
        lhs,
        rhs,
        cash,
        block_length=3,
        replications=250,
        seed=42,
    )
    second = bootstrap_metric_difference(
        lhs,
        rhs,
        cash,
        block_length=3,
        replications=250,
        seed=42,
    )

    assert first == second
    expected = np.mean(
        [
            (left - right).mean() * 252.0
            for left, right in zip(lhs, rhs, strict=True)
        ]
    )
    assert np.isclose(first["annualized_mean_difference"], expected)
    assert first["family_size"] == 2


def test_bootstrap_point_estimate_uses_the_joint_family_intersection():
    dates = pd.bdate_range("2024-01-02", periods=4)
    lhs = [
        pd.Series([1.0, 0.01, 0.03], index=dates[:3]),
        pd.Series([0.02, 0.04, 2.0], index=dates[1:]),
    ]
    rhs = [
        pd.Series([0.0, 0.0, 0.0], index=dates[:3]),
        pd.Series([0.0, 0.0, 0.0], index=dates[1:]),
    ]
    cash = pd.Series([0.001, 0.002, 0.0015, 0.0025], index=dates)

    estimate = bootstrap_metric_difference(
        lhs,
        rhs,
        cash,
        block_length=2,
        replications=100,
        seed=42,
    )

    expected = np.mean([0.02, 0.03]) * 252.0
    assert estimate["observations"] == 2
    assert np.isclose(estimate["annualized_mean_difference"], expected)


@pytest.mark.parametrize("input_position", range(5), ids=["lhs_0", "lhs_1", "rhs_0", "rhs_1", "cash"])
@pytest.mark.parametrize("invalid_dates", ["nonmonotonic", "duplicate", "nat"])
def test_paired_bootstrap_rejects_each_invalid_dated_input_before_alignment(
    input_position, invalid_dates,
):
    dates = pd.bdate_range("2024-01-02", periods=20)
    inputs = [
        pd.Series(np.linspace(-0.01 + i * 0.001, 0.02 + i * 0.002, len(dates)), index=dates)
        for i in range(5)
    ]
    if invalid_dates == "nonmonotonic":
        inputs[input_position] = inputs[input_position].iloc[[1, 0, *range(2, len(dates))]]
    else:
        bad_dates = list(dates)
        bad_dates[0] = dates[1] if invalid_dates == "duplicate" else pd.NaT
        inputs[input_position].index = pd.DatetimeIndex(bad_dates)

    # An inner join could hide NaT or another input's row order.
    with pytest.raises(ValueError, match="dates must be unique, valid, and increasing"):
        bootstrap_metric_difference(
            inputs[:2], inputs[2:4], inputs[4], block_length=3, replications=20, seed=7,
        )


@pytest.mark.parametrize("positional_index", [pd.RangeIndex(20), pd.RangeIndex(19, -1, -1)])
def test_paired_bootstrap_preserves_positional_range_index(positional_index):
    dates = pd.bdate_range("2024-01-02", periods=20)
    lhs = pd.Series(np.linspace(-0.01, 0.02, len(dates)), index=dates)
    rhs = pd.Series(np.sin(np.arange(len(dates))) * 0.005, index=dates)
    cash = pd.Series(0.0, index=dates)
    kwargs = dict(block_length=3, replications=40, seed=7)
    dated = bootstrap_metric_difference([lhs], [rhs], cash, **kwargs)
    positional = [pd.Series(series.to_numpy(), index=positional_index) for series in (lhs, rhs, cash)]
    assert bootstrap_metric_difference([positional[0]], [positional[1]], positional[2], **kwargs) == dated


def test_performance_metrics_use_supplied_cash_excess_returns():
    prices = _trend_panel(periods=30)
    risky = prices[["A"]]
    cash = prices["CASH"].pct_change(fill_method=None).fillna(0.0)
    weights = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([prices.index[0]]))
    result = run_engine(
        weights,
        risky,
        cash,
        cost_rate=0.0,
        engine="event_driven",
    )
    metrics = performance_metrics(result, cash, prices.index)

    returns = result.returns.reindex(prices.index)
    excess = returns - cash.reindex(prices.index)
    expected_sharpe = np.sqrt(252.0) * excess.mean() / excess.std(ddof=1)
    assert metrics["observations"] == len(prices)
    assert metrics["annualized_arithmetic_return"] == pytest.approx(252.0 * returns.mean())
    assert metrics["cash_excess_sharpe"] == pytest.approx(expected_sharpe)


@pytest.mark.parametrize("observations", [30, 252])
@pytest.mark.parametrize("variation", [0.0, 1e-14])
def test_performance_sharpe_distinguishes_constant_from_tiny_variation(
    observations, variation,
):
    index = pd.bdate_range("2024-01-01", periods=observations)
    returns = pd.Series(0.01, index=index)
    returns.iloc[-1] += variation
    cash = pd.Series(0.0, index=index)
    result = SimpleNamespace(
        returns=returns,
        weights=pd.DataFrame({"A": 1.0}, index=index),
        turnover=cash,
        traded_notional=cash,
        costs=cash,
    )

    actual = performance_metrics(result, cash, index)["cash_excess_sharpe"]

    if variation == 0.0:
        assert np.isnan(actual)
    else:
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected)


def test_sample_sharpe_masks_only_constant_draws_per_run():
    excess = np.full((2, 2, 30), 0.01)
    excess[0, 1, -1] += 1e-14
    excess[1, 0, -1] -= 1e-14

    actual = _sample_sharpe(excess)

    assert np.isnan(actual[0, 0])
    assert np.isnan(actual[1, 1])
    for run, draw in ((0, 1), (1, 0)):
        sample = excess[run, draw]
        expected = sample.mean() / sample.std(ddof=1) * np.sqrt(252.0)
        assert np.isfinite(actual[run, draw])
        assert actual[run, draw] == pytest.approx(expected)


def test_bootstrap_rejects_exact_constant_nonzero_excess():
    index = pd.bdate_range("2024-01-01", periods=30)
    with pytest.raises(ValueError, match="Sharpe difference is undefined"):
        bootstrap_metric_difference(
            [pd.Series(0.01, index=index)],
            [pd.Series(np.linspace(-0.01, 0.02, len(index)), index=index)],
            pd.Series(0.0, index=index),
            block_length=30,
            replications=20,
            seed=42,
        )
