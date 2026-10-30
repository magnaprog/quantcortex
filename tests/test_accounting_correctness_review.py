"""Regression oracles for accounting, allocation and model publication."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest
from quantcortex.backtest.engines.vectorized import VectorizedBacktest
from quantcortex.backtest.validation.deflated_sharpe import (
    compute_dsr,
    probabilistic_sharpe_ratio,
)
from quantcortex.portfolio.base import WeightContractViolationError
from quantcortex.portfolio.drl_allocator import DRLAllocator
from quantcortex.portfolio.equal_weight import EqualWeight
from quantcortex.strategies.momentum_ml import MomentumMLStrategy


@pytest.mark.parametrize("failure_stage", ["construction", "learning"])
@pytest.mark.parametrize("columns", [["B", "A"], ["A", "B", "C"]])
def test_failed_drl_retrain_preserves_model_and_asset_schema(
    monkeypatch, failure_stage, columns,
):
    class OldModel:
        def predict(self, observation, deterministic):
            return np.log([0.8, 0.2]), None

    class FailingPPO:
        def __init__(self, *args, **kwargs):
            if failure_stage == "construction":
                raise RuntimeError("training failed")

        def learn(self, **kwargs):
            raise RuntimeError("training failed")

    allocator = DRLAllocator(window=2)
    old_model = OldModel()
    allocator.model = old_model
    allocator._n_assets = 2
    allocator._asset_names = ("A", "B")
    monkeypatch.setattr(allocator, "_make_env", lambda returns: object())
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=FailingPPO))

    changed = pd.DataFrame(0.01, index=range(4), columns=columns)
    with pytest.raises(RuntimeError, match="training failed"):
        allocator.train(changed)

    assert allocator.model is old_model
    assert allocator._n_assets == 2
    assert allocator._asset_names == ("A", "B")
    with pytest.raises(ValueError, match="asset schema"):
        allocator.optimize(changed, previous_weights=np.zeros(len(columns)))
    original = pd.DataFrame(0.01, index=range(4), columns=["A", "B"])
    assert allocator.optimize(original, previous_weights=[0.5, 0.5]) == pytest.approx(
        [0.8, 0.2]
    )


def _momentum_context(strategy):
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "A": [100., 101., 102., 104., 109.],
            "B": [100., 101., 102., 103., 106.],
            "C": [100., 101., 102., 103., 104.],
            "D": [100., 101., 101., 102., 103.],
        },
        index=dates,
    )
    return strategy.build_context(prices, dates[-1])


def test_momentum_strategy_honors_equal_weight_concentration_limit():
    strategy = MomentumMLStrategy(
        optimizer=EqualWeight(weight_bounds=(0., 0.4)),
        top_quantile=0.5, lookback=3, gap=0,
    )
    with pytest.raises(WeightContractViolationError, match="outside bounds"):
        strategy.rebalance(_momentum_context(strategy))


def test_momentum_strategy_honors_equal_weight_subclass():
    class RankedAllocation(EqualWeight):
        def _compute_weights(self, returns, **kwargs):
            assert list(returns.columns) == ["A", "B"]
            return np.array([0.8, 0.2])

    strategy = MomentumMLStrategy(
        optimizer=RankedAllocation(), top_quantile=0.5, lookback=3, gap=0,
    )
    result = strategy.rebalance(_momentum_context(strategy))
    assert result.as_series().to_dict() == pytest.approx(
        {"A": 0.8, "B": 0.2, "C": 0., "D": 0.}
    )


@pytest.mark.parametrize("sign", [1., -1.])
def test_psr_and_dsr_are_undefined_for_negative_variance_estimate(sign):
    # Bias-corrected Pearson kurtosis is -3 here, yielding a negative
    # standard-error square despite finite returns and nonzero dispersion.
    returns = pd.Series([0.01, 0.01, 0.02, 0.02]) * sign
    assert np.isnan(probabilistic_sharpe_ratio(returns))
    assert np.isnan(compute_dsr(returns, n_trials=10))


@pytest.mark.parametrize("invested", [False, True])
def test_vectorized_rejects_nonfinite_derived_asset_returns(invested):
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [1e-308, 1e308, 1e308]}, index=dates)
    weights = (
        pd.DataFrame({"A": [1.]}, index=[dates[0] - pd.Timedelta(days=1)])
        if invested else pd.DataFrame(columns=["A"])
    )
    with pytest.raises(ValueError, match="derived asset returns"):
        VectorizedBacktest(TransactionCostModel(0., 0.)).run(weights, prices)


def test_vectorized_rejects_nonfinite_terminal_nav():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": 100.}, index=dates)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="NAV must remain finite and positive"):
            VectorizedBacktest(TransactionCostModel(0., 0.)).run(
                pd.DataFrame(columns=["A"]), prices,
                cash_returns=pd.Series(1e308, index=dates),
            )


def test_event_engine_reports_solvent_short_drift_without_changing_units():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [100., 100., 130., 130.]}, index=dates)
    weights = pd.DataFrame({"A": [-0.8]}, index=dates[:1])
    result = EventDrivenBacktest(
        TransactionCostModel(0., 0.), capital=1000.,
    ).run(weights, prices)
    assert result.equity_curve.tolist() == pytest.approx([1000., 1000., 760., 760.])
    assert result.weights["A"].iloc[2] == pytest.approx(-1040. / 760.)
    quantities = result.weights["A"] * result.equity_curve / prices["A"]
    assert quantities.tolist() == pytest.approx([0., -8., -8., -8.])
    assert (result.cash_weights * result.equity_curve).tolist() == pytest.approx(
        [1000., 1800., 1800., 1800.]
    )


def test_event_engine_still_rejects_target_outside_per_asset_bounds():
    dates = pd.date_range("2020-01-01", periods=3)
    with pytest.raises(WeightContractViolationError, match="outside bounds"):
        EventDrivenBacktest(TransactionCostModel(0., 0.), max_gross=2.).run(
            pd.DataFrame({"A": [-1.1]}, index=dates[:1]),
            pd.DataFrame({"A": 100.}, index=dates),
        )


def test_event_engine_still_rejects_insolvent_short_book():
    dates = pd.date_range("2020-01-01", periods=3)
    with pytest.raises(ValueError, match="NAV was exhausted"):
        EventDrivenBacktest(TransactionCostModel(0., 0.), capital=1000.).run(
            pd.DataFrame({"A": [-0.8]}, index=dates[:1]),
            pd.DataFrame({"A": [100., 100., 225.]}, index=dates),
        )


@pytest.mark.parametrize("rate", [0.75, 0.9])
def test_event_engine_high_fee_sizing_has_positive_analytic_solution(rate):
    dates = pd.date_range("2020-01-01", periods=3)
    result = EventDrivenBacktest(
        TransactionCostModel(rate, 0.), capital=1000.,
    ).run(
        pd.DataFrame({"A": [1.]}, index=dates[:1]),
        pd.DataFrame({"A": 100.}, index=dates),
    )
    expected = 1000. / (1. + rate)
    assert result.equity_curve.tolist() == pytest.approx(
        [1000., expected, expected], rel=1e-12,
    )
    assert result.weights["A"].iloc[1] == pytest.approx(1.)
    assert result.cash_weights.iloc[1] == pytest.approx(0., abs=1e-12)
    assert result.costs.iloc[1] == pytest.approx(rate / (1. + rate))
    assert result.traded_notional.iloc[1] == pytest.approx(1. / (1. + rate))
    assert result.metadata["cost_aware_sizing"] == "fixed_point_post_cost_nav"


def test_ordinary_cost_sizing_does_not_invoke_root_fallback(monkeypatch):
    def forbidden_fallback(*args, **kwargs):
        pytest.fail("ordinary fixed-point sizing must not use the fallback")

    monkeypatch.setattr("scipy.optimize.brentq", forbidden_fallback)
    dates = pd.date_range("2020-01-01", periods=3)
    result = EventDrivenBacktest(
        TransactionCostModel(0.0013, 0.), capital=1000.,
    ).run(
        pd.DataFrame({"A": [1.]}, index=dates[:1]),
        pd.DataFrame({"A": 100.}, index=dates),
    )
    expected = 1000. / 1.0013
    assert result.equity_curve.tolist() == pytest.approx(
        [1000., expected, expected], rel=1e-12,
    )
    assert result.metadata["cost_aware_sizing"] == "fixed_point_post_cost_nav"


def test_stalled_custom_cost_model_does_not_use_unjustified_root_fallback():
    class CustomCosts(TransactionCostModel):
        pass

    dates = pd.date_range("2020-01-01", periods=3)
    with pytest.raises(ValueError, match="sizing did not converge"):
        EventDrivenBacktest(CustomCosts(0.75, 0.), capital=1000.).run(
            pd.DataFrame({"A": [1.]}, index=dates[:1]),
            pd.DataFrame({"A": 100.}, index=dates),
        )


@pytest.mark.parametrize("rate", [0.5, 0.9])
def test_cost_sizing_still_rejects_exhausted_nav(rate):
    engine = EventDrivenBacktest(TransactionCostModel(rate, 0.), max_gross=2.)
    with pytest.raises(ValueError, match="exhausted portfolio NAV at the liquidation boundary"):
        engine._cost_aware_rebalance(
            np.array([1., -1.]), np.zeros(2),
            prices=np.ones(2), adv=None, capital=1000.,
        )


@pytest.mark.parametrize("rate", [0.5, 0.75])
def test_cost_sizing_sign_reversal_recovers_from_nonpositive_iterate(rate):
    engine = EventDrivenBacktest(TransactionCostModel(rate, 0.), capital=1000.)
    cost, fraction = engine._cost_aware_rebalance(
        np.ones(1), -np.ones(1), prices=np.ones(1), adv=None, capital=1000.,
    )
    # q = 1 - rate * (1 + q) for closing the long and opening the short.
    expected_fraction = (1. - rate) / (1. + rate)
    assert fraction == pytest.approx(expected_fraction, rel=1e-12)
    assert cost.executed_weights == pytest.approx([-expected_fraction], rel=1e-12)
    assert fraction + cost.total_cost == pytest.approx(1., abs=1e-13)

    dates = pd.date_range("2020-01-01", periods=3)
    result = engine.run(
        pd.DataFrame({"A": [1., -1.]}, index=dates[:2]),
        pd.DataFrame({"A": 1.}, index=dates),
    )
    first_nav = 1000. / (1. + rate)
    assert result.equity_curve.tolist() == pytest.approx(
        [1000., first_nav, first_nav * expected_fraction], rel=1e-12,
    )
    assert result.weights["A"].iloc[-1] == pytest.approx(-1., rel=1e-12)
    assert result.cash_weights.iloc[-1] == pytest.approx(2., rel=1e-12)


@pytest.mark.parametrize("rate", [0.5, 0.75])
def test_cost_sizing_leveraged_entry_recovers_from_nonpositive_iterate(rate):
    dates = pd.date_range("2020-01-01", periods=3)
    result = EventDrivenBacktest(
        TransactionCostModel(rate, 0.), capital=1000., max_gross=2.,
    ).run(
        pd.DataFrame({"A": [1.], "B": [1.]}, index=dates[:1]),
        pd.DataFrame({"A": 1., "B": 1.}, index=dates),
    )
    expected_nav = 1000. / (1. + 2. * rate)
    assert result.equity_curve.tolist() == pytest.approx(
        [1000., expected_nav, expected_nav], rel=1e-12,
    )
    assert result.weights.iloc[1].to_numpy() == pytest.approx([1., 1.], rel=1e-12)
    assert result.cash_weights.iloc[1] == pytest.approx(-1., rel=1e-12)
    assert result.costs.iloc[1] == pytest.approx(1. - expected_nav / 1000., rel=1e-12)


@pytest.mark.parametrize("rate", [0.5, 0.75])
def test_custom_cost_model_nonpositive_iterate_retains_explicit_failure(rate):
    class CustomCosts(TransactionCostModel):
        pass

    engine = EventDrivenBacktest(CustomCosts(rate, 0.))
    with pytest.raises(ValueError, match="costs exhausted portfolio NAV"):
        engine._cost_aware_rebalance(
            np.ones(1), -np.ones(1), prices=np.ones(1), adv=None, capital=1000.,
        )
