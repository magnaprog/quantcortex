from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantcortex.research import expansion
from quantcortex.research.expansion import (
    cross_sectional_momentum_targets,
    learned_gbrt_targets,
    monthly_decision_dates,
    short_term_reversal_targets,
    time_series_momentum_targets,
    validate_price_panel,
)

RULES = {
    "time_series": time_series_momentum_targets,
    "cross_sectional": cross_sectional_momentum_targets,
    "reversal": short_term_reversal_targets,
}
CONFIG = {
    "feature_return_windows": [5, 21, 63, 126, 252],
    "feature_volatility_windows": [21, 63],
    "label_horizon_sessions": 21,
    "training_decision_months": 12,
    "minimum_training_decision_months": 2,
    "selection_count": 3,
    "n_estimators": 3,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 2,
    "subsample": 0.8,
}


@pytest.fixture
def prices():
    index = pd.bdate_range("2014-01-02", periods=700)
    t = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "A": 100 * np.exp(0.001 * t),
            "B": 100 * np.exp(-0.001 * t),
            "C": 100 * np.exp(0.02 * np.sin(t / 7)),
        },
        index=index,
    )


@pytest.fixture
def estimator_calls(monkeypatch):
    calls = []

    class RecordingEstimator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.arange(1, len(x) + 1, dtype=float)

    monkeypatch.setattr("sklearn.ensemble.GradientBoostingRegressor", RecordingEstimator)
    return calls


def _bad_dates(index, kind):
    if kind == "unsorted":
        return index[[1, 0, *range(2, len(index))]]
    if kind == "not_datetime":
        return index.astype(str)
    values = list(index)
    values[0] = values[1] if kind == "duplicate" else pd.NaT
    return pd.DatetimeIndex(values)


def _learned(prices, all_decisions, evaluation_decisions):
    return learned_gbrt_targets(
        prices,
        symbols=list(prices.columns),
        all_decisions=all_decisions,
        evaluation_decisions=evaluation_decisions,
        config=CONFIG,
        seed=11,
    )


@pytest.mark.parametrize("generator", [*RULES, "learned"])
@pytest.mark.parametrize("kind", ["unsorted", "duplicate", "nat", "not_datetime"])
def test_target_generators_reject_invalid_price_dates(prices, estimator_calls, generator, kind):
    all_decisions = monthly_decision_dates(prices.index)
    evaluation = all_decisions[-3:]
    malformed = prices.copy()
    malformed.index = _bad_dates(prices.index, kind)
    error = TypeError if kind == "not_datetime" else ValueError

    with pytest.raises(error, match="prices index"):
        if generator == "learned":
            _learned(malformed, all_decisions, evaluation)
        else:
            RULES[generator](malformed, symbols=list(prices), decisions=evaluation)

    assert not estimator_calls


@pytest.mark.parametrize("generator", [*RULES, "learned"])
def test_target_generators_reject_empty_prices(prices, estimator_calls, generator):
    all_decisions = monthly_decision_dates(prices.index)
    evaluation = all_decisions[-3:]

    with pytest.raises(ValueError, match="prices must be a non-empty DataFrame"):
        if generator == "learned":
            _learned(prices.iloc[:0], all_decisions, evaluation)
        else:
            RULES[generator](prices.iloc[:0], symbols=list(prices), decisions=evaluation)

    assert not estimator_calls


@pytest.mark.parametrize("generator", [*RULES, "learned"])
def test_duplicate_price_columns_fail_before_calculation(
    prices, estimator_calls, monkeypatch, generator,
):
    all_decisions = monthly_decision_dates(prices.index)
    evaluation = all_decisions[-3:]
    malformed = pd.concat([prices, prices[["A"]]], axis=1)
    calls = []
    name = "_learned_feature_table" if generator == "learned" else "_decision_position"
    original = getattr(expansion, name)

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(expansion, name, traced)
    with pytest.raises(ValueError, match="prices columns must be unique"):
        if generator == "learned":
            learned_gbrt_targets(
                malformed, symbols=list(prices), all_decisions=all_decisions,
                evaluation_decisions=evaluation, config=CONFIG, seed=11,
            )
        else:
            RULES[generator](malformed, symbols=list(prices), decisions=evaluation)

    assert not calls
    assert not estimator_calls


@pytest.mark.parametrize("generator", RULES)
@pytest.mark.parametrize("kind", ["unsorted", "duplicate", "nat", "not_datetime"])
def test_rules_reject_invalid_decision_dates(prices, generator, kind):
    decisions = _bad_dates(monthly_decision_dates(prices.index)[-3:], kind)
    error = TypeError if kind == "not_datetime" else ValueError

    with pytest.raises(error, match="decisions"):
        RULES[generator](prices, symbols=list(prices), decisions=decisions)


@pytest.mark.parametrize("generator", RULES)
def test_rules_reject_empty_decision_dates(prices, generator):
    with pytest.raises(ValueError, match="decisions must not be empty"):
        RULES[generator](prices, symbols=list(prices), decisions=pd.DatetimeIndex([]))


@pytest.mark.parametrize("schedule", ["all_decisions", "evaluation_decisions"])
@pytest.mark.parametrize("kind", ["unsorted", "duplicate", "nat", "not_datetime"])
def test_learned_rejects_invalid_schedules_before_fit(prices, estimator_calls, schedule, kind):
    all_decisions = monthly_decision_dates(prices.index)
    schedules = {"all_decisions": all_decisions, "evaluation_decisions": all_decisions[-3:]}
    schedules[schedule] = _bad_dates(schedules[schedule], kind)
    error = TypeError if kind == "not_datetime" else ValueError

    with pytest.raises(error, match=schedule):
        _learned(prices, **schedules)

    assert not estimator_calls


@pytest.mark.parametrize("schedule", ["all_decisions", "evaluation_decisions"])
def test_learned_rejects_empty_schedules_before_fit(prices, estimator_calls, schedule):
    all_decisions = monthly_decision_dates(prices.index)
    schedules = {"all_decisions": all_decisions, "evaluation_decisions": all_decisions[-3:]}
    schedules[schedule] = pd.DatetimeIndex([])

    with pytest.raises(ValueError, match=f"{schedule} must not be empty"):
        _learned(prices, **schedules)

    assert not estimator_calls


def test_learned_rejects_missing_requested_decision_before_fit(prices, estimator_calls):
    all_decisions = monthly_decision_dates(prices.index)
    evaluation = all_decisions[-3:]
    omitted = evaluation[1]
    incomplete = all_decisions[all_decisions != omitted]

    with pytest.raises(ValueError, match="evaluation_decisions.*all_decisions"):
        _learned(prices, incomplete, evaluation)

    assert not estimator_calls


def test_learned_rejects_non_price_decision_before_fit(prices, estimator_calls):
    all_decisions = monthly_decision_dates(prices.index)
    evaluation = all_decisions[-3:]
    nonexistent = all_decisions[15] + pd.Timedelta(hours=1)
    malformed = all_decisions.insert(16, nonexistent)

    with pytest.raises(ValueError, match="all_decisions.*price"):
        _learned(prices, malformed, evaluation)

    assert not estimator_calls


def test_future_row_cannot_enter_reversal_lookback():
    index = pd.bdate_range("2020-01-02", periods=360)
    prices = pd.DataFrame(
        {"A": np.linspace(100, 140, len(index)), "B": 100.0, "C": 100.0}, index=index
    )
    prices.iloc[330, 0] = 400.0
    decision = index[300:301]
    expected = short_term_reversal_targets(prices, symbols=list(prices), decisions=decision)
    assert expected.iloc[0].sum() == 0.0

    positions = np.arange(len(index))
    positions[295], positions[330] = positions[330], positions[295]
    unsorted = prices.iloc[positions]
    # Without an ordering check, April's price makes February's A look like a loser.
    with pytest.raises(ValueError, match="prices index"):
        short_term_reversal_targets(unsorted, symbols=list(prices), decisions=decision)

    normalized = validate_price_panel(
        unsorted.assign(SHV=100.0), risky_symbols=list(prices), cash_symbol="SHV"
    )
    actual = short_term_reversal_targets(normalized, symbols=list(prices), decisions=decision)
    pd.testing.assert_frame_equal(actual, expected)


def test_learned_retains_every_valid_requested_decision(prices, estimator_calls):
    all_decisions = monthly_decision_dates(prices.index)
    evaluation = all_decisions[-4:][[0, 2, 3]]

    result = _learned(prices, all_decisions, evaluation)

    pd.testing.assert_index_equal(result.weights.index, evaluation)
    pd.testing.assert_index_equal(result.training_rows.index, evaluation)
    pd.testing.assert_index_equal(result.training_months.index, evaluation)
    assert len(estimator_calls) == len(evaluation)
    assert all(call["random_state"] == 11 for call in estimator_calls)


@pytest.mark.parametrize("generator", RULES)
def test_rule_targets_accept_sorted_gaps_after_loader_normalization(prices, generator):
    prices = prices.drop(prices.index[290:293])
    expected_prices = prices.assign(SHV=100.0)
    normalized = validate_price_panel(
        expected_prices.iloc[::-1], risky_symbols=list(prices), cash_symbol="SHV"
    )
    decisions = monthly_decision_dates(prices.index)[-3:]

    expected = RULES[generator](prices, symbols=list(prices), decisions=decisions)
    actual = RULES[generator](normalized, symbols=list(prices), decisions=decisions)

    pd.testing.assert_frame_equal(actual, expected)
