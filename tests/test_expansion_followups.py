"""Independent, synthetic arithmetic checks for descriptive expansion followups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import run_expansion_experiments as expansion


@pytest.fixture
def yearly_panel():
    dates = pd.to_datetime(
        ["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06"]
    )
    return {
        "cash": pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=dates),
        "series": {
            "learned_gbrt": {
                11: {
                    "baseline": pd.Series(
                        [0.011, -0.018, 0.103, 0.034, -0.045], index=dates
                    ),
                    "same_close": pd.Series(0.5, index=dates),
                }
            }
        },
    }


def test_yearly_performance_handworked_cash_excess_and_weighted_reconciliation(
    yearly_panel,
):
    cash_before = yearly_panel["cash"].copy()
    baseline_before = yearly_panel["series"]["learned_gbrt"][11]["baseline"].copy()

    result = expansion._yearly_performance({"panel_a": yearly_panel})

    assert list(result.columns) == [
        "panel", "strategy", "seed", "year", "observations",
        "annualized_excess_return",
    ]
    expected = pd.DataFrame(
        [
            ("panel_a", "learned_gbrt", "11", 2020, 2, -1.26),
            ("panel_a", "learned_gbrt", "11", 2021, 3, 6.72),
        ],
        columns=result.columns,
    )
    # Daily excess: [.01, -.02] and [.10, .03, -.05]; annualize means by 252.
    pd.testing.assert_frame_equal(
        result.sort_values("year").reset_index(drop=True), expected,
        check_dtype=False, rtol=1e-12, atol=1e-12,
    )
    assert all(isinstance(seed, str) for seed in result["seed"])
    weighted = np.average(
        result["annualized_excess_return"], weights=result["observations"]
    )
    assert weighted == pytest.approx(3.528)  # 252 * .07 / 5.
    assert result["annualized_excess_return"].mean() == pytest.approx(2.73)
    assert weighted != pytest.approx(2.73)
    pd.testing.assert_series_equal(yearly_panel["cash"], cash_before)
    pd.testing.assert_series_equal(
        yearly_panel["series"]["learned_gbrt"][11]["baseline"], baseline_before
    )


def test_yearly_performance_retains_first_return_at_each_year_boundary(yearly_panel):
    original = expansion._yearly_performance({"panel_a": yearly_panel}).set_index(
        "year"
    )
    baseline = yearly_panel["series"]["learned_gbrt"][11]["baseline"]
    baseline.iloc[0] += 0.02
    baseline.iloc[2] += 0.03

    changed = expansion._yearly_performance({"panel_a": yearly_panel}).set_index(
        "year"
    )

    # Each supplied first return contributes, including the very first run date.
    assert changed.loc[2020, "annualized_excess_return"] == pytest.approx(1.26)
    assert changed.loc[2021, "annualized_excess_return"] == pytest.approx(9.24)
    assert (
        changed["annualized_excess_return"] - original["annualized_excess_return"]
    ).tolist() == pytest.approx([2.52, 2.52])
    assert changed["observations"].to_dict() == {2020: 2, 2021: 3}


def test_yearly_performance_keeps_every_panel_strategy_seed_and_observed_year(
    yearly_panel,
):
    cash = yearly_panel["cash"]
    yearly_panel["series"]["learned_gbrt"][29] = {"baseline": cash - 0.01}
    yearly_panel["series"]["ts_momentum"] = {
        "deterministic": {"baseline": cash + 0.02}
    }
    # A one-observation year separated by a missing calendar year stays valid.
    other_dates = pd.to_datetime(["2019-12-31", "2021-01-04"])
    panels = {
        "panel_a": yearly_panel,
        "panel_b": {
            "cash": pd.Series([0.002, 0.003], index=other_dates),
            "series": {
                "learned_gbrt": {
                    "11": {
                        "baseline": pd.Series([0.032, -0.017], index=other_dates)
                    }
                }
            },
        },
    }

    result = expansion._yearly_performance(panels)

    keys = ["panel", "strategy", "seed", "year"]
    assert not result.duplicated(keys).any()
    assert all(isinstance(seed, str) for seed in result["seed"])
    actual = result.set_index(keys)
    expected = {
        ("panel_a", "learned_gbrt", "11", 2020): (2, -1.26),
        ("panel_a", "learned_gbrt", "11", 2021): (3, 6.72),
        ("panel_a", "learned_gbrt", "29", 2020): (2, -2.52),
        ("panel_a", "learned_gbrt", "29", 2021): (3, -2.52),
        ("panel_a", "ts_momentum", "deterministic", 2020): (2, 5.04),
        ("panel_a", "ts_momentum", "deterministic", 2021): (3, 5.04),
        ("panel_b", "learned_gbrt", "11", 2019): (1, 7.56),
        ("panel_b", "learned_gbrt", "11", 2021): (1, -5.04),
    }
    assert set(actual.index) == set(expected)
    for key, (observations, excess_return) in expected.items():
        assert actual.loc[key, "observations"] == observations
        assert actual.loc[key, "annualized_excess_return"] == pytest.approx(
            excess_return
        )


@pytest.mark.parametrize("side", ["cash", "baseline"])
@pytest.mark.parametrize("defect", ["missing_first", "missing_last", "shifted", "extra"])
def test_yearly_performance_rejects_nonidentical_cash_return_indices(
    yearly_panel, side, defect,
):
    owner = yearly_panel if side == "cash" else yearly_panel["series"]["learned_gbrt"][11]
    series = owner[side]
    if defect == "missing_first":
        owner[side] = series.iloc[1:]
    elif defect == "missing_last":
        owner[side] = series.iloc[:-1]
    elif defect == "shifted":
        owner[side] = series.set_axis(series.index + pd.Timedelta(days=1))
    else:
        owner[side] = pd.concat(
            [series, pd.Series([0.0], index=pd.to_datetime(["2021-01-07"]))]
        )

    with pytest.raises(ValueError):
        expansion._yearly_performance({"panel_a": yearly_panel})


@pytest.mark.parametrize("defect", ["duplicate", "reversed", "nat", "empty"])
def test_yearly_performance_rejects_invalid_even_when_indices_match(
    yearly_panel, defect,
):
    cash = yearly_panel["cash"]
    baseline = yearly_panel["series"]["learned_gbrt"][11]["baseline"]
    if defect == "empty":
        yearly_panel["cash"] = cash.iloc[:0]
        yearly_panel["series"]["learned_gbrt"][11]["baseline"] = baseline.iloc[:0]
    else:
        dates = cash.index.tolist()
        if defect == "duplicate":
            dates[1] = dates[0]
        elif defect == "reversed":
            dates.reverse()
        else:
            dates[2] = pd.NaT
        cash.index = baseline.index = pd.DatetimeIndex(dates)

    with pytest.raises(ValueError):
        expansion._yearly_performance({"panel_a": yearly_panel})


def test_yearly_performance_requires_calendar_dates(yearly_panel):
    yearly_panel["cash"].index = pd.RangeIndex(5)
    yearly_panel["series"]["learned_gbrt"][11]["baseline"].index = pd.RangeIndex(5)

    with pytest.raises((TypeError, ValueError)):
        expansion._yearly_performance({"panel_a": yearly_panel})


@pytest.mark.parametrize("side", ["cash", "baseline"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_yearly_performance_rejects_nonfinite_values(yearly_panel, side, nonfinite):
    owner = yearly_panel if side == "cash" else yearly_panel["series"]["learned_gbrt"][11]
    owner[side].iloc[2] = nonfinite

    with pytest.raises(ValueError):
        expansion._yearly_performance({"panel_a": yearly_panel})


@pytest.fixture
def seed_metrics():
    # Values and expected differences are independent of the production SWITCHES.
    variants = (
        "baseline", "same_close", "zero_cash", "zero_cost",
        "costed_comparator", "vectorized",
    )
    runs = [
        ("panel_a", "11", [1.0, -0.5, 2.0, 1.5, 3.0, 0.75]),
        ("panel_a", "29", [-2.0, 1.0, -3.0, -0.5, -4.0, -2.5]),
        ("panel_b", "11", [0.5, 2.5, -0.25, 0.0, -0.5, 1.0]),
        ("panel_b", "29", [3.0, 1.0, 3.5, 2.0, 4.0, 3.0]),
    ]
    return pd.DataFrame(
        [
            {
                "panel": panel, "strategy": "learned_gbrt", "seed": seed,
                "variant": variant, "cash_excess_sharpe": value,
            }
            for panel, seed, values in runs
            for variant, value in zip(variants, values, strict=True)
        ]
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_seed_contract_effects_are_signed_within_panel_and_seed(seed_metrics, shuffle):
    if shuffle:
        seed_metrics = seed_metrics.sample(frac=1.0, random_state=23)
    before = seed_metrics.copy(deep=True)

    result = expansion._seed_contract_effects(seed_metrics)

    assert list(result.columns) == ["panel", "seed", "switch", "sharpe_difference"]
    switches = [
        "same_close_minus_baseline", "zero_cash_minus_baseline",
        "zero_cost_minus_baseline", "strategy_minus_costed_comparator",
        "vectorized_minus_event",
    ]
    assert set(result["switch"]) == set(switches) == set(expansion.SWITCHES)
    assert not result.duplicated(["panel", "seed", "switch"]).any()
    assert len(result) == 20
    expected = {
        ("panel_a", "11"): [-1.5, 1.0, 0.5, -2.0, -0.25],
        ("panel_a", "29"): [3.0, -1.0, 1.5, 2.0, -0.5],
        ("panel_b", "11"): [2.0, -0.75, -0.5, 1.0, 0.5],
        ("panel_b", "29"): [-2.0, 0.5, -1.0, -1.0, 0.0],
    }
    actual = result.set_index(["panel", "seed", "switch"])["sharpe_difference"]
    for (panel, seed), differences in expected.items():
        for switch, difference in zip(switches, differences, strict=True):
            assert actual.loc[(panel, seed, switch)] == pytest.approx(difference)
    # Cancellation in family means must not erase opposing individual effects.
    means = result.groupby(["panel", "switch"])["sharpe_difference"].mean()
    assert means.loc[("panel_a", "same_close_minus_baseline")] == pytest.approx(0.75)
    assert means.loc[("panel_a", "strategy_minus_costed_comparator")] == 0.0
    assert means.loc[("panel_b", "zero_cash_minus_baseline")] == pytest.approx(-0.125)
    pd.testing.assert_frame_equal(seed_metrics, before)


def test_seed_contract_effects_filters_other_strategies_before_validation(seed_metrics):
    irrelevant = pd.DataFrame(
        [{
            "panel": "panel_a", "strategy": "ts_momentum", "seed": "11",
            "variant": "baseline", "cash_excess_sharpe": np.nan,
        }] * 2
    )
    expected = expansion._seed_contract_effects(seed_metrics)

    actual = expansion._seed_contract_effects(pd.concat([irrelevant, seed_metrics]))

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "variant",
    ["baseline", "same_close", "zero_cash", "zero_cost", "costed_comparator", "vectorized"],
)
def test_seed_contract_effects_requires_every_variant_for_each_run(seed_metrics, variant):
    missing = (
        (seed_metrics["panel"] == "panel_b")
        & (seed_metrics["seed"] == "29")
        & (seed_metrics["variant"] == variant)
    )
    incomplete = seed_metrics.loc[~missing]

    with pytest.raises(ValueError):
        expansion._seed_contract_effects(incomplete)


@pytest.mark.parametrize("conflicting", [False, True])
def test_seed_contract_effects_rejects_identical_or_conflicting_duplicates(
    seed_metrics, conflicting,
):
    duplicate = seed_metrics.iloc[[0]].copy()
    if conflicting:
        duplicate["cash_excess_sharpe"] = 99.0

    with pytest.raises(ValueError):
        expansion._seed_contract_effects(pd.concat([seed_metrics, duplicate]))


@pytest.mark.parametrize("column", ["panel", "seed"])
def test_seed_contract_effects_rejects_null_run_identifiers(seed_metrics, column):
    missing_identifier = (
        (seed_metrics["panel"] == "panel_b") & (seed_metrics["seed"] == "29")
    )
    seed_metrics.loc[missing_identifier, column] = None

    # Groupby otherwise silently discards all six rows of this supplied run.
    with pytest.raises(ValueError):
        expansion._seed_contract_effects(seed_metrics)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "variant",
    ["baseline", "same_close", "zero_cash", "zero_cost", "costed_comparator", "vectorized"],
)
def test_seed_contract_effects_rejects_nonfinite_variants(seed_metrics, variant, nonfinite):
    bad_row = (
        (seed_metrics["panel"] == "panel_b")
        & (seed_metrics["seed"] == "29")
        & (seed_metrics["variant"] == variant)
    )
    seed_metrics.loc[bad_row, "cash_excess_sharpe"] = nonfinite

    with pytest.raises(ValueError):
        expansion._seed_contract_effects(seed_metrics)
