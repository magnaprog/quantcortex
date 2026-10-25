from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.research.expansion import (
    FROZEN_PROTOCOL_COMMIT,
    FROZEN_PROTOCOL_SHA256,
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
from scripts.fetch_expansion_data import _adjusted_close, _validate_panel
from scripts.run_expansion_experiments import (
    PANEL_LABELS,
    STRATEGY_LABELS,
    SWITCHES,
    _family_summary,
    _load_panel,
    _load_protocol,
    _plot_baseline,
    _plot_effects,
    _plot_engine_conformance,
    _plot_learned_seeds,
    _plot_seed_contract_effects,
    _plot_yearly_performance,
    _rank_reversals,
    _source_tree_manifest,
    _write_generated_values,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_panel_loader_validates_published_metadata(tmp_path):
    protocol_path = REPO_ROOT / "paper" / "expansion" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dates = pd.bdate_range("2024-01-02", periods=30)
    protocol["data"] = {
        **protocol["data"],
        "start": dates[0].date().isoformat(),
        "evaluation_start": dates[10].date().isoformat(),
        "evaluation_end": dates[-1].date().isoformat(),
        "minimum_pre_evaluation_sessions": 5,
    }
    panel = pd.DataFrame(
        {
            "A": np.linspace(100.0, 110.0, len(dates)),
            "B": np.linspace(90.0, 105.0, len(dates)),
            "SHV": np.linspace(100.0, 100.5, len(dates)),
        },
        index=dates,
    )
    panel.index.name = "date"
    csv_path = tmp_path / "fixture_panel.csv"
    metadata_path = tmp_path / "fixture_panel.metadata.json"
    panel.to_csv(csv_path, float_format="%.17g")
    pre_evaluation = dates < dates[10]
    metadata = {
        "schema_version": 1,
        "panel": "fixture_panel",
        "symbols": ["A", "B", "SHV"],
        "provider": "Yahoo Finance via yfinance",
        "provider_terms_independently_verified": False,
        "retrieved_at": "2026-06-18T22:37:51Z",
        "request": protocol["data"]["provider_request"],
        "protocol_path": "paper/expansion/protocol.json",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "yfinance_version": "1.4.1",
        "terms_urls": [
            "https://ranaroussi.github.io/yfinance/",
            "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
        ],
        "raw_data_committed": False,
        "input_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "provider_rows": len(panel),
        "complete_rows": len(panel),
        "dropped_incomplete_rows": 0,
        "missing_by_symbol": {"A": 0, "B": 0, "SHV": 0},
        "pre_evaluation_sessions": int(pre_evaluation.sum()),
        "pre_evaluation_months": int(dates[pre_evaluation].to_period("M").nunique()),
        "evaluation_sessions": int((~pre_evaluation).sum()),
        "first_date": dates[0].date().isoformat(),
        "last_date": dates[-1].date().isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded, published = _load_panel(
        REPO_ROOT,
        tmp_path,
        "fixture_panel",
        ["A", "B"],
        protocol,
    )
    pd.testing.assert_frame_equal(loaded, panel, check_freq=False)
    assert published == metadata

    tampered = copy.deepcopy(metadata)
    tampered["provider_rows"] += 1
    metadata_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provider-row accounting"):
        _load_panel(REPO_ROOT, tmp_path, "fixture_panel", ["A", "B"], protocol)


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


@pytest.mark.parametrize("track_literal", [False, True])
def test_panel_loader_checks_literal_directory_path(tmp_path, track_literal):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    protocol_path = tmp_path / "paper" / "expansion" / "protocol.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("{}", encoding="ascii")
    panel_dir = tmp_path / "panels[1]"
    neighbor_dir = tmp_path / "panels1"
    for directory in (panel_dir, neighbor_dir):
        directory.mkdir()
        (directory / "fixture.csv").write_text("date,A,SHV\n", encoding="ascii")
    (panel_dir / "fixture.metadata.json").write_text("{}", encoding="ascii")
    tracked = (panel_dir if track_literal else neighbor_dir) / "fixture.csv"
    subprocess.run(
        ["git", "--literal-pathspecs", "add", "--", str(tracked)],
        cwd=tmp_path,
        check=True,
    )

    # The untracked literal must reach metadata validation despite its neighbor.
    message = "raw panel must not be tracked" if track_literal else "unsupported metadata"
    with pytest.raises(ValueError, match=message):
        _load_panel(
            tmp_path, panel_dir, "fixture", ["A"], {"data": {"cash_proxy": "SHV"}}
        )


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


def test_machine_protocol_matches_frozen_model_and_panel_choices():
    protocol_path = REPO_ROOT / "paper" / "expansion" / "protocol.json"
    protocol = _load_protocol(protocol_path)

    assert protocol["status"] == (
        "repository_frozen_prospective_not_externally_registered"
    )
    assert protocol["historical_case_confirmatory"] is False
    assert list(protocol["panels"]) == [
        "us_sector_etfs",
        "country_equity_etfs",
    ]
    learned = protocol["strategies"]["learned_gbrt"]
    assert learned["estimator"] == (
        "sklearn.ensemble.GradientBoostingRegressor"
    )
    assert learned["seeds"] == [11, 29, 47, 71, 97]
    assert protocol["execution"]["cost_per_one_way_gross_notional"] == 0.0013
    assert protocol["uncertainty"] == {
        "method": "joint circular block bootstrap",
        "replications": 5000,
        "primary_block_sessions": 21,
        "sensitivity_block_sessions": [5, 63],
        "seed": 20260618,
        "interval": "two-sided 95 percent percentile",
        "annualized_arithmetic_return": "252 times daily arithmetic mean",
        "sharpe": (
            "sqrt(252) times mean daily strategy-minus-SHV return divided by "
            "sample standard deviation"
        ),
    }
    assert FROZEN_PROTOCOL_COMMIT == (
        "4018f4063f46889f41d6981db5a71079e1dbd713"
    )
    assert FROZEN_PROTOCOL_SHA256 == (
        "e49e41a12a19fa5404a573ba5e21eb8a2888e616985f8c610d9652866923315c"
    )


def test_provider_adjusted_close_selection_and_complete_intersection():
    index = pd.bdate_range("2017-01-02", "2018-03-30")
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], ["A", "B", "SHV"]]
    )
    values = np.tile(np.arange(1, len(columns) + 1, dtype=float), (len(index), 1))
    download = pd.DataFrame(values, index=index, columns=columns)
    download.loc[index[5], ("Adj Close", "B")] = np.nan

    adjusted = _adjusted_close(download, ["A", "B", "SHV"])
    complete, diagnostics = _validate_panel(
        adjusted,
        evaluation_start="2018-01-02",
        evaluation_end="2018-03-30",
        minimum_pre_evaluation_sessions=200,
        minimum_mature_training_months=12,
    )

    assert list(complete.columns) == ["A", "B", "SHV"]
    assert index[5] not in complete.index
    assert diagnostics["dropped_incomplete_rows"] == 1
    assert diagnostics["missing_by_symbol"]["B"] == 1


def test_provider_panel_validation_rejects_a_missing_evaluation_month():
    index = pd.bdate_range("2016-01-04", "2018-03-30")
    frame = pd.DataFrame(100.0, index=index, columns=["A", "B", "SHV"])
    frame = frame.loc[frame.index.to_period("M") != pd.Period("2018-02", freq="M")]

    with pytest.raises(ValueError, match="missing evaluation months: 2018-02"):
        _validate_panel(
            frame,
            evaluation_start="2018-01-02",
            evaluation_end="2018-03-30",
            minimum_pre_evaluation_sessions=252,
            minimum_mature_training_months=24,
        )


def _plot_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    metric_rows = []
    engine_rows = []
    for panel_index, panel in enumerate(PANEL_LABELS):
        for strategy_index, strategy in enumerate(STRATEGY_LABELS):
            for variant_index, variant in enumerate(
                [
                    "baseline",
                    "same_close",
                    "zero_cash",
                    "zero_cost",
                    "costed_comparator",
                    "vectorized",
                ]
            ):
                summary_rows.append(
                    {
                        "panel": panel,
                        "strategy": strategy,
                        "variant": variant,
                        "family_size": 1,
                        "observations": 100 * (panel_index + 1),
                        "annualized_arithmetic_return": (
                            0.01 * (strategy_index + 1) - 0.002 * variant_index
                        ),
                        "cash_excess_sharpe": (
                            0.2 * strategy_index - 0.03 * variant_index
                        ),
                        "cagr": 0.008 * (strategy_index + 1) - 0.001 * variant_index,
                        "max_drawdown": -0.05 * (strategy_index + 1),
                        "mean_risky_exposure": 0.6 + 0.05 * strategy_index,
                        "annualized_gross_traded_notional": 2.5 + strategy_index,
                        "arithmetic_cost_drag": 0.01 + 0.002 * variant_index,
                    }
                )
            if strategy == "learned_gbrt":
                for seed in [11, 29, 47, 71, 97]:
                    metric_rows.append(
                        {
                            "panel": panel,
                            "strategy": "learned_gbrt",
                            "seed": str(seed),
                            "variant": "baseline",
                            "cash_excess_sharpe": 0.01 * seed - panel_index,
                        }
                    )
            else:
                metric_rows.append({
                    "panel": panel, "strategy": strategy, "seed": "deterministic",
                    "variant": "baseline", "cash_excess_sharpe": 0.2 * strategy_index,
                })
            seeds = [11, 29, 47, 71, 97] if strategy == "learned_gbrt" else ["deterministic"]
            for seed in seeds:
                engine_rows.append(
                    {
                        "panel": panel, "strategy": strategy, "seed": str(seed),
                        "max_absolute_daily_return_difference": (
                            0.00001 * (strategy_index + 1)
                        ),
                        "final_wealth_difference": 0.001 * (strategy_index + 1),
                    }
                )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(engine_rows),
    )


def _followup_plot_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    yearly_rows = []
    seed_rows = []
    learned_offsets = dict(zip(
        ["11", "29", "47", "71", "97"], [-0.06, -0.02, 0.01, 0.03, 0.09],
        strict=True,
    ))
    for panel_index, panel in enumerate(PANEL_LABELS):
        for strategy_index, strategy in enumerate(STRATEGY_LABELS):
            seeds = learned_offsets if strategy == "learned_gbrt" else {"deterministic": 0.0}
            for seed, offset in seeds.items():
                for year in range(2018, 2026):
                    yearly_rows.append({
                        "panel": panel, "strategy": strategy, "seed": seed,
                        "year": year, "observations": 250 + year % 3,
                        "annualized_excess_return": (
                            -0.03 + 0.02 * panel_index + 0.01 * strategy_index
                            + 0.005 * (year - 2018) + offset
                        ),
                    })
        for seed_index, seed in enumerate(learned_offsets):
            for switch_index, switch in enumerate(SWITCHES):
                seed_rows.append({
                    "panel": panel, "seed": seed, "switch": switch,
                    "sharpe_difference": (
                        0.02 * (switch_index - 2) + 0.005 * (seed_index - 2)
                        + 0.01 * panel_index
                    ),
                })
    return pd.DataFrame(yearly_rows), pd.DataFrame(seed_rows)


def test_family_summary_counts_seeds_and_baseline_ranks_do_not_change():
    summary, _, _ = _plot_fixture_frames()
    seed_metrics = pd.concat([summary.assign(seed="11"), summary.assign(seed="29")])

    aggregated = _family_summary(seed_metrics)
    ranks = _rank_reversals(aggregated)

    assert (aggregated["family_size"] == 2).all()
    assert len(ranks) == len(PANEL_LABELS) * 6 * len(STRATEGY_LABELS)
    baseline = ranks.loc[ranks["variant"] == "baseline"]
    assert (baseline["rank_change"] == 0).all()
    assert not baseline["baseline_tied"].any()
    assert not baseline["variant_tied"].any()


def test_rank_reversals_disclose_ties_at_exported_precision():
    summary, _, _ = _plot_fixture_frames()
    tied_strategies = sorted(STRATEGY_LABELS)[:2]
    tied_rows = (
        (summary["panel"] == "us_sector_etfs")
        & (summary["variant"] == "costed_comparator")
        & summary["strategy"].isin(tied_strategies)
    )
    summary.loc[tied_rows, "cash_excess_sharpe"] = [
        0.5000000000001,
        0.5000000000002,
    ]

    ranks = _rank_reversals(summary)
    tied_ranks = ranks.loc[
        (ranks["panel"] == "us_sector_etfs")
        & (ranks["variant"] == "costed_comparator")
        & ranks["strategy"].isin(tied_strategies)
    ].sort_values("strategy")

    assert tied_ranks["variant_tied"].all()
    assert not tied_ranks["baseline_tied"].any()
    assert tied_ranks["strategy"].tolist() == tied_strategies
    assert tied_ranks["variant_rank"].tolist() == [1, 2]
    assert set(tied_ranks["rank_method"]) == {
        "descending_12_significant_digit_cash_excess_sharpe_"
        "with_strategy_key_tiebreak"
    }


def test_published_baseline_legend_clears_both_subplots(monkeypatch, tmp_path):
    # Published aggregates fix chart positions, not a new financial experiment.
    summary = pd.read_csv(
        REPO_ROOT / "paper" / "expansion" / "results" / "family_summary.csv"
    )
    inspected = []

    def inspect_figure(fig, path):
        assert path == tmp_path / "baseline_performance"
        fig.set_dpi(180)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend, = fig.legends
        legend_box = legend.get_window_extent(renderer)
        assert len(legend.get_texts()) == 4
        for axis, panel in zip(fig.axes, PANEL_LABELS, strict=True):
            assert not legend_box.overlaps(axis.get_window_extent(renderer))
            frame = summary.loc[
                (summary["panel"] == panel) & (summary["variant"] == "baseline")
            ].set_index("strategy")
            assert len(axis.collections) == 4
            for mark, strategy in zip(axis.collections, STRATEGY_LABELS, strict=True):
                row = frame.loc[strategy]
                np.testing.assert_array_equal(
                    mark.get_offsets(),
                    [[row.annualized_arithmetic_return * 100.0, row.cash_excess_sharpe]],
                )
        inspected.append(True)

    monkeypatch.setattr(
        "scripts.run_expansion_experiments._save_figure", inspect_figure
    )
    _plot_baseline(summary, tmp_path)
    assert inspected == [True]


def test_all_expansion_figures_render_from_complete_tables(tmp_path):
    summary, learned_metrics, engine = _plot_fixture_frames()
    yearly, seed_effects = _followup_plot_fixture_frames()
    effect_rows = []
    for panel in PANEL_LABELS:
        for strategy in STRATEGY_LABELS:
            for switch_index, switch in enumerate(SWITCHES):
                value = (switch_index - 2) * 0.002
                effect_rows.append(
                    {
                        "panel": panel,
                        "strategy": strategy,
                        "switch": switch,
                        "block_length": 21,
                        "annualized_mean_difference": value,
                        "annualized_mean_ci_95_lower": value - 0.01,
                        "annualized_mean_ci_95_upper": value + 0.01,
                        "sharpe_difference": value * 10.0,
                        "sharpe_ci_95_lower": value * 10.0 - 0.1,
                        "sharpe_ci_95_upper": value * 10.0 + 0.1,
                    }
                )
    effects = pd.DataFrame(effect_rows)

    _plot_baseline(summary, tmp_path)
    _plot_effects(effects, tmp_path, metric="return")
    _plot_effects(effects, tmp_path, metric="sharpe")
    _plot_engine_conformance(engine, tmp_path)
    _plot_learned_seeds(learned_metrics, tmp_path)
    _plot_yearly_performance(yearly, tmp_path)
    _plot_seed_contract_effects(seed_effects, tmp_path)

    expected_stems = {
        "baseline_performance", "contract_effects_return", "contract_effects_sharpe",
        "engine_conformance", "learned_seed_sensitivity",
        "yearly_performance", "seed_contract_effects",
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        f"{stem}.{suffix}" for stem in expected_stems for suffix in ("png", "pdf")
    }
    assert all(path.stat().st_size > 1_000 for path in tmp_path.iterdir())


def test_followup_heatmaps_show_family_means_and_signed_seed_cells(monkeypatch, tmp_path):
    yearly, seed_effects = _followup_plot_fixture_frames()
    assert len(yearly) == 128
    assert len(seed_effects) == 50
    inspected = []

    def inspect_figure(fig, path):
        panels = [axis for axis in fig.axes if axis.images]
        assert [axis.get_title() for axis in panels] == list(PANEL_LABELS.values())
        images = [axis.images[0] for axis in panels]
        assert images[0].get_clim() == images[1].get_clim()
        assert images[0].norm.vmin == -images[0].norm.vmax
        values = [np.asarray(artist.get_array()) for artist in images]
        if path.name == "yearly_performance":
            assert [array.shape for array in values] == [(4, 8), (4, 8)]
            assert values[0][0, 0] == pytest.approx(-3.0)
            # Five seed offsets sum to .05: mean .01, displayed as 1 percent.
            assert values[0][3, 0] == pytest.approx(1.0)
            assert values[0][3, 1] == pytest.approx(1.5)
            assert values[1][3, 7] == pytest.approx(6.5)
            for axis in panels:
                assert [tick.get_text() for tick in axis.get_xticklabels()] == [
                    str(year) for year in range(2018, 2026)
                ]
            assert any("(%)" in axis.get_xlabel() for axis in fig.axes)
            precision = 1
        else:
            assert path.name == "seed_contract_effects"
            assert [array.shape for array in values] == [(5, 5), (5, 5)]
            assert values[0][0, 0] == pytest.approx(-0.05)
            assert values[0][4, 4] == pytest.approx(0.05)
            assert values[1][0, 0] == pytest.approx(-0.04)
            for axis in panels:
                assert [tick.get_text() for tick in axis.get_yticklabels()] == [
                    "11", "29", "47", "71", "97"
                ]
            precision = 4
        for axis, array in zip(panels, values, strict=True):
            labels = {text.get_position(): text.get_text() for text in axis.texts}
            for row, column in np.ndindex(array.shape):
                assert labels[(column, row)] == f"{array[row, column]:.{precision}f}"
        inspected.append(path.name)

    monkeypatch.setattr("scripts.run_expansion_experiments._save_figure", inspect_figure)
    _plot_yearly_performance(yearly, tmp_path)
    _plot_seed_contract_effects(seed_effects, tmp_path)
    assert inspected == ["yearly_performance", "seed_contract_effects"]


@pytest.mark.parametrize("shuffle_seed", [0, 31])
@pytest.mark.parametrize("rank_tie", [False, True])
def test_expansion_latex_values_are_derived_from_aggregate_tables(
    tmp_path, shuffle_seed, rank_tie,
):
    summary, learned_metrics, engine = _plot_fixture_frames()
    # Reverse the retroactive ordering and distinguish the two ETF sets.
    summary.loc[summary["variant"] == "same_close", "cash_excess_sharpe"] *= -1
    summary.loc[summary["panel"] == "country_equity_etfs", "cash_excess_sharpe"] -= 1
    baseline = summary["variant"] == "baseline"
    # Equal printed returns must not create a tie; exact drawdown ties share bold.
    summary.loc[baseline & (summary["strategy"] == "ts_momentum"),
                "annualized_arithmetic_return"] = 0.040001
    summary.loc[baseline & (summary["strategy"] == "learned_gbrt"),
                "annualized_arithmetic_return"] = 0.040002
    summary.loc[baseline & (summary["strategy"] == "cross_sectional_momentum"),
                "max_drawdown"] = -0.05
    if rank_tie:
        sector_zero_cost = (
            (summary["panel"] == "us_sector_etfs") & (summary["variant"] == "zero_cost")
        )
        # Ranking records round to 12 significant digits before resolving ties.
        for strategy, value in (
            ("cross_sectional_momentum", 0.60000000000003),
            ("learned_gbrt", 0.60000000000004),
        ):
            summary.loc[sector_zero_cost & (summary["strategy"] == strategy),
                        "cash_excess_sharpe"] = value
    effects = []
    for panel in PANEL_LABELS:
        for strategy in STRATEGY_LABELS:
            for switch_index, switch in enumerate(SWITCHES):
                value = (switch_index - 2) * 0.01
                effects.append(
                    {
                        "panel": panel,
                        "strategy": strategy,
                        "switch": switch,
                        "block_length": 21,
                        "replications": 5000,
                        "annualized_mean_difference": value,
                        "annualized_mean_ci_95_lower": value - 0.001,
                        "annualized_mean_ci_95_upper": value + 0.001,
                        "sharpe_ci_95_lower": value - 0.001,
                        "sharpe_ci_95_upper": value + 0.001,
                    }
                )
    ranks = _rank_reversals(summary)
    path = tmp_path / "generated_values.tex"

    _write_generated_values(
        summary=summary.sample(frac=1, random_state=shuffle_seed),
        effects=pd.DataFrame(effects),
        engine=engine,
        metrics=learned_metrics,
        ranks=ranks.sample(frac=1, random_state=shuffle_seed + 1),
        data_records=[
            {"panel": "us_sector_etfs", "input_sha256": "a" * 64},
            {"panel": "country_equity_etfs", "input_sha256": "b" * 64},
        ],
        path=path,
    )

    generated = path.read_text(encoding="ascii")
    assert "\\newcommand{\\ExpansionPanelCount}{2}" in generated
    assert "\\newcommand{\\ExpansionFamilyCount}{4}" in generated
    assert "\\newcommand{\\ExpansionBootstrapReplications}{5,000}" in generated
    assert f"\\newcommand{{\\ExpansionProtocolDigest}}{{{FROZEN_PROTOCOL_SHA256}}}" in generated
    assert f"\\newcommand{{\\ExpansionSectorInputDigest}}{{{'a' * 64}}}" in generated
    assert f"\\newcommand{{\\ExpansionCountryInputDigest}}{{{'b' * 64}}}" in generated
    assert (
        r"\shortstack[l]{United States\\sectors} & "
        r"Time series momentum"
    ) in generated
    header = r"\newcommand{\ExpansionBaselineRows}{"
    baseline_rows = generated.split(header, 1)[1].split("}\n", 1)[0].splitlines()
    assert baseline_rows[0] == (
        r"\shortstack[l]{United States\\sectors} & "
        r"Time series momentum & "
        r"4.00\% & 0.00 & 0.80\% & \textbf{-5.00\%} \\"
    )
    assert baseline_rows[1].endswith(r" & \textbf{-5.00\%} \\")
    assert baseline_rows[3] == (
        r"\shortstack[l]{United States\\sectors} & "
        r"Gradient boosting & \textbf{4.00\%} & \textbf{0.60}"
        r" & \textbf{3.20\%} & -20.00\% \\"
    )
    assert baseline_rows[7] == (
        r"Country equities & Gradient boosting & "
        r"\textbf{4.00\%} & \textbf{-0.40}"
        r" & \textbf{3.20\%} & -20.00\% \\"
    )
    header = r"\newcommand{\ExpansionTradingRows}{"
    assert generated.count(header) == 1
    trading_rows = generated.split(header, 1)[1].split("}\n", 1)[0].splitlines()
    assert len(trading_rows) == 8
    assert all(len(row.split(" & ")) == 5 and row.endswith(r"\\") for row in trading_rows)
    # .01 cumulative drag over 100 sessions annualizes to 2.52%; 200 to 1.26%.
    assert trading_rows[0] == (
        r"\shortstack[l]{United States\\sectors} & "
        r"Time series momentum & 60.00\% & 2.50 & 2.52\% \\"
    )
    assert trading_rows[4] == (
        r"Country equities & Time series momentum & "
        r"60.00\% & 2.50 & 1.26\% \\"
    )
    header = r"\newcommand{\ExpansionRankRows}{"
    assert generated.count(header) == 1
    rank_rows = generated.split(header, 1)[1].split("}\n", 1)[0].splitlines()
    expected_rank_rows = [
        r"\multicolumn{7}{@{}l@{}}{\textit{United States sectors}} \\",
        r"Time series momentum & "
        r"$0.000$ & 4 & $\mathbf{0.030}$ & 1 & $-0.090$ & 4 \\",
        r"Cross-sectional momentum & "
        r"$0.200$ & 3 & $-0.170$ & 2 & $0.110$ & 3 \\",
        r"Short-term reversal & $0.400$ & 2 & $-0.370$ & 3 & $0.310$ & 2 \\",
        r"Gradient boosting & "
        r"$\mathbf{0.600}$ & 1 & $-0.570$ & 4 & $\mathbf{0.510}$ & 1 \\",
        r"\addlinespace",
        r"\multicolumn{7}{@{}l@{}}{\textit{Country equities}} \\",
        r"Time series momentum & "
        r"$-1.000$ & 4 & $\mathbf{-0.970}$ & 1 & $-1.090$ & 4 \\",
        r"Cross-sectional momentum & "
        r"$-0.800$ & 3 & $-1.170$ & 2 & $-0.890$ & 3 \\",
        r"Short-term reversal & $-0.600$ & 2 & $-1.370$ & 3 & $-0.690$ & 2 \\",
        r"Gradient boosting & "
        r"$\mathbf{-0.400}$ & 1 & $-1.570$ & 4 & $\mathbf{-0.490}$ & 1 \\",
    ]
    if rank_tie:
        expected_rank_rows[2] = expected_rank_rows[2].replace(
            r"$0.110$ & 3", r"$\mathbf{0.600}$ & 1",
        )
        expected_rank_rows[3] = expected_rank_rows[3].replace(
            r"$0.310$ & 2", r"$0.310$ & 3",
        )
        expected_rank_rows[4] = expected_rank_rows[4].replace(
            r"$\mathbf{0.510}$ & 1", r"$\mathbf{0.600}$ & 2",
        )
    assert rank_rows == expected_rank_rows
    assert "ExpansionComparatorRankChanges" not in generated
    assert "95%" not in generated


def test_expansion_source_fingerprint_includes_contract_and_operational_sources():
    manifest = _source_tree_manifest(REPO_ROOT)

    assert "scripts/fetch_expansion_data.py" in manifest["files"]
    assert "scripts/publish_artifact_bundle.py" in manifest["files"]
    assert "scripts/release_expansion_artifacts.sh" in manifest["files"]
    assert "scripts/run_expansion_experiments.py" in manifest["files"]
    assert "paper/preregistration.md" in manifest["files"]
    assert "paper/expansion/protocol.json" in manifest["files"]
    assert "schemas/canonical_target_tape.schema.json" in manifest["files"]
    assert len(manifest["sha256"]) == 64
