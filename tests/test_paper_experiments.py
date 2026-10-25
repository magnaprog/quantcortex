from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantcortex.strategies.multi_asset_rotation import MultiAssetRotation
from scripts.fetch_expansion_data import _is_tracked as _fetch_path_is_tracked
from scripts.generate_report import _ann_sharpe as _report_sharpe
from scripts.run_paper_experiments import (
    BOOTSTRAP_BLOCK_LENGTHS,
    PAPER_MAX_FORWARD_FILL,
    PRIMARY_BOOTSTRAP_BLOCK_LENGTH,
    SOURCE_TREE_FIXED_FILES,
    STRATEGY_PARAMETERS,
    _benchmark_returns,
    _cagr,
    _costed_target_exposure_comparator,
    _engine_difference_summary,
    _git_metadata,
    _git_path_is_tracked,
    _max_drawdown,
    _percentile_errorbar,
    _save_figures,
    _sharpe,
    _subperiod_rows,
    _tex_number,
    circular_block_bootstrap,
    circular_block_bootstrap_frame,
    circular_block_bootstrap_sharpe,
    circular_block_bootstrap_sharpe_frame,
    evaluation_contract,
    invalid_same_close_diagnostic,
    iso_timestamp,
    nonempty_text,
    return_decomposition,
    run_experiments,
    source_tree_manifest,
    verify_expansion_artifacts,
)
from scripts.run_paper_experiments import (
    main as paper_main,
)
from scripts.validate_performance import ann_sharpe as _validation_sharpe


@pytest.mark.parametrize("helper", [_sharpe, _report_sharpe, _validation_sharpe])
@pytest.mark.parametrize("observations", [30, 252])
@pytest.mark.parametrize("variation", [0.0, 1e-14])
def test_scalar_sharpe_distinguishes_constant_from_tiny_variation(
    helper, observations, variation,
):
    returns = pd.Series(0.01, index=pd.RangeIndex(observations))
    returns.iloc[-1] += variation

    actual = helper(returns)

    if variation == 0.0:
        assert np.isnan(actual)
    else:
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected)


def test_circular_block_bootstrap_is_deterministic_and_reports_sign():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015] * 20)
    first = circular_block_bootstrap(
        returns,
        block_length=5,
        replications=200,
        seed=7,
    )
    second = circular_block_bootstrap(
        returns,
        block_length=5,
        replications=200,
        seed=7,
    )

    assert first == second
    assert first["annualized_mean"] == pytest.approx(returns.mean() * 252.0)
    assert first["positive_draw_fraction"] == 1.0
    assert first["ci_95_lower"] < first["ci_95_upper"]


@pytest.mark.parametrize("block_length", [0, 5])
def test_circular_block_bootstrap_rejects_invalid_blocks(block_length):
    with pytest.raises(ValueError, match="block_length"):
        circular_block_bootstrap(
            pd.Series([0.01, 0.02, 0.03, 0.04]),
            block_length=block_length,
            replications=10,
        )


def test_benchmark_returns_share_one_capital_clock():
    index = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 110.0, 99.0, 108.9],
            "QQQ": [100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )

    spy, equal_weight = _benchmark_returns(prices, index[1:])

    assert spy.iloc[0] == pytest.approx(0.10)
    assert _cagr(spy, periods_per_year=3.0) == pytest.approx(0.089)
    expected_curve = prices.div(prices.iloc[0]).mean(axis=1)
    expected = expected_curve.pct_change(fill_method=None).iloc[1:]
    pd.testing.assert_series_equal(equal_weight, expected, check_names=False)


def test_joint_bootstrap_uses_complete_finite_rows():
    frame = pd.DataFrame(
        {
            "first": [0.01, 0.02, -0.01, 0.03],
            "second": [0.02, -0.01, 0.01, 0.00],
        }
    )
    frame["total"] = frame["first"] + frame["second"]

    result = circular_block_bootstrap_frame(
        frame,
        block_length=2,
        replications=100,
        seed=11,
    ).set_index("series")

    assert result.loc["total", "annualized_mean"] == pytest.approx(
        result.loc["first", "annualized_mean"]
        + result.loc["second", "annualized_mean"]
    )
    incomplete = frame.copy()
    incomplete.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        circular_block_bootstrap_frame(
            incomplete,
            block_length=2,
            replications=10,
        )


def test_sharpe_block_bootstrap_is_deterministic_and_reports_sample_statistic():
    returns = pd.Series(
        np.random.default_rng(3).normal(0.001, 0.01, size=150),
        name="excess",
    )
    first = circular_block_bootstrap_sharpe(
        returns,
        block_length=5,
        replications=250,
        seed=17,
    )
    second = circular_block_bootstrap_sharpe(
        returns,
        block_length=5,
        replications=250,
        seed=17,
    )

    assert first == second
    assert first["sample_sharpe"] == pytest.approx(
        returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
    )
    assert first["ci_95_lower"] < first["ci_95_upper"]

    with pytest.raises(ValueError, match="undefined"):
        circular_block_bootstrap_sharpe(
            pd.Series([0.01] * 10),
            block_length=2,
            replications=20,
        )


def test_joint_sharpe_bootstrap_rejects_incomplete_rows():
    frame = pd.DataFrame(
        {
            "first": [0.01, 0.02, -0.01],
            "second": [0.00, np.nan, 0.01],
        }
    )
    with pytest.raises(ValueError, match="complete"):
        circular_block_bootstrap_sharpe_frame(
            frame,
            block_length=2,
            replications=20,
        )


def test_costed_target_exposure_comparator_uses_next_bar_and_charges_rebalances():
    index = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 110.0, 110.0, 121.0, 121.0],
            "B": [100.0, 100.0, 110.0, 110.0, 121.0, 121.0],
        },
        index=index,
    )
    weights = pd.DataFrame(
        {"A": [0.5, 1.0], "B": [0.0, 0.0]},
        index=pd.DatetimeIndex([index[0], index[2]]),
    )
    cash = pd.Series(0.0, index=index)
    evaluation_index = index[2:]

    no_cost = _costed_target_exposure_comparator(
        weights,
        prices,
        cash,
        evaluation_index,
        cost_bps=0.0,
    )
    costed = _costed_target_exposure_comparator(
        weights,
        prices,
        cash,
        evaluation_index,
        cost_bps=13.0,
    )

    expected = pd.Series([0.05, 0.0, 0.10, 0.0], index=evaluation_index)
    pd.testing.assert_series_equal(
        no_cost.returns.reindex(evaluation_index),
        expected,
    )
    assert costed.costs.loc[index[2]] == 0.0
    assert costed.costs.loc[index[3]] > 0.0
    assert _growth_for_test(costed.returns.reindex(evaluation_index)) < (
        _growth_for_test(no_cost.returns.reindex(evaluation_index))
    )


def _growth_for_test(returns: pd.Series) -> float:
    return float((1.0 + returns).prod())


def test_return_decomposition_is_an_exact_daily_identity():
    index = pd.bdate_range("2024-01-01", periods=4)
    cash = pd.Series([0.001, 0.001, 0.001, 0.001], index=index)
    passive = pd.Series([0.01, -0.01, 0.02, 0.00], index=index)
    exposure = pd.Series([0.0, 0.5, 1.0, 0.5], index=index)
    gross = pd.Series([0.001, 0.004, 0.015, -0.002], index=index)
    net = gross - pd.Series([0.0, 0.001, 0.002, 0.001], index=index)

    components, constant_passive = return_decomposition(
        net=net,
        gross=gross,
        cash=cash,
        passive_basket=passive,
        risky_exposure=exposure,
    )

    pd.testing.assert_series_equal(
        components.iloc[:, :4].sum(axis=1),
        components["net_excess_over_cash"],
        check_names=False,
    )
    expected_constant = 0.5 * passive + 0.5 * cash
    pd.testing.assert_series_equal(
        constant_passive,
        expected_constant.rename("constant_exposure_passive_basket"),
    )


def test_same_close_diagnostic_is_explicitly_lookahead():
    index = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=index)
    weights = pd.DataFrame({"A": [1.0]}, index=index[1:2])
    cash = pd.Series(0.0, index=index)

    diagnostic = invalid_same_close_diagnostic(
        weights,
        prices,
        cash,
        cost_bps=0.0,
    )

    assert diagnostic["net"].iloc[0] == 0.0
    assert diagnostic["net"].iloc[1] == pytest.approx(0.10)
    assert diagnostic["exposure"].iloc[1] == pytest.approx(1.0)


def test_drawdown_uses_running_peak():
    returns = pd.Series([0.10, -0.20, 0.10])
    assert _max_drawdown(returns) == pytest.approx(-0.20)
    assert np.isfinite(_cagr(returns))


def test_paper_number_formatting_normalizes_rounded_zero():
    assert _tex_number(-0.000021, digits=2) == "0.00"
    assert _tex_number(-0.1547, digits=4) == r"\mbox{-0.1547}"


def test_paper_provenance_text_rejects_empty_values():
    assert nonempty_text("  Yahoo Finance  ") == "Yahoo Finance"
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
        nonempty_text("   ")


@pytest.mark.parametrize("subdirectory", ["results", "figures"])
def test_verify_expansion_artifacts_rejects_top_level_directory_symlinks(
    tmp_path, monkeypatch, subdirectory,
):
    expansion = tmp_path / "expansion"
    artifacts = {}
    contents = b"synthetic artifact\n"
    for name in ("results", "figures"):
        directory = expansion / name
        directory.mkdir(parents=True)
        (directory / "artifact.txt").write_bytes(contents)
        artifacts[f"{name}/artifact.txt"] = hashlib.sha256(contents).hexdigest()
    (expansion / "results" / "manifest.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="ascii"
    )
    verify_expansion_artifacts(expansion)

    directory = expansion / subdirectory
    stored = expansion / f"stored_{subdirectory}"
    directory.rename(stored)
    directory.symlink_to(stored, target_is_directory=True)

    with pytest.raises(
        SystemExit, match=f"expansion directory is a symlink: {subdirectory}"
    ):
        verify_expansion_artifacts(expansion)

    def unexpected_read_or_traversal(*args, **kwargs):
        pytest.fail("top-level symlinks must be rejected before reading or traversal")

    monkeypatch.setattr(Path, "read_text", unexpected_read_or_traversal)
    monkeypatch.setattr(Path, "rglob", unexpected_read_or_traversal)
    with pytest.raises(
        SystemExit, match=f"expansion directory is a symlink: {subdirectory}"
    ):
        verify_expansion_artifacts(expansion)


@pytest.mark.parametrize("value", ["2026-06-16", "2026-06-16T12:30:00Z"])
def test_paper_retrieval_timestamp_accepts_iso_8601(value):
    assert iso_timestamp(value) == value


def test_paper_retrieval_timestamp_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError, match="ISO-8601"):
        iso_timestamp("June 16, 2026")


def test_paper_strategy_parameters_are_explicit_and_constructible():
    strategy = MultiAssetRotation(**STRATEGY_PARAMETERS)

    assert strategy.top_n_groups == 2
    assert strategy.ir_lookback == 126
    assert strategy.mom_lookback == 126
    assert strategy.mom_gap == 21
    assert strategy.max_position_weight == pytest.approx(0.60)
    assert strategy._hmm.n_states == 3
    assert strategy._hmm.covariance_type == "full"
    assert strategy._hmm.n_iter == 100
    assert strategy._hmm.seed == 42
    assert strategy._hmm.reg_covar == pytest.approx(1e-5)
    assert strategy.regime_feature_vol_lookback == 20
    assert strategy._vix_scaler.floor == pytest.approx(0.3)
    assert strategy._vix_scaler.cap == pytest.approx(1.0)
    assert strategy.vix_proxy_lookback == 21
    assert strategy.required_history == 274
    assert PAPER_MAX_FORWARD_FILL == 0
    assert BOOTSTRAP_BLOCK_LENGTHS == (5, 21, 63)
    assert PRIMARY_BOOTSTRAP_BLOCK_LENGTH == 21


def test_paper_experiment_rejects_insufficient_signal_warmup():
    index = pd.bdate_range("2021-12-01", periods=40)
    prices = pd.DataFrame(
        {
            symbol: 100.0 + np.arange(len(index), dtype=float)
            for symbol in ("QQQ", "VGT", "GLD", "TLT", "SPY", "VIG")
        },
        index=index,
    )
    cash = pd.Series(0.0, index=index)

    with pytest.raises(ValueError, match="require at least 274"):
        run_experiments(
            prices,
            cash,
            evaluation_start=index[20].date().isoformat(),
            evaluation_end=index[-1].date().isoformat(),
            bootstrap_replications=10,
        )


@pytest.mark.parametrize(
    "dates,start,end,missing_period",
    [
        (["2018-01-02", "2018-01-03"], "2018-01-02", "2018-01-03", "2022-2025"),
        (["2022-01-03", "2022-01-04"], "2022-01-03", "2022-01-04", "2018-2021"),
        (["2024-01-02", "2024-01-03"], "2018-01-02", "2025-12-31", "2018-2021"),
    ],
)
def test_paper_experiment_rejects_absent_subperiod_before_strategy_work(
    monkeypatch, dates, start, end, missing_period,
):
    def unexpected_strategy(*args, **kwargs):
        pytest.fail("absent subperiod must fail before constructing strategies")

    monkeypatch.setattr(
        "scripts.run_paper_experiments.MultiAssetRotation", unexpected_strategy
    )
    index = pd.DatetimeIndex(dates)
    prices = pd.DataFrame({"SPY": [100.0, 101.0]}, index=index)
    with pytest.raises(
        ValueError, match=f"fixed subperiod {missing_period}.*{start}.*{end}"
    ):
        run_experiments(
            prices,
            pd.Series(0.0, index=index),
            evaluation_start=start,
            evaluation_end=end,
            bootstrap_replications=10,
        )


def test_subperiod_rows_rejects_empty_series_slice():
    index = pd.bdate_range("2024-01-02", periods=3)
    returns = pd.Series([0.01, -0.01, 0.02], index=index)
    with pytest.raises(ValueError, match="subperiod 2018-2021.*strategy_net"):
        _subperiod_rows(
            {"strategy_net": returns},
            (("2018-2021", "2018-01-01", "2021-12-31"),),
            pd.Series(0.0, index=index),
        )


def test_paper_experiment_rejects_non_shv_before_data_processing(monkeypatch):
    def unexpected_index(*args, **kwargs):
        pytest.fail("unsupported cash proxy must fail before inspecting prices")

    monkeypatch.setattr(
        "scripts.run_paper_experiments._evaluation_index", unexpected_index
    )
    with pytest.raises(ValueError, match="fixed paper protocol requires SHV"):
        run_experiments(
            None,
            None,
            evaluation_start="2018-01-02",
            evaluation_end="2025-12-31",
            bootstrap_replications=10,
            cash_proxy_symbol="BIL",
        )


def test_paper_cli_rejects_non_shv_before_data_processing(
    monkeypatch, tmp_path, capsys,
):
    def unexpected_metadata(*args, **kwargs):
        pytest.fail("unsupported cash proxy must fail before repository or input reads")

    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata", unexpected_metadata
    )
    with pytest.raises(SystemExit) as error:
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv", str(tmp_path / "missing.csv"),
                "--cash-proxy-symbol", "BIL",
                "--data-provider", "test provider",
                "--permission-basis", "test permission",
                "--retrieved-at", "2026-10-08",
                "--adjustment-method", "test adjustment",
            ]
        )
    assert error.value.code == 2
    assert "--cash-proxy-symbol: invalid choice: 'BIL'" in capsys.readouterr().err


def test_source_tree_fingerprint_is_deterministic():
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        "scripts/run_paper_experiments.py",
        "quantcortex/strategies/multi_asset_rotation.py",
    ]
    first = source_tree_manifest(repo_root, paths)
    second = source_tree_manifest(repo_root, paths)

    assert first == second
    assert first["file_count"] == 2
    assert len(first["sha256"]) == 64

    complete = source_tree_manifest(repo_root)
    expected_paths = set(SOURCE_TREE_FIXED_FILES)
    expected_paths.update(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "quantcortex").rglob("*.py")
        if path.is_file()
    )
    assert set(complete["files"]) == expected_paths
    assert complete["file_count"] == len(expected_paths)


def test_git_metadata_captures_cleanliness_before_writes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("source\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "test: source"],
        cwd=tmp_path,
        check=True,
    )

    clean = _git_metadata(tmp_path)
    assert len(clean["source_commit"]) == 40
    assert clean["worktree_clean_at_start"] is True

    (tmp_path / "generated.txt").write_text("artifact\n", encoding="ascii")
    dirty = _git_metadata(tmp_path)
    assert dirty["source_commit"] == clean["source_commit"]
    assert dirty["worktree_clean_at_start"] is False


def test_git_path_tracking_is_measured_from_the_source_repository(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.csv"
    untracked = tmp_path / "untracked.csv"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.csv"
    tracked.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    untracked.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    outside.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.csv"], cwd=tmp_path, check=True)

    assert _git_path_is_tracked(tmp_path, tracked) is True
    assert _git_path_is_tracked(tmp_path, untracked) is False
    assert _git_path_is_tracked(tmp_path, outside) is False


def test_git_path_tracking_fails_closed_on_git_errors(monkeypatch, tmp_path):
    candidate = tmp_path / "prices.csv"
    candidate.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    monkeypatch.setattr(
        "scripts.run_paper_experiments.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stderr="fatal"),
    )

    with pytest.raises(RuntimeError, match="could not determine"):
        _git_path_is_tracked(tmp_path, candidate)


@pytest.mark.parametrize("helper", [_git_path_is_tracked, _fetch_path_is_tracked])
@pytest.mark.parametrize("name", ["prices[1].csv", "prices?.csv", "prices*.csv"])
@pytest.mark.parametrize("track_literal", [False, True])
def test_input_tracking_matches_literal_metacharacter_path(
    tmp_path, helper, name, track_literal,
):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    literal = tmp_path / name
    wildcard_match = tmp_path / "prices1.csv"
    for path in (literal, wildcard_match):
        path.write_text("date,A\n2024-01-01,1\n", encoding="ascii")
    tracked = literal if track_literal else wildcard_match
    subprocess.run(
        ["git", "--literal-pathspecs", "add", "--", tracked.name],
        cwd=tmp_path,
        check=True,
    )

    assert helper(tmp_path, literal) is track_literal


def test_paper_cli_requires_clean_source_before_reading_input(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata",
        lambda _repo_root: {
            "source_commit": "0" * 40,
            "worktree_clean_at_start": False,
        },
    )
    with pytest.raises(RuntimeError, match="clean source worktree"):
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv",
                str(tmp_path / "missing.csv"),
                "--data-provider",
                "test provider",
                "--permission-basis",
                "test permission",
                "--retrieved-at",
                "2026-06-16",
                "--adjustment-method",
                "test adjustment",
                "--require-clean-source",
            ]
        )


def test_paper_cli_rejects_a_git_tracked_price_input(monkeypatch):
    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata",
        lambda _repo_root: {
            "source_commit": "0" * 40,
            "worktree_clean_at_start": True,
        },
    )
    tracked_fixture = (
        Path(__file__).resolve().parent / "fixtures" / "conformance" / "prices.csv"
    )

    with pytest.raises(RuntimeError, match="must not be tracked by Git"):
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv",
                str(tracked_fixture),
                "--data-provider",
                "test provider",
                "--permission-basis",
                "test permission",
                "--retrieved-at",
                "2026-06-16",
                "--adjustment-method",
                "test adjustment",
            ]
        )


def test_paper_cli_writes_a_complete_test_only_artifact_set(monkeypatch, tmp_path):
    warmup = pd.bdate_range(end="2017-12-29", periods=300)
    first_period = pd.bdate_range("2018-01-02", periods=40)
    second_period = pd.bdate_range("2022-01-03", periods=40)
    index = warmup.append(first_period).append(second_period)
    rng = np.random.default_rng(7)
    common = rng.normal(0.0002, 0.006, size=len(index))
    prices = {}
    for offset, symbol in enumerate(["QQQ", "VGT", "GLD", "TLT", "SPY", "VIG"]):
        idiosyncratic = rng.normal(0.0, 0.002 + offset * 0.0001, size=len(index))
        prices[symbol] = 100.0 * np.exp(np.cumsum(common + idiosyncratic))
    prices["SHV"] = 100.0 * np.exp(np.cumsum(np.full(len(index), 0.00005)))
    source = tmp_path / "test_only_prices.csv"
    pd.DataFrame(prices, index=index).rename_axis("date").reset_index().to_csv(
        source,
        index=False,
    )
    output_dir = tmp_path / "paper"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(
        "scripts.run_paper_experiments._git_metadata",
        lambda _repo_root: {
            "source_commit": "1" * 40,
            "worktree_clean_at_start": True,
        },
    )
    monkeypatch.setattr(
        "scripts.run_paper_experiments._threadpool_environment",
        lambda: [{"user_api": "test", "num_threads": 1}],
    )
    from scripts import run_paper_experiments as paper_experiments

    parsed_payloads = []
    original_parser = paper_experiments.target_tape_from_payload

    def recording_parser(payload):
        parsed_payloads.append(payload)
        return original_parser(payload)

    monkeypatch.setattr(paper_experiments, "target_tape_from_payload", recording_parser)

    assert (
        paper_main(
            [
                "run_paper_experiments.py",
                "--prices-csv",
                str(source),
                "--start",
                "2018-01-02",
                "--end",
                "2022-02-25",
                "--output-dir",
                str(output_dir),
                "--data-provider",
                "deterministic synthetic test fixture",
                "--permission-basis",
                "test-only generated data",
                "--retrieved-at",
                "2026-06-16",
                "--adjustment-method",
                "test-only generated positive price paths",
                "--generated-at",
                "2026-06-16T00:00:00Z",
                "--bootstrap-replications",
                "10",
                "--require-clean-source",
            ]
        )
        == 0
    )

    manifest = json.loads(
        (output_dir / "results" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 5
    assert manifest["generated_at"] == "2026-06-16T00:00:00Z"
    assert manifest["generator"]["git"] == {
        "source_commit": "1" * 40,
        "worktree_clean_at_start": True,
    }
    assert manifest["source"]["raw_input_committed"] is False
    assert manifest["source"]["provider"] == "deterministic synthetic test fixture"
    assert manifest["source"]["cash_proxy"] == "SHV"
    assert "results/evaluation_contract.json" in manifest["artifacts"]
    assert "results/target_tape_hashes.json" in manifest["artifacts"]
    assert "results/sharpe_uncertainty.csv" in manifest["artifacts"]
    assert "figures/accounting_summary.pdf" in manifest["artifacts"]
    for relative_path in manifest["artifacts"]:
        assert (output_dir / relative_path).is_file(), relative_path

    target_tapes = manifest["decision_streams"]["variants"]
    assert set(target_tapes) == {"full", "no_regime", "no_vol_scaler", "signal_only"}
    assert all(metadata["schema_version"] == 1 for metadata in target_tapes.values())
    assert all(metadata["decision_count"] > 0 for metadata in target_tapes.values())
    assert all(metadata["symbols"] == sorted(metadata["symbols"]) for metadata in target_tapes.values())
    assert all(
        metadata["record_count"]
        == metadata["decision_count"] * len(metadata["symbols"])
        for metadata in target_tapes.values()
    )
    assert len(parsed_payloads) == 4

    subperiods = pd.read_csv(output_dir / "results" / "subperiods.csv")
    assert set(subperiods["period"]) == {"2018-2021", "2022-2025"}
    assert len(subperiods) == 6
    assert set(subperiods.loc[subperiods["period"] == "2022-2025", "end"]) == {
        "2022-02-25"
    }
    values = (output_dir / "results" / "generated_values.tex").read_text()
    for prefix in ("Strategy", "Matched", "CostedComparator"):
        for metric in ("CAGR", "Sharpe"):
            for suffix in ("FirstHalf", "SecondHalf"):
                assert f"\\newcommand{{\\Paper{prefix}{metric}{suffix}}}" in values

    engine_comparison = pd.read_csv(output_dir / "results" / "engine_comparison.csv")
    for column in (
        "final_cumulative_arithmetic_return_difference",
        "maximum_absolute_cumulative_arithmetic_return_difference",
    ):
        assert engine_comparison[column].nunique() == 1
        assert np.isfinite(engine_comparison[column].iloc[0])


def test_evaluation_contract_separates_attribution_and_costed_comparator():
    contract = evaluation_contract("SHV")

    assert contract["schema_version"] == 1
    assert contract["cash"]["return_proxy"] == "SHV"
    assert contract["overlays"]["may_reduce_risky_exposure"] is True
    assert contract["overlays"]["may_increase_declared_gross_limit"] is False
    assert "block automatic retry" in contract["order_state"][
        "uncertain_submission"
    ]
    comparators = contract["comparators"]
    assert comparators["realized_exposure_attribution_control"] == {
        "purpose": "exact ex-post arithmetic attribution",
        "implementable": False,
        "costed": False,
        "exposure_basis": "strategy realized daily risky exposure",
    }
    target_comparator = comparators["target_exposure_costed_comparator"]
    assert target_comparator["purpose"] == (
        "next-bar costed comparison conditional on strategy target exposure"
    )
    assert target_comparator["implementable"] is True
    assert target_comparator["costed"] is True


def test_evaluation_contract_retains_general_cash_proxy_description():
    assert evaluation_contract("BIL")["cash"]["return_proxy"] == "BIL"


def test_engine_difference_summary_matches_cumulative_path():
    index = pd.bdate_range("2024-01-02", periods=3)
    cumulative, summary = _engine_difference_summary(
        {
            "vectorized": pd.Series([0.01, -0.02, 0.03], index=index),
            "event_driven": pd.Series([0.0, 0.01, -0.01], index=index),
        }
    )

    assert cumulative.to_list() == pytest.approx([0.01, -0.02, 0.02])
    assert summary["final_cumulative_arithmetic_return_difference"] == pytest.approx(
        0.02
    )
    assert summary[
        "maximum_absolute_cumulative_arithmetic_return_difference"
    ] == pytest.approx(0.02)


def test_percentile_errorbar_preserves_outside_bounds_and_sample_points():
    import matplotlib.pyplot as plt

    returns = np.linspace(-0.01, 0.02, 63)
    summary = circular_block_bootstrap_frame(
        pd.DataFrame({"positive": returns, "negative": -returns}),
        block_length=21, replications=1, seed=42,
    )
    points = np.r_[summary["annualized_mean"], 0.0, 0.0, 0.0]
    lower = np.r_[summary["ci_95_lower"], 0.1, -0.3, -0.1]
    upper = np.r_[summary["ci_95_upper"], 0.3, -0.1, 0.1]
    locations = np.arange(len(points))
    assert np.all(lower <= upper)
    assert points[0] < lower[0]
    assert points[1] > upper[1]

    figure, axis = plt.subplots()
    try:
        _percentile_errorbar(
            axis, points, locations, lower, upper,
            fmt="s", color="navy", capsize=2, label="Sample estimate",
        )

        segments = axis.collections[0].get_segments()
        for segment, low, high, location in zip(
            segments, lower, upper, locations, strict=True,
        ):
            np.testing.assert_allclose(
                segment, [[low, location], [high, location]], rtol=0, atol=1e-15,
            )
        markers = [line for line in axis.lines if line.get_marker() == "s"]
        assert len(markers) == 1
        np.testing.assert_array_equal(markers[0].get_xdata(), points)
        np.testing.assert_array_equal(markers[0].get_ydata(), locations)
        assert axis.get_legend_handles_labels()[1] == ["Sample estimate"]
    finally:
        plt.close(figure)


def test_percentile_errorbar_preserves_ordinary_rendering():
    import matplotlib.pyplot as plt

    points = np.array([0.1, -0.2, 0.3])
    lower = np.array([-0.1, -0.2, 0.2])
    upper = np.array([0.2, -0.1, 0.3])
    locations = np.arange(len(points))
    style = {
        "fmt": "s", "color": "navy", "ecolor": "black", "markersize": 4.2,
        "capsize": 2, "elinewidth": 0.9, "label": "Sample estimate",
    }
    rendered = []
    for original in (True, False):
        figure, axis = plt.subplots(figsize=(4, 3))
        try:
            if original:
                axis.errorbar(
                    points, locations,
                    xerr=np.vstack([points - lower, upper - points]), **style,
                )
            else:
                _percentile_errorbar(axis, points, locations, lower, upper, **style)
            axis.legend()
            figure.canvas.draw()
            rendered.append(np.asarray(figure.canvas.buffer_rgba()).copy())
        finally:
            plt.close(figure)

    np.testing.assert_array_equal(rendered[0], rendered[1])


@pytest.mark.parametrize("outside_intervals", [False, True])
def test_paper_figures_are_deterministic_and_use_reviewed_protocol_exports(
    tmp_path,
    monkeypatch,
    outside_intervals,
):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import date2num

    closed_figures = []
    real_close = plt.close
    monkeypatch.setattr(
        plt,
        "close",
        lambda figure=None: closed_figures.append(figure or plt.gcf()),
    )

    index = pd.bdate_range("2024-01-01", periods=6)
    baseline = {
        "gross": pd.Series([0.0, 0.01, -0.005, 0.004, 0.002, 0.003], index=index),
        "net": pd.Series([0.0, 0.009, -0.006, 0.003, 0.001, 0.002], index=index),
        "cash": pd.Series([0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], index=index),
        "exposure": pd.Series([0.0, 0.5, 1.0, 0.3, 0.0, 0.8], index=index),
    }
    matched = pd.Series([0.0, 0.006, -0.002, 0.003, 0.002, 0.004], index=index)
    costed_comparator = pd.Series(
        [0.0, 0.005, -0.0025, 0.0025, 0.0015, 0.0035],
        index=index,
    )
    costs = pd.DataFrame(
        {
            "all_in_cost_bps": [0.0, 13.0],
            "net_cash_excess_sharpe": [0.2, -0.1],
        }
    )
    ablation = pd.DataFrame(
        {
            "variant": ["full", "signal_only"],
            "mean_gross_exposure": [0.3, 0.8],
            "max_drawdown": [-0.09, -0.26],
        }
    )
    ablation_uncertainty = pd.DataFrame(
        {
            "variant": ["full", "signal_only"],
            "annualized_mean": [-0.03, -0.05],
            "ci_95_lower": [-0.05, -0.08],
            "ci_95_upper": [-0.01, -0.02],
        }
    )
    engines = {"vectorized": baseline["net"], "event_driven": baseline["net"]}
    decomposition = pd.DataFrame(
        {
            "component": [
                "active_risky_allocation",
                "dynamic_exposure_timing",
                "passive_risky_exposure",
                "implementation_cost",
                "net_excess_over_cash",
            ],
            "block_length": [21] * 5,
            "annualized_mean": [-0.03, -0.005, 0.04, -0.02, -0.015],
            "ci_95_lower": [-0.05, -0.02, 0.02, -0.03, -0.04],
            "ci_95_upper": [-0.01, 0.01, 0.06, -0.01, 0.01],
        }
    )
    switches = pd.DataFrame(
        {
            "protocol": ["audited", "invalid_same_close"],
            "display_name": ["Audited", "Retroactive signal-close assignment"],
            "diagnostic_class": ["reference", "causally_invalid"],
            "causally_valid": [True, False],
            "shv_excess_sharpe": [-0.1, 0.2],
        }
    )
    bootstrap_sensitivity = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "block_length": block_length,
                "annualized_mean": mean,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
            }
            for comparison, mean, lower, upper in (
                (
                    "strategy_gross_minus_exposure_matched_equal_weight",
                    -0.03,
                    -0.05,
                    -0.01,
                ),
                (
                    "strategy_minus_exposure_matched_equal_weight",
                    -0.05,
                    -0.08,
                    -0.02,
                ),
            )
            for block_length in BOOTSTRAP_BLOCK_LENGTHS
        ]
    )

    if outside_intervals:
        for frame in (decomposition, ablation_uncertainty, bootstrap_sensitivity):
            means = frame["annualized_mean"].to_numpy()
            direction = np.where(np.arange(len(frame)) % 2 == 0, 1.0, -1.0)
            centers = means + direction * 0.03
            frame["ci_95_lower"] = centers - 0.01
            frame["ci_95_upper"] = centers + 0.01
            assert (frame["ci_95_lower"] <= frame["ci_95_upper"]).all()
            assert (
                (means < frame["ci_95_lower"]) | (means > frame["ci_95_upper"])
            ).all()

    first = tmp_path / "first"
    second = tmp_path / "second"
    _save_figures(
        first,
        baseline,
        matched,
        costed_comparator,
        costs,
        ablation,
        ablation_uncertainty,
        engines,
        decomposition,
        switches,
        bootstrap_sensitivity,
    )
    _save_figures(
        second,
        baseline,
        matched,
        costed_comparator,
        costs,
        ablation,
        ablation_uncertainty,
        engines,
        decomposition,
        switches,
        bootstrap_sensitivity,
    )

    for first_path in sorted(first.iterdir()):
        first_bytes = first_path.read_bytes()
        second_path = second / first_path.name
        assert first_bytes == second_path.read_bytes(), first_path.name
        if first_path.suffix == ".pdf":
            assert b"quantcortex" not in first_bytes.lower(), first_path.name

    paper_root = Path(__file__).resolve().parent.parent / "paper"
    for suffix in ("pdf", "png"):
        expected = (paper_root / f"audit_protocol.drawio.{suffix}").read_bytes()
        assert (first / f"audit_protocol.{suffix}").read_bytes() == expected
        assert (second / f"audit_protocol.{suffix}").read_bytes() == expected

    accounting_figures = [
        figure
        for figure in closed_figures
        if len(figure.axes) == 2
        and figure.axes[1].get_title() == "Risky assets and residual cash"
    ]
    assert len(accounting_figures) == 2
    dates = date2num(index)
    for figure in accounting_figures:
        exposure_axis = figure.axes[1]
        risky, cash = [item.get_paths()[0] for item in exposure_axis.collections]
        # Exposure dated t applies to the interval ending at t.
        for position in range(1, len(index)):
            midpoint = (dates[position - 1] + dates[position]) / 2.0
            exposure = baseline["exposure"].iloc[position]
            for weight in (0.1, 0.4, 0.6, 0.9):
                assert risky.contains_point((midpoint, weight)) == (weight < exposure)
                assert cash.contains_point((midpoint, weight)) == (weight > exposure)
        outline, mean = exposure_axis.lines
        np.testing.assert_array_equal(
            outline.get_ydata(), baseline["exposure"].to_numpy()
        )
        np.testing.assert_array_equal(
            outline.get_path().vertices,
            [
                [dates[0], 0.0], [dates[0], 0.5], [dates[1], 0.5],
                [dates[1], 1.0], [dates[2], 1.0], [dates[2], 0.3],
                [dates[3], 0.3], [dates[3], 0.0], [dates[4], 0.0],
                [dates[4], 0.8], [dates[5], 0.8],
            ],
        )
        np.testing.assert_array_equal(mean.get_ydata(), [baseline["exposure"].mean()] * 2)

    attribution_figures = [
        figure
        for figure in closed_figures
        if len(figure.axes) == 2
        and figure.axes[1].get_title() == "Convention checks"
    ]
    assert len(attribution_figures) == 2
    for figure in attribution_figures:
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        diagnostic_axis = figure.axes[1]
        panel_label = next(
            text for text in diagnostic_axis.texts if text.get_text() == "b"
        )
        assert not panel_label.get_window_extent(renderer).overlaps(
            diagnostic_axis.title.get_window_extent(renderer)
        )

    sensitivity_figures = [
        figure
        for figure in closed_figures
        if len(figure.axes) == 2
        and figure.axes[0].get_title() == "Cost sensitivity"
    ]
    assert len(sensitivity_figures) == 2
    for figure in sensitivity_figures:
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        renderer = canvas.get_renderer()
        cost_axis = figure.axes[0]
        legend_box = cost_axis.get_legend().get_window_extent(renderer)
        # The full-height baseline guide must remain clear of the legend.
        assert not legend_box.overlaps(cost_axis.get_window_extent(renderer))
        for axis in figure.axes:
            assert not legend_box.overlaps(axis.title.get_window_extent(renderer))
            for text in axis.texts:
                assert not legend_box.overlaps(text.get_window_extent(renderer))

    for figure in closed_figures:
        real_close(figure)
