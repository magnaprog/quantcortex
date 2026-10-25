#!/usr/bin/env python3
"""Run the frozen evaluation-contract expansion across ETF datasets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from quantcortex.backtest.conformance import (
    target_tape_to_payload,
    weights_to_target_tape,
)
from quantcortex.backtest.metrics.plotting import (
    BRIGHT_COLORS,
    GRID,
    INK,
    MUTED_INK,
    RETURN_DIVERGING_COLORS,
    SPINE,
    add_panel_label,
    apply_plot_style,
    contrasting_text_color,
    style_axis,
)
from quantcortex.data.local_csv import load_price_matrix, sha256_file
from quantcortex.research.expansion import (
    FROZEN_PROTOCOL_COMMIT,
    FROZEN_PROTOCOL_SHA256,
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

VARIANTS = (
    "baseline",
    "same_close",
    "zero_cash",
    "zero_cost",
    "costed_comparator",
    "vectorized",
)
SWITCHES = {
    "same_close_minus_baseline": ("same_close", "baseline"),
    "zero_cash_minus_baseline": ("zero_cash", "baseline"),
    "zero_cost_minus_baseline": ("zero_cost", "baseline"),
    "strategy_minus_costed_comparator": ("baseline", "costed_comparator"),
    "vectorized_minus_event": ("vectorized", "baseline"),
}
STRATEGY_LABELS = {
    "ts_momentum": "TS momentum",
    "cross_sectional_momentum": "Cross-sectional momentum",
    "short_term_reversal": "Short-term reversal",
    "learned_gbrt": "GBRT",
}
STRATEGY_STYLES = {
    name: {"color": color, "marker": marker}
    for name, color, marker in zip(
        STRATEGY_LABELS, BRIGHT_COLORS[:len(STRATEGY_LABELS)], ("o", "s", "^", "D"),
        strict=True,
    )
}
PANEL_LABELS = {
    "us_sector_etfs": "U.S. sector ETFs",
    "country_equity_etfs": "Country equity ETFs",
}
PROVIDER_NAME = "Yahoo Finance via yfinance"
TERMS_URLS = (
    "https://ranaroussi.github.io/yfinance/",
    "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html",
)
SOURCE_FILES = (
    "scripts/fetch_expansion_data.py",
    "scripts/publish_artifact_bundle.py",
    "scripts/release_expansion_artifacts.sh",
    "scripts/run_expansion_experiments.py",
    "paper/preregistration.md",
    "paper/expansion/protocol.json",
    "schemas/canonical_target_tape.schema.json",
    "pyproject.toml",
    "poetry.lock",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_aware_timestamp(value: str, *, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return parsed


def _threadpool_environment() -> list[dict[str, object]]:
    """Return stable BLAS/OpenMP metadata without machine-specific paths."""
    from threadpoolctl import threadpool_info

    keys = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
        "num_threads",
    )
    return [
        {key: entry[key] for key in keys if key in entry}
        for entry in threadpool_info()
    ]


def _git_metadata(repo_root: Path) -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "source_commit": commit,
        "tracked_worktree_clean_at_start": not tracked_status.strip(),
    }


def _source_tree_manifest(repo_root: Path) -> dict[str, object]:
    relative_paths = set(SOURCE_FILES)
    relative_paths.update(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "quantcortex").rglob("*.py")
        if path.is_file()
    )
    files = {}
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"source fingerprint is missing {relative}")
        content_digest = _sha256(path)
        files[relative] = content_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _load_protocol(path: Path) -> dict[str, object]:
    if _sha256(path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError("expansion protocol differs from the pre-data freeze")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported expansion protocol version")
    if protocol.get("status") != (
        "repository_frozen_prospective_not_externally_registered"
    ):
        raise ValueError("expansion protocol is not frozen")
    if protocol.get("historical_case_confirmatory") is not False:
        raise ValueError("historical case must remain non-confirmatory")
    return protocol


def _load_panel(
    repo_root: Path,
    panel_dir: Path,
    panel_name: str,
    risky_symbols: list[str],
    protocol: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = protocol["data"]
    cash_symbol = data["cash_proxy"]
    csv_path = panel_dir / f"{panel_name}.csv"
    metadata_path = panel_dir / f"{panel_name}.metadata.json"
    if not csv_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"missing local panel or metadata for {panel_name}")
    try:
        relative_csv = csv_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative_csv = None
    if relative_csv is not None:
        tracked = subprocess.run(
            [
                "git", "--literal-pathspecs", "ls-files", "--error-unmatch", "--",
                relative_csv,
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if tracked.returncode == 0:
            raise ValueError(f"raw panel must not be tracked: {csv_path}")
        if tracked.returncode != 1:
            raise RuntimeError(f"could not determine tracking status for {csv_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_protocol_digest = _sha256(repo_root / "paper/expansion/protocol.json")
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported metadata version for {panel_name}")
    if metadata.get("panel") != panel_name:
        raise ValueError(f"metadata panel mismatch for {panel_name}")
    if metadata.get("protocol_sha256") != expected_protocol_digest:
        raise ValueError(f"metadata protocol digest mismatch for {panel_name}")
    if metadata.get("input_sha256") != sha256_file(csv_path):
        raise ValueError(f"input digest mismatch for {panel_name}")
    if metadata.get("raw_data_committed") is not False:
        raise ValueError(f"metadata must mark raw {panel_name} data uncommitted")
    symbols = [*risky_symbols, cash_symbol]
    if metadata.get("symbols") != symbols:
        raise ValueError(f"metadata symbol mismatch for {panel_name}")
    if metadata.get("request") != data["provider_request"]:
        raise ValueError(f"metadata request mismatch for {panel_name}")
    if metadata.get("protocol_path") != "paper/expansion/protocol.json":
        raise ValueError(f"metadata protocol path mismatch for {panel_name}")
    if metadata.get("provider") != PROVIDER_NAME:
        raise ValueError(f"metadata provider mismatch for {panel_name}")
    if metadata.get("terms_urls") != list(TERMS_URLS):
        raise ValueError(f"metadata terms URLs mismatch for {panel_name}")
    if not isinstance(metadata.get("yfinance_version"), str) or not metadata[
        "yfinance_version"
    ].strip():
        raise ValueError(f"metadata provider version is missing for {panel_name}")
    if metadata.get("provider_terms_independently_verified") is not False:
        raise ValueError(f"metadata terms status mismatch for {panel_name}")
    _parse_aware_timestamp(metadata.get("retrieved_at"), name="retrieved_at")
    prices = load_price_matrix(
        csv_path,
        symbols=symbols,
        start=data["start"],
        end=data["evaluation_end"],
        max_ffill=None,
        require_complete=True,
    )
    validated = validate_price_panel(
        prices,
        risky_symbols=risky_symbols,
        cash_symbol=cash_symbol,
    )
    evaluation = validated.loc[data["evaluation_start"] : data["evaluation_end"]]
    pre_evaluation_sessions = int(
        (validated.index < pd.Timestamp(data["evaluation_start"])).sum()
    )
    pre_evaluation_months = int(
        validated.index[
            validated.index < pd.Timestamp(data["evaluation_start"])
        ].to_period("M").nunique()
    )
    if pre_evaluation_sessions < data["minimum_pre_evaluation_sessions"]:
        raise ValueError(f"panel {panel_name} has insufficient pre-evaluation history")
    expected_months = pd.period_range(
        pd.Timestamp(data["evaluation_start"]).to_period("M"),
        pd.Timestamp(data["evaluation_end"]).to_period("M"),
        freq="M",
    )
    missing_months = expected_months.difference(evaluation.index.to_period("M").unique())
    if len(missing_months):
        raise ValueError(
            f"panel {panel_name} is missing evaluation months: "
            + ", ".join(str(month) for month in missing_months)
        )
    expected_metadata = {
        "complete_rows": len(validated),
        "evaluation_sessions": len(evaluation),
        "first_date": validated.index[0].date().isoformat(),
        "last_date": validated.index[-1].date().isoformat(),
        "pre_evaluation_months": pre_evaluation_months,
        "pre_evaluation_sessions": pre_evaluation_sessions,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata {key} mismatch for {panel_name}")
    provider_rows = metadata.get("provider_rows")
    dropped_rows = metadata.get("dropped_incomplete_rows")
    missing_by_symbol = metadata.get("missing_by_symbol")
    if (
        isinstance(provider_rows, bool)
        or not isinstance(provider_rows, int)
        or isinstance(dropped_rows, bool)
        or not isinstance(dropped_rows, int)
        or provider_rows != len(validated) + dropped_rows
        or dropped_rows < 0
    ):
        raise ValueError(f"metadata provider-row accounting mismatch for {panel_name}")
    if not isinstance(missing_by_symbol, dict) or set(missing_by_symbol) != set(symbols):
        raise ValueError(f"metadata missingness symbols mismatch for {panel_name}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in missing_by_symbol.values()
    ):
        raise ValueError(f"metadata missingness counts are invalid for {panel_name}")
    if sum(missing_by_symbol.values()) < dropped_rows:
        raise ValueError(f"metadata missingness accounting mismatch for {panel_name}")
    return validated, metadata


def _strategy_targets(
    prices: pd.DataFrame,
    risky_symbols: list[str],
    protocol: dict[str, object],
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict[str, object]]]:
    data = protocol["data"]
    strategies = protocol["strategies"]
    all_decisions = monthly_decision_dates(prices.index, end=data["evaluation_end"])
    evaluation_decisions = monthly_decision_dates(
        prices.index,
        start=data["evaluation_start"],
        end=data["evaluation_end"],
    )
    targets: dict[str, dict[str, pd.DataFrame]] = {
        "ts_momentum": {
            "deterministic": time_series_momentum_targets(
                prices,
                symbols=risky_symbols,
                decisions=evaluation_decisions,
                lookback=strategies["ts_momentum"]["lookback_sessions"],
            )
        },
        "cross_sectional_momentum": {
            "deterministic": cross_sectional_momentum_targets(
                prices,
                symbols=risky_symbols,
                decisions=evaluation_decisions,
                lookback_start=strategies["cross_sectional_momentum"][
                    "lookback_start_sessions"
                ],
                skip_recent=strategies["cross_sectional_momentum"][
                    "skip_recent_sessions"
                ],
                selection_count=strategies["cross_sectional_momentum"][
                    "selection_count"
                ],
            )
        },
        "short_term_reversal": {
            "deterministic": short_term_reversal_targets(
                prices,
                symbols=risky_symbols,
                decisions=evaluation_decisions,
                lookback=strategies["short_term_reversal"]["lookback_sessions"],
                selection_count=strategies["short_term_reversal"][
                    "selection_count"
                ],
            )
        },
        "learned_gbrt": {},
    }
    learned_diagnostics = []
    learned_config = strategies["learned_gbrt"]
    for seed in learned_config["seeds"]:
        result = learned_gbrt_targets(
            prices,
            symbols=risky_symbols,
            all_decisions=all_decisions,
            evaluation_decisions=evaluation_decisions,
            config=learned_config,
            seed=seed,
        )
        targets["learned_gbrt"][str(seed)] = result.weights
        learned_diagnostics.append(
            {
                "seed": int(seed),
                "first_training_rows": int(result.training_rows.iloc[0]),
                "minimum_training_rows": int(result.training_rows.min()),
                "maximum_training_rows": int(result.training_rows.max()),
                "minimum_training_months": int(result.training_months.min()),
                "maximum_training_months": int(result.training_months.max()),
            }
        )
    return targets, learned_diagnostics


def _evaluation_index(prices: pd.DataFrame, protocol: dict[str, object]) -> pd.DatetimeIndex:
    data = protocol["data"]
    index = prices.index[
        (prices.index >= pd.Timestamp(data["evaluation_start"]))
        & (prices.index <= pd.Timestamp(data["evaluation_end"]))
    ]
    if index.empty:
        raise ValueError("evaluation window contains no observations")
    return pd.DatetimeIndex(index)


def _target_hash(weights: pd.DataFrame) -> tuple[str, int]:
    tape = weights_to_target_tape(weights, max_gross=1.0)
    payload = target_tape_to_payload(
        tape,
        max_gross=1.0,
        expected_symbols=list(weights.columns),
    )
    return _canonical_json_digest(payload), int(len(payload["records"]))


def _run_target_variant(
    weights: pd.DataFrame,
    risky_prices: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    cost_rate: float,
) -> dict[str, object]:
    zero_cash = pd.Series(0.0, index=cash_returns.index, name="zero_cash")
    comparator_targets = exposure_matched_comparator_targets(weights)
    return {
        "baseline": run_engine(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
            engine="event_driven",
        ),
        "same_close": invalid_same_close_result(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
        ),
        "zero_cash": run_engine(
            weights,
            risky_prices,
            zero_cash,
            cost_rate=cost_rate,
            engine="event_driven",
        ),
        "zero_cost": run_engine(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=0.0,
            engine="event_driven",
        ),
        "costed_comparator": run_engine(
            comparator_targets,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
            engine="event_driven",
        ),
        "vectorized": run_engine(
            weights,
            risky_prices,
            cash_returns,
            cost_rate=cost_rate,
            engine="vectorized",
        ),
    }


def _panel_experiment(
    panel_name: str,
    prices: pd.DataFrame,
    risky_symbols: list[str],
    protocol: dict[str, object],
) -> dict[str, object]:
    cash_symbol = protocol["data"]["cash_proxy"]
    cost_rate = protocol["execution"]["cost_per_one_way_gross_notional"]
    risky_prices = prices.loc[:, risky_symbols]
    cash_returns = prices[cash_symbol].pct_change(fill_method=None).fillna(0.0)
    cash_returns.name = cash_symbol
    evaluation_index = _evaluation_index(prices, protocol)
    targets, learned_diagnostics = _strategy_targets(
        prices,
        risky_symbols,
        protocol,
    )
    metric_rows = []
    engine_rows = []
    target_rows = []
    variant_series: dict[str, dict[str, dict[str, pd.Series]]] = {}
    for strategy, seed_targets in targets.items():
        variant_series[strategy] = {}
        for seed, weights in seed_targets.items():
            target_digest, record_count = _target_hash(weights)
            target_rows.append(
                {
                    "panel": panel_name,
                    "strategy": strategy,
                    "seed": seed,
                    "sha256": target_digest,
                    "decision_count": int(len(weights)),
                    "record_count": record_count,
                    "symbols": list(weights.columns),
                }
            )
            results = _run_target_variant(
                weights,
                risky_prices,
                cash_returns,
                cost_rate=cost_rate,
            )
            variant_series[strategy][seed] = {}
            for variant in VARIANTS:
                result = results[variant]
                metrics = performance_metrics(result, cash_returns, evaluation_index)
                metric_rows.append(
                    {
                        "panel": panel_name,
                        "strategy": strategy,
                        "seed": seed,
                        "variant": variant,
                        **metrics,
                    }
                )
                variant_series[strategy][seed][variant] = result.returns.reindex(
                    evaluation_index
                )
            event_returns = variant_series[strategy][seed]["baseline"]
            vectorized_returns = variant_series[strategy][seed]["vectorized"]
            engine_rows.append(
                {
                    "panel": panel_name,
                    "strategy": strategy,
                    "seed": seed,
                    "observations": int(len(evaluation_index)),
                    "max_absolute_daily_return_difference": float(
                        (event_returns - vectorized_returns).abs().max()
                    ),
                    "final_wealth_difference": float(
                        (1.0 + vectorized_returns).prod()
                        - (1.0 + event_returns).prod()
                    ),
                    "event_return_sha256": _series_digest(event_returns),
                    "vectorized_return_sha256": _series_digest(vectorized_returns),
                }
            )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "engine": pd.DataFrame(engine_rows),
        "targets": target_rows,
        "series": variant_series,
        "cash": cash_returns.reindex(evaluation_index),
        "learned_diagnostics": learned_diagnostics,
    }


def _series_digest(series: pd.Series) -> str:
    payload = pd.DataFrame(
        {
            "date": series.index.strftime("%Y-%m-%d"),
            "return": series.to_numpy(dtype=float),
        }
    ).to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _family_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"observations", "family_size"}
    ]
    grouped = metrics.groupby(["panel", "strategy", "variant"], sort=True)
    summary = grouped[numeric].mean().reset_index()
    summary["observations"] = grouped["observations"].min().to_numpy()
    summary["family_size"] = grouped.size().to_numpy()
    return summary.loc[
        :,
        [
            "panel",
            "strategy",
            "variant",
            "family_size",
            "observations",
            *numeric,
        ],
    ]


def _contract_effects(
    panel_results: dict[str, dict[str, object]],
    protocol: dict[str, object],
) -> pd.DataFrame:
    uncertainty = protocol["uncertainty"]
    block_lengths = [
        uncertainty["primary_block_sessions"],
        *uncertainty["sensitivity_block_sessions"],
    ]
    rows = []
    for panel_name, panel in panel_results.items():
        for strategy, seeds in panel["series"].items():
            ordered_seeds = sorted(
                seeds,
                key=lambda value: int(value) if value.isdigit() else -1,
            )
            for switch, (lhs_variant, rhs_variant) in SWITCHES.items():
                lhs = [seeds[seed][lhs_variant] for seed in ordered_seeds]
                rhs = [seeds[seed][rhs_variant] for seed in ordered_seeds]
                for block_length in block_lengths:
                    estimate = bootstrap_metric_difference(
                        lhs,
                        rhs,
                        panel["cash"],
                        block_length=block_length,
                        replications=uncertainty["replications"],
                        seed=uncertainty["seed"],
                    )
                    rows.append(
                        {
                            "panel": panel_name,
                            "strategy": strategy,
                            "switch": switch,
                            **estimate,
                        }
                    )
    return pd.DataFrame(rows)


def _rank_reversals(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel, panel_frame in summary.groupby("panel"):
        baseline = panel_frame.loc[panel_frame["variant"] == "baseline"]
        baseline_values = _serialized_metric_values(baseline)
        baseline_order = _metric_order(baseline_values)
        baseline_ties = _metric_ties(baseline_values)
        for variant in VARIANTS:
            variant_frame = panel_frame.loc[panel_frame["variant"] == variant]
            variant_values = _serialized_metric_values(variant_frame)
            order = _metric_order(variant_values)
            variant_ties = _metric_ties(variant_values)
            for strategy in sorted(baseline_order):
                rows.append(
                    {
                        "panel": panel,
                        "variant": variant,
                        "strategy": strategy,
                        "baseline_rank": baseline_order[strategy],
                        "variant_rank": order[strategy],
                        "rank_change": baseline_order[strategy] - order[strategy],
                        "baseline_tied": strategy in baseline_ties,
                        "variant_tied": strategy in variant_ties,
                        "rank_method": (
                            "descending_12_significant_digit_cash_excess_sharpe_"
                            "with_strategy_key_tiebreak"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _yearly_performance(panel_results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Annualize each calendar year's mean baseline return above cash."""
    rows = []
    for panel, result in panel_results.items():
        cash = result["cash"]
        if (
            not isinstance(cash.index, pd.DatetimeIndex)
            or cash.empty
            or not cash.index.is_unique
            or not cash.index.is_monotonic_increasing
            or not np.isfinite(cash.to_numpy(dtype=float)).all()
        ):
            raise ValueError("cash requires finite values on distinct chronological dates")
        for strategy, seeds in result["series"].items():
            for seed, variants in seeds.items():
                returns = variants["baseline"]
                if not returns.index.equals(cash.index):
                    raise ValueError("baseline returns and cash must have identical dates")
                if not np.isfinite(returns.to_numpy(dtype=float)).all():
                    raise ValueError("baseline returns must be finite")
                excess = returns - cash
                for year, values in excess.groupby(excess.index.year):
                    rows.append({
                        "panel": panel,
                        "strategy": strategy,
                        "seed": str(seed),
                        "year": int(year),
                        "observations": len(values),
                        "annualized_excess_return": float(252 * values.mean()),
                    })
    return pd.DataFrame(rows)


def _seed_contract_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute five signed differences in Sharpe relative to cash
    for each seed and ETF set.
    """
    learned = metrics.loc[metrics["strategy"] == "learned_gbrt"]
    if learned.empty or learned[["panel", "seed"]].isna().any().any():
        raise ValueError("learned metrics require nonempty panel and seed identifiers")
    if learned.duplicated(["panel", "seed", "variant"]).any():
        raise ValueError("learned metrics contain duplicate seed and variant rows")
    rows = []
    for (panel, seed), frame in learned.groupby(["panel", "seed"], sort=True):
        values = frame.set_index("variant")["cash_excess_sharpe"]
        if set(values.index) != set(VARIANTS):
            raise ValueError("each learned seed requires all six variants")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError("seed Sharpe values must be finite")
        for switch, (lhs, rhs) in SWITCHES.items():
            rows.append({
                "panel": panel,
                "seed": str(seed),
                "switch": switch,
                "sharpe_difference": float(values[lhs] - values[rhs]),
            })
    return pd.DataFrame(rows)


def _serialized_metric_values(frame: pd.DataFrame) -> dict[str, float]:
    return {
        row.strategy: float(format(float(row.cash_excess_sharpe), ".12g"))
        for row in frame.itertuples(index=False)
    }


def _metric_order(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda strategy: (-values[strategy], strategy))
    return {strategy: rank for rank, strategy in enumerate(ordered, start=1)}


def _metric_ties(values: dict[str, float]) -> set[str]:
    counts = pd.Series(list(values.values())).value_counts()
    tied_values = set(counts.loc[counts > 1].index)
    return {strategy for strategy, value in values.items() if value in tied_values}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _write_generated_values(
    *,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    engine: pd.DataFrame,
    metrics: pd.DataFrame,
    ranks: pd.DataFrame,
    data_records: list[dict[str, object]],
    path: Path,
) -> None:
    """Write paper-facing LaTeX values from the aggregate result tables."""

    strategy_cells = {
        "ts_momentum": "Time series momentum",
        "cross_sectional_momentum": "Cross-sectional momentum",
        "short_term_reversal": "Short-term reversal",
        "learned_gbrt": "Gradient boosting",
    }
    panel_labels = {
        "us_sector_etfs": "United States sectors",
        "country_equity_etfs": "Country equities",
    }
    panel_cells = {
        "us_sector_etfs": r"\shortstack[l]{United States\\sectors}",
        "country_equity_etfs": "Country equities",
    }
    switch_cells = {
        "same_close_minus_baseline": (
            "Retroactive assignment minus execution at the first later close"
        ),
        "zero_cash_minus_baseline": "Zero cash return minus SHV cash return",
        "zero_cost_minus_baseline": (
            "Zero trading cost minus a charge of 13 basis points"
        ),
        "strategy_minus_costed_comparator": (
            "Net strategy minus comparison portfolio"
        ),
        "vectorized_minus_event": "Vectorized engine minus event-driven engine",
    }

    def macro(name: str, value: str | int) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    def percent(value: float) -> str:
        return f"{value * 100.0:.2f}\\%"

    data_by_panel = {str(record["panel"]): record for record in data_records}
    if set(data_by_panel) != set(PANEL_LABELS):
        raise ValueError("generated values require one provenance record per panel")
    input_digests = {}
    for panel, record in data_by_panel.items():
        digest = record.get("input_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid input digest for {panel}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid input digest for {panel}") from exc
        input_digests[panel] = digest

    baseline = summary.loc[summary["variant"] == "baseline"].copy()
    baseline_rows = []
    trading_rows = []
    rank_rows = []
    for panel in PANEL_LABELS:
        frame = baseline.loc[baseline["panel"] == panel].set_index("strategy")
        panel_summary = summary.loc[summary["panel"] == panel].set_index(
            ["strategy", "variant"]
        )
        panel_ranks = ranks.loc[ranks["panel"] == panel].set_index(
            ["strategy", "variant"]
        )
        if rank_rows:
            rank_rows.append(r"\addlinespace")
        rank_rows.append(
            rf"\multicolumn{{7}}{{@{{}}l@{{}}}}{{\textit{{{panel_labels[panel]}}}}} \\"
        )
        for strategy in STRATEGY_LABELS:
            row = frame.loc[strategy]
            performance_cells = []
            for metric in (
                "annualized_arithmetic_return", "cash_excess_sharpe",
                "cagr", "max_drawdown",
            ):
                value = float(row[metric])
                cell = f"{value:.2f}" if metric == "cash_excess_sharpe" else percent(value)
                # Drawdowns are negative: the maximum is the shallowest loss.
                if value == frame[metric].max():
                    cell = rf"\textbf{{{cell}}}"
                performance_cells.append(cell)
            baseline_rows.append(
                " & ".join(
                    [
                        panel_cells[panel],
                        strategy_cells[strategy],
                        *performance_cells,
                    ]
                )
                + r" \\"
            )
            trading_rows.append(
                " & ".join(
                    [
                        panel_cells[panel],
                        strategy_cells[strategy],
                        percent(float(row["mean_risky_exposure"])),
                        f"{float(row['annualized_gross_traded_notional']):.2f}",
                        percent(
                            252.0 * float(row["arithmetic_cost_drag"])
                            / int(row["observations"])
                        ),
                    ]
                )
                + r" \\"
            )

            cells = [strategy_cells[strategy]]
            for variant in ("baseline", "same_close", "zero_cost"):
                key = (strategy, variant)
                value = float(panel_summary.loc[key, "cash_excess_sharpe"])
                score = f"{value:.3f}"
                rank_values = _serialized_metric_values(
                    panel_summary.xs(variant, level="variant").reset_index()
                )
                if rank_values[strategy] == max(rank_values.values()):
                    score = rf"\mathbf{{{score}}}"
                cells.extend([
                    f"${score}$",
                    f"{panel_ranks.loc[key, 'variant_rank']:.0f}",
                ])
            rank_rows.append(" & ".join(cells) + r" \\")

    primary = effects.loc[effects["block_length"] == 21].copy()
    effect_rows = []
    for switch in SWITCHES:
        if effect_rows:
            effect_rows.append(r"\addlinespace")
        effect_rows.append(
            rf"\multicolumn{{4}}{{@{{}}l@{{}}}}{{{switch_cells[switch]}}} \\"
        )
        for metric_label, metric in (
            ("Arithmetic return", "annualized_mean"),
            ("Sharpe ratio", "sharpe"),
        ):
            cells = [metric_label]
            for block_length in (5, 21, 63):
                frame = effects.loc[
                    (effects["switch"] == switch)
                    & (effects["block_length"] == block_length)
                ]
                below = int((frame[f"{metric}_ci_95_upper"] < 0.0).sum())
                above = int((frame[f"{metric}_ci_95_lower"] > 0.0).sum())
                count = int(len(frame))
                cells.append(f"{below}/{count - below - above}/{above}")
            effect_rows.append(
                " & ".join(cells) + r" \\"
            )

    learned = metrics.loc[
        (metrics["strategy"] == "learned_gbrt")
        & (metrics["variant"] == "baseline")
    ]
    seed_ranges = {}
    for panel in PANEL_LABELS:
        values = learned.loc[
            learned["panel"] == panel,
            "cash_excess_sharpe",
        ]
        seed_ranges[panel] = f"{values.min():.2f} to {values.max():.2f}"

    cost_effect = primary.loc[primary["switch"] == "zero_cost_minus_baseline"]
    same_close = primary.loc[
        primary["switch"] == "same_close_minus_baseline"
    ]
    zero_cash = primary.loc[primary["switch"] == "zero_cash_minus_baseline"]
    rank_changes = {
        variant: int(
            (
                ranks.loc[ranks["variant"] == variant, "rank_change"]
                != 0
            ).sum()
        )
        for variant in ("same_close", "zero_cost")
    }
    lines = [
        "% Generated by scripts/run_expansion_experiments.py; do not edit.",
        macro("ExpansionPanelCount", len(PANEL_LABELS)),
        macro("ExpansionFamilyCount", len(STRATEGY_LABELS)),
        macro("ExpansionFamilyPanelCount", len(baseline)),
        macro(
            "ExpansionPositiveBaselineSharpeCount",
            int((baseline["cash_excess_sharpe"] > 0.0).sum()),
        ),
        macro("ExpansionEvaluationSessions", f"{int(baseline['observations'].min()):,}"),
        macro("ExpansionBootstrapReplications", f"{int(primary['replications'].min()):,}"),
        macro("ExpansionProtocolDigest", FROZEN_PROTOCOL_SHA256),
        macro("ExpansionSectorInputDigest", input_digests["us_sector_etfs"]),
        macro(
            "ExpansionCountryInputDigest",
            input_digests["country_equity_etfs"],
        ),
        macro("ExpansionBaselineRows", "\n".join(baseline_rows)),
        macro("ExpansionTradingRows", "\n".join(trading_rows)),
        macro("ExpansionRankRows", "\n".join(rank_rows)),
        macro("ExpansionEffectRows", "\n".join(effect_rows)),
        macro(
            "ExpansionCostEffectRange",
            (
                f"{cost_effect['annualized_mean_difference'].min() * 100.0:.2f} to "
                f"{cost_effect['annualized_mean_difference'].max() * 100.0:.2f}"
                " percentage points"
            ),
        ),
        macro(
            "ExpansionSameCloseBelowCount",
            int((same_close["annualized_mean_ci_95_upper"] < 0.0).sum()),
        ),
        macro(
            "ExpansionSameCloseOverlapCount",
            int(
                (
                    (same_close["annualized_mean_ci_95_lower"] <= 0.0)
                    & (same_close["annualized_mean_ci_95_upper"] >= 0.0)
                ).sum()
            ),
        ),
        macro(
            "ExpansionZeroCashBelowCount",
            int((zero_cash["annualized_mean_ci_95_upper"] < 0.0).sum()),
        ),
        macro(
            "ExpansionZeroCashOverlapCount",
            int(
                (
                    (zero_cash["annualized_mean_ci_95_lower"] <= 0.0)
                    & (zero_cash["annualized_mean_ci_95_upper"] >= 0.0)
                ).sum()
            ),
        ),
        macro(
            "ExpansionEngineMaxDailyBp",
            f"{engine['max_absolute_daily_return_difference'].max() * 10_000.0:.2f}",
        ),
        macro(
            "ExpansionEngineMaxWealthGap",
            percent(float(engine["final_wealth_difference"].abs().max())),
        ),
        macro("ExpansionSectorSeedRange", seed_ranges["us_sector_etfs"]),
        macro("ExpansionCountrySeedRange", seed_ranges["country_equity_etfs"]),
        macro("ExpansionSameCloseRankChanges", rank_changes["same_close"]),
        macro("ExpansionZeroCostRankChanges", rank_changes["zero_cost"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _plot_baseline(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_plot_style("paper")
    baseline = summary.loc[summary["variant"] == "baseline"]
    labels = {
        "ts_momentum": "Time series momentum",
        "cross_sectional_momentum": "Cross-sectional momentum",
        "short_term_reversal": "Short-term reversal",
        "learned_gbrt": "GBRT",
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.1, 3.3), sharex=True, sharey=True)
    for panel_index, panel in enumerate(PANEL_LABELS):
        axis = axes[panel_index]
        frame = baseline.loc[baseline["panel"] == panel].set_index("strategy")
        for strategy in STRATEGY_LABELS:
            row = frame.loc[strategy]
            axis.scatter(
                row.annualized_arithmetic_return * 100.0,
                row.cash_excess_sharpe,
                **STRATEGY_STYLES[strategy],
                label=labels[strategy], s=45, zorder=3,
            )
        axis.margins(x=0.12, y=0.16)
        axis.axhline(0.0, color=SPINE, linewidth=0.8)
        axis.axvline(0.0, color=SPINE, linewidth=0.8)
        axis.set_title(PANEL_LABELS[panel])
        style_axis(axis)
        add_panel_label(axis, chr(ord("a") + panel_index))
    axes[0].set_ylabel("Sharpe relative to SHV")
    fig.supxlabel("Annualized arithmetic return (%)", y=0.20, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.34, top=0.9, wspace=0.14)
    _save_figure(fig, output / "baseline_performance")
    plt.close(fig)


def _plot_effects(effects: pd.DataFrame, output: Path, *, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    apply_plot_style("paper")
    primary = effects.loc[effects["block_length"] == 21].copy()
    if metric == "return":
        estimate = "annualized_mean_difference"
        lower = "annualized_mean_ci_95_lower"
        upper = "annualized_mean_ci_95_upper"
        scale = 100.0
        label = "Annualized arithmetic return difference, percentage points"
        stem = "contract_effects_return"
    elif metric == "sharpe":
        estimate = "sharpe_difference"
        lower = "sharpe_ci_95_lower"
        upper = "sharpe_ci_95_upper"
        scale = 1.0
        label = "Difference in Sharpe relative to SHV"
        stem = "contract_effects_sharpe"
    else:
        raise ValueError("metric must be 'return' or 'sharpe'")
    comparisons = (
        "Retroactive assignment minus first later close: infeasible",
        "Zero cash return minus SHV cash return",
        "Zero trading cost minus a charge of 13 basis points",
        "Net strategy minus comparison portfolio",
        "Vectorized engine minus event-driven engine",
    )
    strategy_labels = (
        "Time series momentum", "Cross-sectional momentum",
        "Short-term reversal", "GBRT",
    )
    fig, axes = plt.subplots(5, 2, figsize=(5.6, 7.1), sharex=True, sharey=True)
    low = float(primary[[estimate, lower]].min().min() * scale)
    high = float(primary[[estimate, upper]].max().max() * scale)
    span = max(high, 0.0) - min(low, 0.0)
    padding = max(span * 0.06, 0.01)
    for row_index, (switch, heading) in enumerate(zip(SWITCHES, comparisons, strict=True)):
        for column_index, panel in enumerate(PANEL_LABELS):
            axis = axes[row_index, column_index]
            frame = primary.loc[
                (primary["panel"] == panel) & (primary["switch"] == switch)
            ].set_index("strategy")
            for strategy_index, strategy in enumerate(STRATEGY_LABELS):
                row = frame.loc[strategy]
                style = STRATEGY_STYLES[strategy]
                y = 3 - strategy_index
                axis.hlines(
                    y, row[lower] * scale, row[upper] * scale,
                    color=style["color"], linewidth=1.0, zorder=2,
                )
                axis.scatter(
                    row[estimate] * scale, y, marker=style["marker"],
                    facecolors="white" if row[lower] <= 0.0 <= row[upper] else style["color"],
                    edgecolors=style["color"], linewidths=0.9, s=20, zorder=3,
                )
            axis.axvline(0.0, color=MUTED_INK, linewidth=0.8, zorder=1)
            axis.set_yticks([3, 2, 1, 0], strategy_labels)
            axis.set_ylim(-0.6, 3.6)
            axis.set_xlim(min(low, 0.0) - padding, max(high, 0.0) + padding)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            style_axis(axis, grid=None)
            axis.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.7)
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", length=0, pad=5)
            if row_index == 0:
                axis.tick_params(axis="x", top=True, labeltop=True, bottom=False)
                axis.spines["bottom"].set_visible(False)
            elif row_index < 4:
                axis.spines["bottom"].set_visible(False)
                axis.tick_params(axis="x", bottom=False)
        axes[row_index, 0].text(
            0.0, 1.55 if row_index == 0 else 1.12, heading, transform=axes[row_index, 0].transAxes,
            ha="left", va="bottom", fontsize=8.0,
        )
    fig.subplots_adjust(left=0.32, right=0.99, bottom=0.12, top=0.87,
                        hspace=0.62, wspace=0.13)
    for column_index, panel in enumerate(PANEL_LABELS):
        box = axes[0, column_index].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.985, PANEL_LABELS[panel],
                 ha="center", fontsize=9.5)
    fig.supxlabel(label, y=0.06, fontsize=8.5)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=INK, label="Interval includes zero"),
            Line2D([], [], marker="o", linestyle="none", color=INK,
                   label="Interval excludes zero"),
        ], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
    )
    _save_figure(fig, output / stem)
    plt.close(fig)


def _plot_engine_conformance(engine: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    apply_plot_style("paper")
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6), sharex="col")
    labels = {
        "ts_momentum": "Time series momentum",
        "cross_sectional_momentum": "Cross-sectional momentum",
        "short_term_reversal": "Short-term reversal",
        "learned_gbrt": "GBRT",
    }
    metrics = (
        ("max_absolute_daily_return_difference", 10_000.0),
        ("final_wealth_difference", 100.0),
    )
    daily_upper = float(
        np.ceil(engine[metrics[0][0]].max() * 10_000.0 / 25.0) * 25.0
    )
    wealth = engine[metrics[1][0]].to_numpy(dtype=float) * 100.0
    wealth_low = min(float(wealth.min()), 0.0)
    wealth_high = max(float(wealth.max()), 0.0)
    wealth_pad = max((wealth_high - wealth_low) * 0.08, 0.05)

    for panel_index, (panel, panel_label) in enumerate(PANEL_LABELS.items()):
        selected = engine.loc[engine["panel"] == panel]
        ordered = []
        row_labels = []
        for strategy in STRATEGY_LABELS:
            family = selected.loc[selected["strategy"] == strategy]
            if strategy == "learned_gbrt":
                family = family.sort_values("seed", key=lambda seed: seed.astype(int))
            for row in family.itertuples(index=False):
                ordered.append(row)
                row_labels.append(
                    f"{labels[strategy]}, seed {row.seed}"
                    if strategy == "learned_gbrt" else labels[strategy]
                )
        positions = np.arange(len(ordered) - 1, -1, -1)
        for column, (metric, scale) in enumerate(metrics):
            axis = axes[panel_index, column]
            for position, row in zip(positions, ordered, strict=True):
                axis.scatter(
                    getattr(row, metric) * scale, position,
                    **STRATEGY_STYLES[row.strategy], s=22, zorder=3,
                )
            axis.set_yticks(positions, row_labels)
            axis.set_ylim(-0.65, len(ordered) - 0.35)
            style_axis(axis, grid="x")
            axis.tick_params(axis="both", labelsize=7.5)
            axis.tick_params(axis="x", labelbottom=True)
            axis.tick_params(axis="y", length=0, pad=5, labelleft=column == 0)
            axis.spines["left"].set_visible(False)
            axis.axvline(0.0, color=MUTED_INK, linewidth=0.8, zorder=1)
            if column == 0:
                axis.set_xlim(-daily_upper * 0.03, daily_upper)
            else:
                axis.set_xlim(wealth_low - wealth_pad, wealth_high + wealth_pad)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.xaxis.set_major_formatter(formatter)
        axes[panel_index, 0].text(
            -0.98, 1.08, panel_label,
            transform=axes[panel_index, 0].transAxes,
            ha="left", va="bottom", fontsize=8.5,
        )

    axes[0, 0].set_title("Maximum absolute\ndaily difference", fontsize=8.5, pad=27)
    axes[0, 1].set_title("Signed terminal\nwealth difference", fontsize=8.5, pad=27)
    axes[1, 0].set_xlabel("Basis points", fontsize=8)
    axes[1, 1].set_xlabel("Percent of initial capital", fontsize=8)
    fig.subplots_adjust(
        left=0.335, right=0.985, top=0.855, bottom=0.115,
        hspace=0.46, wspace=0.24,
    )
    _save_figure(fig, output / "engine_conformance")
    plt.close(fig)


def _plot_learned_seeds(metrics: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_plot_style("paper")
    frame = metrics.loc[
        (metrics["strategy"] == "learned_gbrt")
        & (metrics["variant"] == "baseline")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(5.1, 3.3), sharey=True)
    for panel_index, panel in enumerate(PANEL_LABELS):
        axis = axes[panel_index]
        selected = frame.loc[frame["panel"] == panel].sort_values("seed")
        positions = np.arange(len(selected))
        axis.scatter(
            positions,
            selected["cash_excess_sharpe"],
            **STRATEGY_STYLES["learned_gbrt"],
            s=30,
            zorder=3,
        )
        family_mean = float(selected["cash_excess_sharpe"].mean())
        axis.axhline(
            family_mean,
            color=MUTED_INK,
            linewidth=0.9,
            linestyle="--",
            label="Mean of individual Sharpes",
        )
        axis.axhline(0.0, color=SPINE, linewidth=0.8)
        axis.set_title(PANEL_LABELS[panel])
        rules = metrics.loc[
            (metrics["panel"] == panel) & (metrics["variant"] == "baseline")
            & (metrics["strategy"] != "learned_gbrt")
        ].set_index("strategy")
        for strategy, row in rules.iterrows():
            style = STRATEGY_STYLES[strategy]
            axis.axhline(
                row["cash_excess_sharpe"], color=style["color"], linewidth=0.8,
                linestyle=":",
            )
            axis.scatter(
                -0.4, row["cash_excess_sharpe"], **style, s=20, zorder=3,
                label=STRATEGY_LABELS[strategy].replace("TS momentum", "Time series momentum"),
            )
        axis.set_xlim(-0.8, len(selected) - 0.6)
        axis.set_xlabel("Random seed")
        axis.set_xticks(positions, selected["seed"].astype(int))
        style_axis(axis)
        add_panel_label(axis, chr(ord("a") + panel_index))
    axes[0].set_ylabel("Sharpe relative to SHV")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    _save_figure(fig, output / "learned_seed_sensitivity")
    plt.close(fig)


def _plot_yearly_performance(yearly: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    apply_plot_style("paper")
    grouped = yearly.groupby(["panel", "strategy", "year"])[
        "annualized_excess_return"
    ].mean()
    limit = max(float(grouped.abs().max() * 100), 1.0)
    norm = Normalize(-limit, limit)
    cmap = LinearSegmentedColormap.from_list("excess_return", RETURN_DIVERGING_COLORS)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 3.8))
    labels = [
        "Time series\nmomentum", "Cross-sectional\nmomentum",
        "Short-term\nreversal", "GBRT",
    ]
    for panel_index, (panel, label) in enumerate(PANEL_LABELS.items()):
        axis = axes[panel_index]
        frame = grouped.loc[panel].unstack("year").reindex(STRATEGY_LABELS)
        values = frame.to_numpy(dtype=float) * 100
        if not np.isfinite(values).all():
            raise ValueError("yearly figure requires every family and year")
        artist = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for row, column in np.ndindex(values.shape):
            axis.text(
                column, row, f"{values[row, column]:.1f}",
                ha="center", va="center", fontsize=7.5,
                color=contrasting_text_color(cmap(norm(values[row, column]))),
            )
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xticks(range(len(frame.columns)), frame.columns)
        axis.set_title(label)
        axis.tick_params(length=0, pad=5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        add_panel_label(axis, chr(ord("a") + panel_index))
    fig.subplots_adjust(left=0.23, right=0.99, top=0.88, bottom=0.20, hspace=0.48)
    fig.suptitle("Returns above SHV by calendar year", y=0.99)
    color_axis = fig.add_axes([0.23, 0.07, 0.76, 0.03])
    fig.colorbar(artist, cax=color_axis, orientation="horizontal").set_label(
        "Annualized arithmetic excess return (%)"
    )
    _save_figure(fig, output / "yearly_performance")
    plt.close(fig)


def _plot_seed_contract_effects(effects: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    apply_plot_style("paper")
    limit = max(float(effects["sharpe_difference"].abs().max()), 0.01)
    norm = Normalize(-limit, limit)
    cmap = LinearSegmentedColormap.from_list("sharpe_change", RETURN_DIVERGING_COLORS)
    labels = [
        "Infeasible\nretroactive\nassignment", "Zero cash\ncredit",
        "Zero trading\ncost", "Net strategy\nminus comparison\nportfolio",
        "Vectorized\naccounting",
    ]
    fig = plt.figure(figsize=(5.5, 6.5))
    matrix_axes = [
        fig.add_axes([0.115, 0.745, 0.87, 0.175]),
        fig.add_axes([0.115, 0.455, 0.87, 0.175]),
    ]
    detail_axes = [
        fig.add_axes([0.115, 0.083, 0.355, 0.145]),
        fig.add_axes([0.630, 0.083, 0.355, 0.145]),
    ]
    detail_values = effects.loc[
        effects["switch"] == "vectorized_minus_event", "sharpe_difference"
    ]
    detail_low = min(float(detail_values.min()), 0.0)
    detail_high = max(float(detail_values.max()), 0.0)
    detail_pad = max((detail_high - detail_low) * 0.12, 0.0001)

    for panel_index, (panel, label) in enumerate(PANEL_LABELS.items()):
        axis = matrix_axes[panel_index]
        frame = effects.loc[effects["panel"] == panel].pivot(
            index="seed", columns="switch", values="sharpe_difference"
        )
        frame = frame.loc[sorted(frame.index, key=int), list(SWITCHES)]
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("seed figure requires every seed and comparison")
        artist = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for row, column in np.ndindex(values.shape):
            axis.text(
                column, row, f"{values[row, column]:.4f}",
                ha="center", va="center", fontsize=7.5,
                color=contrasting_text_color(cmap(norm(values[row, column]))),
            )
        axis.set_yticks(range(len(frame)), frame.index)
        axis.set_xticks(range(len(labels)), labels)
        axis.set_ylabel("Seed", fontsize=8)
        axis.set_title(label, fontsize=8.5, pad=7)
        axis.tick_params(length=0, pad=5, labelsize=7.5)
        for spine in axis.spines.values():
            spine.set_visible(False)

        detail = detail_axes[panel_index]
        positions = np.arange(len(frame))
        detail.scatter(
            frame["vectorized_minus_event"].to_numpy(dtype=float), positions,
            **STRATEGY_STYLES["learned_gbrt"], s=19, zorder=3,
        )
        detail.set_yticks(positions, frame.index)
        detail.set_ylim(len(frame) - 0.4, -0.6)
        detail.set_xlim(detail_low - detail_pad, detail_high + detail_pad)
        detail.xaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        detail.xaxis.set_major_formatter(formatter)
        detail.set_title(label, fontsize=8, pad=6)
        detail.set_xlabel("Sharpe difference", fontsize=8)
        detail.set_ylabel("Seed", fontsize=8)
        style_axis(detail, grid="x")
        detail.tick_params(labelsize=7.5)
        detail.tick_params(axis="y", length=0)
        detail.spines["left"].set_visible(False)
        detail.axvline(0.0, color=MUTED_INK, linewidth=0.8, zorder=1)

    color_axis = fig.add_axes([0.115, 0.355, 0.87, 0.015])
    colorbar = fig.colorbar(artist, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Change in Sharpe relative to SHV", fontsize=8, labelpad=4)
    colorbar.ax.tick_params(labelsize=7.5)
    fig.text(
        0.55, 0.279, "Engine detail on an expanded scale: vectorized minus event",
        ha="center", va="center", fontsize=8.5,
    )
    _save_figure(fig, output / "seed_contract_effects")
    plt.close(fig)


def _save_figure(fig, path_without_suffix: Path) -> None:
    fig.savefig(
        path_without_suffix.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )


def _artifact_manifest(root: Path) -> dict[str, str]:
    artifacts = {}
    for directory in (root / "results", root / "figures"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                artifacts[path.relative_to(root).as_posix()] = _sha256(path)
    return artifacts


def run_expansion(
    *,
    repo_root: Path,
    protocol_path: Path,
    panel_dir: Path,
    output_dir: Path,
    generated_at: str,
) -> dict[str, object]:
    """Run the frozen protocol on each ETF dataset and write aggregate artifacts."""
    generated_timestamp = _parse_aware_timestamp(generated_at, name="generated_at")
    git = _git_metadata(repo_root)
    if git["source_commit"] == "unavailable" or not git[
        "tracked_worktree_clean_at_start"
    ]:
        raise ValueError("commit tracked source changes before running expansion")
    protocol = _load_protocol(protocol_path)
    panel_results: dict[str, dict[str, object]] = {}
    data_records = []
    for panel_name, risky_symbols in protocol["panels"].items():
        prices, metadata = _load_panel(
            repo_root,
            panel_dir,
            panel_name,
            list(risky_symbols),
            protocol,
        )
        retrieved_timestamp = _parse_aware_timestamp(
            metadata["retrieved_at"],
            name="retrieved_at",
        )
        if retrieved_timestamp > generated_timestamp:
            raise ValueError(f"panel {panel_name} was retrieved after generation")
        panel_results[panel_name] = _panel_experiment(
            panel_name,
            prices,
            list(risky_symbols),
            protocol,
        )
        data_records.append(
            {
                key: value
                for key, value in metadata.items()
                if key != "local_file"
            }
        )

    metrics = pd.concat(
        [result["metrics"] for result in panel_results.values()],
        ignore_index=True,
    )
    summary = _family_summary(metrics)
    effects = _contract_effects(panel_results, protocol)
    engine = pd.concat(
        [result["engine"] for result in panel_results.values()],
        ignore_index=True,
    )
    ranks = _rank_reversals(summary)
    yearly = _yearly_performance(panel_results)
    seed_effects = _seed_contract_effects(metrics)
    targets = [
        row
        for result in panel_results.values()
        for row in result["targets"]
    ]
    learned_diagnostics = [
        {"panel": panel, **row}
        for panel, result in panel_results.items()
        for row in result["learned_diagnostics"]
    ]

    with tempfile.TemporaryDirectory(prefix="quantcortex-expansion-") as temp:
        generated = Path(temp)
        result_dir = generated / "results"
        figure_dir = generated / "figures"
        result_dir.mkdir(parents=True)
        figure_dir.mkdir(parents=True)
        _write_csv(metrics, result_dir / "seed_variant_metrics.csv")
        _write_csv(summary, result_dir / "family_summary.csv")
        _write_csv(effects, result_dir / "contract_effects.csv")
        _write_csv(engine, result_dir / "engine_conformance.csv")
        _write_csv(ranks, result_dir / "rank_reversals.csv")
        _write_csv(yearly, result_dir / "yearly_performance.csv")
        _write_csv(seed_effects, result_dir / "seed_contract_effects.csv")
        _write_csv(
            pd.DataFrame(learned_diagnostics),
            result_dir / "learned_fit_diagnostics.csv",
        )
        _write_generated_values(
            summary=summary,
            effects=effects,
            engine=engine,
            metrics=metrics,
            ranks=ranks,
            data_records=data_records,
            path=result_dir / "generated_values.tex",
        )
        (result_dir / "target_tape_hashes.json").write_text(
            json.dumps(targets, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (result_dir / "data_provenance.json").write_text(
            json.dumps(data_records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _plot_baseline(summary, figure_dir)
        _plot_effects(effects, figure_dir, metric="return")
        _plot_effects(effects, figure_dir, metric="sharpe")
        _plot_engine_conformance(engine, figure_dir)
        _plot_learned_seeds(metrics, figure_dir)
        _plot_yearly_performance(yearly, figure_dir)
        _plot_seed_contract_effects(seed_effects, figure_dir)
        manifest = {
            "schema_version": 1,
            "generated_at": generated_at,
            "protocol": {
                "path": protocol_path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(protocol_path),
                "freeze_commit": FROZEN_PROTOCOL_COMMIT,
                "status": protocol["status"],
            },
            "git": git,
            "source_tree": _source_tree_manifest(repo_root),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scikit_learn": importlib.metadata.version("scikit-learn"),
                "matplotlib": importlib.metadata.version("matplotlib"),
                "threadpoolctl": importlib.metadata.version("threadpoolctl"),
                "threadpools": _threadpool_environment(),
            },
            "data": data_records,
            "counts": {
                "panels": len(panel_results),
                "strategy_families": len(STRATEGY_LABELS),
                "seed_variant_rows": int(len(metrics)),
                "contract_effect_rows": int(len(effects)),
                "target_tapes": int(len(targets)),
            },
            "artifacts": _artifact_manifest(generated),
        }
        (result_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("results", "figures"):
            destination = output_dir / name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(generated / name, destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("paper/expansion/protocol.json"),
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=Path("local_data/expansion"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/expansion"),
    )
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    manifest = run_expansion(
        repo_root=repo_root,
        protocol_path=args.protocol.resolve(),
        panel_dir=args.panel_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        generated_at=args.generated_at,
    )
    print(
        f"wrote {len(manifest['artifacts'])} aggregate expansion artifacts "
        f"from source commit {manifest['git']['source_commit']}"
    )


if __name__ == "__main__":
    main()
