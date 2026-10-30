"""Tests for the static look-ahead bias detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantcortex.data.processors.lookahead_detector import (
    LookaheadDetector,
    LookaheadViolationError,
)


@pytest.fixture
def feature_frame():
    """A feature frame with two causal columns and one leaking column."""
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.bdate_range("2021-01-01", periods=n)
    base = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=idx)

    df = pd.DataFrame(index=idx)
    # Causal rolling mean: lagged by one bar, with leading NaNs.
    df["causal_ma"] = base.rolling(5).mean().shift(1)
    # Causal return: lagged by one bar.
    df["causal_ret"] = base.pct_change(fill_method=None).shift(1)
    # Tomorrow's value leaks into today's feature and leaves a trailing NaN.
    df["leaky_future"] = base.shift(-1)
    return df, base


def test_detector_flags_future_looking_column(feature_frame):
    df, base = feature_frame
    detector = LookaheadDetector()
    report = detector.scan(df, reference=base)

    assert not report.is_clean
    assert "leaky_future" in report.flagged_columns


def test_detector_clears_causal_columns(feature_frame):
    df, base = feature_frame
    detector = LookaheadDetector()
    report = detector.scan(df, reference=base)

    assert "causal_ma" not in report.flagged_columns
    assert "causal_ret" not in report.flagged_columns


def test_trailing_nan_fingerprint_alone_detects_leak(feature_frame):
    df, _ = feature_frame
    detector = LookaheadDetector()
    # Without a reference or target, only the trailing-NaN check runs.
    report = detector.scan(df[["causal_ma", "leaky_future"]])
    assert "leaky_future" in report.flagged_columns
    assert "causal_ma" not in report.flagged_columns


def test_data_quality_notebook_reports_causal_false_positive(capsys):
    notebook_path = (
        Path(__file__).resolve().parents[1] / "research" / "01_data_quality.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "44598ad5")
    prices = pd.DataFrame(
        {"A": 100 + 10 * np.sin(2 * np.pi * np.arange(300) / 15)},
        index=pd.bdate_range("2021-01-01", periods=300),
    )
    namespace = {"pd": pd, "prices": prices}

    # Run only the diagnostic cell, without fetching prices or saving outputs.
    exec("".join(cell["source"]), namespace)

    assert "causal_ma" in namespace["report"].flagged_columns
    assert "LEAKY_future" in namespace["report"].flagged_columns
    assert "not proof of leakage" in capsys.readouterr().out


def test_near_perfect_target_correlation_flagged():
    rng = np.random.default_rng(1)
    n = 150
    idx = pd.bdate_range("2021-01-01", periods=n)
    target = pd.Series(rng.normal(0, 0.02, n), index=idx)
    df = pd.DataFrame(
        {
            "honest": pd.Series(rng.normal(0, 1, n), index=idx),
            "is_the_target": target * 3.0,  # perfectly correlated transform
        }
    )
    detector = LookaheadDetector()
    report = detector.scan(df, target=target)
    assert "is_the_target" in report.flagged_columns
    assert "honest" not in report.flagged_columns


def test_assert_clean_raises(feature_frame):
    df, base = feature_frame
    detector = LookaheadDetector()
    with pytest.raises(LookaheadViolationError):
        detector.assert_clean(df, reference=base)


def test_detector_rejects_misaligned_reference(feature_frame):
    df, base = feature_frame
    shifted_index = base.copy()
    shifted_index.index = shifted_index.index + pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="exactly match"):
        LookaheadDetector().scan(df, reference=shifted_index)
