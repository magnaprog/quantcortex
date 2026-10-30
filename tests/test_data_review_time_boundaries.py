"""Regression checks for lexical dates and chronological data boundaries."""

import numpy as np
import pandas as pd
import pytest

from quantcortex.data.local_csv import LocalDataError, load_ohlcv_csv, load_price_matrix
from quantcortex.data.processors.lookahead_detector import (
    LookaheadDetector,
    scan_features,
)
from quantcortex.data.processors.pit_enforcer import PITEnforcer, PITViolationError


@pytest.mark.parametrize("ohlcv", [False, True])
@pytest.mark.parametrize("dates", [["20240102", "20240103"], ["2024-01-02", "2024-01-03"]])
def test_csv_calendar_dates_preserve_the_written_days(tmp_path, ohlcv, dates):
    frame = pd.DataFrame({"date": dates, "AAA": [100.0, 101.0]})
    if ohlcv:
        frame = frame.drop(columns="AAA")
        for column in ["open", "high", "low", "close", "adj_close"]:
            frame[column] = [100.0, 101.0]
        frame["volume"] = [10.0, 11.0]
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)

    loader = load_ohlcv_csv if ohlcv else load_price_matrix
    result = loader(path, start="2024-01-02", end="2024-01-03")

    assert result.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


@pytest.mark.parametrize("date", ["20241399", "1704153600"])
def test_csv_invalid_numeric_date_tokens_do_not_become_epoch_dates(tmp_path, date):
    path = tmp_path / "prices.csv"
    path.write_text(f"date,AAA\n{date},100\n", encoding="ascii")

    with pytest.raises(LocalDataError, match="invalid date"):
        load_price_matrix(path)


def test_csv_iso_timezone_dates_still_normalize_to_utc(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,AAA\n2024-01-02T20:00:00-05:00,100\n", encoding="ascii")

    assert load_price_matrix(path).index.tolist() == [pd.Timestamp("2024-01-03T01:00:00")]


@pytest.mark.parametrize("feature_unit,announcement_unit", [("ns", "us"), ("us", "ns"), ("ms", "us"), ("s", "ns")])
def test_pit_merge_normalizes_datetime_units_without_changing_values(feature_unit, announcement_unit):
    features = pd.DataFrame({
        "symbol": ["A"],
        "feature_date": pd.DatetimeIndex(["2024-05-01T08:00:00-04:00"]).as_unit(feature_unit),
    })
    fundamentals = pd.DataFrame({
        "symbol": ["A"],
        "announcement_date": pd.DatetimeIndex(["2024-04-01T12:00:00Z"]).as_unit(announcement_unit),
        "value": [5.0],
    })

    result = PITEnforcer().point_in_time_merge(features, fundamentals)

    assert result.value.tolist() == [5.0]
    assert result.feature_date.iloc[0] == pd.Timestamp("2024-05-01T12:00:00")
    assert result.announcement_date.iloc[0] == pd.Timestamp("2024-04-01T12:00:00")


def test_pit_merge_preserves_submicrosecond_strict_and_exact_boundaries():
    announcement = pd.Timestamp("2024-04-01T12:00:00.000000001")
    features = pd.DataFrame({"symbol": ["A"] * 3, "feature_date": [
        announcement + pd.Timedelta(1, unit="ns"),
        announcement,
        announcement - pd.Timedelta(1, unit="ns"),
    ]})
    fundamentals = pd.DataFrame({"symbol": ["A"], "announcement_date": [announcement], "value": [5.0]})

    strict = PITEnforcer().point_in_time_merge(features, fundamentals)
    exact = PITEnforcer(allow_exact_matches=True).point_in_time_merge(features, fundamentals)

    assert strict.value.iloc[0] == 5.0
    assert strict.value.iloc[1:].isna().all()
    assert exact.value.iloc[:2].tolist() == [5.0, 5.0]
    assert pd.isna(exact.value.iloc[2])
    assert strict.feature_date.tolist() == features.feature_date.tolist()


def test_pit_out_of_range_dates_raise_a_domain_error():
    frame = pd.DataFrame({"symbol": ["A"], "feature_date": ["2500-01-02"], "announcement_date": ["2500-01-01"]})

    with pytest.raises(PITViolationError, match="nanosecond timestamp range"):
        PITEnforcer().enforce(frame)


@pytest.mark.parametrize("entry_point", ["scan", "assert_clean", "scan_features"])
@pytest.mark.parametrize("order", ["descending", "shuffled"])
def test_lookahead_entry_points_reject_unsorted_time_indexes(entry_point, order):
    index = pd.bdate_range("2024-01-01", periods=100)
    reference = pd.Series(np.random.default_rng(7).normal(size=100), index=index)
    features = pd.DataFrame({"future": reference.shift(-1)}).iloc[:-1]
    reference = reference.iloc[:-1]
    positions = np.arange(len(features))[::-1]
    if order == "shuffled":
        positions = np.random.default_rng(8).permutation(len(features))
    features = features.iloc[positions]
    reference = reference.iloc[positions]
    action = scan_features if entry_point == "scan_features" else getattr(LookaheadDetector(), entry_point)

    with pytest.raises(ValueError, match="sorted in increasing order"):
        action(features, reference=reference)


def test_lookahead_chronological_future_and_causal_controls():
    index = pd.bdate_range("2024-01-01", periods=100)
    reference = pd.Series(np.random.default_rng(7).normal(size=100), index=index)
    features = pd.DataFrame({"future": reference.shift(-1), "past": reference.shift(1)})

    result = LookaheadDetector().scan(features, reference=reference)

    assert result.flagged_columns == ["future"]
    assert LookaheadDetector().scan(features[["past"]], reference=reference).is_clean
