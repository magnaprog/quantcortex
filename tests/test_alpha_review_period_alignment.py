"""Quality components require the same available fiscal periods."""

import numpy as np
import pandas as pd
import pytest

from quantcortex.alpha.factors.classical.quality import QualityFactor


def _fundamentals():
    periods = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"])
    fields = {
        "gross_profit": [40., 44., 48., 52., 56.],
        "revenue": [100., 110., 120., 130., 140.],
        "net_income": [1., 2., 3., 4., 5.],
        "operating_cashflow": [1., 2., 3., 4., 5.],
        "book_value": [100., 110., 120., 130., 140.],
        "total_assets": [200., 210., 220., 230., 240.],
    }
    return pd.DataFrame([
        {"symbol": "A", "field": field, "period_end": period,
         "announcement_date": period + pd.Timedelta(days=30), "value": value}
        for field, values in fields.items()
        for period, value in zip(periods, values, strict=True)
    ])


@pytest.mark.parametrize("method,delayed_field,expected", [
    ("gross_margin", "revenue", .4),
    ("gross_margin", "gross_profit", .4),
    ("accruals", "operating_cashflow", 0.),
    ("accruals", "net_income", 0.),
    ("accruals", "total_assets", 0.),
    ("roe", "book_value", 14 / 120),
    ("roe", "net_income", 14 / 120),
])
def test_quality_waits_for_current_quarter_components(method, delayed_field, expected):
    frame = _fundamentals()
    latest = frame.field.eq(delayed_field) & frame.period_end.eq(pd.Timestamp("2024-03-31"))
    frame.loc[latest, "announcement_date"] = pd.Timestamp("2024-05-03")
    dates = pd.to_datetime(["2024-05-01", "2024-05-03", "2024-05-04"])
    actual = getattr(QualityFactor(), method)(frame, index=dates)
    assert actual.A.iloc[:2].isna().all()
    assert actual.A.iloc[2] == pytest.approx(expected)


def test_quality_matches_all_four_quarters_not_just_the_latest_period():
    frame = _fundamentals()
    different_quarter = frame.field.eq("revenue") & frame.period_end.eq(pd.Timestamp("2023-06-30"))
    frame.loc[different_quarter, "period_end"] = pd.Timestamp("2023-06-29")
    actual = QualityFactor().gross_margin(frame, index=pd.to_datetime(["2024-05-01"]))
    assert actual.isna().all(axis=None)


def test_quality_missing_field_does_not_reuse_an_older_complete_window():
    frame = _fundamentals()
    missing = frame.field.eq("revenue") & frame.period_end.eq(pd.Timestamp("2024-03-31"))
    frame = frame.loc[~missing]
    dates = pd.to_datetime(["2024-04-29", "2024-05-01", "2024-05-15"])
    actual = QualityFactor().gross_margin(frame, index=dates)
    assert actual.A.iloc[0] == pytest.approx(.4)
    assert actual.A.iloc[1:].isna().all()


def test_quality_revision_enters_only_after_its_announcement():
    frame = _fundamentals()
    revision = {"symbol": "A", "field": "gross_profit", "period_end": pd.Timestamp("2023-12-31"),
                "announcement_date": pd.Timestamp("2024-05-15"), "value": 104.}
    frame = pd.concat([frame, pd.DataFrame([revision])], ignore_index=True)
    dates = pd.to_datetime(["2024-05-01", "2024-05-15", "2024-05-16"])
    actual = QualityFactor().gross_margin(frame, index=dates)
    np.testing.assert_allclose(actual.A, [.4, .4, 252 / 500])
