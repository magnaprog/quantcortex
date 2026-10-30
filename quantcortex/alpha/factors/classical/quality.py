"""Point-in-time cross-sectional quality factor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.classical._cross_section import (
    cross_sectional_rank,
    cross_sectional_zscore,
)
from quantcortex.alpha.factors.classical._fundamentals import (
    _aligned_quality_panels,
    available_mean,
    panel_axes,
    validate_fundamentals,
)


class QualityFactor:
    """Composite ROE, gross-margin, and accrual-quality factor.

    Income and cash-flow measures are trailing-four-quarter sums. Balance-sheet
    denominators use the average of the latest announced balance and the
    corresponding balance roughly one year earlier. Report revisions enter only
    strictly after their announcement timestamps by default. Components must
    cover the same reporting quarters; incomplete coverage yields missing ratios.
    """

    def compute(
        self,
        fundamentals: pd.DataFrame,
        *,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Compute the composite on availability timestamps or a supplied calendar."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        panels = [
            self.cross_sectional_zscore(self._roe(frame, index, columns)),
            self.cross_sectional_zscore(self._gross_margin(frame, index, columns)),
            self.cross_sectional_zscore(self._accruals(frame, index, columns)),
        ]
        return available_mean(panels)

    def roe(
        self,
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return TTM net income divided by average book equity."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        return self._roe(frame, index, columns)

    def gross_margin(
        self,
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return TTM gross profit divided by TTM revenue."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        return self._gross_margin(frame, index, columns)

    def accruals(
        self,
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex | None = None,
        columns: pd.Index | None = None,
    ) -> pd.DataFrame:
        """Return the negative of TTM accruals divided by average total assets."""
        frame = validate_fundamentals(fundamentals)
        index, columns = panel_axes(frame, index, columns)
        return self._accruals(frame, index, columns)

    @staticmethod
    def _roe(
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        panels = _aligned_quality_panels(
            fundamentals,
            {"net_income": "ttm", "book_value": "average_balance"},
            index,
            columns,
        )
        return QualityFactor._safe_ratio(panels["net_income"], panels["book_value"])

    @staticmethod
    def _gross_margin(
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        panels = _aligned_quality_panels(
            fundamentals,
            {"gross_profit": "ttm", "revenue": "ttm"},
            index,
            columns,
        )
        return QualityFactor._safe_ratio(panels["gross_profit"], panels["revenue"])

    @staticmethod
    def _accruals(
        fundamentals: pd.DataFrame,
        index: pd.DatetimeIndex,
        columns: pd.Index,
    ) -> pd.DataFrame:
        panels = _aligned_quality_panels(
            fundamentals,
            {
                "net_income": "ttm",
                "operating_cashflow": "ttm",
                "total_assets": "average_balance",
            },
            index,
            columns,
        )
        raw_accruals = QualityFactor._safe_ratio(
            panels["net_income"] - panels["operating_cashflow"], panels["total_assets"]
        )
        return -raw_accruals

    @staticmethod
    def _safe_ratio(
        numerator: pd.DataFrame, denominator: pd.DataFrame
    ) -> pd.DataFrame:
        denom = denominator.where(denominator > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator.divide(denom)
        return ratio.replace([np.inf, -np.inf], np.nan)

    cross_sectional_zscore = staticmethod(cross_sectional_zscore)
    rank = staticmethod(cross_sectional_rank)
