"""Cross-sectional price momentum factor.

Implements the classic 12-1 month momentum factor (Jegadeesh & Titman, 1993;
Asness et al., 2013): the cumulative return measured over a long lookback
window while skipping the most recent ``gap`` trading days to avoid the
short-term reversal effect.

A factor value at ``t`` uses prices through ``t - gap``. With ``gap > 0``,
those prices precede ``t``. With ``gap=0``, the value includes the close at
``t`` and is available only after that bar closes, so execution must occur later.

Cross-sectional normalization and price/return helpers are shared with the
other classical factor modules via the private
:mod:`quantcortex.alpha.factors.classical._cross_section` module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.classical._cross_section import (
    cross_sectional_rank,
    cross_sectional_zscore,
    rolling_cov,
    to_returns,
    validate_prices,
)


class MomentumFactor:
    """Cross-sectional 12-1 month price momentum.

    The factor for symbol ``i`` on date ``t`` is the total return realized
    between ``t - lookback`` and ``t - gap``::

        momentum_i(t) = price_i(t - gap) / price_i(t - lookback) - 1

    Excluding the most recent ``gap`` days (typically one month) skips the
    short-term reversal window. A higher score means stronger past performance,
    which is the more attractive end of the momentum factor.

    Parameters
    ----------
    lookback:
        Total formation window in trading days (default 252, i.e. ~12 months).
    gap:
        Number of most-recent trading days to skip (default 21, i.e. ~1 month).
    """

    def __init__(self, lookback: int = 252, gap: int = 21) -> None:
        if (
            isinstance(lookback, (bool, np.bool_))
            or not isinstance(lookback, (int, np.integer))
            or lookback <= 0
        ):
            raise ValueError("lookback must be a positive integer")
        if (
            isinstance(gap, (bool, np.bool_))
            or not isinstance(gap, (int, np.integer))
            or gap < 0
        ):
            raise ValueError("gap must be a non-negative integer")
        if gap >= lookback:
            raise ValueError("gap must be strictly smaller than lookback")
        self.lookback = int(lookback)
        self.gap = int(gap)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute the momentum factor panel.

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date (ascending), with one column
            per symbol.

        Returns
        -------
        pandas.DataFrame
            Factor panel indexed by the same dates as ``prices`` with the same
            columns. Higher values are more attractive. The first ``lookback``
            rows are ``NaN`` because insufficient history exists.
        """
        prices = self._validate_prices(prices)

        # price(t - gap): the numerator excludes the most recent `gap` days.
        # With gap=0, it includes the close at t and requires later execution.
        recent = prices.shift(self.gap)
        # price(t - lookback): the denominator / start of the formation window.
        base = prices.shift(self.lookback)

        with np.errstate(divide="ignore", invalid="ignore"):
            momentum = recent.divide(base) - 1.0

        # Replace overflow from extreme ratios with missing values.
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        return momentum

    def residual_momentum(
        self,
        prices: pd.DataFrame,
        market: pd.Series,
        window: int | None = None,
        market_is_returns: bool | None = None,
    ) -> pd.DataFrame:
        """Momentum of market-model residual returns.

        Residual momentum measures returns left after a factor regression.
        This single-market implementation is related to the residual-momentum
        approach of Blitz, Huij & Martens (2011); it does not reproduce their
        full factor specification or establish a turnover improvement.

        For each symbol, a regression of raw stock returns on market returns,
        with an intercept, is estimated over a trailing window ending at that
        bar. This method does not subtract a risk-free series. The residuals
        are summed over the same formation window as :meth:`compute`.

        Parameters
        ----------
        prices:
            Adjusted close prices indexed by date, columns are symbols.
        market:
            Market (benchmark) price level or return series indexed by the same
            dates.
        window:
            Rolling window (in trading days) for the beta estimation. Defaults to
            ``lookback``.
        market_is_returns:
            Explicitly declare whether ``market`` is a return series (``True``)
            or a price level (``False``). ``None`` is rejected because values
            alone cannot distinguish the two reliably.

        Returns
        -------
        pandas.DataFrame
            Residual-momentum factor panel, higher = better. The module's
            execution-timing requirements apply.
        """
        prices = self._validate_prices(prices)
        if not isinstance(market, pd.Series):
            raise TypeError("market must be a pandas Series")
        window = self.lookback if window is None else window
        if (
            isinstance(window, (bool, np.bool_))
            or not isinstance(window, (int, np.integer))
            or window <= 1
        ):
            raise ValueError("window must be an integer greater than 1")
        window = int(window)

        if not isinstance(market.index, pd.DatetimeIndex):
            raise TypeError("market series must use a DatetimeIndex")
        if market.index.hasnans or market.index.has_duplicates:
            raise ValueError("market series index must contain unique, valid timestamps")
        if not market.index.is_monotonic_increasing:
            raise ValueError("market series dates must be sorted in increasing order")
        if market.index.tz is not None:
            market = market.copy()
            market.index = market.index.tz_convert("UTC").tz_localize(None)
        stock_ret = prices.pct_change(fill_method=None)
        market = market.reindex(prices.index)
        market_ret = self._to_returns(market, market_is_returns)

        # Rolling stock-on-market regression with an intercept, using raw
        # returns without subtracting a risk-free series. Coefficients use
        # returns through t and form the contemporaneous residual at t.
        residuals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        for col in prices.columns:
            y = stock_ret[col]
            x = market_ret.where(y.notna())
            y_pair = y.where(market_ret.notna())
            x_mean = x.rolling(window, min_periods=window).mean()
            x_var = x.rolling(window, min_periods=window).var(ddof=0)
            cov = self._rolling_cov(x, y_pair, window)
            beta = cov.divide(x_var)
            y_mean = y_pair.rolling(window, min_periods=window).mean()
            alpha = y_mean - beta * x_mean
            residuals[col] = y_pair - (alpha + beta * x)

        # Sum residual returns over the same formation interval as price
        # momentum. A full rolling window is required so missing residuals do
        # not disappear inside a skip-na cumulative sum.
        formation = self.lookback - self.gap
        res_mom = residuals.rolling(
            formation, min_periods=formation
        ).sum().shift(self.gap)
        res_mom = res_mom.replace([np.inf, -np.inf], np.nan)
        return res_mom

    # ------------------------------------------------------------------
    # Cross-sectional normalization (shared via _cross_section)
    # ------------------------------------------------------------------
    cross_sectional_zscore = staticmethod(cross_sectional_zscore)
    rank = staticmethod(cross_sectional_rank)

    # ------------------------------------------------------------------
    # Internal helpers (shared via _cross_section)
    # ------------------------------------------------------------------
    _validate_prices = staticmethod(validate_prices)
    _to_returns = staticmethod(to_returns)
    _rolling_cov = staticmethod(rolling_cov)
