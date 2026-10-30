"""Macroeconomic feature engineering with approximate publication lags.

This module turns a wide, date-indexed frame of raw macro/market series
(treasury yields, volatility indices, credit spreads, survey/labor data)
into a panel of derived features that are useful as conditioning variables
for cross-sectional or timing models.

Design principles
-----------------
* **Tolerant of missing inputs.** Real macro data feeds are ragged: not
  every series is always available. Every feature is computed only when its
  required input columns are present; otherwise it is silently skipped.
* **Approximate publication lags.** A feature reported on date ``t`` uses data
  whose approximate availability date is on or before ``t``. Daily market series
  (treasury yields, VIX, credit spreads) carry no lag by default. Monthly
  survey/labor series, however, are *published* well after
  their observation date (e.g. May's UNRATE arrives around the first Friday of
  June), so each lagged series' index is shifted forward by a configurable
  number of calendar days (``publication_lags``) *before* forward-filling onto
  the working index. The default lags (UNRATE: 35 days, PMI/NAPM: 35 days)
  are approximations of typical release schedules, not exact release-calendar
  dates. Forward-filling propagates the last value whose approximate availability
  date has passed; it does not verify publication or remove later revisions.
  We never back-fill.

The accepted (case-insensitive, alias-aware) raw columns include::

    DGS10, DGS2, DGS3MO          treasury constant-maturity yields (%)
    VIXCLS / VIX                 CBOE 30-day implied volatility
    VIX3M                        CBOE 3-month implied volatility
    BAMLH0A0HYM2                 ICE BofA US high-yield OAS (%)
    BAMLC0A0CM                   ICE BofA US investment-grade OAS (%)
    PMI / NAPM / ISM             ISM manufacturing PMI (diffusion index)
    UNRATE                       civilian unemployment rate (%)
    CPIYOY / T10YIE              inflation proxy for the real-rate feature

Aliases retain the supplied source's conventions. For example, ``DTB3`` is
accepted for the three-month yield but is quoted on a discount basis; it is
not identical to the constant-maturity ``DGS3MO`` series.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Approximate number of business days per calendar month / year. Used for the
# lookback windows of change features so the windows mean roughly what their
# names say even though the working index is business-daily.
_BDAYS_PER_MONTH = 21
_BDAYS_PER_YEAR = 252

# Default publication lags (calendar days) applied to series that are released
# well after their observation date. Approximations of typical schedules: the
# unemployment rate for month M is published around the first Friday of M+1
# (~35 days after the observation date the series is stamped with); ISM PMI is
# released on the first business day(s) of the following month.
_DEFAULT_PUBLICATION_LAGS: Dict[str, int] = {"UNRATE": 35, "PMI": 35, "NAPM": 35}

# Canonical name -> list of accepted source aliases (compared case-folded).
_ALIASES: Dict[str, List[str]] = {
    "DGS10": ["DGS10", "GS10", "UST10Y", "Y10"],
    "DGS2": ["DGS2", "GS2", "UST2Y", "Y2"],
    "DGS3MO": ["DGS3MO", "DTB3", "GS3M", "UST3M", "Y3M"],
    "VIX": ["VIXCLS", "VIX", "VIX1M", "VIXINDEX"],
    "VIX3M": ["VIX3M", "VXV", "VIX3MCLS"],
    "HY": ["BAMLH0A0HYM2", "HY_OAS", "HYSPREAD", "HY"],
    "IG": ["BAMLC0A0CM", "IG_OAS", "IGSPREAD", "IG"],
    "PMI": ["PMI", "NAPM", "ISM", "ISMPMI", "NAPMPMI"],
    "UNRATE": ["UNRATE", "UNEMP", "UNEMPLOYMENT"],
    "INFLATION": ["CPIYOY", "T10YIE", "INFLATION", "BREAKEVEN10Y", "EXPINF"],
}


class MacroFeatures:
    """Derive macro features from a wide frame using approximate publication lags.

    Parameters
    ----------
    vix_change_window:
        Lookback in working-index rows for the VIX change feature. Default 5
        (one business week at the default frequency).
    credit_change_window:
        Lookback in working-index rows for the high-yield credit-spread change
        feature. Default 21 (~one month at the default frequency).
    pmi_momentum_periods:
        Lookback in working-index rows for PMI momentum. Defaults to
        ``3 * _BDAYS_PER_MONTH`` (~3 months at the default frequency).
    unrate_change_window:
        Lookback in working-index rows for the unemployment-rate change feature.
        Default ``12 * _BDAYS_PER_MONTH`` (~12 months at the default frequency).
    freq:
        Resampling frequency for the working index. Default ``"B"`` (business
        days). Any pandas offset alias is accepted.
    publication_lags:
        Mapping of raw series name -> publication lag in *calendar days*. Each
        named series' index is shifted forward by its lag before forward
        filling, approximating the date the value actually became public.
        Entries are merged over the defaults
        ``{"UNRATE": 35, "PMI": 35, "NAPM": 35}`` (pass ``{"UNRATE": 0}`` etc.
        to disable a default). Names are matched case-insensitively. A raw
        column's explicit lag takes precedence over its canonical series' lag.
    """

    def __init__(
        self,
        vix_change_window: int = 5,
        credit_change_window: int = 21,
        pmi_momentum_periods: int = 3 * _BDAYS_PER_MONTH,
        unrate_change_window: int = 12 * _BDAYS_PER_MONTH,
        freq: str = "B",
        publication_lags: Optional[Dict[str, int]] = None,
    ) -> None:
        windows = {
            "vix_change_window": vix_change_window,
            "credit_change_window": credit_change_window,
            "pmi_momentum_periods": pmi_momentum_periods,
            "unrate_change_window": unrate_change_window,
        }
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
            for value in windows.values()
        ):
            raise ValueError("macro feature windows must be positive integers")
        self.vix_change_window = int(vix_change_window)
        self.credit_change_window = int(credit_change_window)
        self.pmi_momentum_periods = int(pmi_momentum_periods)
        self.unrate_change_window = int(unrate_change_window)
        self.freq = str(freq)
        raw_lags = {
            **_DEFAULT_PUBLICATION_LAGS,
            **(publication_lags or {}),
        }
        self.publication_lags: Dict[str, int] = {}
        for key, value in raw_lags.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("publication lag names must be non-empty strings")
            if isinstance(value, bool):
                raise ValueError("publication lags must be non-negative integers")
            try:
                lag = int(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "publication lags must be non-negative integers"
                ) from exc
            if lag != value or lag < 0:
                raise ValueError("publication lags must be non-negative integers")
            self.publication_lags[key.strip()] = lag
        try:
            pd.date_range("2000-01-01", periods=2, freq=self.freq)
        except Exception as exc:
            raise ValueError(f"invalid macro feature frequency {self.freq!r}") from exc

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _build_lookup(columns: pd.Index) -> Dict[str, str]:
        """Map stripped, case-folded labels to the original column labels."""
        return {str(c).strip().casefold(): str(c) for c in columns}

    def _resolve(self, macro: pd.DataFrame, canonical: str) -> Optional[pd.Series]:
        """Return the series for ``canonical`` using alias matching, or None."""
        lookup = self._build_lookup(macro.columns)
        for alias in _ALIASES.get(canonical, [canonical]):
            key = alias.casefold()
            if key in lookup:
                return macro[lookup[key]]
        return None

    def _prepare(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Prepare observations on the configured working-frequency index.

        Series with a configured publication lag have their observation dates
        shifted forward by that many calendar days. Values are then propagated
        forward onto the working index. These lags approximate publication dates;
        they do not verify actual availability or remove later data revisions.
        """
        if not isinstance(macro, pd.DataFrame):
            raise TypeError("macro must be a pandas DataFrame")
        if macro.empty:
            return macro.copy()

        df = macro.copy()
        if df.columns.has_duplicates:
            raise ValueError("macro columns must be unique")
        normalized_columns = [str(column).strip().casefold() for column in df.columns]
        if any(not column for column in normalized_columns):
            raise ValueError("macro columns must be non-empty")
        if len(normalized_columns) != len(set(normalized_columns)):
            raise ValueError("macro columns are ambiguous after case normalization")
        index = pd.to_datetime(df.index, errors="coerce", utc=True)
        if index.isna().any():
            raise ValueError("macro index contains invalid timestamps")
        df.index = index.tz_localize(None)
        df = df.sort_index()
        # Collapse duplicate timestamps keeping the last observation.
        df = df[~df.index.duplicated(keep="last")]

        # Missing observations are expected; non-numeric observed values are
        # data errors and must not be silently converted into missing signals.
        numeric = df.apply(pd.to_numeric, errors="coerce")
        if (numeric.isna() & df.notna()).any(axis=None):
            raise ValueError("macro inputs contain non-numeric observations")
        df = numeric
        if np.isinf(df.to_numpy(dtype=float)).any():
            raise ValueError("macro inputs must not contain infinities")

        # Shift lagged series to their approximate publication dates before
        # forward-filling. These offsets do not verify actual release times.
        lags = {
            k.strip().casefold(): int(v) for k, v in self.publication_lags.items()
        }
        cols = []
        for col in df.columns:
            series = df[col]
            key = str(col).strip().casefold()
            lag = lags.get(key)
            if lag is None:
                for canonical, aliases in _ALIASES.items():
                    if key in {alias.casefold() for alias in aliases}:
                        lag = lags.get(canonical.casefold(), 0)
                        break
            lag = 0 if lag is None else lag
            if lag:
                series = series.copy()
                series.index = series.index + pd.Timedelta(days=lag)
            cols.append(series)
        df = pd.concat(cols, axis=1).sort_index()

        # Reindex at the configured frequency after forward filling.
        full_index = pd.date_range(df.index.min(), df.index.max(), freq=self.freq)
        df = df.reindex(df.index.union(full_index)).sort_index()
        df = df.ffill()
        df = df.reindex(full_index)
        return df

    # ------------------------------------------------------------------
    # Per-feature helpers (each accepts the *prepared* frame)
    # ------------------------------------------------------------------
    def yield_curve_slope_10y2y(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """10-year minus 2-year Treasury yield, in percentage points."""
        ten = self._resolve(macro, "DGS10")
        two = self._resolve(macro, "DGS2")
        if ten is None or two is None:
            return None
        return (ten - two).rename("yield_curve_slope_10y2y")

    def slope_10y3m(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """10-year minus 3-month Treasury yield, in percentage points."""
        ten = self._resolve(macro, "DGS10")
        three_mo = self._resolve(macro, "DGS3MO")
        if ten is None or three_mo is None:
            return None
        return (ten - three_mo).rename("slope_10y3m")

    def vix_level(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """Spot 30-day implied volatility index level."""
        vix = self._resolve(macro, "VIX")
        if vix is None:
            return None
        return vix.rename("vix_level")

    def vix_change(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """Change in VIX over ``vix_change_window`` working-index rows."""
        vix = self._resolve(macro, "VIX")
        if vix is None:
            return None
        return vix.diff(self.vix_change_window).rename(
            f"vix_change_{self.vix_change_window}d"
        )

    def vix_term_structure(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """VIX3M minus VIX: an implied-volatility horizon spread.

        These are spot indices at different horizons, not VIX futures prices.
        """
        vix3m = self._resolve(macro, "VIX3M")
        vix = self._resolve(macro, "VIX")
        if vix3m is None or vix is None:
            return None
        return (vix3m - vix).rename("vix_term_structure")

    def credit_spread_level(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """High-yield option-adjusted spread level."""
        hy = self._resolve(macro, "HY")
        if hy is None:
            return None
        return hy.rename("credit_spread_level")

    def credit_spread_change(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """Change in HY spread over ``credit_change_window`` working-index rows."""
        hy = self._resolve(macro, "HY")
        if hy is None:
            return None
        return hy.diff(self.credit_change_window).rename(
            f"credit_spread_change_{self.credit_change_window}d"
        )

    def ig_hy_spread(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """High-yield minus investment-grade option-adjusted spread."""
        hy = self._resolve(macro, "HY")
        ig = self._resolve(macro, "IG")
        if hy is None or ig is None:
            return None
        return (hy - ig).rename("ig_hy_spread")

    def pmi_momentum(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """ISM PMI change over ``pmi_momentum_periods`` working-index rows."""
        pmi = self._resolve(macro, "PMI")
        if pmi is None:
            return None
        return pmi.diff(self.pmi_momentum_periods).rename("pmi_momentum")

    def unrate_change(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """Unemployment-rate change over ``unrate_change_window`` index rows.

        This simple difference does not implement the Sahm rule's comparison
        of three-month averages against their preceding 12-month minimum.
        """
        unrate = self._resolve(macro, "UNRATE")
        if unrate is None:
            return None
        return unrate.diff(self.unrate_change_window).rename(
            f"unrate_change_{self.unrate_change_window // _BDAYS_PER_MONTH}m"
        )

    def real_rate(self, macro: pd.DataFrame) -> Optional[pd.Series]:
        """Real-rate proxy = nominal 10y yield minus an inflation measure.

        If an inflation/breakeven series is present it is used; otherwise this
        returns ``None``. The interpretation depends on the supplied measure
        and its horizon; aliases are not economically interchangeable.
        """
        ten = self._resolve(macro, "DGS10")
        infl = self._resolve(macro, "INFLATION")
        if ten is None or infl is None:
            return None
        return (ten - infl).rename("real_rate")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Compute all available macro features.

        Parameters
        ----------
        macro:
            Wide, date-indexed frame of raw macro/market series. Columns may
            be any subset of the recognized aliases; unrecognized columns are
            ignored. Index need not be regular.

        Returns
        -------
        pandas.DataFrame
            Frame indexed at the configured ``freq``, with one column per
            computable feature. If no feature can be computed an empty (but correctly
            indexed) frame is returned.
        """
        prepared = self._prepare(macro)
        if prepared.empty:
            return pd.DataFrame(index=prepared.index)

        builders = [
            self.yield_curve_slope_10y2y,
            self.slope_10y3m,
            self.vix_term_structure,
            self.vix_level,
            self.vix_change,
            self.credit_spread_level,
            self.credit_spread_change,
            self.ig_hy_spread,
            self.pmi_momentum,
            self.unrate_change,
            self.real_rate,
        ]

        features = []
        for builder in builders:
            series = builder(prepared)
            if series is not None:
                features.append(series)

        if not features:
            return pd.DataFrame(index=prepared.index)

        out = pd.concat(features, axis=1)
        out.index.name = macro.index.name or "date"
        return out
