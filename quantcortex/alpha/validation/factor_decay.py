"""Factor signal-decay analysis.

Compute Information Coefficients (ICs) against forward returns at several
horizons, autocorrelation of an IC series, and the first half-magnitude
crossing of an IC profile. Predictive strength need not decay monotonically.

Conventions
-----------
* ``factor`` and ``returns`` are date x symbol panels. ``returns[t]`` is the
  *single-period* return realized over the period ending at ``t`` (a standard
  return series).
* The **L-period-ahead forward return** for a decision made at date ``t`` is
  the cumulative return from ``t`` to ``t + L`` (for ``L >= 1``), built by
  compounding forward single-period returns and then shifting back to ``t``.
  These are ex-post evaluation labels, unavailable at ``t``. The profile
  starts at lag 1 and excludes the return ending at ``t``. Capturing the full
  labeled interval still requires a signal available before execution at
  ``t``; a feature using that closing price cannot assume a fill at that close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantcortex.alpha.validation.alphalens_report import (
    compute_information_coefficient,
)


class FactorDecay:
    """Measure how a factor's IC decays across forward-return horizons.

    Parameters
    ----------
    ic_method:
        Correlation method passed through to the IC computation
        (``"spearman"`` default, i.e. rank IC).
    """

    def __init__(self, ic_method: str = "spearman") -> None:
        self.ic_method = str(ic_method).lower()
        if self.ic_method not in {"spearman", "pearson"}:
            raise ValueError("ic_method must be 'spearman' or 'pearson'")

    # ------------------------------------------------------------------
    # Forward-return construction
    # ------------------------------------------------------------------
    @staticmethod
    def _forward_return(returns: pd.DataFrame, lag: int) -> pd.DataFrame:
        """Ex-post forward-return label aligned to the decision date.

        Returns the cumulative compounded return over the next ``lag``
        periods (``lag >= 1``), indexed at the decision date ``t``.  Lag 0 is
        deliberately unsupported: it would be the return realized over the
        period ending at ``t``, which is not a tradeable payoff for a
        decision taken at the close of ``t``.
        """
        if isinstance(lag, bool) or int(lag) != lag or lag < 1:
            raise ValueError("lag must be >= 1 (lag 0 is not tradeable)")
        lag = int(lag)
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        numeric = returns.apply(pd.to_numeric, errors="coerce")
        if (numeric.isna() & returns.notna()).any(axis=None):
            raise ValueError("returns contain non-numeric observations")
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any() or (numeric.dropna(how="all") <= -1.0).any(axis=None):
            raise ValueError("returns must be finite when present and greater than -100%")
        # Cumulative gross return over a trailing window of `lag` periods...
        gross = (1.0 + numeric).rolling(window=lag, min_periods=lag).apply(
            np.prod, raw=True
        )
        cum = gross - 1.0
        # ...then shift back by `lag` so the window starting at t is at t.
        return cum.shift(-lag)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def compute(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        max_lag: int = 10,
    ) -> pd.DataFrame:
        """IC decay profile across forward horizons ``1..max_lag``.

        Lag 1 labels the next single-period return. The profile excludes the
        return already realized at the decision date. Signal availability and
        execution delay must be checked separately before treating labels as
        achievable payoffs.

        Parameters
        ----------
        factor, returns:
            Date x symbol panels (see module docstring for return semantics).
        max_lag:
            Largest forward horizon, in periods (default 10).

        Returns
        -------
        pandas.DataFrame
            Indexed by ``lag`` (1..``max_lag``) with columns ``ic_mean``,
            ``ic_std`` and ``icir`` (= ``ic_mean / ic_std``).
        """
        if isinstance(max_lag, bool) or int(max_lag) != max_lag or max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        max_lag = int(max_lag)

        rows = {}
        for lag in range(1, max_lag + 1):
            fwd = self._forward_return(returns, lag)
            ic = compute_information_coefficient(
                factor, fwd, method=self.ic_method
            ).dropna()
            if len(ic):
                mean = float(ic.mean())
                std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
                icir = mean / std if std and not np.isnan(std) else np.nan
            else:
                mean = std = icir = np.nan
            rows[lag] = {"ic_mean": mean, "ic_std": std, "icir": icir}

        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = "lag"
        return out[["ic_mean", "ic_std", "icir"]]

    # ------------------------------------------------------------------
    # Persistence diagnostics
    # ------------------------------------------------------------------
    def ic_autocorrelation(self, ic: pd.Series, max_lag: int = 10) -> pd.Series:
        """Autocorrelation of an IC time series at lags ``1..max_lag``.

        High positive autocorrelation indicates persistence in the IC series at
        the measured lags. Near-zero autocorrelation indicates little linear
        serial association at those lags; it does not establish independence.
        Missing values are removed before shifting, so lag counts retained
        observations rather than elapsed calendar or trading periods.

        Parameters
        ----------
        ic:
            An IC time series (e.g. from
            :func:`compute_information_coefficient`).
        max_lag:
            Maximum autocorrelation lag (default 10).

        Returns
        -------
        pandas.Series
            Autocorrelation indexed by lag ``1..max_lag``.
        """
        if isinstance(max_lag, bool) or int(max_lag) != max_lag or max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        max_lag = int(max_lag)
        if not isinstance(ic, pd.Series):
            raise TypeError("ic must be a pandas Series")
        numeric = pd.to_numeric(ic, errors="coerce")
        if (numeric.isna() & ic.notna()).any():
            raise ValueError("ic must contain numeric observations")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError("ic must not contain infinite values")
        clean = numeric.dropna()
        ac = {lag: clean.autocorr(lag=lag) for lag in range(1, max_lag + 1)}
        out = pd.Series(ac, dtype=float)
        out.index.name = "lag"
        out.name = "ic_autocorr"
        return out

    def half_life(self, decay: pd.DataFrame) -> float:
        """Estimate the first lag where absolute mean IC halves from lag 1.

        We anchor on the lag-1 mean IC (the typical one-period-ahead signal)
        and find the smallest lag ``L`` at which ``|ic_mean(L)|`` falls to or
        below half of ``|ic_mean(1)|``. Linear interpolation between the
        bracketing observed lags gives a fractional crossing lag; missing
        intermediate IC values are skipped. If the IC never decays
        that far within the available lags, ``inf`` is returned; if the
        reference IC magnitude is zero or unavailable, ``nan`` is returned.

        Parameters
        ----------
        decay:
            The frame returned by :meth:`compute` (must contain ``ic_mean``
            and lags including 1).

        Returns
        -------
        float
            Estimated crossing lag in periods, not elapsed periods from lag 1.
        """
        if not isinstance(decay, pd.DataFrame):
            raise TypeError("decay must be a pandas DataFrame")
        if decay.index.has_duplicates:
            raise ValueError("decay lags must be unique")
        try:
            lags_array = np.asarray(decay.index, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("decay lags must be numeric") from exc
        if (
            not np.all(np.isfinite(lags_array))
            or np.any(lags_array < 1)
            or np.any(lags_array != np.floor(lags_array))
        ):
            raise ValueError("decay lags must be positive integers")
        if not decay.index.is_monotonic_increasing:
            raise ValueError("decay lags must be sorted in increasing order")
        if "ic_mean" not in decay.columns or 1 not in decay.index:
            return float("nan")

        series = pd.to_numeric(decay["ic_mean"], errors="coerce")
        if (series.isna() & decay["ic_mean"].notna()).any():
            raise ValueError("decay ic_mean values must be numeric")
        if np.isinf(series.to_numpy(dtype=float)).any():
            raise ValueError("decay ic_mean values must not be infinite")
        series = series.abs()
        ref = series.loc[1]
        if pd.isna(ref) or ref <= 0:
            return float("nan")
        target = ref / 2.0

        lags = [lag for lag in decay.index if lag >= 1]
        prev_lag = 1
        prev_val = series.loc[1]
        for lag in lags:
            if lag == 1:
                continue
            val = series.loc[lag]
            if pd.isna(val):
                continue
            if val <= target:
                # Interpolate between prev_lag (above target) and lag (at/below).
                if prev_val == val:
                    return float(lag)
                frac = (prev_val - target) / (prev_val - val)
                return float(prev_lag + frac * (lag - prev_lag))
            prev_lag, prev_val = lag, val
        return float("inf")

    # ------------------------------------------------------------------
    # Plotting (optional, lazy import)
    # ------------------------------------------------------------------
    def plot(self, decay: pd.DataFrame):
        """Plot the IC decay profile (mean IC with +/- 1 std band).

        Parameters
        ----------
        decay:
            The frame returned by :meth:`compute`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt  # lazy import

        from quantcortex.backtest.metrics.plotting import (
            INK,
            REFERENCE_BLUE,
            plot_style_context,
            style_axis,
        )

        with plot_style_context("notebook"):
            fig, ax = plt.subplots(figsize=(8, 4.2))
            lags = decay.index.to_numpy()
            mean = decay["ic_mean"].to_numpy(dtype=float)
            std = decay["ic_std"].to_numpy(dtype=float)

            ax.plot(
                lags,
                mean,
                marker="o",
                color=REFERENCE_BLUE,
                label="Mean IC",
            )
            ax.fill_between(
                lags,
                mean - std,
                mean + std,
                color=REFERENCE_BLUE,
                alpha=0.16,
                label="+/- 1 std.",
            )
            ax.axhline(0.0, color=INK, lw=0.8)
            ax.set_title("Factor IC decay")
            ax.set_xlabel("Forward-return lag (periods)")
            ax.set_ylabel("Information coefficient")
            style_axis(ax, grid="y")
            ax.legend(loc="best")
            fig.tight_layout()
        return fig
