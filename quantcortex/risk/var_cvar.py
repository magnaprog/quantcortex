"""Value-at-Risk and Conditional Value-at-Risk (Expected Shortfall) estimators.

Three tail-risk estimation families are provided:

* **Historical / empirical** - read VaR straight off the empirical loss
  distribution, without fitting a parametric distribution. Forecast use still
  requires the history to represent the risk being estimated.
* **Parametric (Gaussian)** - assume returns are normal and use closed-form
  quantiles / expected-shortfall formulae.
* **Cornish-Fisher** - a parametric estimate that corrects the Gaussian
  quantile for sample skewness and excess kurtosis.

Sign convention
----------------
Returns are signed (positive = gain, negative = loss).  **All VaR/CVaR figures
are reported as NONNEGATIVE loss estimates.** A VaR of ``0.031`` at
``alpha=0.95`` estimates a 3.1% loss threshold at the 95th percentile for the
input return period, conditional on the estimator's assumptions. If
the relevant quantile is a gain, the reported loss is floored at zero.

Estimators consume only the supplied return history. The caller must ensure
that every observation was available at the decision time. A supplied array
alone cannot establish causality.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["VaRCVaR"]

ArrayLike = Union[pd.Series, np.ndarray, "list[float]"]
TRADING_DAYS = 252


def _as_1d(returns: ArrayLike) -> np.ndarray:
    """Coerce ``returns`` to a finite 1-D ``float64`` array."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("returns is empty.")
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("returns contains no finite observations.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("returns must not contain infinite values")
    return arr


class VaRCVaR:
    """Value-at-Risk / Conditional VaR estimator at confidence level ``alpha``.

    Parameters
    ----------
    alpha:
        Confidence level in ``(0, 1)``, e.g. ``0.95`` for 95% VaR.  The tail
        probability is ``1 - alpha``.

    Notes
    -----
    Loss estimates are floored at zero (see module docstring).
    """

    def __init__(self, alpha: float = 0.95) -> None:
        if isinstance(alpha, (bool, np.bool_)):
            raise TypeError("alpha must be numeric, not boolean")
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly in (0, 1).")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @property
    def tail(self) -> float:
        """Tail probability ``1 - alpha`` (e.g. 0.05 for 95% VaR)."""
        return 1.0 - self.alpha

    # ------------------------------------------------------------------ #
    # Historical / empirical estimators                                  #
    # ------------------------------------------------------------------ #
    def historical_var(self, returns: ArrayLike) -> float:
        """Empirical VaR: the ``(1 - alpha)`` quantile loss of ``returns``.

        Negate the empirical ``tail`` quantile of the return distribution,
        then floor the loss estimate at zero.
        """
        r = _as_1d(returns)
        # The loss threshold is the alpha-th quantile of the *loss* = -return,
        # equivalently the (1-alpha) quantile of the return.
        q = float(np.quantile(r, self.tail, method="linear"))
        return max(-q, 0.0)

    def historical_cvar(self, returns: ArrayLike) -> float:
        """Historical tail-mean estimate of CVaR / Expected Shortfall.

        Negate the mean of returns at or below the interpolated return
        quantile, then floor at zero. This whole-observation tail mean does
        not fractionally weight the quantile boundary as exact discrete
        expected shortfall does, so small samples and ties can differ.
        """
        r = _as_1d(returns)
        threshold = np.quantile(r, self.tail, method="linear")
        tail_losses = r[r <= threshold]
        if tail_losses.size == 0:
            # Degenerate (e.g. tiny sample); fall back to the VaR point itself.
            return max(-float(threshold), 0.0)
        return max(-float(tail_losses.mean()), 0.0)

    # ------------------------------------------------------------------ #
    # Parametric (Gaussian) estimators                                   #
    # ------------------------------------------------------------------ #
    def parametric_var(self, returns: ArrayLike) -> float:
        """Gaussian VaR: ``-(mu + z_{1-alpha} * sigma)``.

        ``z_{1-alpha} = norm.ppf(1 - alpha)`` is negative for typical
        ``alpha``, so the quantile return is below the mean. Negate that
        quantile and floor the loss estimate at zero.
        """
        r = _as_1d(returns)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if r.size > 1 else 0.0
        z = float(stats.norm.ppf(self.tail))
        return max(-(mu + z * sigma), 0.0)

    def parametric_cvar(self, returns: ArrayLike) -> float:
        """Gaussian Expected Shortfall (closed form).

        ``ES = sigma * pdf(z) / (1 - alpha) - mu`` where ``z = norm.ppf(tail)``
        and ``pdf`` is the standard-normal density. The estimate is floored
        at zero.
        """
        r = _as_1d(returns)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if r.size > 1 else 0.0
        z = float(stats.norm.ppf(self.tail))
        pdf = float(stats.norm.pdf(z))
        return max(sigma * pdf / self.tail - mu, 0.0)

    def cornish_fisher_var(self, returns: ArrayLike) -> float:
        """Cornish-Fisher (modified) VaR adjusting the Gaussian quantile.

        The standard-normal quantile ``z`` is expanded with sample skewness
        ``S`` and excess kurtosis ``K`` using a truncated expansion::

            z_cf = z + (z^2 - 1)/6 * S
                     + (z^3 - 3z)/24 * K
                     - (2 z^3 - 5z)/36 * S^2

        VaR is then ``-(mu + z_cf * sigma)``, floored at zero. The expansion
        need not give reliable or ordered tail quantiles for extreme moments.
        """
        r = _as_1d(returns)
        mu = float(r.mean())
        sigma = float(r.std(ddof=1)) if r.size > 1 else 0.0
        if np.ptp(r) <= np.finfo(np.float64).eps * max(1.0, abs(mu)):
            return max(-mu, 0.0)
        # Request excess kurtosis directly with fisher=True and use
        # bias-corrected sample moments for the supplied return history.
        s = float(stats.skew(r, bias=False)) if r.size > 2 else 0.0
        k = float(stats.kurtosis(r, fisher=True, bias=False)) if r.size > 3 else 0.0
        z = float(stats.norm.ppf(self.tail))
        z_cf = (
            z
            + (z**2 - 1.0) / 6.0 * s
            + (z**3 - 3.0 * z) / 24.0 * k
            - (2.0 * z**3 - 5.0 * z) / 36.0 * s**2
        )
        return max(-(mu + z_cf * sigma), 0.0)

    # ------------------------------------------------------------------ #
    # Portfolio-level VaR                                                #
    # ------------------------------------------------------------------ #
    def portfolio_var(
        self,
        weights: ArrayLike,
        returns_matrix: Union[pd.DataFrame, np.ndarray],
        method: str = "historical",
    ) -> float:
        """VaR of the ``weights`` portfolio over a per-asset return matrix.

        Builds the portfolio return series ``returns_matrix @ weights`` then
        dispatches to the chosen single-series estimator.

        Parameters
        ----------
        weights:
            1-D weight vector, ``(n_assets,)``.
        returns_matrix:
            ``(T, n_assets)`` matrix of per-asset returns (DataFrame or array).
        method:
            ``"historical"``, ``"parametric"`` or ``"cornish_fisher"``.
        """
        w = np.asarray(weights, dtype=np.float64)
        rm = np.asarray(returns_matrix, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")
        if rm.ndim != 2:
            raise ValueError("returns_matrix must be 2-D (T, n_assets).")
        if rm.shape[1] != w.size:
            raise ValueError(
                f"weights length {w.size} does not match returns_matrix "
                f"columns {rm.shape[1]}."
            )
        if not np.all(np.isfinite(rm)):
            raise ValueError("returns_matrix must contain only finite values")
        port = rm @ w
        dispatch = {
            "historical": self.historical_var,
            "parametric": self.parametric_var,
            "cornish_fisher": self.cornish_fisher_var,
        }
        if method not in dispatch:
            raise ValueError(
                f"Unknown method {method!r}; expected one of {sorted(dispatch)}."
            )
        return dispatch[method](port)

    # ------------------------------------------------------------------ #
    # Annualization helper                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def annualize(
        var_value: float,
        *,
        periods_per_year: int = TRADING_DAYS,
    ) -> float:
        """Approximate annual VaR/CVaR with the square-root-of-time rule.

        Multiplication by ``sqrt(periods_per_year)`` is exact for additive sums
        of i.i.d. zero-mean Gaussian returns. It is not a general result for
        i.i.d. or compounded simple returns.
        """
        var_value = float(var_value)
        if not np.isfinite(var_value) or var_value < 0.0:
            raise ValueError("var_value must be finite and non-negative")
        if not np.isfinite(periods_per_year) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and positive")
        return var_value * float(np.sqrt(periods_per_year))
