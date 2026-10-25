"""Almgren-Chriss market-impact execution model.

This module separates permanent and temporary impact following Almgren & Chriss
(2000), "Optimal Execution of Portfolio Transactions", *Journal of Risk* 3(2),
5-39. Its one-bar fill rule is a simplified participation-based model, separate
from the paper's linear-impact liquidation schedule:

* **Permanent-impact term**. Named for a lasting market-price shift, but applied
  only to the current fill here. It scales linearly with the
  participation ``q = qty / ADV``::

      permanent = gamma * sigma * q

* **Temporary-impact term**. A price concession for demanding immediacy,
  modeled as a power of participation (concave when ``0 < alpha < 1``)::

      temporary = eta * sigma * sign(q) * |q| ** alpha

  with ``alpha = 0.5`` giving a square-root dependence.

The realized fill price for an order on a bar is the bar close pushed in the
direction of the trade by the sum of permanent and temporary impact (expressed
as a fraction of price)::

    fill = close * (1 + sign(qty) * (|permanent| + |temporary|))

Both components affect only the current fill price in this implementation.
Future bars are unchanged; persistent market-price impact is not simulated.

The closed-form *optimal execution trajectory* of Almgren-Chriss (the schedule
that minimizes a mean-variance combination of impact cost and timing risk) is
provided as :meth:`AlmgrenChriss.optimal_execution_trajectory`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quantcortex.backtest.execution_models.ideal_fill import ExecutionModel

__all__ = ["AlmgrenChriss"]

# Default volatility used when none is supplied / derivable.  Expressed as a
# per-bar return standard deviation; 2% is an uncalibrated fallback.
_DEFAULT_SIGMA = 0.02
# Default participation exponent: 0.5 gives square-root dependence.
_DEFAULT_TEMPORARY_EXPONENT = 0.5


class AlmgrenChriss(ExecutionModel):
    """Simplified permanent + temporary market-impact fill model.

    Parameters
    ----------
    eta:
        Temporary-impact coefficient.  Multiplies ``sigma * participation **
        temporary_exponent``. Default ``0.142``; no calibration is performed.
    gamma:
        Permanent-impact coefficient.  Multiplies ``sigma * participation``.
        Default ``0.314``.
    volatility:
        Optional default per-bar return volatility ``sigma`` used when a bar /
        keyword does not provide one.  ``None`` falls back to a 2% default.
    daily_volume:
        Optional default ADV (shares) used when both the call and the bar omit
        a volume.  ``None`` means the model expects ``adv`` per call.
    temporary_exponent:
        Power applied to participation in the temporary-impact term.  Default
        ``0.5`` (square-root impact).
    """

    def __init__(
        self,
        eta: float = 0.142,
        gamma: float = 0.314,
        volatility: Optional[float] = None,
        daily_volume: Optional[float] = None,
        temporary_exponent: float = _DEFAULT_TEMPORARY_EXPONENT,
    ) -> None:
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (eta, gamma, volatility, daily_volume, temporary_exponent)
            if value is not None
        ):
            raise TypeError("market-impact parameters must be numeric, not boolean")
        if not np.isfinite(eta) or not np.isfinite(gamma) or eta < 0 or gamma < 0:
            raise ValueError("eta and gamma must be non-negative.")
        if not np.isfinite(temporary_exponent) or temporary_exponent <= 0:
            raise ValueError("temporary_exponent must be positive.")
        if volatility is not None and (
            not np.isfinite(volatility) or volatility < 0.0
        ):
            raise ValueError("volatility must be finite and non-negative")
        if daily_volume is not None and (
            not np.isfinite(daily_volume) or daily_volume <= 0.0
        ):
            raise ValueError("daily_volume must be finite and positive")
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.volatility = None if volatility is None else float(volatility)
        self.daily_volume = None if daily_volume is None else float(daily_volume)
        self.temporary_exponent = float(temporary_exponent)

    # ------------------------------------------------------------------ #
    # Impact components (fractional price moves)
    # ------------------------------------------------------------------ #
    def _participation(self, qty: float, adv: float) -> float:
        """Return signed participation ``qty / adv``."""
        if isinstance(qty, (bool, np.bool_)) or isinstance(adv, (bool, np.bool_)):
            raise TypeError("quantity and ADV must be numeric, not boolean")
        if adv is None or not np.isfinite(adv) or adv <= 0:
            raise ValueError("market-impact model requires finite positive ADV")
        if not np.isfinite(qty):
            raise ValueError("quantity must be finite")
        return float(qty) / float(adv)

    def temporary_impact(self, qty: float, adv: float, sigma: float) -> float:
        """Temporary price impact as a fraction of price.

        ``eta * sigma * sign(q) * |q| ** temporary_exponent`` with
        ``q = qty / adv``.  Returns a *signed* fraction (positive for buys).
        """
        q = self._participation(qty, adv)
        if q == 0.0:
            return 0.0
        return float(self.eta * sigma * np.sign(q) * np.abs(q) ** self.temporary_exponent)

    def permanent_impact(self, qty: float, adv: float, sigma: float) -> float:
        """Permanent price impact as a fraction of price.

        ``gamma * sigma * q`` with ``q = qty / adv`` (signed, linear in
        participation).
        """
        q = self._participation(qty, adv)
        if q == 0.0:
            return 0.0
        return float(self.gamma * sigma * q)

    # ------------------------------------------------------------------ #
    # Fill
    # ------------------------------------------------------------------ #
    def _resolve_sigma(self, bar: "pd.Series", kw: dict) -> float:
        sigma = kw.get("sigma")
        if sigma is None and "volatility" in bar.index and pd.notna(bar["volatility"]):
            sigma = bar["volatility"]
        if sigma is None:
            sigma = self.volatility
        if sigma is None:
            sigma = _DEFAULT_SIGMA
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        return sigma

    def _resolve_adv(self, bar: "pd.Series", kw: dict) -> Optional[float]:
        adv = kw.get("adv")
        if adv is None:
            adv = kw.get("volume")
        if adv is None and "volume" in bar.index and pd.notna(bar["volume"]):
            adv = bar["volume"]
        if adv is None:
            adv = self.daily_volume
        return None if adv is None else float(adv)

    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return the close pushed by permanent + temporary impact.

        Uses ``sigma`` and ``adv`` from (in priority order) the keyword
        arguments, the bar (``volatility`` / ``volume`` columns), and finally
        the model's configured defaults.

        The model charges the FULL permanent + temporary impact on this
        one-shot fill. A deliberately conservative convention (the canonical
        Almgren-Chriss accounting embeds only ~half the permanent impact in
        the average execution price of a schedule).
        """
        if isinstance(target_qty, (bool, np.bool_)):
            raise TypeError("target_qty must be numeric, not boolean")
        close = float(bar["close"])
        quantity = float(target_qty)
        if not np.isfinite(close) or close <= 0.0:
            raise ValueError("bar close must be finite and positive")
        if not np.isfinite(quantity):
            raise ValueError("target_qty must be finite")
        if quantity == 0:
            return close

        sigma = self._resolve_sigma(bar, kw)
        adv = self._resolve_adv(bar, kw)

        perm = self.permanent_impact(quantity, adv, sigma)
        temp = self.temporary_impact(quantity, adv, sigma)
        direction = 1.0 if quantity > 0 else -1.0
        total_impact = direction * (abs(perm) + abs(temp))
        fill_price = float(close * (1.0 + total_impact))
        if not np.isfinite(fill_price) or fill_price <= 0.0:
            raise ValueError("market impact produced a non-positive fill price")
        return fill_price

    # ------------------------------------------------------------------ #
    # Optimal execution trajectory (closed form)
    # ------------------------------------------------------------------ #
    def optimal_execution_trajectory(
        self,
        X: float,
        T: float,
        n: int = 50,
        *,
        sigma: float = _DEFAULT_SIGMA,
        risk_aversion: float = 1e-6,
        eta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> "pd.DataFrame":
        """Closed-form Almgren-Chriss optimal liquidation schedule.

        Liquidating ``X`` shares over horizon ``T`` in ``n`` equal intervals
        (``tau = T / n``), the holdings trajectory minimizing a mean-variance
        combination of impact cost and timing risk under a linear temporary-impact
        specification is given below. This schedule does not optimize the
        nonlinear participation-based ``fill`` rule, and its dimensional
        coefficients must be supplied consistently for the chosen asset and
        time unit.

        .. math::

            x_j = X \\, \\frac{\\sinh(\\kappa (T - t_j))}{\\sinh(\\kappa T)},

        where the urgency parameter :math:`\\kappa` solves

        .. math::

            2 \\, (\\cosh(\\kappa \\tau) - 1) =
            \\frac{\\lambda \\, \\sigma^2 \\, \\tau^2}{\\tilde\\eta}, \\qquad
            \\tilde\\eta = \\eta - \\tfrac{1}{2} \\gamma \\tau .

        Parameters
        ----------
        X:
            Total shares to liquidate (positive) or acquire (negative sign just
            flips the schedule).
        T:
            Execution horizon in the chosen time unit.
        n:
            Number of equally spaced trading intervals.
        sigma:
            Price volatility in currency per share per square root of the
            chosen time unit, unlike the fractional return volatility in ``fill``.
        risk_aversion:
            Mean-variance risk-aversion ``lambda``.  Larger => faster (front
            loaded) liquidation; ``->0`` => the linear/uniform (TWAP) schedule.
        eta, gamma:
            Linear-impact coefficients: ``eta`` has units of price times time
            per share traded; ``gamma`` has units of price per share traded.
            If omitted, instance coefficients are reused numerically without
            conversion from the fill rule's participation-based units.

        Returns
        -------
        pandas.DataFrame
            Columns ``time`` (interval endpoints), ``holdings`` ``x_j`` and
            ``trade``. The shares executed during the interval *starting* at
            ``t_j`` (row ``j`` holds ``x_j - x_{j+1}``); the final row's
            ``trade`` is ``0`` since no interval starts at ``T``.
            ``holdings`` runs from ``X`` down to ``0``.
        """
        if not np.isfinite(X) or not np.isfinite(T) or T <= 0:
            raise ValueError("X must be finite and T must be finite and positive")
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("T must be > 0 and n >= 1.")
        n = int(n)
        eta_v = self.eta if eta is None else float(eta)
        gamma_v = self.gamma if gamma is None else float(gamma)
        if (
            not np.isfinite(eta_v)
            or not np.isfinite(gamma_v)
            or eta_v < 0.0
            or gamma_v < 0.0
        ):
            raise ValueError("eta and gamma must be finite and non-negative")
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        if not np.isfinite(risk_aversion) or risk_aversion < 0.0:
            raise ValueError("risk_aversion must be finite and non-negative")

        tau = T / n
        eta_tilde = eta_v - 0.5 * gamma_v * tau
        times = np.arange(0, n + 1) * tau

        if eta_tilde <= 0.0:
            raise ValueError("eta - 0.5 * gamma * tau must be positive")
        # Solve for kappa. Zero risk aversion gives the uniform TWAP schedule.
        if risk_aversion == 0.0 or sigma == 0.0:
            holdings = X * (1.0 - times / T)
        else:
            kappa_tilde_sq = risk_aversion * sigma ** 2 / eta_tilde
            # cosh(kappa*tau) = 1 + 0.5 * kappa_tilde^2 * tau^2
            cosh_arg = 1.0 + 0.5 * kappa_tilde_sq * tau ** 2
            kappa = float(np.arccosh(cosh_arg)) / tau
            if kappa == 0.0:
                holdings = X * (1.0 - times / T)
            else:
                numerator = np.exp(-kappa * times) * (
                    1.0 - np.exp(-2.0 * kappa * (T - times))
                )
                denominator = 1.0 - np.exp(-2.0 * kappa * T)
                holdings = X * numerator / denominator

        holdings[0] = X
        holdings[-1] = 0.0
        trades = -np.diff(holdings, prepend=X)  # x_{j-1} - x_j per interval
        return pd.DataFrame(
            {"time": times, "holdings": holdings, "trade": np.append(trades[1:], 0.0)}
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"AlmgrenChriss(eta={self.eta}, gamma={self.gamma}, "
            f"temporary_exponent={self.temporary_exponent})"
        )
