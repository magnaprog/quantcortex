"""Risk-parity (equal risk contribution) portfolio optimization.

A risk-parity portfolio allocates capital so that every asset contributes the
same amount of risk to the total portfolio variance, rather than equalizing the
dollar weights.  Formally, the *risk contribution* of asset :math:`i` is

.. math::

    \\mathrm{RC}_i = w_i\\,(\\Sigma w)_i,

and these sum to the portfolio variance, :math:`\\sum_i \\mathrm{RC}_i =
w^{\\top}\\Sigma w`.  The equal-risk-contribution (ERC) portfolio is the long-only
vector with :math:`\\mathrm{RC}_i` equal across all assets.

The solver uses a convex log-barrier formulation, normalizes its solution to
full investment, then projects onto configured weight bounds. Binding bounds
can break equality of risk contributions. This implementation supports only
long-only allocation and validates the returned weights against that contract.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from quantcortex.portfolio.base import (
    PortfolioMode,
    PortfolioOptimizer,
    prepare_return_panel,
    validate_return_panel,
)

__all__ = ["RiskParity"]

_RIDGE: float = 1e-8


class RiskParity(PortfolioOptimizer):
    """Equal risk contribution (ERC) optimizer.

    Parameters
    ----------
    mode:
        Must be :attr:`~quantcortex.portfolio.base.PortfolioMode.LONG_ONLY`;
        this implementation rejects market-neutral allocation.
    shrinkage:
        When ``True`` the covariance is estimated with Ledoit-Wolf shrinkage;
        defaults to ``False`` (plain sample covariance) for risk parity.
    max_iter:
        Maximum iterations granted to the underlying L-BFGS-B solver.
    tol:
        Tolerance passed as ``ftol`` and ``gtol`` for the solver's function
        and gradient convergence checks.
    **kw:
        Forwarded to :class:`~quantcortex.portfolio.base.PortfolioOptimizer`
        (``tolerance``, ``weight_bounds``).
    """

    def __init__(
        self,
        mode: Union[PortfolioMode, str] = PortfolioMode.LONG_ONLY,
        *,
        shrinkage: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-10,
        **kw,
    ) -> None:
        super().__init__(mode, **kw)
        if not isinstance(shrinkage, (bool, np.bool_)):
            raise TypeError("shrinkage must be a boolean")
        if self.mode is not PortfolioMode.LONG_ONLY:
            raise ValueError(
                "RiskParity is a long-only construction; "
                f"mode={self.mode.value!r} is not supported."
            )
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
            max_iter, (int, np.integer)
        ):
            raise TypeError("max_iter must be an integer")
        if isinstance(tol, (bool, np.bool_)):
            raise TypeError("tol must be numeric, not boolean")
        self.shrinkage = bool(shrinkage)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not np.isfinite(self.tol) or self.tol <= 0.0:
            raise ValueError("tol must be finite and positive")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def risk_contributions(
        weights: Union[np.ndarray, pd.Series],
        cov: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """Return the per-asset risk contributions ``w_i (Sigma w)_i``.

        The contributions sum to the portfolio variance ``w' Sigma w``.

        Parameters
        ----------
        weights:
            Weight vector of shape ``(n,)``.
        cov:
            Covariance matrix of shape ``(n, n)``.
        """
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        sigma = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        marginal = sigma @ w
        return w * marginal

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _clean(returns: pd.DataFrame) -> pd.DataFrame:
        return prepare_return_panel(returns, name="RiskParity returns")

    def _covariance(self, returns: pd.DataFrame) -> np.ndarray:
        X = self._clean(returns)
        n_assets = X.shape[1]

        if n_assets == 1 or X.shape[0] < 2:
            var = X.var(axis=0, ddof=0).to_numpy(dtype=np.float64)
            sigma = np.diag(np.where(np.isfinite(var) & (var > 0), var, 1.0))
        elif self.shrinkage:
            sigma = LedoitWolf().fit(X.to_numpy(dtype=np.float64)).covariance_
        else:
            sigma = np.cov(X.to_numpy(dtype=np.float64), rowvar=False, ddof=0)

        sigma = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
        sigma = 0.5 * (sigma + sigma.T)
        sigma += _RIDGE * np.eye(n_assets)
        return sigma

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def _solve(self, sigma: np.ndarray) -> np.ndarray:
        """Solve for the equal-risk-contribution weights.

        Uses the standard convex reformulation of the ERC problem

        .. math::

            \\min_{x>0}\\; \\tfrac12 x^{\\top}\\Sigma x
                - \\tfrac1n \\sum_i \\log x_i,

        whose unique positive minimizer has equal risk contributions; the final
        portfolio is recovered by renormalizing ``x`` to sum to one.  The
        log-barrier keeps the optimization variables positive. Configured
        portfolio bounds are applied afterward and can change risk contributions.
        """
        n = sigma.shape[0]
        inv_n = 1.0 / n

        def objective(x: np.ndarray) -> float:
            return 0.5 * float(x @ sigma @ x) - inv_n * float(np.sum(np.log(x)))

        def grad(x: np.ndarray) -> np.ndarray:
            return sigma @ x - inv_n / x

        # Initialize the optimization variables with inverse volatility.
        vol = np.sqrt(np.clip(np.diag(sigma), 1e-18, None))
        x0 = 1.0 / vol

        # Strictly-positive lower bound so log(x) and the gradient stay finite.
        bounds = [(1e-12, None)] * n
        res = minimize(
            objective,
            x0,
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"RiskParity solver failed: {res.message}")
        candidate = res.x

        total = candidate.sum()
        if total <= 0.0 or not np.isfinite(total):
            raise RuntimeError("RiskParity solver returned invalid weights")
        w = candidate / total

        return self._project_configured_bounds(w)

    # ------------------------------------------------------------------
    # PortfolioOptimizer API
    # ------------------------------------------------------------------
    def _compute_weights(self, returns: pd.DataFrame, **kwargs) -> np.ndarray:
        """Compute raw ERC weights satisfying the long-only contract.

        Columns with fewer than 2 finite observations (dead assets) carry no
        usable risk information; forward/backward/zero-filling them would
        create zero-variance pseudo-assets that absorb most of the book.  They
        are excluded from the optimization and reinserted with weight 0.0 at
        their original positions. If all columns are dead, optimization fails.
        """
        df = validate_return_panel(returns, name="RiskParity returns")
        n_total = df.shape[1]
        alive = (df.notna().sum(axis=0) >= 2).to_numpy()

        if not alive.any():
            raise ValueError(
                "RiskParity requires at least one asset with two observations"
            )
        if (~alive).any() and max(0.0, self.weight_bounds[0]) > self.tolerance:
            raise ValueError(
                "RiskParity cannot assign the configured positive minimum weight "
                "to assets without enough observations"
            )

        sub = df.loc[:, df.columns[alive]]
        w_sub = self._optimize_subset(sub)

        # Renormalize the optimized sub-vector to the long-only target sum and
        # re-insert weight 0.0 at the dead-column positions.
        s = float(w_sub.sum())
        if s > 0.0 and np.isfinite(s):
            w_sub = w_sub / s
        out = np.zeros(n_total, dtype=np.float64)
        out[alive] = w_sub
        return out

    def _optimize_subset(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute ERC weights for the live-asset subset."""
        sigma = self._covariance(returns)
        n = sigma.shape[0]

        if n == 1:
            return self._project_configured_bounds(
                np.array([1.0], dtype=np.float64)
            )

        return self._solve(sigma)
