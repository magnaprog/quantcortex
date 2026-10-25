"""Barra-style factor-exposure limiting overlay.

In a Barra (multi-factor) risk model each asset carries a vector of factor
*loadings* (its sensitivity to style/industry/macro factors such as Value,
Momentum, Size, Beta...).  A portfolio's exposure to a factor is the
loading-weighted sum of its positions, ``loadings' * w``. This overlay limits
selected factor exposures to ``[-max_exposure, +max_exposure]``.

This overlay measures portfolio factor exposures and, where any exceeds the
cap, iteratively adjusts weights using least-squares solutions for the
currently offending factors. These steps can redistribute exposure across
assets; the complete procedure is not a global nearest-feasible projection.
A separate backstop prevents the final gross exposure from exceeding the input,
and the result is validated against the exposure contract.

Causality is conditional on the caller supplying point-in-time loadings. The
overlay itself does not access future data.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["FactorExposureLimiter"]


class FactorExposureLimiter:
    """Cap portfolio factor exposures at ``+/-max_exposure`` (Barra-style).

    Parameters
    ----------
    max_exposure:
        Maximum absolute portfolio exposure permitted to any single factor.
    factors:
        Optional list of factor names whose exposures are capped. If ``None``
        (the default), cap exposures for every column of the loadings frame.
    preserve_signs:
        Keep every asset on its original side of zero and leave initially flat
        assets flat. This prevents a risk overlay from creating new long or
        short positions. Disable only when sign-changing hedges are intended.
    """

    def __init__(
        self,
        max_exposure: float = 0.2,
        *,
        factors: Optional[Sequence[str]] = None,
        preserve_signs: bool = True,
    ) -> None:
        if isinstance(max_exposure, (bool, np.bool_)):
            raise TypeError("max_exposure must be numeric, not boolean")
        if not np.isfinite(max_exposure) or max_exposure < 0:
            raise ValueError("max_exposure must be non-negative.")
        self.max_exposure = float(max_exposure)
        if not isinstance(preserve_signs, bool):
            raise TypeError("preserve_signs must be a boolean")
        self.preserve_signs = preserve_signs
        self.factors = list(factors) if factors is not None else None
        if self.factors is not None and (
            len(set(self.factors)) != len(self.factors)
            or any(not isinstance(f, str) or not f for f in self.factors)
        ):
            raise ValueError("factors must contain unique, non-empty names")
        self.last_exposures: Optional[pd.Series] = None

    # ------------------------------------------------------------------ #
    # Exposure measurement                                               #
    # ------------------------------------------------------------------ #
    def compute_exposures(
        self,
        weights: np.ndarray,
        loadings: pd.DataFrame,
    ) -> pd.Series:
        """Portfolio factor exposures ``loadings' * w``.

        Parameters
        ----------
        weights:
            1-D weight vector aligned to the rows (assets) of ``loadings``.
        loadings:
            ``(n_assets, n_factors)`` DataFrame indexed by asset, columns are
            factors.

        Returns
        -------
        pandas.Series
            Exposure per factor, indexed by factor name.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")
        if loadings.shape[0] != w.size:
            raise ValueError(
                f"weights length {w.size} does not match loadings rows "
                f"{loadings.shape[0]}."
            )
        cols = self._selected_columns(loadings)
        L = loadings[cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(L)):
            raise ValueError("loadings must contain only finite values")
        exposures = L.T @ w
        return pd.Series(exposures, index=cols, name="factor_exposure")

    def _selected_columns(self, loadings: pd.DataFrame) -> list:
        """Resolve the factor columns whose exposures are capped for this call."""
        if not isinstance(loadings, pd.DataFrame):
            raise TypeError("loadings must be a pandas DataFrame")
        if loadings.shape[1] == 0:
            raise ValueError("loadings must contain at least one factor column")
        if not loadings.columns.is_unique:
            raise ValueError("loadings factor columns must be unique")
        if any(not isinstance(column, str) or not column for column in loadings.columns):
            raise ValueError("loadings factor columns must be non-empty strings")
        if not loadings.index.is_unique:
            raise ValueError("loadings asset index must be unique")
        if self.factors is None:
            return list(loadings.columns)
        missing = [f for f in self.factors if f not in loadings.columns]
        if missing:
            raise ValueError(f"loadings missing requested factors: {missing}")
        return list(self.factors)

    # ------------------------------------------------------------------ #
    # Capping                                                            #
    # ------------------------------------------------------------------ #
    def apply(
        self,
        weights: np.ndarray,
        loadings: pd.DataFrame,
    ) -> np.ndarray:
        """Adjust ``weights`` so no selected factor exposure exceeds the cap.

        Method
        ------
        Let ``L`` be the ``(n_assets, k)`` loadings of the *offending* factors
        (those whose current exposure violates ``+/-max_exposure``) and ``e`` the
        current exposures of those factors.  We seek the minimal-norm adjustment
        ``delta`` to the weights that drives each offending exposure exactly to
        its signed cap ``t`` (``+max_exposure`` if the exposure was too high,
        ``-max_exposure`` if too low):

            minimize   ||delta||^2
            subject to L' (w + delta) = t          (i.e. L' delta = t - e)

        When the selected columns are linearly independent, the minimum-norm
        solution can be written::

            delta = L (L'L)^-1 (t - e)

        The implementation solves ``L.T * delta = t - e`` with ``lstsq``.
        With dependent columns, this can only approximate incompatible signed
        targets; the final exposure check and uniform shrink enforce the cap.

        Because the projection only constrains the *currently* offending
        factors, it can push other selected factors beyond the cap. We
        therefore **iterate**: after each projection the exposures are
        recomputed and all currently offending factors are projected again
        together, up to ``max_iter`` passes.  If the cap is still violated
        after the loop (e.g. mutually antagonistic loadings), we fall back to
        uniformly scaling the whole weight vector down until the worst
        ``|exposure|`` equals the cap - exposures are linear in ``w`` so this
        shrink is guaranteed feasible and direction-preserving.  Note the
        trade-off: the uniform shrink reduces the weight sum (the shortfall is
        implicitly held as cash) rather than preserving full investment.

        After iteration, sign preservation (when enabled), the input-gross
        backstop, and the per-asset box are applied before the final uniform
        shrink. These operations can change the least-squares solution. The
        result is validated via :func:`enforce_exposure_contract` with a gross
        cap sized to admit an unchanged input book.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")
        if np.any(np.abs(w) > 1.0 + 1e-9):
            raise ValueError("weights must already satisfy the per-asset [-1, 1] box")
        if loadings.shape[0] != w.size:
            raise ValueError("loadings rows must match weights length")
        cols = self._selected_columns(loadings)
        L_all = loadings[cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(L_all)):
            raise ValueError("loadings must contain only finite values")

        exposures = L_all.T @ w

        cap = self.max_exposure
        tol = 1e-12
        in_gross = float(np.abs(w).sum())
        max_gross = max(1.0, in_gross) + 1e-9

        if not np.any(np.abs(exposures) > cap + tol):
            # Exposures already meet the cap; validate the unchanged weights.
            self.last_exposures = pd.Series(
                exposures, index=cols, name="factor_exposure"
            )
            return enforce_exposure_contract(
                w, max_gross=max_gross, name="FactorExposureLimiter"
            )

        adjusted = w.copy()
        max_iter = 50
        for _ in range(max_iter):
            exposures = L_all.T @ adjusted
            offending = np.abs(exposures) > cap + tol
            if not np.any(offending):
                break

            L = L_all[:, offending]          # (n_assets, k)
            e = exposures[offending]         # (k,)
            # Signed target: pull each offending exposure to the nearer cap edge.
            t = np.sign(e) * cap             # (k,)

            rhs = t - e                      # (k,)
            # Solve the stated minimum-norm problem directly. This is more
            # stable than forming the normal-equation Gram matrix.
            delta, *_ = np.linalg.lstsq(L.T, rhs, rcond=None)
            adjusted = adjusted + delta

        if self.preserve_signs:
            lower = np.where(w < 0.0, -1.0, 0.0)
            upper = np.where(w > 0.0, 1.0, 0.0)
            adjusted = np.clip(adjusted, lower, upper)

        # Adjusting a factor tilt can raise gross exposure above the input.
        # Restore that limit by scaling every weight equally, which also
        # reduces each factor exposure in magnitude.
        adj_gross = float(np.abs(adjusted).sum())
        if adj_gross > in_gross > 0.0:
            adjusted = adjusted * (in_gross / adj_gross)

        # Clip weights to [-1, 1] before the final cap check: clipping changes
        # relative weights and can push a factor exposure back above its cap.
        adjusted = np.clip(adjusted, -1.0, 1.0)

        # Factor exposures are linear in weights. Uniform scaling brings the
        # largest absolute exposure to the cap while preserving allocation
        # direction and the [-1, 1] bounds. This also enforces the cap if the
        # iteration above reached max_iter without converging.
        exposures = L_all.T @ adjusted
        worst = float(np.max(np.abs(exposures))) if exposures.size else 0.0
        if worst > cap + tol:
            adjusted = adjusted * (cap / worst)

        final_exposures = L_all.T @ adjusted
        self.last_exposures = pd.Series(
            final_exposures, index=cols, name="factor_exposure"
        )

        return enforce_exposure_contract(
            adjusted, max_gross=max_gross, name="FactorExposureLimiter"
        )
