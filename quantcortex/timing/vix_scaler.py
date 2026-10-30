"""Inverse-volatility (VIX) exposure-scaling timing overlay.

:class:`VIXScaler` scales gross portfolio exposure inversely to the level of
implied volatility (the VIX index, or a volatility proxy in the same units).
Higher values reduce the raw exposure multiplier. This rule does not estimate
the portfolio's realized volatility or guarantee a realized risk target.

Scaling rule
------------
``scale = clip(target_vix / current_vix, floor, cap)``

The raw ratio is below one when ``current_vix > target_vix`` and above one
when ``current_vix < target_vix``. The configured floor and cap then bound it.

The overlay consumes the last supplied VIX value. The caller remains
responsible for publication-time alignment: a same-close VIX observation must
not be used for an order assumed to execute at that same close.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["VIXScaler"]


class VIXScaler:
    """Scale exposure by ``target_vix / current_vix`` within ``[floor, cap]``.

    Parameters
    ----------
    target_vix:
        The VIX level at which the strategy runs at its natural exposure (the
        point where the raw ratio equals 1.0). Defaults to 20 volatility points.
    floor:
        Minimum multiplier returned by :meth:`compute_scale` (default 0.3).
        A positive floor retains some of the input exposure even at high VIX
        levels. Non-positive or non-finite VIX observations also use this floor.
    cap:
        Maximum multiplier (default 1.0). Values above one permit increased
        exposure, subject to the per-asset bounds in :meth:`apply` and any
        downstream gross-exposure limit.
    """

    def __init__(
        self,
        target_vix: float = 20.0,
        floor: float = 0.3,
        cap: float = 1.0,
    ) -> None:
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (target_vix, floor, cap)
        ):
            raise TypeError("VIX controls must be numeric, not boolean")
        if not np.isfinite(target_vix) or target_vix <= 0:
            raise ValueError("target_vix must be finite and positive")
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and non-negative")
        if not np.isfinite(cap) or cap < floor:
            raise ValueError("cap must be finite and >= floor")
        self.target_vix = float(target_vix)
        self.floor = float(floor)
        self.cap = float(cap)

    # ------------------------------------------------------------------ #
    def compute_scale(self, vix: Union[float, pd.Series, np.ndarray]) -> float:
        """Return the clipped exposure multiplier for the latest VIX value."""
        current = self._latest_vix(vix)
        if current <= 0 or not np.isfinite(current):
            # Use the floor for a non-positive or non-finite latest value.
            return self.floor
        raw = self.target_vix / current
        return float(np.clip(raw, self.floor, self.cap))

    def apply(
        self,
        weights: np.ndarray,
        vix: Any = None,
    ) -> np.ndarray:
        """Scale ``weights`` by the inverse-VIX multiplier.

        ``vix`` may be a float, a :class:`pandas.Series`/array (the last value is
        used), or a ``StrategyContext``-like object exposing ``.extra['vix']``.
        The result is validated via :func:`enforce_exposure_contract`.

        When levering up (``cap > 1`` and ``scale > 1``) the scalar scale is
        first capped so no element of the scaled book exceeds the ``[-1, 1]``
        per-asset contract: ``effective_scale = min(scale, 1 / max|w_i|)``.
        The capped scale is applied *unclipped*, preserving the allocation
        proportions. Therefore, concentrated books can receive less leverage
        than the raw ratio requests.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
            raise ValueError("weights must be a non-empty finite 1-D vector")
        scale = self.compute_scale(self._coerce_vix(vix))

        # Cap the scalar so no element leaves the [-1, 1] per-asset contract;
        # the capped scale is applied unclipped to preserve proportions.
        max_abs = float(np.max(np.abs(w))) if w.size else 0.0
        if max_abs > 0.0:
            scale = min(scale, 1.0 / max_abs)

        scaled = w * scale

        input_gross = float(np.abs(w).sum())
        max_gross = max(1.0, input_gross) + 1e-9
        # If cap > 1 the overlay can grow gross; widen the allowance accordingly.
        if self.cap > 1.0:
            max_gross = max(max_gross, input_gross * self.cap + 1e-9)
        return enforce_exposure_contract(
            scaled, max_gross=max_gross, name=type(self).__name__
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _latest_vix(vix: Union[float, pd.Series, np.ndarray]) -> float:
        """Extract the most recent scalar VIX value from a variety of inputs."""
        if isinstance(vix, pd.Series):
            if vix.empty:
                raise ValueError("vix series is empty")
            return float(vix.iloc[-1])
        arr = np.asarray(vix, dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError("vix is empty")
        return float(arr[-1])

    def _coerce_vix(self, vix: Any) -> Union[float, pd.Series, np.ndarray]:
        """Resolve a context-like object to a VIX value if needed."""
        if isinstance(vix, (float, int, np.floating, np.integer, pd.Series, np.ndarray, list, tuple)):
            return vix  # already a usable value/sequence
        extra = getattr(vix, "extra", None)
        if isinstance(extra, dict) and "vix" in extra:
            return extra["vix"]
        raise TypeError(
            "VIXScaler.apply requires a float/Series vix or a context object "
            "exposing .extra['vix']"
        )
