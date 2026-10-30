"""Execution models - the abstract base and the optimistic ``IdealFill``.

An *execution model* maps a desired trade (``target_qty`` shares of a symbol on
a given bar) to an assumed **fill price**. Slippage, participation, and
market-impact assumptions belong here; the price is a simulation input, not
a broker quote or evidence of an achievable execution.

This module defines:

* :class:`ExecutionModel`. The abstract base every model implements.
* :class:`IdealFill`. Fills at the bar close without slippage or market
  impact. Use it as an optimistic reference for sanity checks and for
  comparing assumed execution costs for the same strategy, not as a universal
  upper bound on achievable performance.
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

__all__ = ["ExecutionModel", "IdealFill"]


class ExecutionModel(abc.ABC):
    """Abstract base class for execution / fill models.

    Subclasses translate a signed order quantity into the price at which that
    order is assumed to fill on a particular bar.  Implementations must be
    **causal**: they may use ``bar`` and supplied context available by the
    modeled execution time, never future bars. A completed-bar statistic may
    serve as a fill assumption only under a compatible execution convention.
    """

    @abc.abstractmethod
    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return the per-share fill price for an order.

        Parameters
        ----------
        symbol:
            The asset being traded.  Supplied so a model may key off
            per-symbol calibration (volatility, ADV, ...).
        target_qty:
            Signed order size in shares.  Positive is a buy, negative a sell.
            The sign determines the direction in which slippage / impact pushes
            the fill price (buys fill higher, sells fill lower).
        bar:
            The OHLCV (and optionally ``vwap``/``volume``) row for the bar on
            which the trade executes, subject to the timing convention above.
        **kw:
            Optional extra context (e.g. ``adv``, ``sigma``) used by richer
            models.  Plain models ignore it.

        Returns
        -------
        float
            The fill price per share.
        """
        raise NotImplementedError


class IdealFill(ExecutionModel):
    """Frictionless benchmark: every order fills at the bar close.

    ``IdealFill`` applies **zero slippage and zero market impact**. The fill
    price equals ``bar['close']`` regardless of order size or direction.  This
    is an optimistic reference, not a universal upper bound on performance.
    Comparisons with VWAP participation or Almgren-Chriss models measure
    differences in assumed execution costs for the same strategy.
    """

    def fill(
        self,
        symbol: str,
        target_qty: float,
        bar: "pd.Series",
        **kw,
    ) -> float:
        """Return ``bar['close']`` unchanged (no slippage)."""
        quantity = float(target_qty)
        close = float(bar["close"])
        if not np.isfinite(quantity):
            raise ValueError("target_qty must be finite")
        if not np.isfinite(close) or close <= 0.0:
            raise ValueError("bar close must be finite and positive")
        return close

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "IdealFill()"
