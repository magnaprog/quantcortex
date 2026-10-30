"""Equal-weight (1/N) portfolio optimizer.

The 1/N benchmark (DeMiguel, Garlappi & Uppal, 2009) allocates an identical
fraction of capital to every asset in the investable set. It needs an asset
count, with no return or covariance estimate.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from quantcortex.portfolio.base import PortfolioMode, PortfolioOptimizer

__all__ = ["EqualWeight"]


def _infer_n_assets(returns) -> int:
    """Infer the asset count from an integer or an input container.

    Each DataFrame column counts as an asset; a Series is treated as one
    asset's return history (``n = 1``). A 1-D ndarray could represent one
    asset's history or one observation per asset. For compatibility, it still
    counts each element as an asset (``shape[0]``). Pass a DataFrame, a Series,
    or an explicit ``n_assets`` to avoid that ambiguity.
    """
    if returns is None:
        raise ValueError("EqualWeight needs either `returns` or `n_assets`.")
    if isinstance(returns, (bool, np.bool_)):
        raise TypeError("asset count must be an integer, not boolean")
    if isinstance(returns, (int, np.integer)):
        return int(returns)
    if isinstance(returns, pd.Series):
        # A Series is a single asset's return history, not one asset per row.
        return 1
    shape = getattr(returns, "shape", None)
    if shape is not None:
        # DataFrame / 2-D ndarray -> columns are assets; 1-D ndarray -> elements
        # (ambiguous; see docstring).
        return int(shape[1]) if len(shape) > 1 else int(shape[0])
    return len(returns)


class EqualWeight(PortfolioOptimizer):
    """Allocate ``1/N`` to each asset (long-only by construction)."""

    def __init__(self, **kwargs) -> None:
        # This allocator supports only the long-only 1/N construction.
        kwargs.setdefault("mode", PortfolioMode.LONG_ONLY)
        if PortfolioMode.coerce(kwargs["mode"]) is not PortfolioMode.LONG_ONLY:
            raise ValueError("EqualWeight only supports long_only mode.")
        super().__init__(**kwargs)

    def _compute_weights(
        self,
        returns=None,
        *,
        n_assets: Optional[int] = None,
        **_: Union[int, float],
    ) -> np.ndarray:
        if n_assets is not None and (
            isinstance(n_assets, (bool, np.bool_))
            or not isinstance(n_assets, (int, np.integer))
        ):
            raise TypeError("n_assets must be an integer, not boolean or fractional")
        n = int(n_assets) if n_assets is not None else _infer_n_assets(returns)
        if n <= 0:
            raise ValueError(f"n_assets must be positive, got {n}")
        return np.full(n, 1.0 / n, dtype=np.float64)
