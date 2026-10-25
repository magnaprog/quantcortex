"""Vectorized portfolio backtest engine.

:class:`VectorizedBacktest` runs a target-weight strategy over a daily price
panel using vectorized return calculations with chronological rebalance
processing. It does not track individual share positions or fill prices.

The execution alignment for close-derived signals is: weights decided at
the close of day *t* execute on the first bar strictly after *t* and begin
earning returns after that execution bar. This conservative convention avoids
assuming that the official close was both observed and traded at the same
price. The caller must separately ensure that targets and supplied data use
only information available at the relevant time.

Transaction costs are **mandatory**: a :class:`TransactionCostModel` must be
supplied, and the constructor raises :class:`ValueError` if it is ``None``.
Costs are charged on every rebalance via ``cost_model.apply_costs`` and
subtracted from that period's gross return.

This module also defines :class:`BacktestResult`, the common result container
reused by the event-driven engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.cash import align_cash_returns
from quantcortex.portfolio.base import enforce_exposure_contract

__all__ = ["BacktestResult", "VectorizedBacktest"]


@dataclass
class BacktestResult:
    """Container for the output of a backtest run.

    Attributes
    ----------
    returns:
        Net (after-cost) per-period portfolio returns.
    equity_curve:
        Cumulative NAV: ``(1 + returns).cumprod() * capital``.
    weights:
        Post-trade close weights (date x symbol), which apply to the following
        bar's return.
    gross_returns:
        Per-period portfolio returns *before* transaction costs.
    costs:
        Per-period return drag from transaction costs, measured against the
        prior bar's NAV so that ``returns == gross_returns - costs`` exactly.
    turnover:
        Per-period one-way turnover actually executed.
    traded_notional:
        Per-period gross two-sided traded notional actually executed. This is
        the sum of buy and sell notionals as a fraction of pre-trade NAV.
    asset_contribution:
        Per-period risky-asset contribution before transaction costs.
    cash_contribution:
        Per-period cash-account contribution before transaction costs.
    cash_weights:
        Post-trade cash weight, ``1 - sum(asset weights)``.
    metadata:
        Free-form dict of run parameters (capital, periods_per_year, ...).
    """

    returns: "pd.Series"
    equity_curve: "pd.Series"
    weights: "pd.DataFrame"
    gross_returns: "pd.Series"
    costs: "pd.Series"
    turnover: "pd.Series"
    traded_notional: Optional["pd.Series"] = None
    metadata: dict = field(default_factory=dict)
    asset_contribution: Optional["pd.Series"] = None
    cash_contribution: Optional["pd.Series"] = None
    cash_weights: Optional["pd.Series"] = None

    @property
    def total_return(self) -> float:
        """Total compounded net return over the whole backtest."""
        if len(self.returns) == 0:
            return 0.0
        values = self.returns.dropna().to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < -1.0):
            raise ValueError("backtest returns must be finite and no less than -100%")
        return float((1.0 + self.returns).prod() - 1.0)

    def summary(self) -> dict:
        """Headline performance statistics.

        Returns a dict with ``total_return``, ``cagr``, ``ann_vol``,
        ``sharpe`` and ``max_drawdown``. The Sharpe uses a zero risk-free rate;
        use ``Tearsheet`` with an explicit risk-free series for excess-return
        statistics. This summary does not infer that series from cash accrual.
        """
        r = self.returns.dropna()
        ppy = float(self.metadata.get("periods_per_year", 252))
        values = r.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < -1.0):
            raise ValueError("backtest returns must be finite and no less than -100%")
        if not np.isfinite(ppy) or ppy <= 0.0:
            raise ValueError("periods_per_year must be finite and positive")
        n = len(r)
        if n == 0:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "ann_vol": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        total_growth = float((1.0 + r).prod())
        total_return = total_growth - 1.0
        if (r == -1.0).any():
            cagr = -1.0
        else:
            # Annualize before exponentiating so product underflow is not a wipeout.
            cagr = float(np.expm1(np.log1p(values).mean() * ppy))

        std = float(r.std(ddof=1)) if n > 1 else 0.0
        ann_vol = std * np.sqrt(ppy)
        mean = float(r.mean())
        sharpe = (mean / std) * np.sqrt(ppy) if std > 0 else 0.0

        growth = (1.0 + r).cumprod()
        running_max = growth.cummax().clip(lower=1.0)
        drawdown = growth / running_max - 1.0
        max_dd = float(drawdown.min())

        return {
            "total_return": total_return,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": float(sharpe),
            "max_drawdown": max_dd,
        }


class VectorizedBacktest:
    """Vectorized target-weight backtest with mandatory cost modeling.

    Modeling assumption
    --------------------
    Between rebalances the portfolio is implicitly **reset to the last
    executed weight vector every bar at zero cost**: each period's gross return is
    ``sum(executed_weights * asset_returns)`` with the same weight vector for
    every bar of the segment, plus the residual-cash contribution. There is no
    drift with relative price moves and no cost for the implied daily
    trades that reset the weights.
    This differs from the event-driven engine
    (:class:`~quantcortex.backtest.engines.event_driven.EventDrivenBacktest`), which holds
    fixed share quantities between trades whose portfolio weights drift.

    The vector is ``CostResult.executed_weights``, measured against pre-trade
    NAV by the cost model and reused unchanged after deducting fees. It equals
    the requested target when no execution limit binds. The engine does not
    solve share quantities against post-cost NAV as the event-driven engine
    does; its reported weights describe this constant-weight convention.

    Parameters
    ----------
    cost_model:
        A :class:`TransactionCostModel`.  **Required:** ``None`` raises
        :class:`ValueError` (transaction costs are non-optional in quantcortex).
    capital:
        Starting NAV.  Default ``1e6``.
    periods_per_year:
        Annualization factor for summary statistics.  Default ``252``.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        *,
        capital: float = 1e6,
        periods_per_year: int = 252,
        max_gross: float = 1.0,
    ) -> None:
        if cost_model is None:
            raise ValueError(
                "a TransactionCostModel is required: transaction costs are "
                "mandatory in quantcortex backtests."
            )
        self.cost_model = cost_model
        if isinstance(capital, (bool, np.bool_)):
            raise TypeError("capital must be numeric, not boolean")
        if isinstance(max_gross, (bool, np.bool_)):
            raise TypeError("max_gross must be numeric, not boolean")
        self.capital = float(capital)
        if (
            isinstance(periods_per_year, (bool, np.bool_))
            or not isinstance(periods_per_year, (int, np.integer))
            or periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer")
        self.periods_per_year = int(periods_per_year)
        self.max_gross = float(max_gross)
        if not np.isfinite(self.capital) or self.capital <= 0.0:
            raise ValueError("capital must be finite and positive")
        if not np.isfinite(self.max_gross) or self.max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")

    def run(
        self,
        weights: "pd.DataFrame",
        prices: "pd.DataFrame",
        adv: Optional["pd.DataFrame"] = None,
        cash_returns: Optional["pd.Series"] = None,
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        weights:
            Complete target weights (fractions of NAV) indexed by decision date
            and columns by symbol. Need not be on the daily grid. An on-grid
            close decision executes on the next bar; an off-grid date executes
            on the next available bar. When several distinct decision dates
            snap to the same bar, the last stated target wins.
        prices:
            Daily close prices (date x symbol).
        adv:
            Optional average daily share-volume panel (date x symbol) for the ADV
            liquidity cap.  ``None`` disables the cap.  When the cap binds,
            the engine earns returns on the *executed* (capped) weights, and
            the cap notional is measured against the current running NAV.
        cash_returns:
            Optional per-period simple return of the cash account, indexed on
            every price bar. ``None`` preserves the historical zero-return cash
            assumption. Residual cash weight is ``1 - sum(asset weights)``.

        Returns
        -------
        BacktestResult
        """
        if prices.empty:
            raise ValueError("prices is empty.")

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must use a DatetimeIndex")
        if prices.index.hasnans:
            raise ValueError("prices index must contain valid timestamps")
        prices = prices.copy()
        if prices.index.tz is not None:
            prices.index = prices.index.tz_convert("UTC").tz_localize(None)
        prices = prices.sort_index()
        if prices.index.has_duplicates:
            raise ValueError("prices index must not contain duplicate bars")
        if prices.columns.has_duplicates or prices.shape[1] == 0:
            raise ValueError("prices must have unique, non-empty symbol columns")
        prices = prices.apply(pd.to_numeric, errors="coerce")
        if not np.all(np.isfinite(prices.to_numpy(dtype=float))):
            raise ValueError("prices must contain only finite values")
        if (prices <= 0.0).any().any():
            raise ValueError("prices must be strictly positive")
        symbols = list(prices.columns)
        n = len(prices.index)
        n_sym = len(symbols)

        # Snap each decision date to the first price bar strictly after it.
        # Dates without a later bar are dropped; duplicate snaps keep the last
        # target stated for that execution bar.
        if not isinstance(weights, pd.DataFrame):
            raise TypeError("weights must be a pandas DataFrame")
        if not isinstance(weights.index, pd.DatetimeIndex):
            if weights.empty:
                weights = weights.copy()
                weights.index = prices.index[:0]
            else:
                raise TypeError("weights must use a DatetimeIndex")
        if weights.index.hasnans:
            raise ValueError("weights index must contain valid timestamps")
        if weights.index.has_duplicates:
            raise ValueError("weights index must not contain duplicate decisions")
        if weights.columns.has_duplicates:
            raise ValueError("weights columns must be unique")
        unknown = [column for column in weights.columns if column not in symbols]
        if unknown:
            raise ValueError(f"weights contain unknown symbols: {unknown}")
        weights = weights.copy()
        if weights.index.tz is not None:
            weights.index = weights.index.tz_convert("UTC").tz_localize(None)
        supplied_symbols = list(weights.columns)
        w = weights.sort_index().reindex(columns=symbols)
        missing_symbols = [symbol for symbol in symbols if symbol not in supplied_symbols]
        if missing_symbols:
            w.loc[:, missing_symbols] = 0.0
        w = w.apply(pd.to_numeric, errors="coerce")
        if w[supplied_symbols].isna().any(axis=None):
            raise ValueError(
                "each target row must explicitly specify every supplied symbol"
            )
        if len(w.index) == 0:
            # Keep the all-cash run's empty decision index aligned with prices.
            w.index = prices.index[:0]
        pos = prices.index.searchsorted(w.index, side="right")
        keep = pos < n
        w = w.iloc[keep]
        w.index = prices.index[pos[keep]]
        w = w[~w.index.duplicated(keep="last")]

        reb_positions = prices.index.get_indexer(w.index)
        target_rows = w.to_numpy(dtype=np.float64, copy=True)
        if not np.all(np.isfinite(target_rows)):
            raise ValueError("weights must contain only finite values")
        for position, row in enumerate(target_rows):
            target_rows[position] = enforce_exposure_contract(
                row,
                max_gross=self.max_gross,
                name=f"VectorizedBacktest target row {position}",
            )

        # Close-to-close simple returns; row t is the asset move t-1 -> t.
        asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
        ret_vals = asset_returns.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(ret_vals)):
            raise ValueError("derived asset returns must contain only finite values")
        cash_return_series = align_cash_returns(cash_returns, prices.index)
        cash_ret_vals = cash_return_series.to_numpy(dtype=np.float64)
        price_vals = prices.to_numpy(dtype=np.float64)
        adv_aligned = None
        if adv is not None:
            if not isinstance(adv, pd.DataFrame):
                raise TypeError("adv must be a pandas DataFrame")
            if not isinstance(adv.index, pd.DatetimeIndex):
                raise TypeError("adv must use a DatetimeIndex")
            if adv.index.hasnans or adv.index.has_duplicates:
                raise ValueError("adv index must contain unique valid timestamps")
            if adv.columns.has_duplicates:
                raise ValueError("adv columns must be unique")
            adv = adv.copy()
            if adv.index.tz is not None:
                adv.index = adv.index.tz_convert("UTC").tz_localize(None)
            adv = adv.sort_index().apply(pd.to_numeric, errors="coerce")
            adv_aligned = adv.reindex(columns=symbols).reindex(prices.index).to_numpy(
                dtype=np.float64
            )

        gross_arr = np.zeros(n, dtype=np.float64)
        asset_contribution_arr = np.zeros(n, dtype=np.float64)
        cash_contribution_arr = np.zeros(n, dtype=np.float64)
        costs_arr = np.zeros(n, dtype=np.float64)
        turnover_arr = np.zeros(n, dtype=np.float64)
        traded_notional_arr = np.zeros(n, dtype=np.float64)
        eff = np.zeros((n, n_sym), dtype=np.float64)

        # Track executed weights after any caps and running NAV chronologically.
        # Weights executed at bar p's close earn returns from bar p+1 onward.
        current = np.zeros(n_sym, dtype=np.float64)
        nav = self.capital
        start = 0
        for k, p in enumerate(reb_positions):
            # Bars [start, p] earn returns on the previously executed weights.
            asset_seg = ret_vals[start : p + 1] @ current
            cash_seg = cash_ret_vals[start : p + 1] * (1.0 - float(current.sum()))
            seg = asset_seg + cash_seg
            if not np.all(np.isfinite(seg)):
                raise ValueError("portfolio gross returns must contain only finite values")
            if np.any(seg <= -1.0):
                raise ValueError("portfolio gross return reached or fell below -100%")
            gross_arr[start : p + 1] = seg
            asset_contribution_arr[start : p + 1] = asset_seg
            cash_contribution_arr[start : p + 1] = cash_seg
            if len(seg) > 1:
                nav *= float(np.prod(1.0 + seg[:-1]))
            # Mark-to-market NAV at the close of the rebalance bar (pre-cost):
            # this is the capital base for the ADV liquidity cap.
            nav_mtm = nav * (1.0 + float(seg[-1]))
            row_adv = None if adv_aligned is None else adv_aligned[p]
            result = self.cost_model.apply_costs(
                current,
                target_rows[k],
                prices=price_vals[p],
                adv=row_adv,
                capital=nav_mtm,
                max_gross=self.max_gross,
            )
            # ``total_cost`` is a fraction of the current pre-trade NAV. Store
            # costs as a fraction of prior-bar NAV so gross - cost equals the
            # reported period return exactly.
            period_cost = result.total_cost * (1.0 + float(seg[-1]))
            costs_arr[p] = period_cost
            turnover_arr[p] = result.turnover
            traded_notional_arr[p] = result.traded_notional
            eff[start:p] = current
            nav *= (1.0 + float(seg[-1])) * (1.0 - result.total_cost)
            if not np.isfinite(nav) or nav <= 0.0:
                raise ValueError("transaction costs exhausted portfolio NAV")
            current = np.asarray(result.executed_weights, dtype=np.float64)
            eff[p] = current
            start = p + 1
        if start < n:
            asset_seg = ret_vals[start:] @ current
            cash_seg = cash_ret_vals[start:] * (1.0 - float(current.sum()))
            seg = asset_seg + cash_seg
            if not np.all(np.isfinite(seg)):
                raise ValueError("portfolio gross returns must contain only finite values")
            if np.any(seg <= -1.0):
                raise ValueError("portfolio gross return reached or fell below -100%")
            gross_arr[start:] = seg
            asset_contribution_arr[start:] = asset_seg
            cash_contribution_arr[start:] = cash_seg
            eff[start:] = current

        gross = pd.Series(gross_arr, index=prices.index)
        costs = pd.Series(costs_arr, index=prices.index)
        turnover = pd.Series(turnover_arr, index=prices.index)
        traded_notional = pd.Series(traded_notional_arr, index=prices.index)
        effective_weights = pd.DataFrame(
            eff, index=prices.index, columns=symbols
        )
        asset_contribution = pd.Series(asset_contribution_arr, index=prices.index)
        cash_contribution = pd.Series(cash_contribution_arr, index=prices.index)
        cash_weights = 1.0 - effective_weights.sum(axis=1)

        net = gross - costs
        equity_curve = (1.0 + net).cumprod() * self.capital
        if not np.all(np.isfinite(equity_curve)) or (equity_curve <= 0.0).any():
            raise ValueError("portfolio NAV must remain finite and positive")

        metadata = {
            "capital": self.capital,
            "periods_per_year": self.periods_per_year,
            "engine": "vectorized",
            "n_periods": int(len(net)),
            "n_rebalances": int(len(reb_positions)),
            "execution_timing": "next_bar_close",
            "cash_return_source": cash_return_series.name,
        }

        return BacktestResult(
            returns=net,
            equity_curve=equity_curve,
            weights=effective_weights,
            gross_returns=gross,
            costs=costs,
            turnover=turnover,
            traded_notional=traded_notional,
            metadata=metadata,
            asset_contribution=asset_contribution,
            cash_contribution=cash_contribution,
            cash_weights=cash_weights,
        )
