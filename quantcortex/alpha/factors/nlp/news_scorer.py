"""News-headline sentiment scoring with date and timestamp filters.

:class:`NewsScorer` turns timestamped news headlines into a sentiment signal
for each date and symbol, suitable for use as an alpha factor. It delegates
text scoring to a
:class:`~quantcortex.alpha.factors.nlp.finbert_sentiment.FinBERTSentiment`
instance, then aggregates the headline scores into a daily panel.

Causality
---------
Default daily aggregation includes headlines through each UTC calendar date,
including any published after that day's market close. Such a score becomes
available only after the included headlines. Supplying ``as_of`` instead
filters exact publication timestamps at that cutoff. Optional exponential
time-decay weighting uses calendar-day ages and a configurable half-life.
Source publication times require separate verification.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from quantcortex.alpha.factors.nlp.finbert_sentiment import FinBERTSentiment


class NewsScorer:
    """Score news headlines and aggregate them by date or an explicit cutoff.

    Parameters
    ----------
    sentiment:
        A :class:`FinBERTSentiment` instance used to score text. If ``None``
        (default), a deterministic lexicon scorer is constructed. Pass an
        explicit ``backend="transformers"`` scorer to require FinBERT.
    half_life:
        Half-life, in days, of the exponential recency weighting used by the
        time-decayed aggregation helpers. With a half-life of ``h``, a headline
        ``h`` days older than the evaluation date receives half the weight of a
        same-day headline. Must be positive.
    """

    def __init__(
        self,
        sentiment: Optional[FinBERTSentiment] = None,
        half_life: float = 3.0,
    ) -> None:
        half_life = self._validate_half_life(half_life)
        self.sentiment = (
            sentiment
            if sentiment is not None
            else FinBERTSentiment(backend="lexicon")
        )
        self.half_life = half_life

    # ------------------------------------------------------------------
    # Headline scoring
    # ------------------------------------------------------------------
    def score_headlines(self, headlines: Sequence[str]) -> np.ndarray:
        """Score a list of headlines to sentiment in ``[-1, 1]``.

        Parameters
        ----------
        headlines:
            Iterable of headline strings.

        Returns
        -------
        numpy.ndarray
            One sentiment score per headline in ``[-1, 1]``.
        """
        if isinstance(headlines, str):
            headlines = [headlines]
        headlines = list(headlines)
        if not headlines:
            return np.empty(0, dtype=float)
        return self.sentiment.score(headlines)

    # ------------------------------------------------------------------
    # Recency weighting helper
    # ------------------------------------------------------------------
    def recency_weight(
        self,
        age_days: np.ndarray,
        half_life: Optional[float] = None,
    ) -> np.ndarray:
        """Exponential time-decay weight for a headline of a given age.

        The weight for a headline published ``age_days`` before the evaluation
        date is ``0.5 ** (age_days / half_life)``, so it is 1.0 for same-day news
        and halving every ``half_life`` days. Negative ages (future-dated
        headlines) receive zero weight. Publication-time accuracy remains the
        caller's responsibility.

        Parameters
        ----------
        age_days:
            Non-negative age (in days) of each headline relative to the
            evaluation date. May be a scalar or array-like.
        half_life:
            Override for the instance ``half_life``.

        Returns
        -------
        numpy.ndarray
            Weights in ``[0, 1]`` aligned to ``age_days``.
        """
        hl = (
            self.half_life
            if half_life is None
            else self._validate_half_life(half_life)
        )
        ages = np.asarray(age_days, dtype=float)
        if not np.all(np.isfinite(ages)):
            raise ValueError("age_days must contain only finite values")
        weights = np.power(0.5, ages / hl)
        # Future-dated (negative age) news must not leak: zero its weight.
        weights = np.where(ages < 0, 0.0, weights)
        return weights

    @staticmethod
    def _validate_half_life(value: object) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("half_life must be numeric, not boolean")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("half_life must be numeric") from exc
        if not np.isfinite(parsed) or parsed <= 0.0:
            raise ValueError("half_life must be finite and positive")
        return parsed

    # ------------------------------------------------------------------
    # Daily aggregation
    # ------------------------------------------------------------------
    def aggregate_daily(
        self,
        news_df: pd.DataFrame,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        headline_col: str = "headline",
        time_decay: bool = False,
        lookback_days: Optional[int] = None,
        as_of: Optional[object] = None,
    ) -> pd.DataFrame:
        """Aggregate headline sentiment into a ``date x symbol`` panel.

        Each cell is the mean or recency-weighted mean of headline sentiment.
        By default, only headlines on the same UTC calendar date contribute.
        With ``lookback_days``, prior dates in the inclusive lookback window
        also contribute. Without ``as_of``, the panel covers observed dates and
        symbols, and daily outputs include headlines published after market
        close on the output date.

        Supplying ``as_of`` applies the exact ``published_at <= as_of`` cutoff,
        even with ``time_decay=False``. The returned index still uses the UTC
        calendar date, not the exact cutoff timestamp.

        Parameters
        ----------
        news_df:
            Long-format news table with at least date, symbol and headline
            columns (names configurable below).
        date_col, symbol_col, headline_col:
            Column names in ``news_df``.
        time_decay:
            If ``True``, weight each headline by its exponential recency weight
            (see :meth:`recency_weight`) before averaging. Age is measured in
            UTC calendar days relative to the output date, or the UTC calendar
            date of ``as_of`` when supplied. Same-date headlines have weight
            1.0. If ``False``, use an unweighted mean.
        lookback_days:
            If set, each output date ``t`` includes headlines with UTC calendar
            dates in ``[t - lookback_days, t]``, inclusive. With ``as_of``, the
            exact publication-time cutoff also applies. If ``None`` (default),
            use same-date headlines without ``as_of`` and all news up to the
            cutoff with ``as_of``.
        as_of:
            Optional exact evaluation timestamp, converted to UTC. If provided,
            returns a single-row panel containing only headlines with
            ``published_at <= as_of``, regardless of ``time_decay``. A supplied
            ``lookback_days`` also bounds the window by UTC calendar date.

        Returns
        -------
        pandas.DataFrame
            Panel indexed by timezone-naive UTC calendar date with one column
            per symbol of mean or decay-weighted mean sentiment. Cells with no
            contributing news are ``NaN``.
        """
        self._validate_news(news_df, date_col, symbol_col, headline_col)
        if lookback_days is not None:
            if isinstance(lookback_days, bool):
                raise ValueError("lookback_days must be a non-negative integer")
            try:
                parsed_lookback = int(lookback_days)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "lookback_days must be a non-negative integer"
                ) from exc
            if parsed_lookback != lookback_days or parsed_lookback < 0:
                raise ValueError("lookback_days must be a non-negative integer")
            lookback_days = parsed_lookback

        df = news_df[[date_col, symbol_col, headline_col]].copy()
        df.columns = ["date", "symbol", "headline"]
        # Normalize to calendar dates: intraday timestamps would otherwise
        # produce one "daily" row per distinct timestamp instead of per day.
        published_at = pd.to_datetime(df["date"], utc=True)
        if published_at.isna().any():
            raise ValueError("news dates must be valid timestamps")
        df["published_at"] = published_at
        df["date"] = published_at.dt.tz_convert(None).dt.normalize()
        df = df.dropna(subset=["symbol", "headline"])
        if (
            ~df["symbol"].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
        ).any():
            raise ValueError("news symbols must be non-empty strings")
        if (
            ~df["headline"].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
        ).any():
            raise ValueError("news headlines must be non-empty strings")
        df["symbol"] = df["symbol"].str.strip()
        df["headline"] = df["headline"].str.strip()
        # Keep the first publication, so a later copy cannot hide available news.
        df = df.sort_values("published_at", kind="stable")
        df = df.drop_duplicates(subset=["date", "symbol", "headline"])
        if df.empty:
            return pd.DataFrame(dtype=float)

        # Score every unique headline once for efficiency, then map back.
        unique_headlines = pd.unique(df["headline"].astype(str))
        scores = self.sentiment.score(list(unique_headlines))
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(unique_headlines),) or not np.all(np.isfinite(scores)):
            raise ValueError("sentiment backend returned malformed scores")
        score_map = dict(zip(unique_headlines, scores, strict=True))
        df["sentiment"] = df["headline"].astype(str).map(score_map).astype(float)

        symbols = pd.Index(sorted(pd.unique(df["symbol"])))

        # --- Single as-of evaluation date ---------------------------------
        if as_of is not None:
            asof_exact = pd.to_datetime(as_of, utc=True)
            if pd.isna(asof_exact):
                raise ValueError("as_of must be a valid timestamp")
            asof_ts = pd.Timestamp(asof_exact).tz_convert(None).normalize()
            window = df[df["published_at"] <= asof_exact]
            if lookback_days is not None:
                lower = asof_ts - pd.Timedelta(days=int(lookback_days))
                window = window[window["date"] >= lower]
            row = self._aggregate_window(window, asof_ts, symbols, time_decay)
            return pd.DataFrame([row], index=pd.Index([asof_ts], name="date"))

        # --- Per-date panel ----------------------------------------------
        eval_dates = pd.Index(sorted(pd.unique(df["date"])), name="date")

        if lookback_days is None and not time_decay:
            # Fast path: plain same-day mean via pivot.
            panel = (
                df.groupby(["date", "symbol"])["sentiment"]
                .mean()
                .unstack("symbol")
            )
            return panel.reindex(index=eval_dates, columns=symbols)

        out = pd.DataFrame(np.nan, index=eval_dates, columns=symbols, dtype=float)
        for t in eval_dates:
            if lookback_days is None:
                window = df[df["date"] == t]
            else:
                lower = t - pd.Timedelta(days=int(lookback_days))
                window = df[(df["date"] <= t) & (df["date"] >= lower)]
            row = self._aggregate_window(window, t, symbols, time_decay)
            out.loc[t] = pd.Series(row)
        return out

    # ------------------------------------------------------------------
    # Internal aggregation of a causal window into one row
    # ------------------------------------------------------------------
    def _aggregate_window(
        self,
        window: pd.DataFrame,
        eval_date: pd.Timestamp,
        symbols: pd.Index,
        time_decay: bool,
    ) -> dict:
        """Aggregate a selected window of headlines into a row dict."""
        row = {sym: np.nan for sym in symbols}
        if window.empty:
            return row

        if time_decay:
            age = (eval_date - window["date"]).dt.days.to_numpy(dtype=float)
            weights = self.recency_weight(age)
        else:
            weights = np.ones(len(window), dtype=float)

        tmp = window[["symbol", "sentiment"]].copy()
        tmp["w"] = weights
        tmp["ws"] = tmp["w"] * tmp["sentiment"]

        grouped = tmp.groupby("symbol")
        wsum = grouped["w"].sum()
        wssum = grouped["ws"].sum()
        for sym in wsum.index:
            denom = wsum.loc[sym]
            row[sym] = float(wssum.loc[sym] / denom) if denom > 0 else np.nan
        return row

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_news(
        news_df: pd.DataFrame,
        date_col: str,
        symbol_col: str,
        headline_col: str,
    ) -> None:
        if not isinstance(news_df, pd.DataFrame):
            raise TypeError("news_df must be a pandas DataFrame")
        missing = [
            c for c in (date_col, symbol_col, headline_col) if c not in news_df.columns
        ]
        if missing:
            raise ValueError(f"news_df is missing required columns: {missing}")
