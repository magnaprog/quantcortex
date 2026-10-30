"""Walk-forward cross-validation splitter for time-series strategy research.

:class:`WalkForwardOptimizer` produces successive ``(train, test)`` index folds
with each test block following its training block in positional order. Callers
must supply chronologically ordered samples.
It supports two window schemes and positional buffers motivated by the leakage
controls discussed in Lopez de Prado, *Advances in Financial Machine Learning*
(2018):

* **expanding**. The training window stays anchored at index 0 and grows with
  each fold, using all history available so far.
* **rolling**. The training window is a fixed-length block of ``train_size``
  bars that slides forward each fold, discarding older data to adapt to regime
  changes.

Two positional buffers separate train from test:

* **embargo**. ``embargo_gap`` bars separate the raw training block from the
  test block. This buffer does not guarantee independence across the boundary.
* **purge**. The splitter removes the last ``label_horizon`` training samples
  as a buffer for forward labels. The caller must choose a horizon that covers
  the labels' actual information span; the splitter does not inspect labels.

There are ``label_horizon + embargo_gap`` excluded observations between the
last retained training index and the first test index.
"""

from __future__ import annotations

from typing import Iterator, Tuple, Union

import numpy as np
import pandas as pd

__all__ = ["WalkForwardOptimizer"]

IndexLike = Union[int, "pd.Index", np.ndarray, list]


class WalkForwardOptimizer:
    """Generate walk-forward splits with positional purge and gap buffers.

    Parameters
    ----------
    train_size:
        Number of bars in the training window.  In ``rolling`` mode this is the
        fixed window length; in ``expanding`` mode it is the size of the first
        window (the window then grows).
    test_size:
        Number of bars in each test window.
    embargo_gap:
        Number of bars between the raw training end and test start, in addition
        to the samples removed by ``label_horizon``. Default ``21``. This is a
        pre-test gap, not a general post-test embargo for arbitrary CV folds.
    mode:
        ``"expanding"`` (anchored, growing train) or ``"rolling"`` (fixed-size
        sliding train).  Default ``"expanding"``.
    label_horizon:
        Forward-return label horizon in bars.  The last ``label_horizon``
        training samples are purged so their label windows do not overlap the
        test region.  Default ``1``.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        *,
        embargo_gap: int = 21,
        mode: str = "expanding",
        label_horizon: int = 1,
    ) -> None:
        values = {
            "train_size": train_size,
            "test_size": test_size,
            "embargo_gap": embargo_gap,
            "label_horizon": label_horizon,
        }
        if any(
            isinstance(value, bool) or int(value) != value
            for value in values.values()
        ):
            raise ValueError("window and gap parameters must be integers")
        if train_size < 1:
            raise ValueError("train_size must be >= 1.")
        if test_size < 1:
            raise ValueError("test_size must be >= 1.")
        if embargo_gap < 0:
            raise ValueError("embargo_gap must be >= 0.")
        if label_horizon < 0:
            raise ValueError("label_horizon must be >= 0.")
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'.")
        self.train_size = int(train_size)
        self.test_size = int(test_size)
        self.embargo_gap = int(embargo_gap)
        self.mode = mode
        self.label_horizon = int(label_horizon)

    @staticmethod
    def _resolve_n(index_or_n: IndexLike) -> int:
        """Return the number of positional samples for an int or index/array."""
        if isinstance(index_or_n, (int, np.integer)):
            n = int(index_or_n)
        else:
            n = len(index_or_n)
        if n < 0:
            raise ValueError("sample count must be non-negative")
        return n

    def n_splits(self, index_or_n: IndexLike) -> int:
        """Number of folds produced for ``n`` samples (or a given index).

        Counts the folds yielded by :meth:`split`, which omits a fold when
        purging empties its training window.
        """
        return sum(1 for _ in self.split(index_or_n))

    def split(
        self, index_or_n: IndexLike
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` positional integer arrays per fold.

        The test window walks forward by ``test_size`` bars each fold until the
        data is exhausted.  Training indices are purged of their last
        ``label_horizon`` samples and an ``embargo_gap`` is left before the test
        window begins.

        Parameters
        ----------
        index_or_n:
            Either the integer sample count ``n`` or a pandas ``Index`` / array
            / list whose *positional* indices are used.

        Yields
        ------
        (train_idx, test_idx):
            Tuples of 1-D ``numpy`` integer arrays.
        """
        n = self._resolve_n(index_or_n)
        gap = self.embargo_gap

        train_end = self.train_size  # exclusive end of the raw training block
        while True:
            test_start = train_end + gap
            test_end = test_start + self.test_size
            if test_end > n:
                break

            if self.mode == "expanding":
                train_start = 0
            else:  # rolling
                train_start = train_end - self.train_size

            # Remove the configured positional buffer for forward labels.
            purged_train_end = train_end - self.label_horizon
            if purged_train_end <= train_start:
                # No training samples survive purging for this fold; advance.
                train_end += self.test_size
                continue

            train_idx = np.arange(train_start, purged_train_end, dtype=np.int64)
            test_idx = np.arange(test_start, test_end, dtype=np.int64)
            yield train_idx, test_idx

            train_end += self.test_size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"WalkForwardOptimizer(train_size={self.train_size}, "
            f"test_size={self.test_size}, embargo_gap={self.embargo_gap}, "
            f"mode='{self.mode}', label_horizon={self.label_horizon})"
        )
