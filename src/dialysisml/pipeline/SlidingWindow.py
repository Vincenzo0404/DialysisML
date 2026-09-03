from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from dialysisml.features import META_COLUMNS

# What makes a series: windows are cut inside one, never across two.
GROUP_COLUMNS = ["patient", "t_event"]
TIME_COLUMN = "t_session"


class SlidingWindow:
    """Sliding windows over a DataFrame, materialised only when asked.

    Holds a reference to the frame plus one integer per window — its first row —
    so instances can share the same data without copying it. Iterating yields
    one window at a time as a view; `materialize` is where the copy happens.

    A window never crosses a series boundary.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        size: int = 30,
        stride: int = 1,
        target_columns: Sequence[str] = ("tte",),
        meta_columns: Sequence[str] = META_COLUMNS,
    ):
        if size < 2:
            raise ValueError(f"size must be at least 2, got {size}")
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")

        self.size = size
        self.stride = stride
        self.target_columns = list(target_columns)
        self.meta_columns = list(meta_columns)

        # a series has to occupy consecutive rows: windows are cut as slices,
        # so a series split in two would silently mix in other patients' data
        self.df = df.sort_values([*GROUP_COLUMNS, TIME_COLUMN]).reset_index(drop=True)
        self.starts = self._find_starts()

    @property
    def span(self) -> int:
        """Rows a window covers, which exceeds `size` when stride > 1."""
        return (self.size - 1) * self.stride + 1

    @property
    def feature_columns(self) -> list[str]:
        """Everything that is neither metadata nor a target.

        Read off the frame rather than declared, so a feature added upstream
        needs no change here.
        """
        excluded = {*self.meta_columns, *self.target_columns}
        return [c for c in self.df.columns if c not in excluded]

    def _find_starts(self) -> np.ndarray:
        """First row of every window, series by series.

        The last `span - 1` rows of a series cannot open a window, so the
        starts are not contiguous: this is the only place that knows it.
        """
        starts: list[int] = []
        for _, group in self.df.groupby(GROUP_COLUMNS, sort=False):
            rows = group.index.to_numpy()
            if len(rows) < self.span:
                continue
            first = int(rows[0])
            starts.extend(range(first, first + len(rows) - self.span + 1))
        return np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yields (window, targets) one at a time.

        The window is `(size, n_features)` and is a view: slicing an ndarray,
        even with a step, does not copy. Targets are `(n_targets,)`.
        """
        # converted once: doing it per window is the expensive mistake here
        values = self.df[self.feature_columns].to_numpy(dtype=np.float32)
        targets = self.df[self.target_columns].to_numpy()

        for start in self.starts:
            end = start + self.span
            # a window is labelled at its last row, the session it predicts from
            yield values[start : end : self.stride], targets[end - 1]

    def materialize(self) -> tuple[np.ndarray, np.ndarray]:
        """The whole dataset at once."""
        if len(self) == 0:
            raise ValueError("no series long enough to produce a window")

        windows, targets = zip(*self)
        return np.stack(windows), np.stack(targets)

    def meta(self) -> pd.DataFrame:
        """One row per window: whose it is, and when it ends."""
        anchors = self.starts + (self.span - 1)
        return self.df.loc[anchors, self.meta_columns].reset_index(drop=True)
