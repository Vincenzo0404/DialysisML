"""Reducing every window to one row of features.

A window varies in length, so nothing here materialises it. The least-squares
line over a window depends on five sums, and a window is a contiguous range of
rows: each sum is then the difference of two cumulative sums, computed for
every window at once.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandera.pandas as pa

from dialysisml.pipeline.SlidingWindow import SlidingWindow
from dialysisml.schema import Kind, Role, columns, select


def linear_fit(
    values: np.ndarray, t: np.ndarray, start: np.ndarray, anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope, fitted value at the anchor, and rmse of every window and column.

    `values` is `(rows, columns)` and `t` the time axis in days, both over the
    whole frame; `start` and `anchor` bound the windows, both inclusive.
    """
    n = (anchor - start + 1).astype(np.float64)[:, None]
    t = t[:, None]

    def window_sums(a: np.ndarray) -> np.ndarray:
        """`a` summed over each window, from the cumulative sum of `a`.

        The leading row of zeros is what lets a window starting at row 0 take
        the difference without a special case.
        """
        cumulative = np.zeros((len(a) + 1, a.shape[1]))
        np.cumsum(a, axis=0, out=cumulative[1:])
        return cumulative[anchor + 1] - cumulative[start]

    # one at a time: each cumulative sum is as large as the frame, and only
    # its window totals are needed afterwards
    sum_t = window_sums(t)
    sum_y = window_sums(values)
    sum_tt = window_sums(t * t)
    sum_ty = window_sums(t * values)
    sum_yy = window_sums(values * values)

    # the denominator is n squared times the variance of t, so it vanishes only
    # if every session of a window fell on one day -- the view keeps a single
    # row per patient per day, so it cannot
    slope = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t)
    at_zero = (sum_y - slope * sum_t) / n
    intercept = at_zero + slope * t[anchor]

    residual = (
        sum_yy
        - 2 * at_zero * sum_y
        - 2 * slope * sum_ty
        + at_zero * at_zero * n
        + 2 * at_zero * slope * sum_t
        + slope * slope * sum_tt
    )
    # an exact fit lands on a small negative number rather than on zero, and
    # the root of that would be a NaN quietly entering the features
    return slope, intercept, np.sqrt(np.maximum(residual, 0.0) / n)


class WindowTransformer(ABC):
    """Turns the windows of a `SlidingWindow` into `(windows, features)`."""

    @abstractmethod
    def transform(
        self, window: SlidingWindow
    ) -> tuple[np.ndarray, pa.DataFrameSchema]: ...


class RegressionFeatures(WindowTransformer):
    """A line fitted over each window, plus the value it ends on.

    The fit only says something about a quantity that moves continuously, so
    it runs on the numeric features. Every feature also contributes its last
    value: what a measure reads *now* is not what its trend says, and for a
    one-hot column it is the only thing there is to take.
    """

    def transform(
        self, window: SlidingWindow
    ) -> tuple[np.ndarray, pa.DataFrameSchema]:
        numeric = select(window.schema, role=Role.FEATURE, kind=Kind.NUMERIC)
        features = select(window.schema, role=Role.FEATURE)

        slope, intercept, rmse = linear_fit(
            window.column_values(numeric), window.t, window.start, window.anchor
        )
        last = window.at_anchor(features)

        # this order has to match how the schema below is built, or a feature
        # importance would be read against the wrong name
        values = np.hstack([slope, intercept, rmse, last]).astype(np.float32)
        schema = pa.DataFrameSchema(
            {
                **columns(
                    float,
                    Role.FEATURE,
                    Kind.NUMERIC,
                    [f"{c}_{stat}" for stat in ("slope", "intercept", "rmse") for c in numeric],
                ),
                # the last value keeps the kind of what it came from: a one-hot
                # column stays binary, so a scaler still leaves it alone
                **{f"{c}_last": window.schema.columns[c] for c in features},
            }
        )
        return values, schema
