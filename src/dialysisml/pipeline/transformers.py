"""Transformations that reduce a window to a feature vector.

A window transformer runs in three phases, each knowing something the previous
one could not:

    __init__   Hydra, at startup — the parameters, and column *names*
    bind       once per fold — the real columns, resolved to indices
    transform  per window or per block — only the values

`bind` exists because the column names are known at runtime only: how many
there are depends on what `drop_columns` removed and how many categories the
encoder found.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


def linear_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares line fit of every (window, feature) series against t = 0..W-1.

    Takes `(N, W, F)` and returns slope, intercept and RMSE, each `(N, F)`.
    t = 0 is the oldest session of the window, t = W-1 the current one.

    The RMSE is a root-*mean*-square: it stays on the scale of the feature
    instead of growing with sqrt(W), so it is comparable across window sizes.
    """
    N, W, F = X.shape
    if W < 2:
        raise ValueError(f"a line needs at least 2 points, got a window of {W}")

    # float32, not the default int64: an int `t` promotes every (N, W, F)
    # temporary below to float64 and doubles the peak memory of the fit
    t = np.arange(W, dtype=np.float32)
    t_mean = t.mean()
    y_mean = X.mean(axis=1)

    cov = np.sum((t - t_mean).reshape(1, W, 1) * (X - y_mean.reshape(N, 1, F)), axis=1)
    var_t = np.sum((t - t_mean) ** 2)

    slope = cov / var_t
    intercept = y_mean - slope * t_mean

    fitted = slope.reshape(N, 1, F) * t.reshape(1, W, 1) + intercept.reshape(N, 1, F)
    rmse = np.sqrt(np.mean((X - fitted) ** 2, axis=1))

    return slope, intercept, rmse


class WindowTransformer(ABC):
    """Reduces `(size, n_features)` to a flat feature vector.

    Subclasses implement `_bind` and `_transform`; the base handles resolving
    names to indices and accepting both a single window and a block of them.
    """

    _feature_names: list[str] | None = None
    _output_names: list[str] | None = None

    def bind(self, feature_names: Sequence[str]) -> "WindowTransformer":
        """Resolves column names to indices and works out the output names.

        Called once per fold, before any transform. Validates here so a wrong
        column name fails immediately instead of producing wrong indices.
        """
        self._feature_names = list(feature_names)
        self._bind()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """`(n, size, F) -> (n, K)`, or `(size, F) -> (K,)`.

        Accepting both shapes lets the same instance serve a per-window loop
        and a materialised block.
        """
        if self._output_names is None:
            raise RuntimeError(f"{type(self).__name__}: call bind() first")

        single = X.ndim == 2
        if single:
            X = X[None]  # (size, F) -> (1, size, F)

        out = self._transform(X)
        return out[0] if single else out

    @property
    def output_names(self) -> list[str]:
        """One name per produced column, in the order `transform` builds them."""
        if self._output_names is None:
            raise RuntimeError(f"{type(self).__name__}: call bind() first")
        return self._output_names

    def _resolve(self, columns: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
        """Column names to positions; `None` means every column."""
        names = self._feature_names
        assert names is not None

        if columns is None:
            return np.arange(len(names)), list(names)

        position = {name: i for i, name in enumerate(names)}
        missing = [c for c in columns if c not in position]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: columns not among the features: {missing}"
            )
        return np.array([position[c] for c in columns]), list(columns)

    @abstractmethod
    def _bind(self) -> None: ...

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray: ...


class RegressionFeatureTransformer(WindowTransformer):
    """Summarises each window with a linear fit, plus the last known value.

    Regression only makes sense on numeric columns; a one-hot column is
    constant within a window, so its slope is always zero. Those columns still
    have to lose the time axis — otherwise nothing concatenates — and taking
    their last value is the reduction that fits.
    """

    def __init__(
        self,
        *,
        regression_columns: Sequence[str] | None = None,
        last_value_columns: Sequence[str] | None = None,
        suffix: str = "",
    ):
        self.regression_columns = regression_columns
        self.last_value_columns = last_value_columns
        self.suffix = suffix

    def _bind(self) -> None:
        self._reg_idx, reg_names = self._resolve(self.regression_columns)
        self._last_idx, last_names = self._resolve(self.last_value_columns)

        # this order must match how _transform stacks the blocks, or SHAP will
        # attribute importances to the wrong features
        self._output_names = (
            [f"{c}_slope{self.suffix}" for c in reg_names]
            + [f"{c}_intercept{self.suffix}" for c in reg_names]
            + [f"{c}_rmse{self.suffix}" for c in reg_names]
            + [f"{c}_last{self.suffix}" for c in last_names]
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        blocks = []

        if len(self._reg_idx):
            blocks.extend(linear_fit(X[:, :, self._reg_idx]))
        if len(self._last_idx):
            blocks.append(X[:, -1, self._last_idx])

        return np.hstack(blocks)


class MultiSizeRegrTransformer(WindowTransformer):
    """The same summary over several nested windows, all ending at the anchor.

    Windows of different length sharing an endpoint are nested: the last `w`
    steps of a window of size W *are* the window of size w for that session.
    So the last value is added once, not once per size.
    """

    def __init__(
        self,
        *,
        slices: Sequence[int] = (30, 60, 90),
        regression_columns: Sequence[str] | None = None,
        last_value_columns: Sequence[str] | None = None,
    ):
        if not slices:
            raise ValueError("slices must not be empty")
        if len(set(slices)) != len(slices):
            raise ValueError(f"slices must not repeat a size, got {tuple(slices)}")

        self.slices = tuple(sorted(slices))
        self.regression_columns = regression_columns
        self.last_value_columns = last_value_columns

    def _bind(self) -> None:
        assert self._feature_names is not None

        # one regression per size, none of them taking the last value
        self._parts = [
            RegressionFeatureTransformer(
                regression_columns=self.regression_columns,
                last_value_columns=(),
                suffix=f"_w{size}",
            ).bind(self._feature_names)
            for size in self.slices
        ]

        self._last_idx, last_names = self._resolve(self.last_value_columns)

        self._output_names = [
            name for part in self._parts for name in part.output_names
        ] + [f"{c}_last" for c in last_names]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        window = X.shape[1]
        if self.slices[-1] > window:
            raise ValueError(
                f"slice {self.slices[-1]} exceeds the window size {window}"
            )

        # tail, not head: every sub window must end at the current session
        blocks = [
            part.transform(X[:, -size:, :])
            for size, part in zip(self.slices, self._parts)
        ]
        if len(self._last_idx):
            blocks.append(X[:, -1, self._last_idx])

        return np.hstack(blocks)
