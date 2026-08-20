from abc import ABC
from typing import Sequence

import numpy as np


class WindowTransformer(ABC):
    """Base class for named, cacheable transformations of windowed data.

    Override `transform_X` and/or `transform_y` depending on which side of
    the data the transformer changes; `WindowStore` inspects which ones were
    overridden to decide what to compute and what to persist to disk.
    """

    def __str__(self) -> str:
        return type(self).__name__

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return X

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return y

    @property
    def transforms_X(self) -> bool:
        return type(self).transform_X is not WindowTransformer.transform_X

    @property
    def transforms_y(self) -> bool:
        return type(self).transform_y is not WindowTransformer.transform_y


# ---Transformers---
def linear_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares line fit of every (window, feature) series against t = 0..W-1.

    Takes X of shape (N, W, F) and returns (a, b, c), each of shape (N, F):
    slope, intercept and RMSE of the residuals. t = 0 is the oldest session of
    the window, t = W-1 the current one.
    """
    N, W, F = X.shape
    if W < 2:
        raise ValueError(f"a line needs at least 2 points, got a window of {W}")

    t = np.arange(W)

    t_mean = np.mean(t)
    y_mean = np.mean(X, axis=1)

    t_diff = (t - t_mean).reshape(1, W, 1)
    y_diff = X - y_mean.reshape(N, 1, F)

    cov = np.sum(t_diff * y_diff, axis=1)
    var_t = np.sum((t - t_mean) ** 2)

    # a (Slope), b (Intercept)
    a = cov / var_t
    b = y_mean - a * t_mean

    # c (RMSE)
    y_hat = a.reshape(N, 1, F) * t.reshape(1, W, 1) + b.reshape(N, 1, F)
    residuals = X - y_hat
    c = np.sqrt(np.mean(residuals**2, axis=1))

    return a, b, c


class RegressionFeatureTransformer(WindowTransformer):
    """Summarizes each window/feature with a linear fit (slope, intercept, RMSE) plus the last value."""

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        a, b, c = linear_fit(X)

        # curr (Ultimo valore temporale della finestra)
        curr = X[:, -1, :]

        # Appiattimento e concatenazione (Ordine: tutti gli 'a', tutti i 'b', ecc.)
        return np.hstack([a, b, c, curr])

    def output_feature_names(self, input_feature_names: Sequence[str]) -> list[str]:
        return (
            [f"{f}_slope" for f in input_feature_names]
            + [f"{f}_intercept" for f in input_feature_names]
            + [f"{f}_rmse" for f in input_feature_names]
            + [f"{f}_last" for f in input_feature_names]
        )


class MultiSizeRegrTransformer(WindowTransformer):
    def __init__(self, slices: Sequence[int] = (30, 60, 90)):
        self._slices = slices
        self.slice_transformer = RegressionFeatureTransformer()

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        N, W, F = X.shape
        if max(self._slices) > W:
            raise ValueError(
                f"slice {max(self._slices)} exceeds the stored window size {W}"
            )

        sub_arrays = []
        last_curr = np.empty((N, F))

        for slice in self._slices:

            X_slice = X[:, -slice:, :]  # extract sub windows
            X_slice_regr = self.slice_transformer.transform_X(X_slice)
            # the base transformer returns 4 blocks of F columns: [a | b | c | curr]
            last_curr = X_slice_regr[:, -F:]
            sub_arrays.append(X_slice_regr[:, :-F])  # skip "curr"

        sub_arrays.append(last_curr)

        return np.hstack(sub_arrays)

    def __str__(self) -> str:
        # the sizes change the output, so they must change the on-disk cache key too
        return f"{type(self).__name__}({','.join(str(s) for s in self._slices)})"
