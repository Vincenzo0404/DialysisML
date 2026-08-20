"""Tests for RegressionFeatureTransformer.

Two complementary kinds:
- example based: hand-built input whose exact answer we know a priori. Strongest
  oracle, since it depends on no other implementation, but covers few cases.
- differential: real data compared against np.polyfit. Covers the whole dataset,
  but would not notice if both implementations were wrong the same way.
"""

import numpy as np
import pytest

from dialysisml.config import DATA_PATH
from dialysisml.window import (
    MultiSizeRegrTransformer,
    RegressionFeatureTransformer,
    WindowConfig,
    WindowStore,
)

# The transformer concatenates 4 side-by-side blocks, each as wide as the
# number of features: [slope | intercept | rmse | last].
BLOCKS = ("slope", "intercept", "rmse", "last")


def block(out: np.ndarray, name: str, n_features: int) -> np.ndarray:
    """Extracts one of the 4 blocks from the transformer output."""
    i = BLOCKS.index(name)
    return out[:, i * n_features : (i + 1) * n_features]


# ---------------------------------------------------------------------------
# Example based
# ---------------------------------------------------------------------------


def test_perfect_line_and_constant_feature():
    """feature 0: y = 3t + 5 -> slope 3, intercept 5, rmse 0, last 3*29+5 = 92
    feature 1: y = 7 -> slope 0, intercept 7, rmse 0, last 7

    Pins the maths, the block order and per-feature independence at once.
    """
    W = 30
    t = np.arange(W)  # t = [0, 1, 2, ..., 29]
    X = np.empty((1, W, 2))  # 1 window, 30 steps, 2 features
    X[0, :, 0] = 3.0 * t + 5.0  # feature 0: perfect line
    X[0, :, 1] = 7.0  # feature 1: constant

    out = RegressionFeatureTransformer().transform_X(X)  # apply the transformer

    #                 slope     intercept    rmse       last
    expected = np.array([[3.0, 0.0, 5.0, 7.0, 0.0, 0.0, 92.0, 7.0]])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_nonzero_rmse():
    """y = [0, 1, 0] over t = [0, 1, 2]: no line fits, so rmse > 0.

    t_mean = 1, y_mean = 1/3
    cov = (-1)(-1/3) + (0)(2/3) + (1)(-1/3) = 0   -> slope 0
    intercept = y_mean - slope * t_mean = 1/3
    residuals = [-1/3, 2/3, -1/3] -> rmse = sqrt((1/9 + 4/9 + 1/9) / 3) = sqrt(2)/3
    """
    X = np.array([[[0.0], [1.0], [0.0]]])  # 1 window, 3 steps, 1 feature

    out = RegressionFeatureTransformer().transform_X(X)

    slope, intercept, rmse, last = out[0]
    assert slope == pytest.approx(0.0, abs=1e-12)
    assert intercept == pytest.approx(1 / 3)
    assert rmse == pytest.approx(np.sqrt(2) / 3)
    assert last == pytest.approx(0.0)


def test_output_shape():
    """(N, window, F) must become (N, F * 4)."""
    out = RegressionFeatureTransformer().transform_X(np.zeros((7, 30, 33)))
    assert out.shape == (7, 33 * 4)


# ---------------------------------------------------------------------------
# Differential
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_windows() -> np.ndarray:
    """Raw windows from data/, read once for the whole file.

    Skips rather than regenerating, so tests never hit the database.
    """
    conf = WindowConfig(size=30, stride=1)
    npz = DATA_PATH / str(conf) / f"{conf}.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not found: generate the windows first")
    return WindowStore(conf).X_train


def test_matches_polyfit(real_windows):
    """Compare against np.polyfit on every window and every feature."""
    X = real_windows
    N, W, F = X.shape
    t = np.arange(W)

    out = RegressionFeatureTransformer().transform_X(X)

    # np.polyfit fits many series at once when they are stacked as columns:
    # (N, W, F) -> (W, N*F), so a single lstsq covers the whole dataset.
    # Column j holds pair (window n, feature f) with j = n*F + f, hence reshape(N, F).
    Y = X.transpose(1, 0, 2).reshape(W, N * F)
    slope_ref, intercept_ref = np.polyfit(t, Y, 1)

    residuals = Y - (t[:, None] * slope_ref + intercept_ref)
    rmse_ref = np.sqrt((residuals**2).mean(axis=0))

    np.testing.assert_allclose(
        block(out, "slope", F), slope_ref.reshape(N, F), atol=1e-9
    )
    np.testing.assert_allclose(
        block(out, "intercept", F), intercept_ref.reshape(N, F), atol=1e-9
    )
    np.testing.assert_allclose(block(out, "rmse", F), rmse_ref.reshape(N, F), atol=1e-9)
    # last is a plain copy, so it must match bit for bit
    np.testing.assert_array_equal(block(out, "last", F), X[:, -1, :])


# ---------------------------------------------------------------------------
# MultiSizeRegrTransformer
# ---------------------------------------------------------------------------
# Output layout for slices (s1, ..., sk), every block F columns wide:
#   [a_s1 | b_s1 | c_s1 | a_s2 | b_s2 | c_s2 | ... | curr]
# `curr` appears once, not once per slice.

STATS = ("slope", "intercept", "rmse")


def ms_block(out: np.ndarray, i: int, name: str, n_features: int) -> np.ndarray:
    """Extracts the block of statistic `name` for the i-th slice."""
    j = i * len(STATS) + STATS.index(name)
    return out[:, j * n_features : (j + 1) * n_features]


def ms_curr(out: np.ndarray, n_features: int) -> np.ndarray:
    return out[:, -n_features:]


def test_multisize_output_shape():
    """(N, W, F) must become (N, F * (3 * n_slices + 1)): 9 stats + curr per feature."""
    out = MultiSizeRegrTransformer((30, 60, 90)).transform_X(np.zeros((7, 90, 33)))
    assert out.shape == (7, 33 * (3 * 3 + 1)) == (7, 330)


def test_multisize_single_slice_matches_base_transformer():
    """With one slice covering the whole window it degenerates to step 1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(11, 30, 4))

    np.testing.assert_allclose(
        MultiSizeRegrTransformer((30,)).transform_X(X),
        RegressionFeatureTransformer().transform_X(X),
        atol=1e-12,
    )


def test_multisize_slices_the_tail_not_the_head():
    """The windows are nested and all end at the current session.

    Feature 0 is flat at 0 for the first 60 sessions, then rises with slope 2.
    So the W=30 fit sees only the rising part, W=90 sees the whole thing, and
    the value at t=W-1 is the same for all three. Head-slicing would invert
    this: W=30 would report slope 0, the trend from three months ago.
    """
    W = 90
    X = np.zeros((1, W, 1))
    X[0, 60:, 0] = 2.0 * np.arange(30)  # 0..58 over the last 30 sessions

    out = MultiSizeRegrTransformer((30, 60, 90)).transform_X(X)

    # W=30 covers exactly the rising segment -> slope 2, perfect fit
    assert ms_block(out, 0, "slope", 1)[0, 0] == pytest.approx(2.0)
    assert ms_block(out, 0, "rmse", 1)[0, 0] == pytest.approx(0.0, abs=1e-12)

    # W=60 covers 30 flat + 30 rising -> shallower, and no line fits.
    # cov = sum(t*y) - n*t_mean*y_mean = 43210 - 60*29.5*14.5 = 17545
    # var_t = n*(n^2 - 1)/12 = 60*3599/12 = 17995
    assert ms_block(out, 1, "slope", 1)[0, 0] == pytest.approx(17545 / 17995)
    assert ms_block(out, 1, "rmse", 1)[0, 0] > 1.0

    # W=90 is shallower still: the flat stretch drags the long-run trend down
    assert 0.0 < ms_block(out, 2, "slope", 1)[0, 0] < ms_block(out, 1, "slope", 1)[0, 0]

    # the current session is shared by every window
    assert ms_curr(out, 1)[0, 0] == pytest.approx(58.0)


def test_multisize_str_carries_the_sizes():
    """str() keys the on-disk cache, so different sizes must not collide."""
    assert str(MultiSizeRegrTransformer((30, 60, 90))) != str(
        MultiSizeRegrTransformer((10, 20))
    )
    assert "30,60,90" in str(MultiSizeRegrTransformer((30, 60, 90)))


def test_multisize_rejects_slice_larger_than_the_window():
    """numpy clamps out-of-range slices silently, which would yield equal blocks."""
    with pytest.raises(ValueError, match="exceeds the stored window size"):
        MultiSizeRegrTransformer((30, 60, 90)).transform_X(np.zeros((2, 30, 3)))


@pytest.fixture(scope="module")
def real_windows_90() -> np.ndarray:
    """Raw size-90 windows from data/, read once for the whole file."""
    conf = WindowConfig(size=90, stride=1)
    npz = DATA_PATH / str(conf) / f"{conf}.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not found: generate the windows first")
    return WindowStore(conf).X_train


def test_multisize_blocks_equal_the_base_transformer_on_each_tail(real_windows_90):
    """The nesting property, on real data.

    The last `s` steps of a 90-window *are* the s-window of the same session, so
    each block must equal step 1 applied to that tail. This is what makes one
    size-90 store enough to cover all three scales.
    """
    X = real_windows_90
    F = X.shape[2]
    slices = (30, 60, 90)

    out = MultiSizeRegrTransformer(slices).transform_X(X)
    base = RegressionFeatureTransformer()

    for i, s in enumerate(slices):
        ref = base.transform_X(X[:, -s:, :])
        for name in STATS:
            np.testing.assert_allclose(
                ms_block(out, i, name, F),
                block(ref, name, F),
                atol=1e-9,
                err_msg=f"{name} block differs for slice {s}",
            )

    # curr is a plain copy of the current session, so it must match bit for bit
    np.testing.assert_array_equal(ms_curr(out, F), X[:, -1, :])
