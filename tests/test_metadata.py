"""Tests for the per-window metadata carried by a WindowStore.

The metadata is aligned to the windows by position only: nothing in the types
enforces it, so these tests pin the invariant that everything else relies on.
"""

import numpy as np
import pytest

from dialysisml.config import DATA_PATH, META_COLUMNS
from dialysisml.window import (
    MultiSizeRegrTransformer,
    RegressionFeatureTransformer,
    WindowConfig,
    WindowStore,
)

CONF = WindowConfig(size=90, stride=1)


@pytest.fixture(scope="module")
def store() -> WindowStore:
    """Raw size-90 store from data/, read once for the whole file."""
    npz = DATA_PATH / str(CONF) / f"{CONF}.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not found: generate the windows first")
    return WindowStore(CONF)


def test_meta_has_the_declared_columns(store):
    assert list(store.meta_train.columns) == list(META_COLUMNS)
    assert list(store.meta_test.columns) == list(META_COLUMNS)


def test_one_meta_row_per_window(store):
    assert len(store.meta_train) == len(store.X_train) == len(store.y_train)
    assert len(store.meta_test) == len(store.X_test) == len(store.y_test)


@pytest.mark.parametrize(
    "transformer", [RegressionFeatureTransformer(), MultiSizeRegrTransformer()]
)
def test_meta_survives_a_transform(store, transformer):
    """A transformer rewrites columns, never rows, so the metadata still fits."""
    transformed = WindowStore(CONF, transformer)

    assert len(transformed.meta_train) == len(transformed.X_train)
    assert len(transformed.meta_test) == len(transformed.X_test)
    # and it is the same metadata, not merely the same length
    assert transformed.meta_train.equals(store.meta_train)


def test_anchor_is_the_current_session(store):
    """t must be the window's LAST session, not its first.

    The target is the days from the anchor to the event, so if `t` were taken
    from the wrong row of the window this identity would break.
    """
    for split in ("train", "test"):
        meta = getattr(store, f"meta_{split}")
        y = getattr(store, f"y_{split}").ravel()
        days = (meta["t_out"] - meta["t"]).dt.days
        np.testing.assert_array_equal(days.to_numpy(), y)
        assert (meta["t_start"] < meta["t"]).all()


def test_no_patient_is_in_both_splits(store):
    """split_series splits on patient; without this the test set means nothing."""
    assert not set(store.meta_train["patient"]) & set(store.meta_test["patient"])
    assert store.stats().loc["patient_overlap", "train"] == 0


def test_windows_of_a_series_are_consecutive(store):
    """series_idx must count 0..n-1 inside each series, with no gaps."""
    for _, g in store.meta_train.groupby(["patient", "event"], sort=False):
        idx = g["series_idx"].to_numpy()
        np.testing.assert_array_equal(idx, np.arange(len(idx)))
        assert g["series_len"].nunique() == 1
        assert len(idx) == g["series_len"].iloc[0] - CONF.size + 1


def test_stats_and_per_patient_agree(store):
    stats = store.stats()
    per_patient = store.per_patient("train")

    assert len(per_patient) == stats.loc["patients", "train"]
    assert per_patient["windows"].sum() == stats.loc["windows", "train"]
    assert len(store.groups("train")) == len(store.X_train)
