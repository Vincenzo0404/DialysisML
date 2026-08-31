"""SlidingWindow: index arithmetic, where a mistake is silent.

Nothing here raises when the windows come out wrong — the shapes still fit
and training still runs, only on the wrong rows. So these tests check the
three things that cannot be read off a shape: which rows a window covers,
which row labels it, and that it never spans two series.
"""

import numpy as np
import pandas as pd
import pytest

from dialysisml.features import META_COLUMNS
from dialysisml.pipeline.SlidingWindow import SlidingWindow


def with_marker(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `row_id`, a feature whose value *is* the row's position.

    Lets a test say which rows a window landed on by reading the window.
    """
    df = df.sort_values(["patient", "event", "t_session"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=float)
    return df


def series_lengths(df: pd.DataFrame) -> list[int]:
    return df.groupby(["patient", "event"]).size().tolist()


# -- construction ----------------------------------------------------------


@pytest.mark.parametrize("size", [-1, 0, 1])
def test_rejects_size_below_two(windowable, size):
    with pytest.raises(ValueError, match="size must be at least 2"):
        SlidingWindow(windowable, size=size)


@pytest.mark.parametrize("stride", [-1, 0])
def test_rejects_non_positive_stride(windowable, stride):
    with pytest.raises(ValueError, match="stride must be at least 1"):
        SlidingWindow(windowable, stride=stride)


@pytest.mark.parametrize(
    ("size", "stride", "expected"),
    [(2, 1, 2), (30, 1, 30), (3, 2, 5), (4, 3, 10)],
)
def test_span_accounts_for_stride(windowable, size, stride, expected):
    assert SlidingWindow(windowable, size=size, stride=stride).span == expected


# -- how many windows ------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "stride"), [(2, 1), (3, 1), (4, 2), (8, 1), (5, 3)]
)
def test_window_count_is_per_series(windowable, size, stride):
    """Each series contributes `len - span + 1` windows, never fewer nor more."""
    sw = SlidingWindow(windowable, size=size, stride=stride)
    expected = sum(max(0, n - sw.span + 1) for n in series_lengths(windowable))
    assert len(sw) == expected


def test_series_shorter_than_span_produce_nothing(make_merged_frame):
    df = make_merged_frame(n_patients=2, sessions_per_patient=4)
    df["tte"] = 1.0
    assert len(SlidingWindow(df, size=5)) == 0


def test_materialize_refuses_an_empty_window_set(make_merged_frame):
    df = make_merged_frame(n_patients=1, sessions_per_patient=3)
    df["tte"] = 1.0
    with pytest.raises(ValueError, match="no series long enough"):
        SlidingWindow(df, size=10).materialize()


# -- which rows a window covers -------------------------------------------


def test_window_rows_are_consecutive(windowable):
    df = with_marker(windowable)
    sw = SlidingWindow(df, size=3)
    marker = sw.feature_columns.index("row_id")

    for window, _ in sw:
        rows = window[:, marker]
        assert np.array_equal(np.diff(rows), np.ones(len(rows) - 1))


def test_stride_skips_rows_but_keeps_the_anchor(windowable):
    """With stride 2 a window covers 5 rows and keeps 3, the last one included."""
    df = with_marker(windowable)
    sw = SlidingWindow(df, size=3, stride=2)
    marker = sw.feature_columns.index("row_id")

    for start, (window, _) in zip(sw.starts, sw):
        rows = window[:, marker]
        assert np.array_equal(rows, np.arange(start, start + 5, 2))


def test_window_never_crosses_a_series(make_merged_frame):
    """Two events per patient: a window straddling the boundary would mix the
    tail of one series with the head of the next."""
    df = make_merged_frame(n_patients=3, sessions_per_patient=12, events_per_patient=3)
    df["tte"] = 1.0
    df = with_marker(df)

    sw = SlidingWindow(df, size=3)
    # a series id the window can be read off, since patient/event are metadata
    series = df.groupby(["patient", "event"], sort=False).ngroup().to_numpy()
    marker = sw.feature_columns.index("row_id")

    for window, _ in sw:
        rows = window[:, marker].astype(int)
        assert len(set(series[rows])) == 1


# -- which row labels a window --------------------------------------------


def test_target_comes_from_the_last_row(windowable):
    df = with_marker(windowable)
    sw = SlidingWindow(df, size=4)
    marker = sw.feature_columns.index("row_id")
    tte = df["tte"].to_numpy()

    for window, target in sw:
        last_row = int(window[-1, marker])
        assert target[0] == pytest.approx(tte[last_row])


def test_meta_is_aligned_with_the_windows(windowable):
    """Row i of `meta()` must describe window i — that is what ties a
    prediction back to a patient."""
    df = with_marker(windowable)
    sw = SlidingWindow(df, size=3)
    marker = sw.feature_columns.index("row_id")
    meta = sw.meta()

    assert len(meta) == len(sw)
    for i, (window, _) in enumerate(sw):
        last_row = int(window[-1, marker])
        assert meta.loc[i, "patient"] == sw.df.loc[last_row, "patient"]
        assert meta.loc[i, "t_session"] == sw.df.loc[last_row, "t_session"]


# -- what counts as a feature ---------------------------------------------


def test_feature_columns_exclude_meta_and_targets(windowable):
    sw = SlidingWindow(windowable, size=3)
    assert set(sw.feature_columns).isdisjoint(META_COLUMNS)
    assert "tte" not in sw.feature_columns


def test_a_second_target_column_stays_out_of_the_features(windowable):
    """A column derived from the target leaks unless it is declared here."""
    df = windowable.copy()
    df["tte_months"] = df["tte"] / 30

    sw = SlidingWindow(df, size=3, target_columns=("tte", "tte_months"))
    assert "tte_months" not in sw.feature_columns
    assert sw.materialize()[1].shape[1] == 2


# -- shapes ----------------------------------------------------------------


def test_materialize_shapes(windowable):
    sw = SlidingWindow(windowable, size=4, stride=2)
    X, y = sw.materialize()

    assert X.shape == (len(sw), 4, len(sw.feature_columns))
    assert y.shape == (len(sw), 1)
    assert X.dtype == np.float32


# -- robustness to input order --------------------------------------------


def test_row_order_of_the_input_does_not_matter(windowable):
    """Windows are cut as slices, so the frame is sorted internally; a shuffled
    input must give the same windows, not different ones."""
    df = with_marker(windowable)
    shuffled = df.sample(frac=1.0, random_state=1)

    X_ordered, y_ordered = SlidingWindow(df, size=3).materialize()
    X_shuffled, y_shuffled = SlidingWindow(shuffled, size=3).materialize()

    np.testing.assert_array_equal(X_ordered, X_shuffled)
    np.testing.assert_array_equal(y_ordered, y_shuffled)


def test_the_input_frame_is_not_modified(windowable):
    before = windowable.copy()
    SlidingWindow(windowable, size=3).materialize()
    pd.testing.assert_frame_equal(windowable, before)
