"""Post-split steps: what gets fitted on what, and what must stay untouched.

These steps are the ones that can leak. Every transformer here is fitted on
the training rows and applied to both sides, so the tests are about the
boundary — a statistic that came from the test rows is a silent leak, and so
is a column that got transformed when it should not have been.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from dialysisml.pipeline.steps import (
    apply_sklearn_transformer,
    drop_incomplete_rows,
    flag_missing_values,
)
from dialysisml.preprocessing import Winsorizer


@pytest.fixture
def split(make_merged_frame):
    """Two disjoint patient groups on clearly different scales.

    The scales differ so a statistic borrowed from the test side shows up as
    a wrong number rather than as noise.
    """
    train = make_merged_frame(n_patients=2, sessions_per_patient=20, seed=1)
    test = make_merged_frame(n_patients=2, sessions_per_patient=20, seed=2)
    test["patient"] = test["patient"].str.replace("P", "Q")
    test["hb"] = test["hb"] * 100
    train["tte"], test["tte"] = 10.0, 10.0
    return train, test


# -- fitted on train, applied to both -------------------------------------


def test_statistics_come_from_train_only(split):
    train, test = split
    scaler = StandardScaler()

    out_train, _ = apply_sklearn_transformer(
        train, test, transformer=scaler, columns=["hb"]
    )

    assert scaler.mean_[0] == pytest.approx(train["hb"].mean())
    assert out_train["hb"].mean() == pytest.approx(0.0, abs=1e-9)


def test_test_rows_are_not_recentred(split):
    """The test side keeps its own distribution: it is transformed with the
    train statistics, not standardised in its own right."""
    train, test = split

    _, out_test = apply_sklearn_transformer(
        train, test, transformer=StandardScaler(), columns=["hb"]
    )

    assert abs(out_test["hb"].mean()) > 1.0


def test_inputs_are_not_modified_in_place(split):
    train, test = split
    before_train, before_test = train.copy(), test.copy()

    apply_sklearn_transformer(
        train, test, transformer=StandardScaler(), columns=["hb"]
    )

    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(test, before_test)


def test_unknown_column_raises(split):
    train, test = split
    with pytest.raises(ValueError, match="columns not in the data"):
        apply_sklearn_transformer(
            train, test, transformer=StandardScaler(), columns=["nope"]
        )


# -- what the default column set picks up ---------------------------------


def test_default_leaves_the_meta_columns_alone(split):
    """`columns=None` means every numeric column; the meta columns are objects
    and datetimes, so they survive."""
    train, test = split

    out_train, out_test = apply_sklearn_transformer(
        train, test, transformer=StandardScaler()
    )

    for column in ("patient", "t_session", "t_event", "event", "event_type"):
        pd.testing.assert_series_equal(out_train[column], train[column])
        pd.testing.assert_series_equal(out_test[column], test[column])


def test_default_does_not_touch_the_target(split):
    """`tte` is numeric, so `columns=None` reaches it. Scaling the target — and
    scaling the test targets with the train statistics — changes what the
    metrics mean."""
    train, test = split

    out_train, out_test = apply_sklearn_transformer(
        train, test, transformer=StandardScaler()
    )

    pd.testing.assert_series_equal(out_train["tte"], train["tte"])
    pd.testing.assert_series_equal(out_test["tte"], test["tte"])


def test_winsorizer_does_not_clip_the_target(split):
    """Clipping `tte` to the train quantiles moves the test labels themselves,
    which no metric can recover from."""
    train, test = split
    rng = np.random.default_rng(0)
    train["tte"] = rng.gamma(2.0, 50.0, len(train))
    test["tte"] = rng.gamma(2.0, 50.0, len(test))

    out_train, out_test = apply_sklearn_transformer(
        train, test, transformer=Winsorizer(lower=0.01, upper=0.99)
    )

    pd.testing.assert_series_equal(out_train["tte"], train["tte"])
    pd.testing.assert_series_equal(out_test["tte"], test["tte"])


def test_winsorizer_does_not_flatten_a_rare_missing_flag(make_merged_frame):
    """A `*_missing` flag is 0/1 and numeric, so `columns=None` clips it too.
    Below 1% of ones the 99th percentile is 0 and the whole column goes to
    zero — the flag stops saying anything."""
    train = make_merged_frame(n_patients=5, sessions_per_patient=60, seed=1)
    test = make_merged_frame(n_patients=2, sessions_per_patient=60, seed=2)

    # a feature missing on a handful of rows only
    train.loc[train.index[:2], "hb"] = np.nan
    train = flag_missing_values(train, columns=["hb"])
    test = flag_missing_values(test, columns=["hb"])

    assert train["hb_missing"].mean() < 0.01, "fixture must keep the flag rare"

    out_train, _ = apply_sklearn_transformer(
        train, test, transformer=Winsorizer(lower=0.01, upper=0.99)
    )

    assert out_train["hb_missing"].sum() == train["hb_missing"].sum()


# -- dropping rows ---------------------------------------------------------


def test_drop_incomplete_rows_uses_only_the_listed_columns(split):
    train, test = split
    train.loc[train.index[0], "hb"] = np.nan
    train.loc[train.index[1], "bmi"] = np.nan

    out_train, _ = drop_incomplete_rows(train, test, columns=["hb"])

    assert len(out_train) == len(train) - 1
    assert out_train["bmi"].isna().sum() == 1
