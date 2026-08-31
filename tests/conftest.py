"""Synthetic frames shared by the pipeline tests.

The pipeline is a chain of pure DataFrame -> DataFrame steps, so every test
starts from a frame built here rather than from the snapshot: what a step does
is far easier to assert on ten rows you wrote yourself.
"""

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

DEFAULT_FEATURES = ("hb", "bmi")


def make_merged(
    *,
    n_patients: int = 3,
    sessions_per_patient: int = 10,
    events_per_patient: int = 1,
    features: Sequence[str] = DEFAULT_FEATURES,
    missing_rate: float = 0.0,
    days_between: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """A frame shaped like `read_raw_data`'s output: every session already
    joined to the adverse event that follows it.

    Each patient's sessions are cut into `events_per_patient` consecutive
    blocks, one event per block, so `(patient, event)` names a series exactly
    as it does downstream. `event` ids repeat across patients on purpose —
    that is what the real data does, and a step that keys on `event` alone
    has to break on it.
    """
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(days_between, "D")

    rows = []
    for p in range(n_patients):
        # staggered starts, so patients are interleaved once sorted by date
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(p, "D")
        times = pd.date_range(start, periods=sessions_per_patient, freq=step)

        for block, positions in enumerate(
            np.array_split(np.arange(sessions_per_patient), events_per_patient)
        ):
            if len(positions) == 0:
                continue
            t_event = times[positions[-1]] + step
            for i in positions:
                rows.append(
                    {
                        "patient": f"P{p}",
                        "t_session": times[i],
                        "t_event": t_event,
                        "event": f"E{block}",
                        "event_type": "death",
                    }
                )

    df = pd.DataFrame(rows)

    for name in features:
        values = rng.normal(loc=10.0, scale=2.0, size=len(df))
        if missing_rate:
            values[rng.random(len(df)) < missing_rate] = np.nan
        df[name] = values

    return df


@pytest.fixture
def make_merged_frame():
    """The factory itself, for tests that need their own shape."""
    return make_merged


@pytest.fixture
def merged() -> pd.DataFrame:
    """Three patients, one event each, ten sessions apiece."""
    return make_merged()


@pytest.fixture
def windowable() -> pd.DataFrame:
    """A frame ready for `SlidingWindow`: meta columns, features and `tte`.

    `tte` is filled by hand rather than by `compute_tte` so the window tests
    stay independent of it, and so a target can be recognised by its value.
    """
    df = make_merged(n_patients=2, sessions_per_patient=8)
    df["tte"] = (df["t_event"] - df["t_session"]).dt.days.astype(float)
    return df
