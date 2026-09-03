from typing import Callable

import pandas as pd


def scale_tte(df: pd.DataFrame, divisor: int = 30) -> pd.DataFrame:
    """Rescale tte by divisor."""

    if "tte" not in df.columns:
        raise ValueError("Missing column: 'tte'.")
    if divisor <= 0:
        raise ValueError(f"`divisor` must be positive, got {divisor}.")

    df = df.copy()
    df["tte"] = df["tte"] / divisor
    return df


def compute_tte(
    df: pd.DataFrame,
    *,
    min_tte: int = 1,
) -> pd.DataFrame:
    """Days from each session to the next adverse event.

    `min_tte=1` drops the sessions held on the event day itself, whose
    time-to-event is zero. The threshold is always in days, so it keeps the
    same meaning whatever `tte_transformation` does.

    `tte_transformation` rescales the target after filtering: pass
    `tte_to_months` to train on months instead of days.
    """
    required = {"t_session", "t_event", "has_event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["tte"] = (df["t_event"] - df["t_session"]).dt.days
    df = df[df["tte"] >= min_tte].copy()

    return df
