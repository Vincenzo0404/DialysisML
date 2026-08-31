from typing import Callable

import pandas as pd


def tte_to_months(tte: pd.Series) -> pd.Series:
    """Rescale a time-to-event expressed in days into 30-day months."""
    return tte / 30


def compute_tte(
    df: pd.DataFrame,
    *,
    min_days: int = 1,
    tte_transformation: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame:
    """Days from each session to the next adverse event.

    `min_days=1` drops the sessions held on the event day itself, whose
    time-to-event is zero. The threshold is always in days, so it keeps the
    same meaning whatever `tte_transformation` does.

    `tte_transformation` rescales the target after filtering: pass
    `tte_to_months` to train on months instead of days.
    """
    required = {"t_session", "t_event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["tte"] = (df["t_event"] - df["t_session"]).dt.days
    df = df[df["tte"] >= min_days].copy()
    if tte_transformation is not None:
        df["tte"] = tte_transformation(df["tte"])
    return df
