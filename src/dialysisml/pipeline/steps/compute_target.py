import pandas as pd


def compute_tte(df: pd.DataFrame, *, min_days: int = 1) -> pd.DataFrame:
    """Days from each session to the next adverse event.

    `min_days=1` drops the sessions held on the event day itself, whose
    time-to-event is zero.
    """
    required = {"t_session", "t_event"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["tte"] = (df["t_event"] - df["t_session"]).dt.days
    return df[df["tte"] >= min_days]
