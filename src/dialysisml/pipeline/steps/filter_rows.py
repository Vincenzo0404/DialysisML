from typing import Sequence

import pandas as pd


def drop_incomplete_rows(
    train: pd.DataFrame, test: pd.DataFrame, *, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drops rows containing at least one null value."""
    columns = list(columns)
    return train.dropna(subset=columns), test.dropna(subset=columns)


def drop_short_censored_sessions(
    df: pd.DataFrame, *, min_days: int = 365, cap_days: bool = True
) -> pd.DataFrame:
    """Removes censored sessions for patients with series shorter than `min_days` and caps their TTE to `min_days`"""
    # validate inputs
    required = ("has_event", "tte")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns:  {missing}.")
    if min_days < 0:
        raise ValueError(f"`min_days` has to be positive {min_days} was given.")

    df = df.copy()

    # a censored session is valid only if it has `min_days` days of data ahead of it.
    valid = df["has_event"] | df["tte"].gt(min_days - 1)
    df = df[valid]

    if cap_days:
        # cap tte to min_days
        beyond_horizion = df["tte"] > min_days
        df.loc[beyond_horizion, "tte"] = min_days
        assert (df["tte"] <= min_days).all()

    return df
