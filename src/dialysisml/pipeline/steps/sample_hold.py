import pandas as pd

from dialysisml.features import META_COLUMNS


def sample_hold(df: pd.DataFrame) -> pd.DataFrame:
    if not set(["patient", "t_session"]) <= set(df.columns):
        raise ValueError("Missing patient in df")

    cols_to_fill = df.columns.difference(META_COLUMNS)
    df[cols_to_fill] = df.groupby("patient")[cols_to_fill].ffill()
    df[cols_to_fill] = df.groupby("patient")[cols_to_fill].bfill()

    return df
