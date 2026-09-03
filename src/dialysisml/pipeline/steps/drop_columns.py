from typing import Sequence

import pandas as pd

from dialysisml.features import META_COLUMNS


def drop_columns(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] = (),
    meta_columns: Sequence[str] = META_COLUMNS,
) -> pd.DataFrame:
    """Drops the listed columns and keeps everything else.

    Dropping rather than selecting means a column added to the materialized
    view reaches the model on its own; name it here when it should not.
    """
    cols = set(columns)

    protected = cols & set(meta_columns)
    if protected:
        raise ValueError(f"meta columns cannot be dropped: {sorted(protected)}")

    missing_meta = [c for c in meta_columns if c not in df.columns]
    if missing_meta:
        raise ValueError(f"meta columns not in the data: {missing_meta}")

    unknown = cols - set(df.columns)
    if unknown:
        raise ValueError(f"columns to drop not in the data: {sorted(unknown)}")

    return df[[c for c in df.columns if c not in columns]]
