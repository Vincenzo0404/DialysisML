from typing import Sequence

import pandas as pd


def drop_incomplete_rows(
    train: pd.DataFrame, test: pd.DataFrame, *, columns: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drops rows containing at least one null value."""
    columns = list(columns)
    return train.dropna(subset=columns), test.dropna(subset=columns)
