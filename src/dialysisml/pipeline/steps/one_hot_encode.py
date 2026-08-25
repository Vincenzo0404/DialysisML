from typing import Sequence

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Replaces categorical columns with their one-hot encoding.
    """
    columns = list(columns)

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"columns not in the data: {sorted(missing)}")

    if drop_missing:
        df = df.dropna(subset=columns)
    elif df[columns].isna().any().any():
        raise ValueError(
            f"missing values in {columns}: pass drop_missing=True or fill them first"
        )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[columns])

    result = df.drop(columns=columns)
    result[list(encoder.get_feature_names_out(columns))] = encoded
    return result