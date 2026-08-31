from typing import Any, Sequence

import pandas as pd


def apply_sklearn_transformer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    # left untyped: neither BaseEstimator nor TransformerMixin declares fit,
    # so any sklearn annotation here fights the type checker for nothing
    transformer: Any,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits a sklearn transformer on train and applies it on both train and test sets"""
    if columns is None:
        columns = list(train.select_dtypes("number").columns)
    else:
        columns = list(columns)

    missing = set(columns) - set(train.columns)
    if missing:
        raise ValueError(f"columns not in the data: {sorted(missing)}")

    fitted = transformer.fit(train[columns])

    train, test = train.copy(), test.copy()
    train[columns] = fitted.transform(train[columns])
    test[columns] = fitted.transform(test[columns])

    return train, test
