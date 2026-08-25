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
    """Fits a scikit-learn transformer on the training rows, applies it to both.

    The fit sees only `train`, so leakage is impossible by construction. The
    result is written back into the same columns, keeping both frames intact
    for the steps that follow — the windowing still needs the column names.

    Leaving `columns` out selects every numeric column, which is what an
    imputer wants; a scaler names its own so different ones can be tried.

    Only transformers that preserve the number of columns fit this shape;
    a one-hot encoder needs its own step.
    """
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
