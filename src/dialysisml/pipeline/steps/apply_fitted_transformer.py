from typing import Any, Sequence

import pandas as pd

from dialysisml.features import META_COLUMNS, TARGET_COLUMNS
from dialysisml.pipeline.steps.flag_missing_values import MISSING_SUFFIX


def default_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column that is a feature, which is not every numeric one.

    `columns=None` exists so a new column from the view is scaled without
    touching a config, which makes what it must leave out worth stating: the
    targets, numeric like everything else and scaled they would change what the
    metrics mean, and the missing flags, whose rare 1s a winsorizer flattens to
    a constant. The meta columns survive on their type alone, but naming them
    keeps that from being an accident.
    """
    excluded = {*META_COLUMNS, *TARGET_COLUMNS}
    return [
        column
        for column in df.select_dtypes("number").columns
        if column not in excluded and not column.endswith(MISSING_SUFFIX)
    ]


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
    # an explicit list stays explicit: naming a target there is a deliberate
    # choice, and only the default has to protect it
    columns = default_columns(train) if columns is None else list(columns)

    missing = set(columns) - set(train.columns)
    if missing:
        raise ValueError(f"columns not in the data: {sorted(missing)}")

    fitted = transformer.fit(train[columns])

    train, test = train.copy(), test.copy()
    train[columns] = fitted.transform(train[columns])
    test[columns] = fitted.transform(test[columns])

    return train, test
