from typing import Any, Sequence

from dialysisml.schema import Frame, Kind, Role, select


def apply_sklearn_transformer(
    train: Frame,
    test: Frame,
    *,
    transformer: Any,
    columns: Sequence[str] | None = None,
) -> tuple[Frame, Frame]:
    """Fits a sklearn transformer on train and applies it on both sides.

    Defaults to the numeric features: the binary ones are flags and one-hot
    columns, which a winsorizer would flatten and a scaler only shifts.
    """
    if columns is None:
        columns = select(train.schema, role=Role.FEATURE, kind=Kind.NUMERIC)
    columns = list(columns)

    fitted = transformer.fit(train.data[columns])

    train_data = train.data.copy()
    train_data[columns] = fitted.transform(train_data[columns])

    test_data = test.data.copy()
    test_data[columns] = fitted.transform(test_data[columns])

    return train.update(train_data), test.update(test_data)
