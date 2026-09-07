from sklearn.preprocessing import OneHotEncoder

from dialysisml.schema import Frame, Kind, Role, columns, select


def one_hot_encode(frame: Frame, *, drop_missing: bool = True) -> Frame:
    """Replaces every categorical feature with its one-hot encoding."""
    categorical = select(frame.schema, role=Role.FEATURE, kind=Kind.CATEGORICAL)
    data = frame.data

    if drop_missing:
        data = data.dropna(subset=categorical)
    elif data[categorical].isna().any().any():
        raise ValueError(f"missing values in {categorical}: pass drop_missing=True")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(data[categorical])
    names = list(encoder.get_feature_names_out(categorical))

    data = data.drop(columns=categorical)
    data[names] = encoded

    return frame.update(data, add=columns(float, Role.FEATURE, Kind.BINARY, names))
