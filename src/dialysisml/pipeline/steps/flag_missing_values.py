from dialysisml.schema import Frame, Kind, Role, columns, select

MISSING_SUFFIX = "_missing"


def flag_missing_values(frame: Frame) -> Frame:
    """Adds a `{col}_missing` flag for every numeric feature."""
    data = frame.data.copy()
    flags = []

    for column in select(frame.schema, role=Role.FEATURE, kind=Kind.NUMERIC):
        flag = f"{column}{MISSING_SUFFIX}"
        data[flag] = data[column].isna().astype(float)
        flags.append(flag)

    return frame.update(data, add=columns(float, Role.FEATURE, Kind.BINARY, flags))
