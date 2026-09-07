from dialysisml.schema import Frame, Role, select


def sample_hold(frame: Frame) -> Frame:
    """Carries each feature forward, then backward, within a patient."""
    data = frame.data.copy()
    features = select(frame.schema, role=Role.FEATURE)

    data[features] = data.groupby("patient")[features].ffill()
    data[features] = data.groupby("patient")[features].bfill()

    return frame.update(data)
