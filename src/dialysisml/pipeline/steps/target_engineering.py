import numpy as np

from dialysisml.schema import Frame, Kind, Role, columns


def scale_tte(
    frame: Frame, *, divisor: int = 30, source: str = "tte", into: str = "tte_months"
) -> Frame:
    """Which `divisor`-day interval the event falls in, counting from 1.

    Not a duration: `1` means "within `divisor` days". Never 0 -- the first
    interval is 1 -- which keeps a relative-error loss off a zero denominator,
    as long as `source` itself is positive.
    """
    if divisor <= 0:
        raise ValueError(f"`divisor` must be positive, got {divisor}.")

    data = frame.data.copy()
    data[into] = np.ceil(data[source] / divisor).astype(int)

    return frame.update(data, add=columns(int, Role.LABEL, Kind.NUMERIC, [into]))


def cap_tte(
    frame: Frame, *, cap: int = 365, source: str = "tte", into: str = "tte_capped"
) -> Frame:
    """`min(source, cap)` as a new column: the restricted time to event."""
    data = frame.data.copy()
    data[into] = data[source].clip(upper=cap)

    return frame.update(data, add=columns(int, Role.LABEL, Kind.NUMERIC, [into]))
