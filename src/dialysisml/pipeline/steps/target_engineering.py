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

    return frame.update(data, add=columns(int, Role.META, Kind.NUMERIC, [into]))


def build_buckets(
    frame: Frame, *, thresholds: tuple[int, ...], tte_col_name: str = "tte"
) -> Frame:
    """One column per interval: did the event fall inside it, as far as we know."""
    if any(t <= 0 for t in thresholds):
        raise ValueError(f"Thresholds must be non-negative, got: {thresholds}")

    if list(thresholds) != sorted(set(thresholds)):
        raise ValueError(f"Thresholds must be strictly increasing, got: {thresholds}")

    data = frame.data.copy()
    tte, observed = data[tte_col_name], ~data["censored"]

    names = []
    for lower, upper in zip((0, *thresholds), thresholds):
        # `tte` counts whole days, so `(lower, upper]` is days lower+1..upper
        name = f"event_in_d{lower + 1}_{upper}"
        # default to NaN (to handle censored)
        data[name] = np.nan
        # mark TTE NOT happening in interval
        data.loc[tte >= upper, name] = 0.0
        # mark TTE happening in interval
        data.loc[observed & (tte > lower) & (tte <= upper), name] = 1.0
        names.append(name)

    return frame.update(
        data, add=columns(float, Role.LABEL, Kind.BINARY, names, nullable=True)
    )
