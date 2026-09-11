from typing import Sequence

from dialysisml.schema import Frame, Kind, Role, select


def drop_incomplete_rows(
    frame: Frame, *, columns: Sequence[str] | None = None
) -> Frame:
    """Drops rows still missing a value after sample_hold.

    The default -- every numeric feature -- keeps 0.6% of these rows, so a
    panel that has to be present is normally named instead.
    """
    if columns is None:
        columns = select(frame.schema, role=Role.FEATURE, kind=Kind.NUMERIC)
    return frame.update(frame.data.dropna(subset=list(columns)))


def drop_sessions_near_event(frame: Frame, *, min_days: int = 1) -> Frame:
    """Drops sessions whose event is less than `min_days` away.

    `min_days=1` drops the sessions held on the event day itself.
    """
    return frame.update(frame.data[frame.data["tte"] >= min_days])


def drop_early_censored_sessions(frame: Frame, *, horizon_days: int = 365) -> Frame:
    """Keeps only the sessions whose outcome within the horizon is known."""
    data = frame.data
    return frame.update(data[~data["censored"] | (data["tte"] >= horizon_days)])
