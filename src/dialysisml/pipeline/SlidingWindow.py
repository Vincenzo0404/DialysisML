import logging
from typing import Sequence

import numpy as np
import pandas as pd

from dialysisml.schema import Frame, Role, select

# What makes a series: windows are cut inside one, never across two.
GROUP_COLUMNS = ["patient", "t_event"]
TIME_COLUMN = "t_session"

logger = logging.getLogger(__name__)


class SlidingWindow:
    """Windows of a fixed span in days, one per session that has the history.

    A session anchors a window when its series reaches back at least `days`
    before it — not `days` worth of sessions, one session older than that. The
    window holds the sessions in `(t - days, t]`, so it varies in length with
    how often the patient came.

    Nothing is materialised: the windows are two arrays of row positions into
    a frame sorted by series and date, which is what lets a transformer sum
    over them without copying anything.
    """

    def __init__(self, frame: Frame, *, days: int = 30, min_sessions: int = 2):
        if days < 1:
            raise ValueError(f"days must be at least 1, got {days}")

        self.days = days
        self.schema = frame.schema
        # a series has to occupy consecutive rows: a window is a range of
        # positions, so a series split in two would take in other patients
        self.df = frame.data.sort_values([*GROUP_COLUMNS, TIME_COLUMN]).reset_index(
            drop=True
        )

        day = self.df[TIME_COLUMN].to_numpy("datetime64[D]").astype("int64")
        # counted from the earliest session rather than from the epoch: the
        # fit subtracts nearly equal quantities, and smaller numbers there
        # lose fewer digits to the cancellation
        self.t = (day - day.min()).astype(np.float64)
        self.start, self.anchor = self._bounds(day, min_sessions)

    def _bounds(self, day: np.ndarray, min_sessions: int) -> tuple[np.ndarray, ...]:
        """First and last row of every window, as positions in `self.df`."""
        starts, anchors = [], []

        for _, group in self.df.groupby(GROUP_COLUMNS, sort=False):
            rows = group.index.to_numpy()
            t = day[rows]

            # first row still inside the window of each session
            first_inside = np.searchsorted(t, t - self.days, side="right")
            # an anchor needs a session older than the window, not merely
            # `days` worth of sessions: `first_inside > 0` says one exists
            eligible = np.flatnonzero(first_inside > 0)

            starts.append(rows[first_inside[eligible]])
            anchors.append(rows[eligible])

        start = np.concatenate(starts)
        anchor = np.concatenate(anchors)

        enough = anchor - start + 1 >= min_sessions
        if not enough.all():
            logger.info(
                "%d of %d windows dropped: fewer than %d sessions in %d days",
                (~enough).sum(),
                len(enough),
                min_sessions,
                self.days,
            )
        return start[enough], anchor[enough]

    def __len__(self) -> int:
        return len(self.anchor)

    def column_values(self, columns: Sequence[str]) -> np.ndarray:
        """`(rows, columns)` over the whole frame, for a transformer to sum."""
        return self.df[list(columns)].to_numpy(np.float64)

    def at_anchor(self, columns: Sequence[str]) -> np.ndarray:
        """`(windows, columns)`: the value each window ends on."""
        return self.df.loc[self.anchor, list(columns)].to_numpy()

    def targets(self, columns: Sequence[str]) -> np.ndarray:
        """The labels a run predicts, read at the session it predicts from."""
        labels = select(self.schema, role=Role.LABEL)
        not_labels = [c for c in columns if c not in labels]
        if not_labels:
            raise ValueError(f"targets that are not labels: {not_labels}")
        return self.at_anchor(columns)

    def meta(self) -> pd.DataFrame:
        """One row per window: whose it is, and when it ends."""
        return self.df.loc[
            self.anchor, select(self.schema, role=Role.META)
        ].reset_index(drop=True)
