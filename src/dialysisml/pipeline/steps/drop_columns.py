from typing import Sequence

from dialysisml.schema import Frame


def drop_columns(frame: Frame, *, columns: Sequence[str] = ()) -> Frame:
    """Drops the listed columns and keeps everything else."""
    return frame.update(frame.data.drop(columns=list(columns)))
