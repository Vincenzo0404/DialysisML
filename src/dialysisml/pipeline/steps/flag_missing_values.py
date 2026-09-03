from typing import Sequence

import pandas as pd

# What names a flag rather than a measurement. Declared here because this is
# what produces them, and read where a step has to leave them alone.
MISSING_SUFFIX = "_missing"


def flag_missing_values(df: pd.DataFrame, *, columns: Sequence[str]) -> pd.DataFrame:
    """Adds a `{col}_missing` flag for every column in `columns`.

    Row-wise and stateless: whether a row's own value is missing never
    depends on any other row, so this is safe to compute once here rather
    than separately per split.

    Run after sample_hold, this flags a feature never measured for that
    patient at all — not merely absent on this particular session, which
    sample_hold already recovers from a nearby one.
    """
    flags = df[list(columns)].isna().astype(int).add_suffix(MISSING_SUFFIX)
    return pd.concat([df, flags], axis=1)
