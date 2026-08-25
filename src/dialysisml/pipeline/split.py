"""Patient-grouped splitting, the second macro step.

Sessions of one patient are highly correlated, so a patient appearing in both
train and test would be memorised rather than predicted. Splitting the rows of
the DataFrame — rather than the windows built later — also keeps the split
independent of the window size, so results across sizes stay comparable.
"""

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


class PatientGroupSplit:
    """Yields (train_idx, test_idx) over the rows of the DataFrame.

    k=1 gives a single train/test split, k>1 a GroupKFold. Both keep every
    session of a patient on the same side.
    """

    def __init__(self, k: int = 1, seed: int = 42, train_size: float = 0.6):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not 0.0 < train_size < 1.0:
            raise ValueError(f"train_size must be in (0, 1), got {train_size}")

        self.k = k
        self.seed = seed
        self.train_size = train_size

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = df["patient"].to_numpy()

        if self.k == 1:
            splitter = GroupShuffleSplit(
                n_splits=1, train_size=self.train_size, random_state=self.seed
            )
        else:
            splitter = GroupKFold(n_splits=self.k)

        yield from splitter.split(groups, groups=groups)

    def __len__(self) -> int:
        return self.k
