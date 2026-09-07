from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from dialysisml.adapters import ModelAdapter
from dialysisml.metrics import Metric


@dataclass
class RSFAdapter(ModelAdapter):
    """A random survival forest, judged by the concordance index.

    The only adapter that reads two targets: a censored session carries the
    follow-up as `tte` and a True `censored`, and the forest is the model
    that can tell the two apart. Set
    `window_conf.target_columns: [censored, tte]`, or every row will say the
    same thing.
    """

    n_estimators: int = 1000
    min_samples_split: int = 10
    min_samples_leaf: int = 15
    n_jobs: int = 5
    seed: int = 42

    def __str__(self) -> str:
        return f"RSF_{self.n_estimators}trees"

    @property
    def score_metric_name(self) -> str:
        return "c_index"

    @staticmethod
    def _to_survival(y: np.ndarray) -> np.ndarray:
        """`(n, 2)` of (censored, tte) to the structured array sksurv reads.

        The order is the one `target_columns` declares, so getting it backwards
        is a silent mistake: the check below is what makes it loud.
        """
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(
                "a survival model needs two targets: set "
                "window_conf.target_columns to [censored, tte], got an array "
                f"of shape {y.shape}"
            )
        return Surv.from_arrays(event=y[:, 0].astype(bool), time=y[:, 1].astype(float))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        y_train, y_test = self._to_survival(y_train), self._to_survival(y_test)

        model = RandomSurvivalForest(
            self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )

        model.fit(X_train, y_train)

        # one row rather than one per epoch — the forest is fitted in a single
        # shot — but with the columns main.py logs and reports on
        return pd.DataFrame(
            [
                {
                    "epoch": 0,
                    "train_c_index": model.score(X_train, y_train),
                    "test_c_index": model.score(X_test, y_test),
                }
            ]
        )
