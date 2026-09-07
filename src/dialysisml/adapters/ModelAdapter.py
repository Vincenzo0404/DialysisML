from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"


class ResultSchema(pa.DataFrameModel):
    """Schema of dataframe returned by train method."""

    iteration: Series[int] = pa.Field(ge=0)
    split: Series[str] = pa.Field(isin=list(Split))
    metric: Series[str]
    value: Series[float]

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["iteration", "split", "metric"]


@dataclass(frozen=True)
class TrainingResult:
    history: DataFrame[ResultSchema]
    best_iteration: int
    total_iterations: int

    def pivot(self, by: Sequence[str] = ("split", "metric")) -> pd.DataFrame:
        """Pivots history returning it's wide form, one column per `by` combination."""
        history = ResultSchema.validate(self.history)
        wide = history.pivot(columns=list(by), index="iteration", values="value")
        # flat names: a MultiIndex is not a metric name MLflow can take
        wide.columns = ["_".join(column) for column in wide.columns]
        return wide


class ModelAdapter(ABC):
    """Wraps an ML model, used to have a unified API over different libraries."""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[Any, TrainingResult]:
        """The model at its `best_iteration`, and what happened along the way.

        Returned rather than stored on `self`: one adapter serves every fold,
        so a field would keep only the last one.
        """

    @abstractmethod
    def log_model(self, model: Any, name: str = "model") -> None:
        """Attaches the model to the active MLflow run.

        Abstract because there is no shared way to serialise these: torch,
        xgboost and sklearn each have their own, and MLflow a flavor for each.
        """

    @abstractmethod
    def __str__(self) -> str:
        pass
