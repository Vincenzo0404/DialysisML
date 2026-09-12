from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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


class ModelAdapter[T](ABC):
    """Wraps an ML model to have a unified API over different libraries."""

    _model: T | None = None

    @property
    def model(self) -> T:
        if self._model is None:
            raise ValueError(f"model was not assigned.")
        return self._model

    @model.setter
    def model(self, value: T) -> None:
        self._model = value

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingResult:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
