import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator, Self, Sequence

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series

from omegaconf import OmegaConf

import dialysisml.metrics as metrics


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
    _registry: dict[str, type["ModelAdapter"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelAdapter._registry[cls.__name__] = cls

    @classmethod
    def resolve(cls, name: str) -> type["ModelAdapter"]:
        """The adapter class registered under `name`.

        The class rather than an instance: `load` is a classmethod, so rebuilding
        a saved adapter needs the type, and going through `create` would build an
        instance only to throw it away.
        """
        if name not in cls._registry.keys():
            known = ", ".join(sorted(cls._registry)) or "none"
            raise ValueError(f"{name} is not a known ModelAdapter. Known: {known}.")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """Instantiates an adapter based on it's class name."""
        return cls.resolve(name)(*args, **kwargs)

    @property
    def model(self) -> T:
        if self._model is None:
            raise ValueError(f"model was not assigned.")
        return self._model

    @model.setter
    def model(self, value: T) -> None:
        self._model = value

    def dump_params(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            # plain (non-`_target_`) config fields, e.g. hidden_layers, stay
            # as OmegaConf containers rather than native list/tuple/dict
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            if callable(value):
                # Hydra `_partial_: true` targets arrive wrapped in a
                # functools.partial (sub)class, not the bare function.
                fn = getattr(value, "func", value)
                return {"$fn": fn.__name__}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value

        return {f.name: encode(getattr(self, f.name)) for f in fields(self)}

    def load_params(self, params: dict[str, Any]) -> None:
        def decode(value: Any) -> Any:
            if isinstance(value, dict) and "$fn" in value:
                return getattr(metrics, value["$fn"])
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value

        for f in fields(self):
            if f.name in params:
                setattr(self, f.name, decode(params[f.name]))

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

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


if __name__ == "__main__":
    reg = ModelAdapter._registry
    print(reg)
