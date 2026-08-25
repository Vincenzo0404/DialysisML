from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from dialysisml.metrics import Metric
from dialysisml.modelconf import BaseModelConfig


def metric_name(metric: Metric) -> str:
    """Name of a metric, whether a plain function or a functools.partial.

    Hydra builds metrics with `_partial_: true`, and a partial has no
    __name__: without this the metric columns would be unusable.
    """
    return getattr(metric, "__name__", None) or metric.func.__name__  # type: ignore[attr-defined]


class ModelAdapter(ABC):
    def __init__(self, model_config: BaseModelConfig):
        self.model_config = model_config

    @abstractmethod
    def train(
        self,
        metrics: list[Metric],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        pass

    def compute_metrics_record(
        self,
        metrics: list[Metric],
        iteration: int,
        y_train: torch.Tensor,
        preds_train: torch.Tensor,
        y_test: torch.Tensor,
        preds_test: torch.Tensor,
    ) -> dict:
        """
        Computes metrics for both training and testing datasets and returns a record dictionary.
        """
        record: dict[str, float] = {"epoch": iteration}

        # Compute additional metrics for training data
        for metric in metrics:
            record[f"train_{metric_name(metric)}"] = float(metric(preds_train, y_train))

        # Compute additional metrics for testing data
        for metric in metrics:
            record[f"test_{metric_name(metric)}"] = float(metric(preds_test, y_test))

        return record
