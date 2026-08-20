from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from dialysisml.metrics import Metric
from dialysisml.modelconf import BaseModelConfig


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
            record[f"train_{metric.__name__}"] = float(metric(preds_train, y_train))

        # Compute additional metrics for testing data
        for metric in metrics:
            record[f"test_{metric.__name__}"] = float(metric(preds_test, y_test))

        return record
