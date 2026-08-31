import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from dialysisml.adapters.ModelAdapter import ModelAdapter, metric_name
from dialysisml.metrics import Metric, mae

logger = logging.getLogger(__name__)


@dataclass
class EstimatorAdapter(ModelAdapter):
    """Predicts one constant fitted on y_train: the floor every model must clear."""

    estimator: Callable[[np.ndarray], float] = np.median
    # no loss to optimise, so what it is judged by is stated instead of derived
    score_metric: Metric = mae

    def __str__(self) -> str:
        # metric_name despite the name: an estimator is a partial too
        return f"Const_{metric_name(self.estimator)}"

    def train(
        self,
        metrics: list[Metric],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        # y_train only: fitting the constant on the test side would be leakage
        constant = float(self.estimator(y_train))
        logger.info("costante predetta: %.3f", constant)

        # The targets arrive as (N, 1) while the prediction is a scalar: without
        # flattening, the subtraction inside a metric broadcasts to (N, N) and
        # the mean comes out over a matrix -- wrong, but plausible enough to miss.
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1)

        record = self.compute_metrics_record(
            metrics=metrics,
            iteration=1,
            y_train=y_train_t,
            preds_train=torch.full_like(y_train_t, constant),
            y_test=y_test_t,
            preds_test=torch.full_like(y_test_t, constant),
        )
        # A single row: there is nothing to iterate. The rest of the pipeline
        # reads an epoch frame either way, so nothing downstream special-cases it.
        return pd.DataFrame([record])
