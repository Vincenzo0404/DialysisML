from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from dialysisml.metrics import Metric


def metric_name(metric: Metric) -> str:
    """Name of a metric, whether a plain function or a functools.partial.

    Hydra builds metrics with `_partial_: true`, and a partial has no
    __name__: without this the metric columns would be unusable.
    """
    return getattr(metric, "__name__", None) or metric.func.__name__  # type: ignore[attr-defined]


class ModelAdapter(ABC):
    """Wrapper to different libraries models, used to have a unified API for training.

    No constructor: each adapter takes the hyperparameters it needs, named as it
    needs them.
    """

    @abstractmethod
    def train(
        self,
        metrics: list[Metric],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """One row per iteration: `epoch`, plus a column per metric per side."""

    @abstractmethod
    def __str__(self) -> str:
        pass

    # The measure this model is judged by, reported as `test_score_*`. Only the
    # name: an adapter with a loss derives it from there, one without states it.
    score_metric_name: str

    def compute_metrics_record(
        self,
        metrics: list[Metric],
        iteration: int,
        y_train: torch.Tensor,
        preds_train: torch.Tensor,
        y_test: torch.Tensor,
        preds_test: torch.Tensor,
        score_metric: Metric | None = None,
    ) -> dict:
        """Metrics for both sides, plus the score the model is judged by.

        `score_metric` is the callable behind `score_metric_name`, passed by the
        adapter that has one: the base knows the name only, and a name cannot be
        evaluated. Folded in by name, not identity, because Hydra builds it and
        the metrics as separate partials.
        """
        if score_metric is not None and metric_name(score_metric) not in map(
            metric_name, metrics
        ):
            metrics = [*metrics, score_metric]

        record: dict[str, float] = {"epoch": iteration}

        for metric in metrics:
            record[f"train_{metric_name(metric)}"] = float(metric(preds_train, y_train))
            record[f"test_{metric_name(metric)}"] = float(metric(preds_test, y_test))

        return record
