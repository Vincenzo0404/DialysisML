import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from dialysisml.adapters.ModelAdapter import ModelAdapter
from dialysisml.metrics import Metric, mae, metric_name

logger = logging.getLogger(__name__)

# TODO fix later


@dataclass
class EstimatorAdapter(ModelAdapter):
    """Predicts one constant fitted on y_train: the floor every model must clear."""

    estimator: Callable[[np.ndarray], float] = np.median
    # no loss to optimise, so what it is judged by is stated instead of derived

    @property
    def score_metric_name(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Const_{self.estimator.__name__}"
