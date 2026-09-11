import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from mlflow import xgboost as mlflow_xgboost

from dialysisml.adapters.ModelAdapter import (
    ModelAdapter,
    ResultSchema,
    Split,
    TrainingResult,
)
from dialysisml.metrics import Metric, mae, mape, metric_name

logger = logging.getLogger(__name__)


@dataclass
class XGBRegressorAdapter(ModelAdapter):
    """Gradient-boosted trees over the same window features FFNNAdapter takes.

    One target column, so this is the regression counterpart of the per-bucket
    classifier: it exists to say whether that formulation's poor showing came
    from the target or from the loss the network was given.
    """

    n_estimators: int = 800
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    min_child_weight: float = 50.0
    early_stopping_rounds: int = 30
    random_seed: int = 42
    # `reg:gamma` predicts through a log link, so it cannot return a negative
    # time, and its deviance depends on `y/pred` alone -- a relative error
    objective: str = "reg:squaredlogerror"
    eval_metric: str = "mape"
    # `1/y**2` under a squared loss is exactly the squared percentage error.
    # Only meaningful with `reg:squarederror`: the other objectives are
    # already relative, and weighting them again would double the tilt.
    relative_weights: bool = False
    metrics: list[Metric] = field(default_factory=lambda: [mape, mae])

    def __str__(self) -> str:
        return f"XGBReg_{self.objective.partition(':')[2]}_d={self.max_depth}"

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[Any, TrainingResult]:
        if y_train.ndim == 2 and y_train.shape[1] != 1:
            raise ValueError(
                f"this adapter predicts one target, got {y_train.shape[1]}: "
                "narrow the formulation's labels or set target_columns"
            )
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            objective=self.objective,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=1.0 / y_train**2 if self.relative_weights else None,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        best = model.best_iteration
        preds = {
            Split.TRAIN: (model.predict(X_train, iteration_range=(0, best + 1)), y_train),
            Split.VAL: (model.predict(X_test, iteration_range=(0, best + 1)), y_test),
        }

        records: list[dict] = []
        for split, (prediction, truth) in preds.items():
            for metric in self.metrics:
                records.append(
                    {
                        "iteration": 0,
                        "split": split,
                        "metric": metric_name(metric),
                        "value": float(
                            metric(
                                torch.tensor(prediction, dtype=torch.float32),
                                torch.tensor(truth, dtype=torch.float32),
                            )
                        ),
                    }
                )
            # nothing bounds a squared-error prediction below: reported rather
            # than clipped, since a negative time is a fact about the fit
            records.append(
                {
                    "iteration": 0,
                    "split": split,
                    "metric": "pred_min",
                    "value": float(prediction.min()),
                }
            )

        logger.info("%s: %d trees, val pred range [%.2f, %.2f]", self,
                    best + 1, preds[Split.VAL][0].min(), preds[Split.VAL][0].max())

        return model, TrainingResult(
            history=ResultSchema.validate(pd.DataFrame(records)),
            best_iteration=0,
            total_iterations=1,
        )

    def log_model(self, model: Any, name: str = "model") -> None:
        mlflow_xgboost.log_model(model, name=name)
