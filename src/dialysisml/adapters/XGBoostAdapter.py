import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from dialysisml.adapters.ModelAdapter import ModelAdapter
from dialysisml.metrics import Metric, mae

logger = logging.getLogger(__name__)


@dataclass
class XGBoostAdapter(ModelAdapter):
    """Gradient-boosted trees over the same flattened window features as FFNNAdapter."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    # minimum summed hessian a leaf needs to keep splitting: with ~900 train
    # patients behind ~100k correlated windows, a low value lets a split
    # isolate one patient's quirk instead of a pattern several share
    min_child_weight: float = 1.0
    # stops boosting once the held-out side stalls, so `n_estimators` only
    # has to be an upper bound rather than a value to tune
    early_stopping_rounds: int = 20
    random_seed: int = 42
    # xgboost's own training objective; not one of dialysisml.metrics, so unlike
    # FFNNAdapter's loss_function it cannot double as what the run is judged by
    objective: str = "reg:absoluteerror"
    # what it is judged by is stated instead of derived, same as EstimatorAdapter
    score_metric: Metric = mae

    @property
    def score_metric_name(self) -> str:
        return metric_name(self.score_metric)

    def __str__(self) -> str:
        return f"XGB_d={self.max_depth}_n={self.n_estimators}"

    def train(
        self,
        metrics: list[Metric],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            objective=self.objective,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # best_iteration, not n_estimators: early stopping keeps every tree it
        # grew, so scoring past it would judge the model on an overfit tail
        best_round = model.best_iteration
        preds_train = model.predict(X_train, iteration_range=(0, best_round + 1))
        preds_test = model.predict(X_test, iteration_range=(0, best_round + 1))

        record = self.compute_metrics_record(
            metrics=metrics,
            iteration=best_round,
            y_train=torch.tensor(y_train, dtype=torch.float32),
            preds_train=torch.tensor(preds_train, dtype=torch.float32),
            y_test=torch.tensor(y_test, dtype=torch.float32),
            preds_test=torch.tensor(preds_test, dtype=torch.float32),
            score_metric=self.score_metric,
        )
        # a single row: boosting has no notion of "epoch" the rest of the
        # pipeline understands, so it is reported the way EstimatorAdapter's
        # one-shot fit is -- at its best round rather than at every round
        return pd.DataFrame([record])
