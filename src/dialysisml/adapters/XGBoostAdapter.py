import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import pyfunc
from sklearn.metrics import roc_auc_score

from dialysisml.adapters.ModelAdapter import (
    ModelAdapter,
    ResultSchema,
    Split,
    TrainingResult,
)

logger = logging.getLogger(__name__)


def _predict(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """The positive class probability, from the trees up to `best_iteration`."""
    # get best iteration from training
    best = getattr(model, "best_iteration", None)
    # pick best tree as the last tree to use to predict probabilities
    end = None if best is None else (0, best + 1)
    # return positive class predicted probabilities from the model (contributions are made up to the best tree)
    return model.predict_proba(X, iteration_range=end)[:, 1]


class BucketClassifiers(pyfunc.PythonModel):
    """The run's model: a hazard curve, computed by one booster per bucket.

    One artifact rather than K, because a single hazard means little on its own
    -- the survival through bucket `k` is the running product of `1 - hazard`,
    so what the run predicts is the whole row. The boosters stay reachable for
    a TreeExplainer, which can only take one of them at a time.
    """

    def __init__(self, models: list[xgb.XGBClassifier], example: np.ndarray):
        self.models = models
        # a handful of rows, so the logged model carries a schema
        self.example = example

    @property
    def names(self) -> list[str]:
        """`b1..bK`: the adapter sees arrays, so the column names it would
        rather use never reach it. The index is the bucket's own order."""
        return [f"b{i}" for i in range(1, len(self.models) + 1)]

    def hazards(self, X: np.ndarray) -> np.ndarray:
        """`(rows, buckets)`: each bucket's probability of the event inside it."""
        return np.column_stack([_predict(model, X) for model in self.models])

    # unannotated on purpose: mlflow reads the hints to infer a signature,
    # and warns on every one it cannot turn into a schema
    def predict(self, context, model_input, params=None):
        return self.hazards(np.asarray(model_input))


@dataclass
class XGBoostAdapter(ModelAdapter):
    """One binary booster per bucket of a discrete-time hazard formulation."""

    # maximum number of trees who partecipate in the ensemble
    n_estimators: int = 300
    # maximum number of splits for each tree
    max_depth: int = 4
    # weight assigned to each tree contribution
    learning_rate: float = 0.05
    # fraction of number of rows in the original dataset used to train each treee
    subsample: float = 0.8
    # fraction of columns used to train each the tree
    colsample_bytree: float = 0.8
    # L2 regularization
    reg_lambda: float = 1.0
    min_child_weight: float = 50.0
    # stops training when `early_stopping_rounds` are run without improving `eval_metric`
    early_stopping_rounds: int = 30
    random_seed: int = 42
    # loss function to minimize
    objective: str = "binary:logistic"
    # metric by which the best model is selected
    eval_metric: str = "logloss"

    def __str__(self) -> str:
        return f"XGBBuckets_d={self.max_depth}_mcw={self.min_child_weight:g}"

    def _classifier(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
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

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[Any, TrainingResult]:
        if y_train.ndim != 2:
            raise ValueError(
                "this adapter predicts one bucket per target column, got an "
                f"array of shape {y_train.shape}"
            )

        models: list[xgb.XGBClassifier] = []
        records: list[dict] = []

        # for each bucket
        for k in range(y_train.shape[1]):
            # Group splits in dict marking missing targets (possible NaNs for censoring)
            train_known = ~np.isnan(y_train[:, k])
            val_known = ~np.isnan(y_test[:, k])
            sides = {
                Split.TRAIN: (X_train, y_train, train_known),
                Split.VAL: (X_test, y_test, val_known),
            }
            # for each split check both classes appear at least once in the k-th bucket
            for split, (X, y, known) in sides.items():
                if len(np.unique(y[known, k])) < 2:
                    raise ValueError(
                        f"bucket {k + 1} has a single class on the {split} side "
                        f"({int(known.sum())} usable rows): widen the bucket or "
                        "shorten the horizon"
                    )

            model = self._classifier()
            # train k-th classifier filtering NaNs in the k-th bucket
            model.fit(
                X_train[train_known],
                y_train[train_known, k],  # binary classifier only on k-th bucket
                eval_set=[(X_test[val_known], y_test[val_known, k])],
                verbose=False,
            )
            models.append(model)
            logger.info(
                "bucket %d: %d/%d usable rows, %d trees",
                k + 1,
                int(train_known.sum()),
                len(y_train),
                model.best_iteration + 1,
            )

            # compute metrics per split
            for split, (X, y, known) in sides.items():
                records.append(
                    {
                        "iteration": 0,
                        "split": split,
                        "metric": f"roc_auc_b{k + 1}",
                        "value": float(
                            roc_auc_score(y[known, k], _predict(model, X[known]))
                        ),
                    }
                )

        # report mean auc
        for split in (Split.TRAIN, Split.VAL):
            buckets = [r["value"] for r in records if r["split"] == split]
            records.append(
                {
                    "iteration": 0,
                    "split": split,
                    "metric": "roc_auc_mean",
                    "value": float(np.mean(buckets)),
                }
            )

        # one row per metric rather than one per round: the K boosters stop at
        # K different rounds, so there is no shared iteration axis to report on
        return BucketClassifiers(models, X_test[:5]), TrainingResult(
            history=ResultSchema.validate(pd.DataFrame(records)),
            best_iteration=0,
            total_iterations=1,
        )

    def log_model(self, model: Any, name: str = "model") -> None:
        # pyfunc, not the xgboost flavor: what is being logged is the curve
        # over every bucket, and no single booster is that
        pyfunc.log_model(name=name, python_model=model, input_example=model.example)
