import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xgboost as xgb
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


@dataclass
class XGBoostAdapter(ModelAdapter[list[xgb.XGBClassifier]]):
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
    # used to convert probabilities to TTE
    prob_threshold: float = 0.5

    _hazards = None

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
    ) -> TrainingResult:
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

        self.model = models
        # one row per metric rather than one per round: the K boosters stop at
        # K different rounds, so there is no shared iteration axis to report on
        return TrainingResult(
            history=ResultSchema.validate(pd.DataFrame(records)),
            best_iteration=0,
            total_iterations=1,
        )

    def hazards(self, X: np.ndarray):
        """Calculates hazard values for each classifier.

        `Returns:`
            ndarray of shape (X.shape[0], n_classifiers)
        """
        classifiers = self.model
        hazards = []

        for classifier in classifiers:
            # [:, 1] contains hazard
            hazard = classifier.predict_proba(X)[:, 1]
            hazards.append(hazard)

        return np.stack(hazards, axis=1)

    def survival(self, X: np.ndarray, hazards: np.ndarray | None = None) -> np.ndarray:
        """Calculates survival probability at time t."""
        if hazards is None:
            hazards = self.hazards(X)
        hazards_inv = np.ones(hazards.shape, dtype=np.float32) - hazards

        # first bucket survival values
        survs = [hazards_inv[:, 0]]
        # for each other bucket
        for i in range(1, hazards.shape[1]):
            surv_i = survs[i - 1] * hazards_inv[:, i]
            survs.append(surv_i)

        return np.stack(survs, 1)

    def predict(
        self,
        X: np.ndarray,
        threshold: float | None = None,
        surv_probs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Transforms survival probabilities into forecasted bucket the event will fall in"""
        if threshold is None:
            threshold = self.prob_threshold
        if surv_probs is None:
            surv_probs = self.survival(X)

        s1, s2 = surv_probs.shape, X.shape
        assert s1[0] == s2[0], f"X has {s2[0]} rows, while surv_values has {s1[0]}."

        n_buckets = surv_probs.shape[1]
        below = surv_probs < threshold
        # first col index to have surv < threshold
        first_bucket = np.argmax(below, axis=1)
        # which rows are never below the threshold
        never_below = ~below.any(axis=1)
        # for those never below, forecasted bucket is considered over the last one
        first_bucket[never_below] = n_buckets
        first_bucket += 1

        return first_bucket

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        boosters = self.model
        assert (
            boosters is not None and len(boosters) != 0
        ), "There is no booster to save."

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("params.json", json.dumps(self.dump_params()))
            for i, model in enumerate(boosters):
                raw: bytearray = model.get_booster().save_raw(raw_format="ubj")
                zf.writestr(f"bucket_{i:02d}.ubj", bytes(raw))

    @classmethod
    def load(cls, path: Path) -> Self:
        boosters: list[xgb.XGBClassifier] = []
        instance = cls()
        with zipfile.ZipFile(path, mode="r") as zf:
            if "params.json" in zf.namelist():
                instance.load_params(json.loads(zf.read("params.json")))
            else:
                logger.warning(
                    f"params.json was not found in {path.resolve()} loading default params."
                )
            names = sorted(n for n in zf.namelist() if n.startswith("bucket_"))
            for name in names:
                raw: bytes = zf.read(name)
                # through the wrapper, not by assigning `_Booster`: predict_proba
                # reads `n_classes_`, which only load_model restores
                m = xgb.XGBClassifier()
                m.load_model(bytearray(raw))
                boosters.append(m)
        instance.model = boosters
        return instance


if __name__ == "__main__":
    from dialysisml.adapters import XGBoostAdapter

    print(type.__dict__)
