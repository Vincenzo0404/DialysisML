import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dialysisml.adapters.ModelAdapter import (
    ModelAdapter,
    ResultSchema,
    Split,
    TrainingResult,
)
from dialysisml.metrics import Metric, mae, mape, metric_name
from dialysisml.models import FeedForwardNN, Normalized

logger = logging.getLogger(__file__)


@dataclass
class FFNNAdapter(ModelAdapter):
    """A feed-forward network, one hidden layer per entry in `hidden_layers`.

    A dataclass so the fields keep the types and defaults a config object used
    to give them, without a second class to declare them in.
    """

    metrics: list[Metric] = field(default_factory=lambda: [mape, mae])
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 40
    random_seed: int = 42
    loss_function: Metric = mape

    def __str__(self) -> str:
        return f"FFNN_hl={list(self.hidden_layers)}"

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[Any, TrainingResult]:
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive. {self.epochs} was given.")

        # sets the seed to initialize the model
        torch.manual_seed(self.random_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1).to(device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # initialize model. The standardisation lives inside it, fitted on the
        # training rows: the saved model then takes raw features, and SHAP
        # attributes to those rather than to z-scores.
        input_dim = X_train.shape[1]
        model = Normalized(
            FeedForwardNN(
                input_dim=input_dim,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
            ),
            mean=X_train.mean(axis=0),
            std=X_train.std(axis=0),
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        patience = 15
        best_loss = float("inf")
        best_epoch = 1
        # a copy, not a reference: the weights keep changing after this epoch
        best_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
        # training loop
        records: list[dict] = []
        epoch = 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                # Squeeze preds from (batch_size, 1) to (batch_size,) to avoid broadcasting
                batch_preds = model(batch_X).squeeze(-1)
                assert (
                    batch_preds.shape == batch_y.shape
                ), f"Batch shapes differ: preds {batch_preds.shape}, true {batch_y.shape}"
                loss = self.loss_function(batch_preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            train_loss /= X_train_t.size(0)

            model.eval()
            with torch.no_grad():
                # Recompute predictions to evaluate the model
                train_preds_t: torch.Tensor = model(X_train_t)
                test_preds_t: torch.Tensor = model(X_test_t)

                train_preds_t = train_preds_t.squeeze(-1)
                test_preds_t = test_preds_t.squeeze(-1)

                # each side against its own target: the two have different lengths
                assert (
                    train_preds_t.shape == y_train_t.shape
                ), f"Train shapes differ: preds {train_preds_t.shape}, true {y_train_t.shape}"
                assert (
                    test_preds_t.shape == y_test_t.shape
                ), f"Test shapes differ: preds {test_preds_t.shape}, true {y_test_t.shape}"
                # Compute loss on test set
                test_loss = float(self.loss_function(test_preds_t, y_test_t))

            # one row per metric per side
            sides = {
                Split.TRAIN: (train_preds_t, y_train_t),
                Split.VAL: (test_preds_t, y_test_t),
            }
            for split, (preds, y) in sides.items():
                for metric in self.metrics:
                    records.append(
                        {
                            "iteration": epoch - 1,
                            "split": split,
                            "metric": metric_name(metric),
                            "value": float(metric(preds, y)),
                        }
                    )

            # stops after `patience` epochs that beat no minimum
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        # the returned model is the one `best_iteration` and the reported
        # `best_*` metrics describe, not whatever the last epoch left behind
        model.load_state_dict(best_state)

        return model, TrainingResult(
            history=ResultSchema.validate(pd.DataFrame(records)),
            best_iteration=best_epoch - 1,
            total_iterations=epoch,
        )

    def log_model(self, model: Any, name: str = "model") -> None:
        # `pickle`, not the default `pt2`: torch.export returns an
        # ExportedProgram, which is not an nn.Module and which SHAP cannot hook.
        # `.cpu()` so the artifact loads on a machine without a GPU too --
        # training is over by now, so moving it back costs nothing.
        mlflow.pytorch.log_model(model.cpu(), name=name, serialization_format="pickle")
