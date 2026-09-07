import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dialysisml.adapters.ModelAdapter import ModelAdapter
from dialysisml.metrics import Metric, mae
from dialysisml.models import LSTM

# TODO fix api later

logger = logging.getLogger(__file__)


@dataclass
class LSTMAdapter(ModelAdapter):
    """An LSTM over the raw window `(n, size, n_features)`.

    Unlike FFNNAdapter it is not handed a vector of derived features: it wants
    the sequence, so the configured window transformer must keep the time axis.
    """

    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 40
    random_seed: int = 42
    # doubles as the loss: unlike the FFNN there is no separate one to derive from
    score_metric: Metric = mae

    @property
    def score_metric_name(self) -> str:
        return metric_name(self.score_metric)

    def __str__(self) -> str:
        return f"LSTM_h={self.hidden_size}_n={self.num_layers}"

    def _scale(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardises per feature, fitted on the training windows.

        StandardScaler only takes a matrix, so the timesteps are stacked as
        rows and the shape restored afterwards: one mean and one scale per
        feature, shared by every position in the window.
        """
        n_features = X_train.shape[-1]
        scaler = StandardScaler()

        train = scaler.fit_transform(X_train.reshape(-1, n_features))
        test = scaler.transform(X_test.reshape(-1, n_features))

        return train.reshape(X_train.shape), test.reshape(X_test.shape)

    @torch.no_grad()
    def _predict(self, model: LSTM, X: torch.Tensor) -> torch.Tensor:
        """Predictions in batches, and only the predictions kept.

        A forward pass over a whole split would hold the recurrent state of
        every timestep at once, which the flat input of the FFNN never does.
        """
        model.eval()
        return torch.cat(
            [
                model(X[i : i + self.batch_size])
                for i in range(0, len(X), self.batch_size)
            ]
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        # a flattening window transformer would otherwise fail deep inside
        # nn.LSTM, where the shape no longer says what went wrong
        if X_train.ndim != 3:
            raise ValueError(
                f"{self} needs windows (n, size, n_features), got {X_train.shape}: "
                "the configured window transformation flattens the time axis"
            )

        # sets the seed which to initialize the model
        torch.manual_seed(self.random_seed)

        X_train_scaled, X_test_scaled = self._scale(X_train, X_test)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device} to train {str(self)}.")

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1).to(device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = LSTM(
            input_dim=X_train_scaled.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        patience = 15
        min_delta = 0.01
        best_test_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        records = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                batch_preds = model(batch_X)
                loss = self.score_metric(batch_preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            train_loss /= X_train_t.size(0)

            train_preds_t = self._predict(model, X_train_t)
            test_preds_t = self._predict(model, X_test_t)
            test_loss = float(self.score_metric(test_preds_t, y_test_t))

            # L'early stopping segue la loss di TEST: quella di training cala
            # anche mentre il modello va in overfitting, quindi la patience su
            # di essa non scatta mai quando la generalizzazione si e' fermata.
            if test_loss < best_test_loss * (1 - min_delta):
                best_test_loss = test_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            record = self.compute_metrics_record(
                iteration=epoch,
                y_train=y_train_t,
                preds_train=train_preds_t,
                y_test=y_test_t,
                preds_test=test_preds_t,
                metrics=metrics,
                score_metric=self.score_metric,
            )

            records.append(record)

            if epochs_without_improvement >= patience:
                break

        df = pd.DataFrame(records)
        df.attrs["best_test_loss"] = best_test_loss
        df.attrs["best_epoch"] = best_epoch
        return df
