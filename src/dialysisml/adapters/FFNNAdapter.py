import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dialysisml.adapters import ModelAdapter
from dialysisml.metrics import Metric, mape
from dialysisml.models import FeedForwardNN

logger = logging.getLogger(__file__)


@dataclass
class FFNNAdapter(ModelAdapter):
    """A feed-forward network, one hidden layer per entry in `hidden_layers`.

    A dataclass so the fields keep the types and defaults a config object used
    to give them, without a second class to declare them in.
    """

    # Architettura a imbuto (es. 132 -> 128 -> 64 -> 32)
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 40
    random_seed: int = 42
    score_metric: Metric = mape

    def __str__(self) -> str:
        return f"FFNN_hl={list(self.hidden_layers)}"

    def train(
        self,
        metrics: list[Metric],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:

        # sets the seed which to initialize the model
        torch.manual_seed(self.random_seed)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device} to train {str(self)}.")

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1).to(device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 4. INIZIALIZZAZIONE MODELLO
        input_dim = X_train_scaled.shape[1]
        model = FeedForwardNN(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        patience = 20
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

            model.eval()
            with torch.no_grad():
                # Ricalcoliamo le predizioni sull'INTERO dataset di Train e Test
                train_preds_t: torch.Tensor = model(X_train_t)
                test_preds_t: torch.Tensor = model(X_test_t)

                # Calcoliamo la loss ufficiale sul Test set
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

            # --- 3. REGISTRAZIONE METRICHE ---
            record = self.compute_metrics_record(
                iteration=epoch,
                y_train=y_train_t,
                preds_train=train_preds_t,
                y_test=y_test_t,
                preds_test=test_preds_t,
                metrics=metrics,
            )

            records.append(record)

            if epochs_without_improvement >= patience:
                break

        df = pd.DataFrame(records)
        df.attrs["best_test_loss"] = best_test_loss
        df.attrs["best_epoch"] = best_epoch
        return df
