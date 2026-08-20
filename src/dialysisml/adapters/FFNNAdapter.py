import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dialysisml.adapters.ModelAdapter import ModelAdapter
from dialysisml.metrics import Metric
from dialysisml.modelconf import FFNNConfig
from dialysisml.models import FeedForwardNN


class FFNNAdapter(ModelAdapter):
    model_config: FFNNConfig  # TODO fix type hinting for subclasses of BaseModelConfig

    def __init__(self, model_config: FFNNConfig):
        super().__init__(model_config)

    def train(
        self,
        metrics: list[Metric],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:

        # Scale dataset
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. PREPARAZIONE PYTORCH
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # I target arrivano come (N, 1) mentre la rete produce (N,): senza appiattirli
        # la sottrazione nella loss farebbe broadcasting a (N, N), confrontando ogni
        # target con ogni predizione.
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1).to(device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(
            dataset, batch_size=self.model_config.batch_size, shuffle=True
        )

        # 4. INIZIALIZZAZIONE MODELLO
        input_dim = X_train_scaled.shape[1]
        model = FeedForwardNN(
            input_dim=input_dim,
            hidden_layers=self.model_config.hidden_layers,
            dropout=self.model_config.dropout,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
        )

        records = []
        for epoch in range(1, self.model_config.epochs + 1):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                batch_preds = model(batch_X)
                loss = self.model_config.loss_function(batch_preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            # Loss media ponderata sul totale dei campioni di train
            train_loss /= X_train_t.size(0)

            model.eval()
            with torch.no_grad():
                # Ricalcoliamo le predizioni sull'INTERO dataset di Train e Test
                train_preds_t: torch.Tensor = model(X_train_t)
                test_preds_t: torch.Tensor = model(X_test_t)

                # Calcoliamo la loss ufficiale sul Test set
                test_loss = float(
                    self.model_config.loss_function(test_preds_t, y_test_t)
                )

            # --- 3. REGISTRAZIONE METRICHE ---
            record = self.compute_metrics_record(
                iteration=epoch,
                y_train=y_train_t,
                preds_train=train_preds_t,
                y_test=y_test_t,
                preds_test=test_preds_t,
                metrics=metrics,
            )

            record["train_loss"] = train_loss
            record["test_loss"] = test_loss

            records.append(record)

        return pd.DataFrame(records)
