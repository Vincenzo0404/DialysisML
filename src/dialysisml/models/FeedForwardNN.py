import torch
import torch.nn as nn


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout: float):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Ultimo layer: output singolo per la regressione (Time-To-Event)
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Squeeze per portare l'output da (Batch, 1) a (Batch,)
        return self.network(x).squeeze(-1)
