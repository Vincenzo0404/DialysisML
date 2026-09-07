import torch.nn as nn
from torch import Tensor


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], dropout: float):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer is a single scalar
        layers.append(nn.Linear(current_dim, 1))

        # self.network returns a tensor of size (batch_size, n_outputs) which in this case
        # is equeal to (batch_size, 1) since we only have a single output
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # Do not squeeze predictions, shape (batch_size, 1) is needed for SHAP library
        return self.network(x)
