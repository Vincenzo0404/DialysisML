import torch
import torch.nn as nn


class LSTM(nn.Module):
    """A recurrent encoder over the window, plus a linear head.

    `hidden_size` is a scalar, not a list as in FeedForwardNN: nn.LSTM stacks
    layers of the same width, so depth and width are two separate numbers.

    `output_dim` is the only axis that changes with the target: 1 for the TTE,
    k for k cutoffs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_n is (num_layers, batch, hidden): the last layer at the final step,
        # which is what the head reads. Taking it from here rather than from the
        # output sequence keeps the whole (batch, size, hidden) tensor out of it.
        _, (h_n, _) = self.lstm(x)

        # squeeze(-1) drops the trailing axis for a single output, and is a
        # no-op when output_dim > 1
        return self.head(self.dropout(h_n[-1])).squeeze(-1)
