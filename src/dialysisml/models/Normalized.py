import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Normalized(nn.Module):
    """Standardises its input before the wrapped module sees it.

    The statistics are buffers, not parameters: training leaves them alone, but
    `state_dict` carries them, so a saved model takes raw features and there is
    nothing to remember when reloading it.
    """

    def __init__(self, module: nn.Module, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.module = module
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        # a column that never varies would divide by zero; StandardScaler does
        # the same, which is why the scaler never surfaced the two constant ones
        scale = np.where(std == 0.0, 1.0, std)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        mean = self.get_buffer("mean")
        scale = self.get_buffer("scale")
        return self.module((x - mean) / scale)
