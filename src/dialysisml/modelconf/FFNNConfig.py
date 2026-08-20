from dataclasses import dataclass

from dialysisml.metrics import Metric, mape
from dialysisml.modelconf.BaseModelConfig import BaseModelConfig


@dataclass
class FFNNConfig(BaseModelConfig):
    name: str = "FFNN_Regression"
    # Architettura a imbuto (es. 132 -> 128 -> 64 -> 32)
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.2
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 40
    loss_function: Metric = mape

    def __str__(self):
        return f"NN_layers{self.hidden_layers}_Drop{self.dropout}"
