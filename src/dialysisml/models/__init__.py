"""PyTorch model definitions."""

from dialysisml.models.FeedForwardNN import FeedForwardNN

# NOTA: lstm.py non e' esportato, importa ancora config.LSTMConfig (rimossa).
__all__ = ["FeedForwardNN"]
