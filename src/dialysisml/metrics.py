from typing import Callable

import torch

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mape(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Percentage Error (MAPE) metric.

    Args:
        y_hat (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.

    Returns:
        torch.Tensor: MAPE loss value.
    """
    epsilon = 1e-8  # Small value to avoid division by zero
    return torch.mean(torch.abs((y_true - y_hat) / (y_true + epsilon))) * 100


def mae(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (MAE) metric.

    Args:
        y_hat (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.

    Returns:
        torch.Tensor: MAE value.
    """
    return torch.mean(torch.abs(y_true - y_hat))


def rmse(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error (RMSE) metric.

    Args:
        y_hat (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.

    Returns:
        torch.Tensor: RMSE value.
    """
    return torch.sqrt(torch.mean((y_true - y_hat) ** 2))


def huber(y_hat: torch.Tensor, y_true: torch.Tensor, delta: float = 4.0) -> torch.Tensor:
    """Huber loss: quadratic within `delta` of the target, linear beyond it.

    Like RMSE for small errors, like MAE for large ones. `delta` is in the
    target's own unit: 4.0 is the MAD of tte in months. On a target in days it
    must be rescaled, or every error falls inside it and this becomes MSE/2.
    """
    err = y_true - y_hat
    abs_err = torch.abs(err)
    quadratic = 0.5 * err**2
    linear = delta * (abs_err - 0.5 * delta)
    return torch.mean(torch.where(abs_err <= delta, quadratic, linear))


def msle(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Logarithmic Error: squared error on log(1 + days).

    The relative-error objective MAPE was reaching for, without its pathology:
    log compresses the target's right tail instead of dividing by it, so a
    patient one day from the event cannot contribute an unbounded gradient.

    Predictions pass through softplus so the log always receives a positive
    argument. A hard clamp would work too, but it zeroes the gradient of any
    negative prediction, which can strand the model early in training.
    """
    return torch.mean(
        (torch.log1p(torch.nn.functional.softplus(y_hat)) - torch.log1p(y_true)) ** 2
    )
