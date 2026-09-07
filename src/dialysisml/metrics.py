from functools import partial
from typing import Callable

import torch

Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def metric_name(metric: Metric | partial) -> str:
    """Name of a metric, whether a plain function or a functools.partial. (to handle hydra's _partial_)"""
    if isinstance(metric, partial):
        return metric.func.__name__
    return metric.__name__


def mape(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    epsilon = 1e-8  # Small value to avoid division by zero
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    return torch.mean(torch.abs(y_true - y_pred))


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def huber(
    y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 4.0
) -> torch.Tensor:

    err = y_true - y_pred
    abs_err = torch.abs(err)
    quadratic = 0.5 * err**2
    linear = delta * (abs_err - 0.5 * delta)
    return torch.mean(torch.where(abs_err <= delta, quadratic, linear))


def msle(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

    return torch.mean(
        (torch.log1p(torch.nn.functional.softplus(y_pred)) - torch.log1p(y_true)) ** 2
    )
