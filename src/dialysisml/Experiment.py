from dataclasses import asdict

import mlflow
import pandas as pd

from dialysisml.adapters import AdapterFactory, ModelAdapter
from dialysisml.metrics import Metric
from dialysisml.modelconf import BaseModelConfig
from dialysisml.window import WindowStore


def _stringify(params: dict) -> dict[str, str]:
    """Normalizes parameter values to strings (MLflow only accepts strings)."""
    stringified = {}
    for key, value in params.items():
        if callable(value):
            stringified[key] = value.__name__
        elif isinstance(value, (tuple, list)):
            stringified[key] = ",".join(str(v) for v in value)
        else:
            stringified[key] = str(value)
    return stringified


class Experiment:
    """Trains one model configuration on one window store, tracking it on MLflow."""

    def __init__(
        self,
        model_config: BaseModelConfig,
        window_store: WindowStore,
    ):
        self.model_config = model_config
        self.window_store = window_store
        self.model_adapter: ModelAdapter = AdapterFactory.create_model_adapter(
            model_config
        )

    def params(self) -> dict[str, str]:
        """Window store + model config parameters, normalized to strings."""
        return _stringify({**self.window_store.params(), **asdict(self.model_config)})

    def _log_metrics(self, df_results: pd.DataFrame):
        """Logs every metric column of the results as an MLflow metric, stepped by epoch."""
        for row in df_results.to_dict(orient="records"):
            step = int(row["epoch"])
            for col_name, value in row.items():
                if col_name != "epoch" and pd.notna(value):
                    mlflow.log_metric(str(col_name), float(value), step=step)

    def run(self, metrics: list[Metric]) -> pd.DataFrame:
        """Trains the model, logs params and per-epoch metrics, returns the results.

        The returned DataFrame carries the experiment parameters as extra columns,
        repeated on every row, so results of several experiments can be concatenated.
        """
        params = self.params()
        windows = self.window_store

        with mlflow.start_run(run_name=str(self)):
            mlflow.log_params(params)

            df_results = self.model_adapter.train(
                metrics,
                windows.X_train,
                windows.y_train,
                windows.X_test,
                windows.y_test,
            )

            self._log_metrics(df_results)

        return df_results.assign(**params)

    def __str__(self):
        return f"{self.model_config.name}_{self.window_store}"
