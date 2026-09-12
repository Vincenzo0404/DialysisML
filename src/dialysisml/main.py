"""Entry point: unfitted steps, patient-grouped split, fitted steps, training.

A run needs a formulation, which is the target it trains on and the MLflow
experiment it lands in: `formulation=first_event_cap365/ffnn`. One adapter per
run: sweep with `--multirun adapter.dropout=0.0,0.3`.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True  # ensures copies are made only when really needed

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from mlflow.data.pandas_dataset import from_pandas
from omegaconf import DictConfig, OmegaConf

from dialysisml import config
from dialysisml.adapters import ModelAdapter, ResultSchema, TrainingResult
from dialysisml.conf.schemas import Config, register
from dialysisml.pipeline.run_steps import run_fitted, run_unfitted, split_frame
from dialysisml.reporting import metadata_collector
from dialysisml.schema import Role, select

register()

# hydra.main installs its own logging config, so the level and the handler are
# already set by the time main runs.
logger = logging.getLogger(__name__)

# The config group holding the formulations. Named once: it appears both as a
# key of `runtime.choices` and as the directory the experiment name comes from.
FORMULATION_GROUP = "formulation"


def log_training_results(
    results: pd.DataFrame, best_iteration: int, total_iterations: int
) -> None:
    """Logs TrainingResults to MLFlow, with it's best iteration.
    Results must be in wide form.
    """

    # a one-shot fit has no iteration to pick: `best_X` would only repeat `X`
    if len(results) > 1:
        # loc, not iloc: best_idx is an iteration, which the pivot made the index
        best = results.loc[best_iteration]
        mlflow.log_metrics({f"best_{name}": value for name, value in best.items()})  # type: ignore
        mlflow.log_metric("best_iteration", best_iteration)
    mlflow.log_metric("total_iterations", total_iterations)

    for iteration, row in results.iterrows():
        mlflow.log_metrics(row.to_dict(), step=int(iteration))  # type: ignore


def log_dataset(df: pd.DataFrame, name: str):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Hint: Inferred schema contains integer column"
        )
        dataset = from_pandas(df, name=name)
        mlflow.log_input(dataset, context=name)


def choice_params(cfg: Config) -> dict[str, Any]:
    """Returns a dict containing current run mappings then logged as params to MLFlow."""
    choices = HydraConfig.get().runtime.choices
    params: dict[str, Any] = {}
    for key, value in choices.items():
        # the formulation is not a modelling choice within its own experiment:
        # it is the experiment, and the sweep file is already a tag
        if key.startswith("hydra/") or key.startswith(FORMULATION_GROUP):
            continue
        # keep the group, drop where it lands: MLflow rejects `@` in a param name
        params[key.partition("@")[0]] = value

    params["window_days"] = cfg.window_conf.days
    return params


def formulation_choice() -> tuple[str, str]:
    """`(formulation, sweep file)` behind the `formulation=` option."""
    choice = str(HydraConfig.get().runtime.choices.get(FORMULATION_GROUP, ""))
    formulation, _, sweep_file = choice.partition("/")
    return formulation, sweep_file


def adapter_params(adapter: DictConfig) -> dict[str, Any]:
    """The adapter's hyperparameters, as sortable MLflow columns.

    From the config node, not the adapter: `_target_` and the nested partials
    read better as the names of what they point at.
    """
    params: dict[str, Any] = {}
    container = OmegaConf.to_container(adapter, resolve=True)
    assert isinstance(container, dict), "adapter must be a mapping"
    for key, value in container.items():
        if str(key).startswith("_"):
            continue
        # a nested _target_ (the loss, say) reads better as what it points at
        if isinstance(value, dict) and "_target_" in value:
            value = value["_target_"].rsplit(".", 1)[-1]
        params[f"adapter.{key}"] = value
    return params


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    # the formulation folder, not a name written in the YAML: `experiment_name`
    # is only the fallback for a run composed without one
    formulation, sweep_file = formulation_choice()
    mlflow.set_experiment(cfg.experiment_name or formulation)

    adapter: ModelAdapter = instantiate(cfg.adapter)

    with mlflow.start_run(run_name=str(adapter)) as run:
        run_id = run.info.run_id
        mlflow.log_params(choice_params(cfg))
        mlflow.log_params(adapter_params(cfg.adapter))
        # resolve=True expands ${seed} and ${features:...}
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")  # type: ignore

        collector = metadata_collector()

        frame = run_unfitted(cfg, collector)
        log_dataset(frame.data, "unfitted")
        logger.info("unfitted:\n%s", collector.to_console())
        collector.to_mlflow()

        train, test = split_frame(cfg, frame)
        windowed_train, windowed_test = run_fitted(cfg, train, test)

        features = select(windowed_train.schema, role=Role.FEATURE)
        # None unless a formulation overrides: normally the targets are
        # whatever the unfitted steps marked `Role.LABEL`
        targets = (
            list(cfg.target_columns)
            if cfg.target_columns
            else select(windowed_train.schema, role=Role.LABEL)
        )
        X_train = windowed_train.data[features].to_numpy(np.float32)
        X_test = windowed_test.data[features].to_numpy(np.float32)
        y_train = windowed_train.data[targets].to_numpy()
        y_test = windowed_test.data[targets].to_numpy()

        # the row frames, not the windowed ones: the funnel counts sessions
        fold_report = metadata_collector()
        fold_report.add_record(
            train.data, idx="train", windows=len(X_train), features=X_train.shape[1]
        )
        fold_report.add_record(
            test.data, idx="test", windows=len(X_test), features=X_test.shape[1]
        )

        # the order the model expects its columns in: without it a reloaded
        # model has 149 anonymous features and no way to notice a mismatch
        mlflow.log_dict({"features": features}, "features.json")

        # --- TRAINING ---
        logger.info(f"Training {str(adapter)}: ...")
        started = time.perf_counter()
        results = adapter.train(X_train, y_train, X_test, y_test)
        adapter.save(config.SAVED_ADAPTERS / run_id)

        log_training_results(
            results.pivot(), results.best_iteration, results.total_iterations
        )

        # measure training time
        fold_report.add_record(None, time.perf_counter() - started, idx="training")
        logger.info(fold_report.to_console())


if __name__ == "__main__":
    main()
