"""Entry point: presplit, patient-grouped split, per-fold pipeline, training.

A run needs a formulation, which is the target it trains on and the MLflow
experiment it lands in: `formulation=first_event_cap365/ffnn`. One adapter per
run: sweep with `--multirun adapter.dropout=0.0,0.3`.
"""

import logging
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd

from dialysisml.pipeline.split import PatientGroupSplit

pd.options.mode.copy_on_write = True  # ensures copies are made only when really needed

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from mlflow.data.pandas_dataset import from_pandas
from omegaconf import DictConfig, OmegaConf

from dialysisml import config
from dialysisml.adapters import ModelAdapter, ResultSchema, TrainingResult
from dialysisml.conf.schemas import Config, register
from dialysisml.pipeline.SlidingWindow import SlidingWindow
from dialysisml.reporting import metadata_collector

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

    # adapter keeps no state between calls to `train` method
    adapter: ModelAdapter = instantiate(cfg.adapter)

    with mlflow.start_run(run_name=str(adapter)):

        mlflow.log_params(choice_params(cfg))
        mlflow.log_params(adapter_params(cfg.adapter))
        # resolve=True expands ${seed} and ${features:...}
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")  # type: ignore

        collector = metadata_collector()

        # -- SOURCE --
        sessions, events = instantiate(cfg.data_source)()
        for step in instantiate(cfg.event_steps):
            events = step(events)
        # partial, not a call: run() needs the step itself to time and name it
        frame = collector.run(partial(instantiate(cfg.series), sessions, events))
        log_dataset(frame.data, "raw_df")

        # -- PRE SPLIT --
        for step in instantiate(cfg.presplit.steps):
            frame = collector.run(step, frame)
        frame.validate()
        log_dataset(frame.data, "presplit")

        logger.info("presplit:\n%s", collector.to_console())
        collector.to_mlflow()

        # -- SPLIT --
        splitter: PatientGroupSplit = instantiate(cfg.split)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(frame.data)):
            train = frame.update(frame.data.iloc[train_idx])
            test = frame.update(frame.data.iloc[test_idx])

            fold_report = metadata_collector()
            # -- POST SPLIT --

            # apply transformations fitted on training set
            for step in instantiate(cfg.postsplit.fitted_transformations):
                train, test = step(train, test)

            # make windows
            window_conf = OmegaConf.to_container(cfg.window_conf)
            assert isinstance(window_conf, dict), "window_conf must be a mapping"
            window_conf = {str(key): val for key, val in window_conf.items()}

            windows_train = SlidingWindow(train, **window_conf)
            windows_test = SlidingWindow(test, **window_conf)

            targets = list(cfg.target_columns)
            y_train, y_test = windows_train.targets(targets), windows_test.targets(
                targets
            )

            transformer = instantiate(cfg.postsplit.window_transformation)
            X_train, feature_schema = transformer.transform(windows_train)
            X_test, _ = transformer.transform(windows_test)

            fold_report.add_record(
                train.data, idx="train", windows=len(X_train), features=X_train.shape[1]
            )
            fold_report.add_record(
                test.data, idx="test", windows=len(X_test), features=X_test.shape[1]
            )

            # the order the model expects its columns in: without it a reloaded
            # model has 149 anonymous features and no way to notice a mismatch
            mlflow.log_dict({"features": list(feature_schema.columns)}, "features.json")

            # --- TRAINING ---
            logger.info(f"Training {str(adapter)}: ...")
            started = time.perf_counter()
            model, results = adapter.train(X_train, y_train, X_test, y_test)
            adapter.log_model(model)

            history_pivoted = results.pivot()
            log_training_results(
                # pivot results to wide form
                history_pivoted,
                results.best_iteration,
                results.total_iterations,
            )

            # measure training time
            fold_report.add_record(None, time.perf_counter() - started, idx="training")
            logger.info(fold_report.to_console())


if __name__ == "__main__":
    main()
