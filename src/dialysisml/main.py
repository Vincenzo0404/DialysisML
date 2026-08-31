"""Entry point: presplit, patient-grouped split, per-fold pipeline, training.

One model per run: sweep with `--multirun model.dropout=0.0,0.3`.
"""

import logging
import time
from functools import partial
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from mlflow.data.pandas_dataset import from_pandas
from omegaconf import DictConfig, OmegaConf

from dialysisml import config, features
from dialysisml.adapters import ModelAdapter, metric_name
from dialysisml.conf.schemas import register
from dialysisml.pipeline.SlidingWindow import SlidingWindow
from dialysisml.pipeline.steps.read_raw_data import read_raw_data
from dialysisml.reporting import (
    aggregate_folds,
    epochs_collector,
    metadata_collector,
)

register()

# hydra.main installs its own logging config, so the level and the handler are
# already set by the time main runs.
logger = logging.getLogger(__name__)

# Lets a YAML write `columns: ${features:ROBUST_COLUMNS}`, so the column lists
# stay defined only in features.py instead of being duplicated per config.
OmegaConf.register_new_resolver("features", lambda name: getattr(features, name))


def log_results(results: pd.DataFrame, fold: int) -> None:
    """One MLflow series per metric, stepped by epoch, plus the curve as a table.

    The series is what the UI charts; the table is what `load_table` concatenates
    across a sweep. `log_table` appends, so the folds pile up on their own.
    """
    for row in results.to_dict("records"):
        step = int(row["epoch"])
        for key, value in row.items():
            if key != "epoch" and pd.notna(value):
                mlflow.log_metric(f"fold{fold}/{key}", float(value), step=step)

    mlflow.log_table(results.assign(fold=fold), "epochs.json")


import warnings


def log_dataset(df: pd.DataFrame, name: str):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Hint: Inferred schema contains integer column"
        )
        dataset = from_pandas(df, name=name)
        mlflow.log_input(dataset, context=name)


def choice_params(cfg: DictConfig) -> dict[str, str]:
    """Returns a dict containing current run mappings then logged as params to MLFlow."""
    choices = HydraConfig.get().runtime.choices
    params: dict[str, str] = {}
    for key, value in choices.items():
        # the experiment file is provenance, not a modelling choice: it goes to a tag
        if key.startswith("hydra/") or key == "experiment":
            continue
        # keep the group, drop where it lands: MLflow rejects `@` in a param name
        params[key.partition("@")[0]] = value

    params["window_size"] = cfg.window_conf.size
    return params


def sweep_id() -> str:
    """`<date>/<time>` of the launch, shared by every job of one multirun.

    From the resolved `runtime.output_dir`: `hydra.sweep.dir` is stored unresolved,
    so reading it would re-resolve `now` and give each job a different id.
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    # MULTIRUN adds a per-job subdir; a single run has none
    if HydraConfig.get().mode == RunMode.MULTIRUN:
        output_dir = output_dir.parent
    return "/".join(output_dir.parts[-2:])


def model_params(model: DictConfig) -> dict[str, str]:
    """The adapter's hyperparameters, as sortable MLflow columns.

    From the config node, not the adapter: `_target_` and the nested partials
    read better as the names of what they point at.
    """
    params: dict[str, str] = {}
    container = OmegaConf.to_container(model, resolve=True)
    assert isinstance(container, dict), "model must be a mapping"
    for key, value in container.items():
        if str(key).startswith("_"):
            continue
        # a nested _target_ (the loss, say) reads better as what it points at
        if isinstance(value, dict) and "_target_" in value:
            value = value["_target_"].rsplit(".", 1)[-1]
        params[f"model.{key}"] = value
    return params


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment_name)

    # once, not per fold: it holds no state between folds, and the run name has
    # to exist before the run is opened
    adapter: ModelAdapter = instantiate(cfg.model)

    with mlflow.start_run(run_name=str(adapter)):
        mlflow.log_params(choice_params(cfg))
        mlflow.log_params(model_params(cfg.model))
        # which measure test_score_* refers to, and the key to group runs by
        mlflow.log_param("score_metric", metric_name(adapter.score_metric))
        # tags, not params: they say which launch a run came from, not what it is
        mlflow.set_tags(
            {
                "sweep": sweep_id(),
                "experiment_file": HydraConfig.get().runtime.choices.get(
                    "experiment", ""
                ),
            }
        )
        # resolve=True expands ${seed} and ${features:...}
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")  # type: ignore

        collector = metadata_collector()

        # -- PRE SPLIT --
        # partial, not a call: run() needs the step itself to time and name it
        df = collector.run(partial(read_raw_data, fromdb=cfg.fromdb))
        log_dataset(df, "raw_df")

        for step in instantiate(cfg.presplit.steps):
            df = collector.run(step, df)
        log_dataset(df, "after_presplit")

        logger.info("presplit:\n%s", collector.to_console())
        collector.to_mlflow()

        metrics = instantiate(cfg.metrics)

        # -- SPLIT --
        fold_summaries = []
        splitter = instantiate(cfg.split)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df)):
            train, test = df.iloc[train_idx], df.iloc[test_idx]

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

            # apply window transformers
            transformer = instantiate(cfg.postsplit.window_transformation)
            transformer.bind(windows_train.feature_columns)
            assert windows_train.feature_columns == windows_test.feature_columns

            X_train, y_train = windows_train.materialize()
            X_test, y_test = windows_test.materialize()
            X_train, X_test = transformer.transform(X_train), transformer.transform(
                X_test
            )

            fold_report.add_record(
                train, idx="train", windows=len(X_train), features=X_train.shape[1]
            )
            fold_report.add_record(
                test, idx="test", windows=len(X_test), features=X_test.shape[1]
            )

            # --- training ---
            started = time.perf_counter()
            results = adapter.train(metrics, X_train, y_train, X_test, y_test)
            log_results(results, fold)

            # measure training time
            fold_report.add_record(None, time.perf_counter() - started, idx="training")
            logger.info("fold %d:\n%s", fold, fold_report.to_console())

            summary = epochs_collector(results, metric_name(adapter.score_metric))
            fold_summaries.append(summary)
            logger.info("fold %d results:\n%s", fold, summary.to_console())

        cv = aggregate_folds(fold_summaries)
        # a table, not text: `mlflow.load_table` can concatenate this artifact
        # across every run of a sweep, which an opaque blob of CSV cannot be
        mlflow.log_table(cv.to_frame(), "run_metrics_aggregated.json")
        cv.to_mlflow_scalars()
        logger.info("cross-validated:\n%s", cv.to_console())


if __name__ == "__main__":
    main()
