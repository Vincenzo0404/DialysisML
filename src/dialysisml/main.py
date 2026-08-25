"""Entry point: presplit, patient-grouped split, per-fold pipeline, training.

One model per run: sweep with `--multirun model.model_config.dropout=0.0,0.3`.
"""

import logging
import time

import hydra
import mlflow
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dialysisml import config, features
from dialysisml.conf.schemas import register
from dialysisml.pipeline.SlidingWindow import SlidingWindow
from dialysisml.pipeline.steps.read_raw_data import read_raw_data
from dialysisml.reporting import MetricCollector

register()

# hydra.main installs its own logging config, so the level and the handler are
# already set by the time main runs.
logger = logging.getLogger(__name__)

# Lets a YAML write `columns: ${features:ROBUST_COLUMNS}`, so the column lists
# stay defined only in features.py instead of being duplicated per config.
OmegaConf.register_new_resolver("features", lambda name: getattr(features, name))


def log_results(results: pd.DataFrame, fold: int) -> None:
    """One MLflow series per metric, stepped by epoch.

    The fold prefix keeps the curves apart inside a single run.
    """
    for row in results.to_dict("records"):
        step = int(row["epoch"])
        for key, value in row.items():
            if key != "epoch" and pd.notna(value):
                mlflow.log_metric(f"fold{fold}/{key}", float(value), step=step)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("resolved config:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.model.model_config.name):
        # resolve=True expands ${seed} and ${features:...}, otherwise the
        # logged file would keep the interpolations instead of the values
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")

        collector = MetricCollector()

        started = time.perf_counter()

        # -- PRE SPLIT --
        df = read_raw_data(fromdb=cfg.fromdb)
        # no input frame: the source has nothing to compare against
        collector.record("read_raw_data", None, df, time.perf_counter() - started)

        for step in instantiate(cfg.presplit.steps):
            df = collector.run(step, df)

        logger.info("presplit:\n%s", collector.to_console())
        collector.to_csv(config.RESULTS_PATH / "pipeline_funnel.csv")
        collector.to_mlflow()

        metrics = instantiate(cfg.metrics)

        # -- SPLIT --
        splitter = instantiate(cfg.split)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(df)):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            # its own collector: the presplit one is already flushed, and these
            # rows describe a fold rather than the funnel
            fold_report = MetricCollector()
            # -- POST SPLIT --

            # apply transformations fitted on training set
            for step in instantiate(cfg.postsplit.fitted_transformations):
                train, test = step(train, test)

            # make windows
            window_conf = OmegaConf.to_container(cfg.postsplit.window_conf)
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

            # no before frame: a split produces its side, it does not filter one
            fold_report.record(
                "train", None, train, windows=len(X_train), features=X_train.shape[1]
            )
            fold_report.record(
                "test", None, test, windows=len(X_test), features=X_test.shape[1]
            )

            # --- training ---
            adapter = instantiate(cfg.model)
            started = time.perf_counter()
            results = adapter.train(metrics, X_train, y_train, X_test, y_test)
            log_results(results, fold)

            last = results.iloc[-1]
            fold_report.record(
                "training",
                None,
                None,
                time.perf_counter() - started,
                **{
                    k: round(float(last[k]), 2)
                    for k in results.columns
                    if k != "epoch" and pd.notna(last[k])
                },
            )
            logger.info("fold %d:\n%s", fold, fold_report.to_console())


if __name__ == "__main__":
    main()
