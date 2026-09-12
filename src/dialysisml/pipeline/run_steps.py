"""The pipeline as plain functions, so a run and an analysis take one path.

`main.py` owns the Hydra entry point and the MLflow bookkeeping; what reshapes
the data lives here, callable from a notebook with a composed or logged `cfg`.
"""

from functools import partial
from typing import Any, cast

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dialysisml.conf.schemas import Config, register
from dialysisml.config import HYDRA_CONF_DIR
from dialysisml.pipeline.SlidingWindow import SlidingWindow
from dialysisml.reporting import MetricCollector
from dialysisml.schema import Frame, Role, select


def run_unfitted(cfg: Config, collector: MetricCollector | None = None) -> Frame:
    """Source and single-frame steps: the frame as it reaches the split."""
    sessions, events = instantiate(cfg.data_source)()
    for step in instantiate(cfg.event_steps):
        events = step(events)

    # partial, not a call: `run` needs the step itself to time and name it
    build = partial(instantiate(cfg.series), sessions, events)
    frame = collector.run(build) if collector else build()

    for step in instantiate(cfg.unfitted):
        frame = collector.run(step, frame) if collector else step(frame)
    return frame.validate()


def split_frame(cfg: Config, frame: Frame) -> tuple[Frame, Frame]:
    """Train and test, grouped by patient. One pair: the project runs k=1."""
    train_idx, test_idx = next(iter(instantiate(cfg.split).split(frame.data)))
    return (
        frame.update(frame.data.iloc[train_idx]),
        frame.update(frame.data.iloc[test_idx]),
    )


def run_fitted(cfg: Config, train: Frame, test: Frame) -> tuple[Frame, Frame]:
    """Steps fitted on train, then the windowing: one row per window.

    The windowing closes the pipeline although it fits nothing: `linear_fit`
    sums over a window through cumulative sums, so one missing value would
    propagate through the column, and filling it needs a train-fitted imputer.
    """
    for step in instantiate(cfg.fitted):
        train, test = step(train, test)

    window_conf = OmegaConf.to_container(cfg.window_conf)
    assert isinstance(window_conf, dict), "window_conf must be a mapping"
    window_conf = {str(key): value for key, value in window_conf.items()}

    transformer = instantiate(cfg.window_transformation)
    return (
        transformer.transform(SlidingWindow(train, **window_conf)),
        transformer.transform(SlidingWindow(test, **window_conf)),
    )


def x_y_split(cfg: Config, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    features = select(frame.schema, role=Role.FEATURE)
    targets = (
        list(cfg.target_columns)
        if cfg.target_columns
        else select(frame.schema, role=Role.LABEL)
    )
    X = frame.data[features].to_numpy(np.float32)
    y = frame.data[targets].to_numpy()

    return X, y


def build_hydra_cfg(overrides: dict[str, str]) -> Config:
    """A cfg composed outside a run, for notebooks and analyses.

    `register` first: `base_config` and `base_split` live in the ConfigStore,
    not in a YAML, so composing without it fails on the split.
    """
    register()
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CONF_DIR)):
        # a cast, not a conversion: compose returns the DictConfig that `Config`
        # only validates, and every function here is annotated with the schema
        return cast(
            Config,
            compose(
                config_name="config",
                overrides=[f"{key}={val}" for key, val in overrides.items()],
            ),
        )
