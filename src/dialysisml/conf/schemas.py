"""Config schemas registered in Hydra's ConfigStore."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class SplitConfig:
    """How patients are divided. k=1 means a single train/test split."""

    _target_: str = "dialysisml.pipeline.split.PatientGroupSplit"
    k: int = 1
    seed: int = 42
    train_size: float = 0.6


@dataclass
class WindowConfig:
    """How windows are cut out of a fold's rows.

    A span in days, not in sessions: how often a patient came should not
    change what a window covers.
    """

    days: int = 30
    min_sessions: int = 2


@dataclass
class Config:
    # The three source stages: read the two frames, decide which events count,
    # join them into one series per patient. Part of the formulation rather
    # than a config group: they say what y means.
    data_source: Any = None
    event_steps: list[Any] = field(default_factory=list)
    series: Any = None
    # Steps taking one frame, so nothing they do can carry across the split.
    unfitted: list[Any] = field(default_factory=list)
    split: SplitConfig = field(default_factory=SplitConfig)
    # Steps fitted on the training rows and applied to both sides.
    fitted: list[Any] = field(default_factory=list)
    # Outside both lists: not a frame-to-frame step but the change of
    # representation that follows them, from a frame to the windowed array.
    window_conf: WindowConfig = field(default_factory=WindowConfig)
    window_transformation: Any = None
    # Which labels this run predicts. Null means every column the unfitted
    # steps marked `Role.LABEL`; set it only to predict a subset of them.
    target_columns: list[str] | None = None
    # An adapter, not a model: it owns the training loop and the scoring, and
    # the network it may wrap lives in `dialysisml.models`.
    adapter: Any = None
    # Fallback only: the MLflow experiment is normally the formulation folder,
    # so nothing has to keep two files saying the same name.
    experiment_name: str | None = None
    fromdb: bool = False
    seed: int = 42


def register() -> None:
    """Called once before Hydra composes the configuration."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="split", name="base_split", node=SplitConfig)
