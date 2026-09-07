"""Config schemas registered in Hydra's ConfigStore."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class PresplitConfig:
    """Presplit data processing pipeline configuration"""

    steps: list[Any] = field(default_factory=list)


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
class PostsplitConfig:
    """Postsplit data processing pipeline configuration.

    Three sub-steps in a fixed order — fit the transformers, build the windows,
    transform the windows — each configurable on its own. A single transformer,
    not a list: one that flattens the time axis is only valid last, so the
    composition belongs inside the transformer rather than in the config.
    """

    fitted_transformations: list[Any] = field(default_factory=list)
    window_transformation: Any = None


@dataclass
class Config:
    # The three source stages: read the two frames, decide which events count,
    # join them into one series per patient. Part of the formulation rather
    # than a config group: they say what y means.
    data_source: Any = None
    event_steps: list[Any] = field(default_factory=list)
    series: Any = None
    presplit: PresplitConfig = field(default_factory=PresplitConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    postsplit: PostsplitConfig = field(default_factory=PostsplitConfig)
    # Top level: shared by every postsplit variant, not one thing to repeat
    # inside each of them.
    window_conf: WindowConfig = field(default_factory=WindowConfig)
    # Which labels this run predicts.
    target_columns: list[str] = field(default_factory=lambda: ["tte"])
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
