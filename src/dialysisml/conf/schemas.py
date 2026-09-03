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

    Every column derived from the target must be listed here, even one not
    used as a label: `feature_columns` is everything else, so a forgotten one
    ends up among the inputs and leaks.
    """

    size: int = 30
    stride: int = 1
    target_columns: list[str] = field(default_factory=lambda: ["tte"])


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
    # The step producing the raw frame. Part of the formulation rather than a
    # config group: which events count as adverse is part of what y means.
    data_source: Any = None
    presplit: PresplitConfig = field(default_factory=PresplitConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    postsplit: PostsplitConfig = field(default_factory=PostsplitConfig)
    # Top level: shared by every postsplit variant, not one thing to repeat
    # inside each of them.
    window_conf: WindowConfig = field(default_factory=WindowConfig)
    # An adapter, not a model: it owns the training loop and the scoring, and
    # the network it may wrap lives in `dialysisml.models`.
    adapter: Any = None
    metrics: list[Any] = field(default_factory=list)
    # Fallback only: the MLflow experiment is normally the formulation folder,
    # so nothing has to keep two files saying the same name.
    experiment_name: str = "scratch"
    fromdb: bool = False
    seed: int = 42


def register() -> None:
    """Called once before Hydra composes the configuration."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="split", name="base_split", node=SplitConfig)
