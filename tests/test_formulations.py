"""Every formulation composes, instantiates, and sweeps keys that exist.

Hydra composes by name, so a renamed config option or a renamed keyword leaves
a dangling reference that nothing notices until a run is launched — after the
presplit has already spent minutes reading and reshaping the data. These tests
are the fast version of that discovery: they build everything a run builds,
short of touching the dataset.

`instantiate` alone is not enough. A `_partial_` step binds its keywords at
call time, so a misspelled one survives instantiation, and a swept key that no
config defines is just a string until the sweeper expands it. Both are checked
explicitly below.
"""

import inspect
from functools import partial
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# imported for its side effects: registers the schemas and the ${features:}
# resolver the configs interpolate through
import dialysisml
import dialysisml.main  # noqa: F401

# `conf` is a namespace package, so its own `__file__` is None
CONF_DIR = Path(dialysisml.__file__).parent / "conf"
FORMULATION_DIR = CONF_DIR / "formulation"

# `_base` and friends are included by the sweeps, never selected on their own
FORMULATIONS = sorted(
    str(path.relative_to(FORMULATION_DIR).with_suffix(""))
    for path in FORMULATION_DIR.rglob("*.yaml")
    if not path.name.startswith("_")
)


def compose_formulation(name: str, hydra_config: bool = False) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(
            "config",
            overrides=[f"formulation={name}"],
            return_hydra_config=hydra_config,
        )


def check_call_signature(step) -> None:
    """A `_partial_` target accepts any keyword until it is called.

    `bind_partial` is what a call would do, minus the call: it raises on a
    keyword the target has no parameter for, which is how a renamed argument
    shows up.
    """
    if not isinstance(step, partial):
        return
    inspect.signature(step.func).bind_partial(*step.args, **step.keywords)


def test_formulations_are_discovered() -> None:
    """A typo in the layout would otherwise leave every test below vacuous."""
    assert FORMULATIONS


@pytest.mark.parametrize("name", FORMULATIONS)
def test_composes(name: str) -> None:
    """The defaults resolve and every interpolation has something to point at."""
    cfg = compose_formulation(name)
    # to_container, not resolve: in place, a `${features:}` list would be
    # written back into a node typed from the interpolation string. This is
    # also the form main.py logs the config with.
    OmegaConf.to_container(cfg, resolve=True)

    assert cfg.presplit.steps, "a formulation without a presplit has no target"
    assert cfg.window_conf.target_columns


@pytest.mark.parametrize("name", FORMULATIONS)
def test_instantiates(name: str) -> None:
    """Every `_target_` imports, and every bound keyword exists on it."""
    cfg = compose_formulation(name)

    check_call_signature(instantiate(cfg.data_source))
    for step in instantiate(cfg.presplit.steps):
        check_call_signature(step)
    for step in instantiate(cfg.postsplit.fitted_transformations):
        check_call_signature(step)
    for metric in instantiate(cfg.metrics):
        check_call_signature(metric)

    instantiate(cfg.postsplit.window_transformation)
    instantiate(cfg.split)
    # a real call, not a partial: a keyword the adapter has no field for
    # raises right here
    instantiate(cfg.adapter)


@pytest.mark.parametrize("name", FORMULATIONS)
def test_swept_keys_exist(name: str) -> None:
    """A sweep over a key no config defines would silently add a new one.

    `adapter.score_metric` on an adapter that only has `loss_function` is the
    case this catches: the sweeper sets it, the run logs it, and nothing about
    the model changes.
    """
    with_hydra = compose_formulation(name, hydra_config=True)
    swept = OmegaConf.select(with_hydra, "hydra.sweeper.params") or {}

    cfg = compose_formulation(name)
    missing = [key for key in swept if OmegaConf.select(cfg, key) is None]
    assert not missing, f"{name} sweeps keys absent from the config: {missing}"
