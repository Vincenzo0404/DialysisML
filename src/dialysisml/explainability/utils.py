import warnings
from typing import cast

import mlflow
import mlflow.artifacts
import numpy as np
import shap
import torch
from omegaconf import DictConfig, OmegaConf

from dialysisml import config
from dialysisml.adapters import ModelAdapter
from dialysisml.conf.schemas import Config
from dialysisml.pipeline.run_steps import (
    run_fitted,
    run_unfitted,
    split_frame,
    x_y_split,
)
from dialysisml.schema import Role, select


def load_adapter(run_id: str) -> ModelAdapter:
    """The adapter a run trained, rebuilt from disk.

    The weights are no longer an MLflow artifact: training writes them to
    `saved_adapters` under the run id. MLflow still says *which* adapter wrote
    them, through the tag `main.py` sets on the run, and that name is what the
    registry turns back into a class.
    """
    try:
        tags = mlflow.get_run(run_id).data.tags
    except Exception as error:
        raise LookupError(f"MLflow knows no run {run_id}.") from error

    name = tags.get("adapter")
    if name is None:
        raise LookupError(
            f"run {run_id} carries no `adapter` tag: it predates the tag, so "
            "nothing records which class its archive holds"
        )

    path = config.SAVED_ADAPTERS / run_id
    if not path.exists():
        raise LookupError(
            f"run {run_id} is an {name} but has no archive at {path}: the run "
            "and `saved_adapters` are out of sync"
        )
    return ModelAdapter.resolve(name).load(path)


def list_adapters() -> None:
    """The runs whose adapter can be rebuilt, and whether it is on disk."""
    for experiment in mlflow.search_experiments():
        runs = mlflow.search_runs([experiment.experiment_id])
        if runs.empty or "tags.adapter" not in runs:
            continue
        for _, run in runs[runs["tags.adapter"].notna()].iterrows():
            run_id = str(run["run_id"])
            name = str(run["tags.adapter"])
            # a neural net is the only thing `get_shap_values` can explain today
            mark = "shap" if name == "FFNNAdapter" else "    "
            on_disk = (config.SAVED_ADAPTERS / run_id).exists()
            # the bucket adapter's run name is its whole dataclass repr, which
            # would push everything else off the line
            run_name = str(run.get("tags.mlflow.runName", ""))[:40]
            print(
                f"{run_id}  {mark}  {name:<15} {experiment.name:<20} "
                f"{run_name}{'' if on_disk else '  (no archive)'}"
            )


def get_shap_values(
    run_id: str,
    background: int = 200,
    samples: int = 1000,
    seed: int = 42,
) -> shap.Explanation:
    """Compute shapley values.
    Args:
        `run_id`: MLFlow run id from which to take the adapter.
        `background`: size of background set used to compute empirical distribution
            for marginalization.
        `samples`: number of rows in X_test to compute shapley values on.
        `seed`: random seed.

    Returns:
        A `shap.Explanation` of shape (samples, features): the values, the
        baseline they are measured against, the rows they describe, and the
        feature names.
    """

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if run_id is None:
        raise ValueError("a run id is required")

    adapter = load_adapter(run_id)
    model = adapter.model
    if not isinstance(model, torch.nn.Module):
        raise NotImplementedError(
            f"run {run_id} holds a {type(adapter).__name__}, whose model is a "
            f"{type(model).__name__}: only neural adapters can be explained"
        )
    print("adapter:", type(adapter).__name__)

    # retrieve pipeline config
    logged = OmegaConf.create(mlflow.artifacts.load_text(f"runs:/{run_id}/config.yaml"))
    assert isinstance(logged, DictConfig), "the logged config must be a mapping"

    cfg = cast(Config, logged)

    # replay the run's pipeline to rebuild the arrays it trained on
    frame = run_unfitted(cfg)
    train, test = split_frame(cfg, frame)
    train, test = run_fitted(cfg, train, test)

    names = select(train.schema, role=Role.FEATURE)
    X_train, _ = x_y_split(cfg, train)
    X_test, _ = x_y_split(cfg, test)

    # check features match
    logged_names = mlflow.artifacts.load_dict(f"runs:/{run_id}/features.json")[
        "features"
    ]
    if names != logged_names:
        raise ValueError("rebuilt features differ from the ones the run logged")
    print(
        f"rebuilt {len(X_train)} train and {len(X_test)} test windows, "
        f"{len(names)} features"
    )

    rng = np.random.default_rng(seed)
    # background set used to get empirical features and target distribution
    reference = X_train[rng.choice(len(X_train), background, replace=False)]
    # feature rows to explain
    explained = X_test[rng.choice(len(X_test), samples, replace=False)]

    # make tensors
    device = next(model.parameters()).device
    model.eval()
    reference = torch.tensor(reference, dtype=torch.float32, device=device)
    explained = torch.tensor(explained, dtype=torch.float32, device=device)

    # compute shapley values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unrecognized nn.Module")
        explainer = shap.DeepExplainer(model, reference)
        shapley_values = np.asarray(explainer.shap_values(explained))

    shapley_values = shapley_values.squeeze(-1)
    assert shapley_values.shape == (
        samples,
        len(names),
    ), f"unexpected shape {shapley_values.shape}"

    # E[f(X)] of the background set
    baseline = np.asarray(explainer.expected_value).reshape(-1)[0]  # type: ignore
    return shap.Explanation(
        values=shapley_values,
        base_values=np.repeat(baseline, samples),
        data=explained.cpu().numpy(),
        feature_names=names,
    )
