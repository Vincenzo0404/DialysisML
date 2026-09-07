import warnings
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import shap
import torch
from hydra.utils import instantiate
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

from dialysisml import config
from dialysisml.pipeline.SlidingWindow import SlidingWindow

# Wich SHAP algorithm to use for each Library
SHAP_ALGORITHMS = {
    "pytorch": (mlflow.pytorch.load_model, shap.DeepExplainer),  # type: ignore
    "xgboost": (mlflow.xgboost.load_model, shap.TreeExplainer),  # type: ignore
    "sklearn": (mlflow.sklearn.load_model, shap.TreeExplainer),  # type: ignore
}


def load_model(run_id: str) -> tuple[Any, type[shap.Explainer]]:
    """The run's model, and the explainer its flavor calls for.

    MLflow records the flavor alongside the model, so nothing here has to know
    in advance which library trained it.
    """
    uri = f"runs:/{run_id}/model"
    try:
        flavors = mlflow.models.get_model_info(uri).flavors  # type: ignore
    except MlflowException as error:
        raise LookupError(f"run {run_id} logged no model.") from error

    for name, (loader, explainer) in SHAP_ALGORITHMS.items():
        if name in flavors:
            return loader(uri), explainer
    raise LookupError(f"no explainer for any of {list(flavors)}")


def list_models() -> None:
    """The runs that carry a model, newest first."""
    for experiment in mlflow.search_experiments():
        for model in mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id], output_format="list"
        ):
            assert isinstance(model.source_run_id, str), "Model has no run_id ."
            run = mlflow.get_run(model.source_run_id)
            print(
                f"{model.source_run_id}  {experiment.name:<16}"
                f"{run.data.tags.get('mlflow.runName', '')}"
            )


def rebuild_features(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Replays the run's pipeline to rebuild the arrays it trained on."""
    sessions, events = instantiate(cfg.data_source)()
    for step in instantiate(cfg.event_steps):
        events = step(events)

    frame = instantiate(cfg.series)(sessions, events)
    for step in instantiate(cfg.presplit.steps):
        frame = step(frame)

    train_idx, test_idx = next(iter(instantiate(cfg.split).split(frame.data)))
    train = frame.update(frame.data.iloc[train_idx])
    test = frame.update(frame.data.iloc[test_idx])
    for step in instantiate(cfg.postsplit.fitted_transformations):
        train, test = step(train, test)

    window_conf = OmegaConf.to_container(cfg.window_conf)
    assert isinstance(window_conf, dict), "window_conf must be a mapping"
    window_conf = {str(key): value for key, value in window_conf.items()}

    transformer = instantiate(cfg.postsplit.window_transformation)
    X_train, schema = transformer.transform(SlidingWindow(train, **window_conf))
    X_test, _ = transformer.transform(SlidingWindow(test, **window_conf))
    return X_train, X_test, list(schema.columns)


def get_shap_values(
    run_id: str, background: int = 200, samples: int = 1000, seed: int = 42
) -> shap.Explanation:
    """Compute shapley values.

    Args:
        `run_id`: MLFlow run id from which to take the model.
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

    model, explainer = load_model(run_id)
    print("model:", type(model).__name__)

    cfg = OmegaConf.create(mlflow.artifacts.load_text(f"runs:/{run_id}/config.yaml"))
    assert isinstance(cfg, DictConfig), "the logged config must be a mapping"
    X_train, X_test, names = rebuild_features(cfg)

    # check features match
    logged = mlflow.artifacts.load_dict(f"runs:/{run_id}/features.json")["features"]
    if names != logged:
        raise ValueError("rebuilt features differ from the ones the run logged")
    print(
        f"rebuilt {len(X_train)} train and {len(X_test)} test windows, "
        f"{len(names)} features"
    )

    rng = np.random.default_rng(seed)
    reference = X_train[rng.choice(len(X_train), background, replace=False)]
    explained = X_test[rng.choice(len(X_test), samples, replace=False)]

    # Convert data to tensors if model is a pytorch module
    if isinstance(model, torch.nn.Module):
        model.eval()
        reference, explained = torch.tensor(reference), torch.tensor(explained)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unrecognized nn.Module")
        fitted = explainer(model, reference)
        values = fitted.shap_values(explained)  # type: ignore

    baseline = np.asarray(fitted.expected_value).reshape(-1)[0]  # type: ignore
    return shap.Explanation(
        values=np.asarray(values).reshape(samples, len(names)),
        base_values=np.repeat(baseline, samples),
        data=np.asarray(explained),
        feature_names=names,
    )
