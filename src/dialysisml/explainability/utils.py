import warnings
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import shap
import torch
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig, OmegaConf

from dialysisml import config
from dialysisml.pipeline.run_steps import run_fitted, run_unfitted, split_frame
from dialysisml.schema import Role, select

# Wich SHAP algorithm to use for each Library
SHAP_ALGORITHMS = {
    "pytorch": (mlflow.pytorch.load_model, shap.DeepExplainer),  # type: ignore
    "xgboost": (mlflow.xgboost.load_model, shap.TreeExplainer),  # type: ignore
    "sklearn": (mlflow.sklearn.load_model, shap.TreeExplainer),  # type: ignore
}


def load_model(
    run_id: str, bucket: int | None = None
) -> tuple[Any, type[shap.Explainer]]:
    """The run's model, and the explainer its flavor calls for.

    MLflow records the flavor alongside the model, so nothing here has to know
    in advance which library trained it. A model that wraps one booster per
    bucket needs `bucket` to say which to explain: a TreeExplainer takes one
    tree model, not a curve over several.
    """
    uri = f"runs:/{run_id}/model"
    try:
        flavors = mlflow.models.get_model_info(uri).flavors  # type: ignore
    except MlflowException as error:
        raise LookupError(f"run {run_id} logged no model.") from error

    for library, (loader, explainer) in SHAP_ALGORITHMS.items():
        if library in flavors:
            return loader(uri), explainer

    models = getattr(mlflow.pyfunc.load_model(uri).unwrap_python_model(), "models", None)
    if models is None:
        raise LookupError(f"no explainer for any of {list(flavors)}")
    if bucket is None:
        raise ValueError(
            f"this run's model holds {len(models)} boosters: pass bucket=1.."
            f"{len(models)} to pick which one to explain"
        )
    return models[bucket - 1], shap.TreeExplainer


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
    train, test = run_fitted(cfg, *split_frame(cfg, run_unfitted(cfg)))
    names = select(train.schema, role=Role.FEATURE)
    return (
        train.data[names].to_numpy(np.float32),
        test.data[names].to_numpy(np.float32),
        names,
    )


def get_shap_values(
    run_id: str,
    background: int = 200,
    samples: int = 1000,
    seed: int = 42,
    bucket: int | None = None,
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

    model, explainer = load_model(run_id, bucket)
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
