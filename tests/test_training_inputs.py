"""Tests that Experiment + FFNNAdapter feed the network the right data.

The data never leaves train(), so we cannot compare a return value. Instead we
inject spies into the three seams the code already offers:
- FeedForwardNN, monkeypatched, records what every forward() receives;
- loss_function, a config field, records the shapes it is called with;
- metrics, a run() argument, records the full target vectors.

The oracle is rebuilt from the RAW windows, so the test also proves the store
served transformed data rather than the raw one.
"""

import functools
from importlib import import_module

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from dialysisml.config import DATA_PATH
from dialysisml.Experiment import Experiment
from dialysisml.metrics import mape
from dialysisml.modelconf import FFNNConfig
from dialysisml.models import FeedForwardNN
from dialysisml.window import RegressionFeatureTransformer, WindowConfig, WindowStore


@pytest.fixture(scope="module")
def stores() -> tuple[WindowStore, WindowStore]:
    """Raw and transformed stores from data/, read once for the whole file."""
    conf = WindowConfig(size=30, stride=1)
    npz = DATA_PATH / str(conf) / f"{conf}.npz"
    if not npz.exists():
        pytest.skip(f"{npz} not found: generate the windows first")
    return WindowStore(conf), WindowStore(conf, RegressionFeatureTransformer())


def spying_network(seen: dict):
    """FeedForwardNN subclass recording its input_dim and every forward() input."""

    class SpyFFNN(FeedForwardNN):
        def __init__(self, input_dim, hidden_layers, dropout):
            super().__init__(input_dim, hidden_layers, dropout)
            seen["input_dim"] = input_dim

        def forward(self, x):
            seen["forwards"].append(x.detach().cpu().numpy())
            return super().forward(x)

    return SpyFFNN


def test_network_is_fed_the_transformed_windows(stores, tmp_path, monkeypatch):
    raw, transformed = stores
    transformer = RegressionFeatureTransformer()

    # --- oracle: rebuilt independently, starting from the RAW windows ---
    scaler = StandardScaler()
    expected_train = scaler.fit_transform(transformer.transform_X(raw.X_train))
    expected_test = scaler.transform(transformer.transform_X(raw.X_test))
    n_features = raw.X_train.shape[2]

    # --- spies ---
    seen: dict = {"forwards": [], "input_dim": None}
    loss_shapes: list[tuple] = []
    seen_targets: list[np.ndarray] = []

    @functools.wraps(mape)  # keeps __name__ == "mape", used for the column names
    def spy_loss(y_hat, y_true):
        loss_shapes.append((tuple(y_hat.shape), tuple(y_true.shape)))
        return mape(y_hat, y_true)

    @functools.wraps(mape)
    def spy_metric(preds, y_true):
        seen_targets.append(y_true.detach().cpu().numpy())
        return mape(preds, y_true)

    # Patch the name where it is USED, not where the class is defined.
    # import_module is needed because the package attribute FFNNAdapter is the
    # class, which shadows the submodule of the same name.
    adapter_module = import_module("dialysisml.adapters.FFNNAdapter")
    monkeypatch.setattr(adapter_module, "FeedForwardNN", spying_network(seen))
    # keep MLflow out of the project's mlruns/; setenv is undone after the test
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())

    config = FFNNConfig(
        name="spy",
        hidden_layers=(8,),
        dropout=0.0,
        epochs=1,
        batch_size=8192,
        loss_function=spy_loss,
    )
    Experiment(config, transformed).run([spy_metric])

    # --- 1. the network is sized on the transformed features, not the raw ones ---
    assert seen["input_dim"] == 4 * n_features == 132  # not 33, and not 30*33

    # --- 2. the end-of-epoch eval pass sees the whole matrix, unshuffled ---
    full_train = [a for a in seen["forwards"] if a.shape[0] == len(expected_train)]
    full_test = [a for a in seen["forwards"] if a.shape[0] == len(expected_test)]
    assert len(full_train) == 1, "expected exactly one full-train forward per epoch"
    assert len(full_test) == 1

    np.testing.assert_allclose(full_train[0], expected_train, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(full_test[0], expected_test, rtol=1e-4, atol=1e-4)

    # --- 3. every training batch is made of rows of the expected matrix ---
    batches = [a for a in seen["forwards"] if a.shape[0] < len(expected_train)]
    assert sum(len(a) for a in batches) >= len(expected_train)
    assert all(a.shape[1] == 132 for a in batches)

    # --- 4. targets arrive whole and flat (a (N, 1) would broadcast in the loss) ---
    assert any(
        np.array_equal(t, raw.y_train.reshape(-1).astype(np.float32))
        for t in seen_targets
    ), "the full y_train never reached the metrics"
    assert all(shapes[0] == shapes[1] for shapes in loss_shapes), (
        f"loss called with mismatched shapes: {loss_shapes}"
    )
