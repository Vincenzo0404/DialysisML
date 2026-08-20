from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from dialysisml.config import DATA_PATH
from dialysisml.features import FINAL_FEATURES, RAW_FEATURES
from dialysisml.window.transformers import WindowTransformer


class Windows(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class Split(NamedTuple):
    """One dataset split: the windows, their targets, and one metadata row each.

    `meta` is aligned to `X` and `y` by position: row i of `meta` describes
    window `X[i]`. Nothing may reorder or filter one without the others.
    """

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


@dataclass()
class WindowConfig:
    """Configuration for windowing time series data."""

    stride: int = 1
    size: int = 30
    raw_features: tuple[str, ...] = tuple(RAW_FEATURES)
    features: tuple[str, ...] = tuple(FINAL_FEATURES)
    # How to handle values still missing after the per-patient ffill/bfill.
    # False drops the row, which removes every patient who was never measured
    # for even one feature; True fills with the training-set median instead.
    impute: bool = False

    def __str__(self):
        name = f"size={self.size}_stride={self.stride}"
        return f"{name}_imputed" if self.impute else name


class WindowStore:
    """Stores and manages windowed data for analysis."""

    def __init__(
        self, window_conf: WindowConfig, transformer: Optional[WindowTransformer] = None
    ):
        self.window_conf = window_conf
        self.transformer = transformer
        self.path = DATA_PATH / str(window_conf)
        self._transformed_cache: dict[str, Windows] = {}
        self._load_windows()

        # if a transformer is provided, load its transformed windows directly
        if transformer is not None:
            self.X_train, self.y_train, self.X_test, self.y_test = (
                self.get_transformed_windows(transformer)
            )

        # A transformer rewrites columns, never rows, so the metadata stays
        # aligned. Everything below (stats, groups) depends on that.
        assert len(self.meta_train) == len(self.X_train)
        assert len(self.meta_test) == len(self.X_test)

    def _raw_path(self) -> Path:
        return self.path / f"{self.window_conf}.npz"

    def _meta_path(self, split: str) -> Path:
        # metadata belongs to the WindowConfig, not to the transformer: every
        # transformer of a given config produces the same rows in the same order
        return self.path / f"{self.window_conf}_meta_{split}.parquet"

    def _transform_path(self, transformer: WindowTransformer) -> Path:
        return self.path / f"{self.window_conf}_{transformer}.npz"

    def _load_windows(self):
        # deferred: data_pipeline imports WindowConfig/WindowStore from this module,
        # so importing make_windows at module load time would create a circular import
        from dialysisml.data_pipeline import make_windows

        raw_path = self._raw_path()

        # the windows and their metadata are always written together, so a
        # missing metadata file means the cache predates it and must be rebuilt
        cached = raw_path.exists() and all(
            self._meta_path(s).exists() for s in ("train", "test")
        )

        if not cached:
            print(f"Making windows with {self.window_conf} ...")
            train, test = make_windows(
                fromdb=False, show_df=False, window_conf=self.window_conf
            )
            self.X_train, self.y_train, self.meta_train = train
            self.X_test, self.y_test, self.meta_test = test

            self.path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                raw_path,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
            )
            self.meta_train.to_parquet(self._meta_path("train"))
            self.meta_test.to_parquet(self._meta_path("test"))

        else:
            windows = np.load(raw_path)
            self.X_train = windows["X_train"]
            self.y_train = windows["y_train"]
            self.X_test = windows["X_test"]
            self.y_test = windows["y_test"]
            self.meta_train = pd.read_parquet(self._meta_path("train"))
            self.meta_test = pd.read_parquet(self._meta_path("test"))

    def get_transformed_windows(self, transformer: WindowTransformer) -> Windows:
        """Returns the transformer's output, computing and caching it if needed.

        Cached in memory (per WindowStore instance) and on disk (per WindowConfig
        folder), keyed by `str(transformer)`. Only arrays the transformer actually
        changes (X and/or y) are ever computed or persisted; the rest are reused
        from the raw windows.
        """
        name = str(transformer)

        # Check if the transformed windows are already cached in memory
        if name in self._transformed_cache:
            return self._transformed_cache[name]

        transform_path = self._transform_path(transformer)

        # Check if the transformed windows are already cached on disk
        if transform_path.exists():
            cached = np.load(transform_path)
            X_train = cached["X_train"] if transformer.transforms_X else self.X_train
            X_test = cached["X_test"] if transformer.transforms_X else self.X_test
            y_train = cached["y_train"] if transformer.transforms_y else self.y_train
            y_test = cached["y_test"] if transformer.transforms_y else self.y_test
        else:

            to_save = {}
            # Compute the transformed windows and cache them on disk if they don't exist
            if transformer.transforms_X:
                X_train = transformer.transform_X(self.X_train)
                X_test = transformer.transform_X(self.X_test)
                to_save["X_train"] = X_train
                to_save["X_test"] = X_test
            else:
                X_train, X_test = self.X_train, self.X_test

            if transformer.transforms_y:
                y_train = transformer.transform_y(self.y_train)
                y_test = transformer.transform_y(self.y_test)
                to_save["y_train"] = y_train
                to_save["y_test"] = y_test
            else:
                y_train, y_test = self.y_train, self.y_test

            if to_save:
                self.path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(transform_path, **to_save)

        result = Windows(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        self._transformed_cache[name] = result
        return result

    def available_transforms(self) -> list[str]:
        """Lists transformer names with data already cached on disk for this WindowConfig."""
        prefix = f"{self.window_conf}_"
        return [p.stem[len(prefix) :] for p in self.path.glob(f"{prefix}*.npz")]

    def stats(self) -> pd.DataFrame:
        """Descriptive statistics of both splits, one column per split.

        `windows_per_patient` is the one to watch: windows of the same series
        overlap by all but a few sessions, so a high value means the number of
        rows badly overstates how much independent data there is.
        """
        columns = {}
        for split in ("train", "test"):
            meta = getattr(self, f"meta_{split}")
            y = getattr(self, f"y_{split}").ravel()
            n_patients = meta["patient"].nunique()
            columns[split] = {
                "windows": len(meta),
                "patients": n_patients,
                "series": meta.groupby(["patient", "event"]).ngroups,
                "windows_per_patient": round(len(meta) / n_patients, 1),
                "first_session": meta["t"].min(),
                "last_session": meta["t"].max(),
                "target_min": int(y.min()),
                "target_median": int(np.median(y)),
                "target_max": int(y.max()),
            }

        stats = pd.DataFrame(columns)
        # the split is by patient, so this must be 0 for the test set to mean
        # anything: it is an assumption of split_series, checked here
        stats.loc["patient_overlap"] = len(
            set(self.meta_train["patient"]) & set(self.meta_test["patient"])
        )
        return stats

    def per_patient(self, split: str = "train") -> pd.DataFrame:
        """One row per patient, for ad-hoc questions `stats` does not answer."""
        meta = getattr(self, f"meta_{split}")
        y = getattr(self, f"y_{split}").ravel()

        return (
            meta.assign(target=y)
            .groupby("patient")
            .agg(
                windows=("series_idx", "size"),
                series=("event", "nunique"),
                sessions=("series_len", "max"),
                first_session=("t", "min"),
                last_session=("t", "max"),
                target_median=("target", "median"),
            )
        )

    def groups(self, split: str = "train") -> np.ndarray:
        """Patient of every window, to group a cross validation by patient.

        Pass as the `groups` argument of scikit-learn's GroupKFold so that no
        patient is split across folds.
        """
        return getattr(self, f"meta_{split}")["patient"].to_numpy()

    def params(self) -> dict:
        """Identifying parameters of this store, for experiment logging."""
        conf = asdict(self.window_conf)
        conf.pop("raw_features", None)
        conf.pop("features", None)

        return {
            **conf,
            "transformer": str(self.transformer) if self.transformer else "none",
            "X_train_shape": self.X_train.shape,
            "y_train_shape": self.y_train.shape,
        }

    def __str__(self):
        if self.transformer is None:
            return str(self.window_conf)
        return f"{self.window_conf}_{self.transformer}"
