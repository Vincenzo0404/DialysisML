"""Metrics collection module."""

import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from dialysisml.pipeline.SlidingWindow import GROUP_COLUMNS

Step = Callable[..., pd.DataFrame]

logger = logging.getLogger(__name__)


def step_name(step: Step) -> str:
    """Name of a step"""
    return getattr(step, "__name__", None) or step.func.__name__  # type: ignore[attr-defined]


def _is_integral(column: pd.Series) -> bool:
    """Whether every value present is a whole number, missing ones aside."""
    values = column.dropna()
    return len(values) > 0 and all(
        isinstance(v, (int, float)) and float(v).is_integer() for v in values
    )


def profile(
    df: pd.DataFrame, metrics: dict[str, Callable]
) -> dict[str, Optional[float]]:
    """Compute given metrics over the dataset."""

    record = {}
    for metric_name, metric_fn in metrics.items():
        val = metric_fn(df)
        record[metric_name] = val
    return record


@dataclass(frozen=True)
class Record:
    """One row: `idx` names it, `data` is everything measured about it."""

    idx: str
    data: dict[str, Optional[float]]
    seconds: float


class MetricCollector:
    """Holds the records in order. Rendering lives in the `to_*` methods.

    `metrics` is what to measure on every frame it is handed; without it a
    record carries only what the caller passes as `extra`.
    """

    def __init__(self, metrics: Optional[dict[str, Callable]] = None) -> None:
        self.records: list[Record] = []
        self.metrics = metrics

    # -- collecting ------------------------------------------------------

    def add_record(
        self,
        df: Optional[pd.DataFrame] = None,
        seconds: float = 0.0,
        *,
        idx: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Adds a row. `idx` labels it for `aggregate_folds`/`to_mlflow_scalars`
        to key on; omit it when a collector only ever holds one row — there is
        nothing to group or prefix against, so a position-based label is fine.
        """
        # round floats
        extra = {
            key: round(val, 3) if isinstance(val, float) else val
            for key, val in extra.items()
        }
        if idx is None:
            idx = str(len(self.records))
        data = {**extra}
        if self.metrics is not None and df is not None:
            data.update(profile(df, self.metrics))
        self.records.append(Record(idx, data, seconds))

    def run(
        self, step: Step, df: pd.DataFrame | None = None, **extra: Any
    ) -> pd.DataFrame:
        """Runs a step, times it, records it, returns its output.

        `df=None` marks a source step: it produces a frame instead of taking
        one, so it is called with no argument.
        """
        started = time.perf_counter()
        result = step() if df is None else step(df)
        finished = time.perf_counter()

        self.add_record(result, finished - started, idx=step_name(step), **extra)
        return result

    # -- rendering -------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """Returns records list as a dataframe."""
        rows = []
        for record in self.records:
            row: dict[str, Any] = {
                "idx": record.idx,
                "seconds": round(record.seconds, 2),
            }
            for measure, value in record.data.items():
                row[measure] = value
            rows.append(row)

        frame = pd.DataFrame(rows)
        # a column of zeros says nothing: an epochs summary has no durations
        if "seconds" in frame and (frame["seconds"] == 0).all():
            frame = frame.drop(columns="seconds")
        # remove None cols
        return frame.dropna(axis=1, how="all")

    def to_console(self, columns: Sequence[str] | None = None) -> str:
        frame = self.to_frame()
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]

        # a count turns into a float wherever a step left it missing, which
        # would print every row as `113864.0`. Formatting to text rather than
        # back to int: an int column holding a NaN is promoted to float again.
        counts = frame.drop(columns=["idx", "seconds"], errors="ignore")
        integral = [c for c in counts.columns if _is_integral(counts[c])]
        frame[integral] = counts[integral].map(
            lambda v: "-" if pd.isna(v) else str(int(v))
        )

        return frame.to_string(
            index=False, na_rep="-", float_format=lambda v: f"{v:.2f}"
        )

    def to_csv(self, path: str | Path) -> pd.DataFrame:
        frame = self.to_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return frame

    def to_mlflow(self, artifact_name: str = "pipeline_funnel.csv") -> None:
        """Attaches the funnel to the active run.

        The table as an artifact, since it depends on the configured steps;
        the last row as metrics, so runs stay comparable in the runs table
        without opening it. Metrics rather than params: these are measured
        floats, and params compare as strings.
        """
        import mlflow

        frame = self.to_frame()
        mlflow.log_text(frame.to_csv(index=False), artifact_name)

    def to_mlflow_scalars(self, prefix: str = "") -> None:
        """Every record as a `{prefix}{idx}_{measure}` single-valued metric."""
        import mlflow

        for record in self.records:
            for measure, value in record.data.items():
                if value is not None and pd.notna(value):
                    mlflow.log_metric(f"{prefix}{record.idx}_{measure}", float(value))


# custom metrics
def unique_values(df: pd.DataFrame, col: str) -> int:
    return df[col].nunique(dropna=True)


def count_groups(df: pd.DataFrame, by: Sequence[str]) -> int:
    # dropna=False or a censored series, whose `event` is null, is not counted
    return df.groupby(by=list(by), dropna=False).ngroups


def column_stat(
    df: pd.DataFrame, *, col: str, stat: Callable[[pd.Series], Any]
) -> Optional[float]:
    """One statistic of one column, or None where the column is not there yet.

    The target is built partway through the presplit, so the steps before it
    have nothing to describe. None leaves those cells empty rather than
    failing, the same way a step that drops a column does.
    """
    if col not in df.columns:
        return None
    return float(stat(df[col]))


# The target's shape, step by step. One column per statistic rather than a
# `describe` off to the side: which step flattens the distribution is the thing
# worth seeing, and that only shows up next to the row counts that explain it.
# `p25` and not `25%`: MLflow rejects `%` in a metric name.
TTE_STATS: dict[str, Callable[[pd.Series], Any]] = {
    "min": pd.Series.min,
    "p25": lambda values: values.quantile(0.25),
    "p50": lambda values: values.quantile(0.50),
    "p75": lambda values: values.quantile(0.75),
    "max": pd.Series.max,
    "mean": pd.Series.mean,
    "std": pd.Series.std,
}

# What a frame is worth measuring by, wherever it appears in the pipeline.
FRAME_METRICS: dict[str, Callable] = {
    "rows": len,
    "patients": partial(unique_values, col="patient"),
    **{
        f"tte_{name}": partial(column_stat, col="tte", stat=stat)
        for name, stat in TTE_STATS.items()
    },
}


# -- preconfigured collectors ---------------------------------------------
# Functions, not instances: a collector accumulates records, so a module-level
# one would be shared across folds and across the jobs of a sweep.


def metadata_collector() -> MetricCollector:
    """Tracks how a frame narrows: the presplit funnel, or a fold's two sides."""
    return MetricCollector(FRAME_METRICS)


def epochs_collector(results: pd.DataFrame, score: str) -> MetricCollector:
    """Best value and epoch of the measure the model is judged by.

    Only that one: the per-epoch table keeps every metric, and a best per metric
    is a column you cannot read across models that optimise different things.
    """
    column = f"test_{score}"
    best = results.loc[results[column].idxmin()]
    collector = MetricCollector()
    collector.add_record(
        idx="test_score",
        best_value=float(best[column]),
        best_epoch=float(best["epoch"]),
        total_epochs=len(results),
    )
    return collector


def aggregate_folds(folds: Sequence[MetricCollector]) -> MetricCollector:
    """Mean and stddev of every measure across the folds of a cross-validation."""
    if len(folds) == 1:
        return folds[0]

    frame = pd.concat([fold.to_frame() for fold in folds], ignore_index=True)
    collector = MetricCollector()

    for reduction, group in frame.groupby("idx", sort=False):
        measures = group.drop(columns=["idx", "seconds"], errors="ignore")
        collector.add_record(None, idx=f"{reduction}_mean", **measures.mean().to_dict())
        if len(folds) > 1:
            collector.add_record(
                None, idx=f"{reduction}_std", **measures.std().to_dict()
            )

    return collector
