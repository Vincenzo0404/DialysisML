"""Metrics collection module."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

Step = Callable[[pd.DataFrame], pd.DataFrame]


def step_name(step: Step) -> str:
    """Name of a step"""
    return getattr(step, "__name__", None) or step.func.__name__  # type: ignore[attr-defined]


def _is_integral(column: pd.Series) -> bool:
    """Whether every value present is a whole number, missing ones aside."""
    values = column.dropna()
    return len(values) > 0 and all(
        isinstance(v, (int, float)) and float(v).is_integer() for v in values
    )


def profile(df: pd.DataFrame | None) -> dict[str, int | None]:
    """What can be measured on any DataFrame, whatever the step did to it."""
    if df is None:
        return {"rows": None, "patients": None, "series": None}

    out: dict[str, int | None] = {"rows": len(df), "patients": None, "series": None}
    if "patient" in df.columns:
        out["patients"] = int(df["patient"].nunique())
    if {"patient", "event"}.issubset(df.columns):
        out["series"] = int(df.groupby(["patient", "event"], sort=False).ngroups)
    return out


@dataclass(frozen=True)
class StepReport:
    """One step's before/after. Deltas are derived, never supplied."""

    step: str
    before: dict[str, int | None]
    after: dict[str, int | None]
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Holds the reports in order. Rendering lives in the `to_*` methods."""

    MEASURES = ("rows", "patients", "series")

    def __init__(self) -> None:
        self.reports: list[StepReport] = []

    # -- collecting ------------------------------------------------------

    def record(
        self,
        step: str,
        before: pd.DataFrame | None,
        after: pd.DataFrame | None,
        seconds: float = 0.0,
        **extra: Any,
    ) -> None:
        self.reports.append(
            StepReport(step, profile(before), profile(after), seconds, extra)
        )

    def run(self, step: Step, df: pd.DataFrame | None, **extra: Any) -> pd.DataFrame:
        """Runs a step, times it, records it, returns its output."""
        started = time.perf_counter()
        result = step(df)  # type: ignore[arg-type]
        self.record(step_name(step), df, result, time.perf_counter() - started, **extra)
        return result

    # -- rendering -------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """One row per step, with the deltas computed here.

        No step ever has to know how much it discarded: that is a function of
        its before and after.
        """
        rows = []
        for report in self.reports:
            row: dict[str, Any] = {
                "step": report.step,
                "seconds": round(report.seconds, 2),
            }
            for measure in self.MEASURES:
                before, after = report.before[measure], report.after[measure]
                row[measure] = after
                row[f"{measure}_delta"] = (
                    after - before if before is not None and after is not None else None
                )
            row.update(report.extra)
            rows.append(row)

        frame = pd.DataFrame(rows)
        # a measure absent from every step says nothing: drop the empty columns
        return frame.dropna(axis=1, how="all")

    def to_console(self, columns: Sequence[str] | None = None) -> str:
        frame = self.to_frame()
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]

        # a count turns into a float wherever a step left it missing, which
        # would print every row as `113864.0`. Formatting to text rather than
        # back to int: an int column holding a NaN is promoted to float again.
        counts = frame.drop(columns=["step", "seconds"], errors="ignore")
        integral = [c for c in counts.columns if _is_integral(counts[c])]
        frame[integral] = counts[integral].map(
            lambda v: "-" if pd.isna(v) else str(int(v))
        )

        return frame.fillna("-").to_string(index=False)

    def to_csv(self, path: str | Path) -> pd.DataFrame:
        frame = self.to_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return frame

    def to_mlflow(self, artifact_name: str = "pipeline_funnel.csv") -> None:
        """Logs the funnel as an artifact, plus the final counts as params.

        Not `log_metric`: these are not a time series, and MLflow's step axis
        would be meaningless for them.
        """
        import mlflow

        frame = self.to_frame()
        mlflow.log_text(frame.to_csv(index=False), artifact_name)

        if not frame.empty:
            last = frame.iloc[-1]
            for measure in self.MEASURES:
                if measure in frame.columns and pd.notna(last[measure]):
                    mlflow.log_param(f"final_{measure}", int(last[measure]))
