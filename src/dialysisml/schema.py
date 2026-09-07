"""What each column is, carried alongside the frame it describes.

Schemas are `pandera.DataFrameSchema`; `role` and `kind` live in each column's
`metadata`. The schema's column order is what the windowed array follows, so
the frame's own column order is free.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Sequence

import pandas as pd
import pandera.pandas as pa


class Role(StrEnum):
    FEATURE = "feature"
    LABEL = "label"
    META = "meta"


class Kind(StrEnum):
    NUMERIC = "numeric"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    IDENTIFIER = "identifier"


def columns(
    dtype: Any, role: Role, kind: Kind, names: Sequence[str], **kwargs: Any
) -> dict[str, pa.Column]:
    """Columns sharing a role and a kind, for a `DataFrameSchema` mapping."""
    return {
        name: pa.Column(dtype, metadata={"role": role, "kind": kind}, **kwargs)
        for name in names
    }


def select(schema: pa.DataFrameSchema, **criteria: Any) -> list[str]:
    """The names whose metadata matches every criterion, in schema order."""
    return [
        name
        for name, column in schema.columns.items()
        if all((column.metadata or {}).get(k) == v for k, v in criteria.items())
    ]


@dataclass(frozen=True)
class Frame:
    """A DataFrame and the schema of its columns."""

    data: pd.DataFrame
    schema: pa.DataFrameSchema

    def update(
        self, data: pd.DataFrame, *, add: dict[str, pa.Column] | None = None
    ) -> "Frame":
        """The frame after a step.

        Removals follow from the data; anything new has to be declared.
        """
        schema = self.schema if add is None else self.schema.add_columns(add)
        removed = [c for c in schema.columns if c not in data.columns]
        return Frame(data, schema.remove_columns(removed) if removed else schema)

    def validate(self) -> "Frame":
        """Dtypes, nullability and value checks. Not free -- roughly a second
        on the full frame -- so call it at the boundaries, not per step."""
        self.schema.validate(self.data)
        return self
