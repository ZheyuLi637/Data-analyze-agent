from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import pandas as pd

from agent.date_utils import parse_mixed_dates


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    date_columns: list[str]
    missing_values: dict[str, int]
    missing_percent: dict[str, float]
    date_parse_percent: dict[str, float]
    date_invalid_examples: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


def perceive_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Convert a dataframe into the compact state used by the agent."""
    numeric_columns = list(df.select_dtypes(include="number").columns)
    date_columns, date_parse_percent, date_invalid_examples = _detect_date_columns(df)
    categorical_columns = [
        column
        for column in df.select_dtypes(include=["object", "category", "bool"]).columns
        if column not in date_columns
    ]
    missing_values = df.isna().sum().astype(int).to_dict()
    row_count = int(len(df))

    if row_count:
        missing_percent = {
            column: round((count / row_count) * 100, 2)
            for column, count in missing_values.items()
        }
    else:
        missing_percent = {column: 0.0 for column in df.columns}

    return DatasetProfile(
        row_count=row_count,
        column_count=int(len(df.columns)),
        columns=list(df.columns),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        date_columns=date_columns,
        missing_values=missing_values,
        missing_percent=missing_percent,
        date_parse_percent=date_parse_percent,
        date_invalid_examples=date_invalid_examples,
    )


def _detect_date_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, float], dict[str, list[str]]]:
    date_columns: list[str] = []
    date_parse_percent: dict[str, float] = {}
    date_invalid_examples: dict[str, list[str]] = {}
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            continue
        name_hint = any(token in column.lower() for token in ("date", "time", "day"))
        if not name_hint and not pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = parse_mixed_dates(series)
        date_parse_percent[column] = parsed.success_percent
        date_invalid_examples[column] = parsed.invalid_examples
        if parsed.success_percent >= 50:
            date_columns.append(column)
    return date_columns, date_parse_percent, date_invalid_examples
