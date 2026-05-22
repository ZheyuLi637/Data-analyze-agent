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
    numeric_parse_percent: dict[str, float]
    date_parse_percent: dict[str, float]
    date_invalid_examples: dict[str, list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


def perceive_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Convert a dataframe into the compact state used by the agent."""
    numeric_columns = list(df.select_dtypes(include="number").columns)
    inferred_numeric, numeric_parse_percent = _detect_numeric_strings(df, numeric_columns)
    numeric_columns = numeric_columns + inferred_numeric
    date_columns, date_parse_percent, date_invalid_examples = _detect_date_columns(df)
    categorical_columns = [
        column
        for column in df.select_dtypes(include=["object", "category", "bool"]).columns
        if column not in date_columns and column not in numeric_columns
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
        numeric_parse_percent=numeric_parse_percent,
        date_parse_percent=date_parse_percent,
        date_invalid_examples=date_invalid_examples,
    )


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
    )
    is_percent = cleaned.str.endswith("%")
    cleaned = cleaned.str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce").astype(float)
    numeric.loc[is_percent & numeric.notna()] = numeric.loc[is_percent & numeric.notna()] / 100
    return numeric


def _detect_numeric_strings(df: pd.DataFrame, existing_numeric: list[str]) -> tuple[list[str], dict[str, float]]:
    inferred: list[str] = []
    parse_percent: dict[str, float] = {column: 100.0 for column in existing_numeric}
    for column in df.select_dtypes(include=["object", "category"]).columns:
        series = df[column].dropna()
        if series.empty:
            continue
        numeric = clean_numeric_series(series)
        success_percent = round(float(numeric.notna().mean() * 100), 1)
        if success_percent >= 80:
            inferred.append(column)
            parse_percent[column] = success_percent
    return inferred, parse_percent


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
