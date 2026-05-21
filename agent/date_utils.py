from __future__ import annotations

from dataclasses import dataclass
import warnings

import pandas as pd


COMMON_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y.%m.%d",
    "%d %b %Y",
    "%b %d %Y",
)


@dataclass
class DateParseResult:
    parsed: pd.Series
    success_percent: float
    invalid_examples: list[str]


def parse_mixed_dates(series: pd.Series) -> DateParseResult:
    values = series.dropna()
    if values.empty:
        return DateParseResult(pd.Series(dtype="datetime64[ns]"), 0.0, [])

    parsed_values = values.apply(_parse_one_date)
    success_percent = round(float(parsed_values.notna().mean() * 100), 1)
    invalid_examples = values[parsed_values.isna()].astype(str).drop_duplicates().head(3).tolist()
    return DateParseResult(parsed_values, success_percent, invalid_examples)


def _parse_one_date(value) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    if not text or text.lower() in {"missing", "unknown", "none", "nan", "null"}:
        return pd.NaT

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed

    for fmt in COMMON_DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return pd.NaT
