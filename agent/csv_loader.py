from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd


@dataclass
class LoadedCSV:
    dataframe: pd.DataFrame | None
    source_name: str
    source_id: str
    error: str | None = None


def load_csv_path(path: Path, has_header: bool = True) -> LoadedCSV:
    return load_csv_bytes(path.read_bytes(), path.name, has_header)


def load_csv_bytes(content: bytes, source_name: str, has_header: bool = True) -> LoadedCSV:
    source_id = _source_id(content, source_name, has_header)
    if not content.strip():
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error="Uploaded CSV is empty. Choose a CSV with headers or at least one data row.",
        )

    try:
        dataframe = pd.read_csv(BytesIO(content), header=0 if has_header else None)
    except UnicodeDecodeError:
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error="Uploaded CSV could not be decoded as text. Save it as UTF-8 CSV and upload again.",
        )
    except pd.errors.EmptyDataError:
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error="Uploaded CSV has no readable columns. Check whether the file is empty or malformed.",
        )
    except pd.errors.ParserError as exc:
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error=f"Uploaded CSV could not be parsed: {exc}",
        )

    if not has_header:
        dataframe.columns = [f"column_{index + 1}" for index in range(len(dataframe.columns))]

    return LoadedCSV(
        dataframe=dataframe,
        source_name=source_name,
        source_id=source_id,
    )


def _source_id(content: bytes, source_name: str, has_header: bool) -> str:
    digest = hashlib.sha256(content).hexdigest()[:16]
    header_flag = "header" if has_header else "no-header"
    return f"{source_name}:{digest}:{header_flag}"
