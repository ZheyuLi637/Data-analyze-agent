from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd


DEFAULT_MAX_ANALYSIS_ROWS = 5000


@dataclass
class LoadedCSV:
    dataframe: pd.DataFrame | None
    source_name: str
    source_id: str
    error: str | None = None
    original_row_count: int = 0
    sampled: bool = False
    header_mode: str = "present"
    detected_header_row: int = 0
    notes: list[str] | None = None


def load_csv_path(
    path: Path,
    has_header: bool | str = True,
    max_rows: int = DEFAULT_MAX_ANALYSIS_ROWS,
) -> LoadedCSV:
    return load_csv_bytes(path.read_bytes(), path.name, has_header, max_rows=max_rows)


def load_csv_bytes(
    content: bytes,
    source_name: str,
    has_header: bool | str = True,
    max_rows: int = DEFAULT_MAX_ANALYSIS_ROWS,
) -> LoadedCSV:
    header_mode = _normalize_header_mode(has_header)
    source_id = _source_id(content, source_name, header_mode, max_rows)
    if not content.strip():
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error="Uploaded CSV is empty. Choose a CSV with headers or at least one data row.",
            header_mode=header_mode,
        )

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error="Uploaded CSV could not be decoded as text. Save it as UTF-8 CSV and upload again.",
            header_mode=header_mode,
        )

    lines = text.splitlines()
    detected_header_row = _detect_header_row(lines) if header_mode == "auto" else 0
    header = None if header_mode == "absent" else 0
    skiprows = detected_header_row if header_mode in {"auto", "present"} else 0
    original_row_count = _estimate_data_rows(lines, skiprows, header_mode)
    sampled = original_row_count > max_rows
    notes = _load_notes(header_mode, detected_header_row, original_row_count, max_rows)

    try:
        dataframe = pd.read_csv(BytesIO(content), header=header, skiprows=skiprows, nrows=max_rows)
    except pd.errors.EmptyDataError:
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error="Uploaded CSV has no readable columns. Check whether the file is empty or malformed.",
            header_mode=header_mode,
            detected_header_row=detected_header_row,
            original_row_count=original_row_count,
            notes=notes,
        )
    except pd.errors.ParserError as exc:
        return LoadedCSV(
            dataframe=None,
            source_name=source_name,
            source_id=source_id,
            error=f"Uploaded CSV could not be parsed: {exc}",
            header_mode=header_mode,
            detected_header_row=detected_header_row,
            original_row_count=original_row_count,
            notes=notes,
        )

    if header_mode == "absent":
        dataframe.columns = [f"column_{index + 1}" for index in range(len(dataframe.columns))]

    return LoadedCSV(
        dataframe=dataframe,
        source_name=source_name,
        source_id=source_id,
        original_row_count=original_row_count,
        sampled=sampled,
        header_mode=header_mode,
        detected_header_row=detected_header_row,
        notes=notes,
    )


def _normalize_header_mode(has_header: bool | str) -> str:
    if has_header is True:
        return "present"
    if has_header is False:
        return "absent"
    if has_header in {"present", "absent", "auto"}:
        return has_header
    raise ValueError("header mode must be True, False, 'present', 'absent', or 'auto'")


def _detect_header_row(lines: list[str]) -> int:
    best_index = 0
    best_score = -1
    for index, line in enumerate(lines[:25]):
        row = _parse_csv_line(line)
        if len(row) < 2:
            continue
        next_row = _parse_csv_line(lines[index + 1]) if index + 1 < len(lines) else []
        score = _header_score(row, next_row)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _parse_csv_line(line: str) -> list[str]:
    try:
        return next(csv.reader([line]))
    except csv.Error:
        return []


def _header_score(row: list[str], next_row: list[str]) -> int:
    cleaned = [cell.strip() for cell in row]
    score = 0
    score += len(cleaned)
    score += 4 if len(set(cleaned)) == len(cleaned) else -4
    score += sum(1 for cell in cleaned if cell and not _looks_number(cell))
    score += 3 if next_row and len(next_row) == len(row) else -2
    score -= sum(1 for cell in cleaned if not cell)
    return score


def _looks_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _estimate_data_rows(lines: list[str], skiprows: int, header_mode: str) -> int:
    non_empty = [line for line in lines[skiprows:] if line.strip()]
    if header_mode in {"present", "auto"} and non_empty:
        return max(0, len(non_empty) - 1)
    return len(non_empty)


def _load_notes(header_mode: str, detected_header_row: int, original_row_count: int, max_rows: int) -> list[str]:
    notes: list[str] = []
    if header_mode == "auto" and detected_header_row > 0:
        notes.append(f"Skipped {detected_header_row} preamble row(s) before the detected header.")
    if original_row_count > max_rows:
        notes.append(f"Loaded the first {max_rows} rows for analysis out of about {original_row_count} data rows.")
    return notes


def _source_id(content: bytes, source_name: str, header_mode: str, max_rows: int) -> str:
    digest = hashlib.sha256(content).hexdigest()[:16]
    return f"{source_name}:{digest}:{header_mode}:{max_rows}"
