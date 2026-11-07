"""Column profiling for the ``ingest`` node.

Profiles are the *only* view of the data the mapping LLM gets: header names, inferred
types, null/distinct counts and a few sample values. Samples are raw file content, so the
ingest-time input-safety screen must run over them before they reach a prompt.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .loader import read_csv, sha256_file
from .schema import Source

_INT = re.compile(r"^[+-]?\d+$")
_DEC = re.compile(r"^[+-]?[£$€]?\d{1,3}(,\d{3})*(\.\d+)?%?$|^[+-]?[£$€]?\d+(\.\d+)?%?$")
_DATE_FORMATS = ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d")


class ColumnProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    inferred_type: str
    non_null: int
    null_count: int
    distinct_count: int
    samples: list[str]
    date_formats: list[str] = []
    """For date columns: every candidate ``strptime`` format that parses all values."""


class FileProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Source
    file_name: str
    sha256: str
    row_count: int
    columns: list[ColumnProfile]


def _is_iso_date(v: str) -> bool:
    try:
        date.fromisoformat(v)
        return True
    except ValueError:
        return False


def _formats_parsing_all(values: list[str]) -> list[str]:
    ok = []
    for fmt in _DATE_FORMATS:
        try:
            for v in values:
                datetime.strptime(v, fmt)
        except ValueError:
            continue
        ok.append(fmt)
    return ok


def _infer(values: list[str]) -> tuple[str, list[str]]:
    if not values:
        return "empty", []
    if all(_is_iso_date(v) for v in values):
        return "date", ["ISO-8601"]
    formats = _formats_parsing_all(values)
    if formats:
        return "date", formats
    if all(_INT.match(v) for v in values):
        return "integer", []
    if all(_DEC.match(v) for v in values):
        return "decimal", []
    return "string", []


def profile_csv(path: Path, source: Source, sample_size: int = 3) -> FileProfile:
    headers, rows = read_csv(path)
    columns = []
    for h in headers:
        values = [r[h].strip() for r in rows]
        non_null = [v for v in values if v]
        samples: list[str] = []
        for v in non_null:
            if v not in samples:
                samples.append(v)
            if len(samples) == sample_size:
                break
        inferred, formats = _infer(non_null)
        columns.append(
            ColumnProfile(
                name=h,
                inferred_type=inferred,
                non_null=len(non_null),
                null_count=len(values) - len(non_null),
                distinct_count=len(set(non_null)),
                samples=samples,
                date_formats=formats,
            )
        )
    return FileProfile(
        source=source,
        file_name=Path(path).name,
        sha256=sha256_file(path),
        row_count=len(rows),
        columns=columns,
    )
