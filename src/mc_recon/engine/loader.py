"""Parse raw CSVs through a confirmed :class:`ColumnMapping` into typed records.

Fails loudly (:class:`ParseError` / :class:`MappingError` / :class:`EngineInputError`) on
anything it cannot parse unambiguously. Normalisation is limited to formatting that cannot
change meaning: loan-id case/whitespace, thousands separators, currency and percent signs.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import InvestorRecord, LoadedInputs, PaymentRecord, ServicingRecord
from .schema import (
    ColumnMapping,
    EngineInputError,
    FieldType,
    ParseError,
    Source,
    SourceMapping,
    field_specs,
)

_WS = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return ``(headers, rows)``. Rows with the wrong number of cells are an error."""
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        try:
            headers = [h.strip() for h in next(reader)]
        except StopIteration:
            raise EngineInputError(f"{path}: file is empty") from None
        if len(set(headers)) != len(headers):
            raise EngineInputError(f"{path}: duplicate column headers {headers}")
        rows: list[dict[str, str]] = []
        for i, cells in enumerate(reader, start=1):
            if not any(c.strip() for c in cells):
                continue  # blank trailing line
            if len(cells) != len(headers):
                raise EngineInputError(
                    f"{path}: data row {i} has {len(cells)} cells, expected {len(headers)}"
                )
            rows.append(dict(zip(headers, cells, strict=True)))
    return headers, rows


def read_headers(path: Path) -> list[str]:
    return read_csv(path)[0]


def normalize_loan_id(value: str) -> str:
    return _WS.sub("", value).upper()


def _parse_decimal(text: str, strip_chars: str) -> Decimal:
    cleaned = text.strip()
    for ch in strip_chars:
        cleaned = cleaned.replace(ch, "")
    value = Decimal(cleaned)  # raises InvalidOperation
    if not value.is_finite():
        raise InvalidOperation
    return value


def _parse_value(
    raw: str, ftype: FieldType, date_format: str | None, *, source: Source, row: int, column: str
) -> object:
    text = raw.strip()
    try:
        match ftype:
            case FieldType.LOAN_ID:
                value = normalize_loan_id(text)
                if not value:
                    raise ValueError("empty loan id")
                return value
            case FieldType.STRING:
                return text
            case FieldType.MONEY:
                return _parse_decimal(text, ",£$€ ")
            case FieldType.RATE:
                return _parse_decimal(text, "% ")
            case FieldType.DATE:
                if date_format is None:
                    return date.fromisoformat(text)
                return datetime.strptime(text, date_format).date()
    except (ValueError, InvalidOperation) as exc:
        reason = str(exc) or f"not a valid {ftype.value}"
        raise ParseError(source, row, column, raw, reason) from None
    raise AssertionError(f"unhandled field type {ftype}")


def _parse_rows(
    source: Source, rows: list[dict[str, str]], sm: SourceMapping
) -> list[dict[str, object]]:
    specs = field_specs(source)
    parsed: list[dict[str, object]] = []
    for i, raw_row in enumerate(rows, start=1):
        record: dict[str, object] = {"row": i}
        for name, column in sm.columns.items():
            spec = specs[name]
            raw = raw_row[column]
            if not raw.strip():
                if spec.required:
                    raise ParseError(source, i, column, raw, f"required field {name!r} is empty")
                record[name] = None
                continue
            record[name] = _parse_value(
                raw, spec.type, sm.date_format, source=source, row=i, column=column
            )
        parsed.append(record)
    return parsed


def _unique(source: Source, records: list, attr: str) -> None:
    seen: dict[str, int] = {}
    for r in records:
        key = getattr(r, attr)
        if key in seen:
            raise EngineInputError(
                f"{source}: {attr} {key!r} appears on rows {seen[key]} and {r.row}; "
                "expected one row per key for the period"
            )
        seen[key] = r.row


def load_inputs(paths: Mapping[Source, Path], mapping: ColumnMapping) -> LoadedInputs:
    missing = [s.value for s in Source if s not in paths]
    if missing:
        raise EngineInputError(f"missing input files for: {missing}")

    tables = {s: read_csv(paths[s]) for s in Source}
    mapping.validate_against({s: headers for s, (headers, _) in tables.items()})

    servicing = [
        ServicingRecord(**r)
        for r in _parse_rows(Source.SERVICING, tables[Source.SERVICING][1], mapping.servicing)
    ]
    payments = [
        PaymentRecord(**r)
        for r in _parse_rows(Source.PAYMENTS, tables[Source.PAYMENTS][1], mapping.payments)
    ]
    investor = [
        InvestorRecord(**r)
        for r in _parse_rows(Source.INVESTOR, tables[Source.INVESTOR][1], mapping.investor)
    ]
    _unique(Source.SERVICING, servicing, "loan_id")
    _unique(Source.INVESTOR, investor, "loan_id")
    _unique(Source.PAYMENTS, payments, "txn_id")

    return LoadedInputs(
        servicing=tuple(servicing),
        payments=tuple(payments),
        investor=tuple(investor),
        file_hashes={s: sha256_file(paths[s]) for s in Source},
    )
