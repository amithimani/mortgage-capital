"""Canonical input schema and the column mapping that binds raw files to it.

The LLM *proposes* a :class:`ColumnMapping`; an analyst confirms it; the engine only ever
consumes a validated mapping. :meth:`ColumnMapping.validate_against` fails loudly with every
problem at once, so a bad proposal is rejected before any number is parsed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Source(StrEnum):
    SERVICING = "servicing"
    PAYMENTS = "payments"
    INVESTOR = "investor"


class FieldType(StrEnum):
    STRING = "string"
    LOAN_ID = "loan_id"
    DATE = "date"
    MONEY = "money"
    RATE = "rate"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: FieldType
    description: str
    required: bool = True


CANONICAL_SCHEMA: dict[Source, tuple[FieldSpec, ...]] = {
    Source.SERVICING: (
        FieldSpec("loan_id", FieldType.LOAN_ID, "Loan account identifier"),
        FieldSpec("pool_id", FieldType.STRING, "Securitisation / funding pool"),
        FieldSpec("due_date", FieldType.DATE, "Scheduled payment due date for the period"),
        FieldSpec("scheduled_amount", FieldType.MONEY, "Scheduled payment amount"),
        FieldSpec(
            "closing_balance", FieldType.MONEY, "Principal balance after the scheduled payment"
        ),
        FieldSpec("base_rate", FieldType.RATE, "Reference/base rate, percent"),
        FieldSpec("margin", FieldType.RATE, "Contractual margin over base, percent"),
        FieldSpec("applied_rate", FieldType.RATE, "Rate actually charged, percent"),
        FieldSpec(
            "mandate_ref", FieldType.STRING, "Direct debit mandate reference", required=False
        ),
    ),
    Source.PAYMENTS: (
        FieldSpec("txn_id", FieldType.STRING, "Unique payment transaction id"),
        FieldSpec("loan_id", FieldType.LOAN_ID, "Loan the payment is allocated to"),
        FieldSpec("value_date", FieldType.DATE, "Date funds were received"),
        FieldSpec("amount", FieldType.MONEY, "Amount received"),
        FieldSpec("channel", FieldType.STRING, "Payment channel, e.g. DD / FPS / CHQ"),
        FieldSpec(
            "mandate_ref", FieldType.STRING, "Direct debit mandate reference", required=False
        ),
    ),
    Source.INVESTOR: (
        FieldSpec("loan_id", FieldType.LOAN_ID, "Loan identifier"),
        FieldSpec("pool_id", FieldType.STRING, "Pool the loan is reported under"),
        FieldSpec("period_end", FieldType.DATE, "Reporting period end date"),
        FieldSpec("reported_rate", FieldType.RATE, "Rate reported to the investor, percent"),
        FieldSpec("reported_margin", FieldType.RATE, "Margin reported to the investor, percent"),
        FieldSpec("scheduled_amount", FieldType.MONEY, "Scheduled payment reported to investor"),
        FieldSpec("closing_balance", FieldType.MONEY, "Closing principal balance reported"),
    ),
}


def field_specs(source: Source) -> dict[str, FieldSpec]:
    return {f.name: f for f in CANONICAL_SCHEMA[source]}


def describe_schema() -> dict[str, list[dict[str, object]]]:
    """JSON-friendly description of the canonical schema (used in LLM payloads)."""
    return {
        source.value: [
            {
                "name": f.name,
                "type": f.type.value,
                "required": f.required,
                "description": f.description,
            }
            for f in specs
        ]
        for source, specs in CANONICAL_SCHEMA.items()
    }


class EngineInputError(ValueError):
    """Base class for malformed engine input. The engine never guesses past these."""


class MappingError(EngineInputError):
    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Invalid column mapping:\n  - " + "\n  - ".join(self.problems))


class ParseError(EngineInputError):
    def __init__(self, source: Source, row: int, column: str, value: str, reason: str):
        self.source, self.row, self.column, self.value, self.reason = (
            source,
            row,
            column,
            value,
            reason,
        )
        super().__init__(f"{source.value} row {row}, column {column!r}: {reason} (value={value!r})")


class SourceMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: dict[str, str]
    """canonical field name -> header in the raw file"""

    date_format: str | None = None
    """``strptime`` format for every date column in this file; ``None`` means ISO 8601."""


class ColumnMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    servicing: SourceMapping
    payments: SourceMapping
    investor: SourceMapping

    def for_source(self, source: Source) -> SourceMapping:
        return getattr(self, source.value)

    def problems(self, headers: Mapping[Source, Sequence[str]]) -> list[str]:
        """Every reason this mapping can't be applied to files with ``headers``."""
        problems: list[str] = []
        for source in Source:
            sm = self.for_source(source)
            specs = field_specs(source)
            file_headers = set(headers.get(source, ()))
            for name in sorted(set(sm.columns) - set(specs)):
                problems.append(f"{source}: unknown canonical field {name!r}")
            for spec in specs.values():
                if spec.required and spec.name not in sm.columns:
                    problems.append(f"{source}: required field {spec.name!r} is not mapped")
            for name, column in sorted(sm.columns.items()):
                if column not in file_headers:
                    problems.append(
                        f"{source}: field {name!r} mapped to {column!r}, which is not a header"
                    )
            used: dict[str, str] = {}
            for name, column in sorted(sm.columns.items()):
                if column in used:
                    problems.append(
                        f"{source}: column {column!r} mapped to both {used[column]!r} and {name!r}"
                    )
                used[column] = name
        return problems

    def validate_against(self, headers: Mapping[Source, Sequence[str]]) -> None:
        problems = self.problems(headers)
        if problems:
            raise MappingError(problems)
