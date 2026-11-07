"""Typed records, discrepancies and results. All money and rates are ``Decimal``."""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from .schema import Source
from .taxonomy import DiscrepancyKind

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class ServicingRecord(BaseModel):
    model_config = _FROZEN

    row: int
    loan_id: str
    pool_id: str
    due_date: date
    scheduled_amount: Decimal
    closing_balance: Decimal
    base_rate: Decimal
    margin: Decimal
    applied_rate: Decimal
    mandate_ref: str | None = None


class PaymentRecord(BaseModel):
    model_config = _FROZEN

    row: int
    txn_id: str
    loan_id: str
    value_date: date
    amount: Decimal
    channel: str
    mandate_ref: str | None = None


class InvestorRecord(BaseModel):
    model_config = _FROZEN

    row: int
    loan_id: str
    pool_id: str
    period_end: date
    reported_rate: Decimal
    reported_margin: Decimal
    scheduled_amount: Decimal
    closing_balance: Decimal


class LoadedInputs(BaseModel):
    model_config = _FROZEN

    servicing: tuple[ServicingRecord, ...]
    payments: tuple[PaymentRecord, ...]
    investor: tuple[InvestorRecord, ...]
    file_hashes: dict[Source, str]
    """sha256 of each raw file — recorded in the run manifest."""


class EvidenceRef(BaseModel):
    """Pointer to a raw-file row. With the file hash, this is enough to re-derive the finding."""

    model_config = _FROZEN

    source: Source
    row: int
    """1-based data row number (header excluded)."""
    key: str
    """loan id (servicing / investor) or txn id (payments)."""
    fields: dict[str, str] = Field(default_factory=dict)


class Discrepancy(BaseModel):
    model_config = _FROZEN

    id: str
    kind: DiscrepancyKind
    loan_id: str | None
    summary: str
    expected: dict[str, str] = Field(default_factory=dict)
    actual: dict[str, str] = Field(default_factory=dict)
    evidence: tuple[EvidenceRef, ...]

    @staticmethod
    def make_id(
        kind: DiscrepancyKind, loan_id: str | None, scope: str, evidence: tuple[EvidenceRef, ...]
    ) -> str:
        """Stable id: same inputs -> same id, so remediation-log writes are idempotent."""
        refs = ";".join(f"{e.source.value}:{e.row}" for e in evidence)
        digest = hashlib.sha256(f"{kind.value}|{loan_id or ''}|{scope}|{refs}".encode()).hexdigest()
        return f"D-{digest[:12]}"


class ReconConfig(BaseModel):
    model_config = _FROZEN

    amount_tolerance: Decimal = Decimal("0.01")
    balance_tolerance: Decimal = Decimal("0.05")
    rate_tolerance: Decimal = Decimal("0.005")
    """Percentage points."""
    match_window_days: int = 3
    """A payment matches a due date if its value date is within ± this many days."""
    duplicate_window_days: int = 5


class ReconciliationResult(BaseModel):
    model_config = _FROZEN

    engine_version: str
    config: ReconConfig
    file_hashes: dict[Source, str]
    discrepancies: tuple[Discrepancy, ...]
    stats: dict[str, int]
