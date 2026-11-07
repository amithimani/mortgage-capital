"""Deterministic reconciliation engine. No LLM, no I/O beyond reading the input files."""

from .loader import load_inputs, normalize_loan_id
from .models import (
    Discrepancy,
    EvidenceRef,
    InvestorRecord,
    LoadedInputs,
    PaymentRecord,
    ReconciliationResult,
    ReconConfig,
    ServicingRecord,
)
from .profile import ColumnProfile, FileProfile, profile_csv
from .reconcile import ENGINE_VERSION, reconcile
from .schema import (
    CANONICAL_SCHEMA,
    ColumnMapping,
    EngineInputError,
    MappingError,
    ParseError,
    Source,
    SourceMapping,
)
from .taxonomy import DiscrepancyKind, ExceptionCategory, allowed_categories, constrain_category

__all__ = [
    "CANONICAL_SCHEMA",
    "ENGINE_VERSION",
    "ColumnMapping",
    "ColumnProfile",
    "Discrepancy",
    "DiscrepancyKind",
    "EngineInputError",
    "EvidenceRef",
    "ExceptionCategory",
    "FileProfile",
    "InvestorRecord",
    "LoadedInputs",
    "MappingError",
    "ParseError",
    "PaymentRecord",
    "ReconConfig",
    "ReconciliationResult",
    "ServicingRecord",
    "Source",
    "SourceMapping",
    "allowed_categories",
    "constrain_category",
    "load_inputs",
    "normalize_loan_id",
    "profile_csv",
    "reconcile",
]
