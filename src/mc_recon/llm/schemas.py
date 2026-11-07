"""Structured outputs for each LLM task. Strict: unknown fields are rejected."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..engine.schema import ColumnMapping
from ..engine.taxonomy import ExceptionCategory

_STRICT = ConfigDict(extra="forbid")


class MappingProposal(BaseModel):
    """``propose_mapping`` output. Applied only after analyst confirmation."""

    model_config = _STRICT

    mapping: ColumnMapping
    notes: str = Field(default="", max_length=2000)


class ClassificationOutput(BaseModel):
    """``classify_exception`` output. ``category`` is re-checked by ``constrain_category``."""

    model_config = _STRICT

    category: ExceptionCategory
    reason: str = Field(max_length=500)


class RationaleOutput(BaseModel):
    """``draft_rationale`` output. ``cited_evidence`` must reference the discrepancy's evidence."""

    model_config = _STRICT

    rationale: str = Field(min_length=1, max_length=2000)
    remediation_note: str = Field(min_length=1, max_length=2000)
    cited_evidence: list[str] = Field(min_length=1)
    """Evidence keys (``source:row``) the rationale relies on."""
