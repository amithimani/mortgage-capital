"""Builders for :class:`LLMRequest` payloads.

The graph nodes and the oracle-recording builder both use these, so a recorded response is
keyed by exactly the input the live node will send.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..engine.models import Discrepancy
from ..engine.profile import FileProfile
from ..engine.schema import describe_schema
from ..engine.taxonomy import allowed_categories
from .provider import PROMPT_VERSIONS, LLMRequest, LLMTask


def evidence_key(source: str, row: int) -> str:
    return f"{source}:{row}"


def mapping_request(profiles: Sequence[FileProfile]) -> LLMRequest:
    return LLMRequest(
        task=LLMTask.PROPOSE_MAPPING,
        prompt_version=PROMPT_VERSIONS[LLMTask.PROPOSE_MAPPING],
        payload={
            "canonical_schema": describe_schema(),
            "files": [
                p.model_dump(mode="json", exclude={"sha256"})
                for p in sorted(profiles, key=lambda p: p.source.value)
            ],
        },
    )


def classification_request(d: Discrepancy) -> LLMRequest:
    return LLMRequest(
        task=LLMTask.CLASSIFY_EXCEPTION,
        prompt_version=PROMPT_VERSIONS[LLMTask.CLASSIFY_EXCEPTION],
        payload={
            "discrepancy": d.model_dump(mode="json"),
            "allowed_categories": [c.value for c in allowed_categories(d.kind)],
        },
    )


def rationale_request(d: Discrepancy, category: str) -> LLMRequest:
    """Evidence rows only — the rationale is grounded in nothing else."""
    return LLMRequest(
        task=LLMTask.DRAFT_RATIONALE,
        prompt_version=PROMPT_VERSIONS[LLMTask.DRAFT_RATIONALE],
        payload={
            "discrepancy_id": d.id,
            "category": category,
            "summary": d.summary,
            "expected": d.expected,
            "actual": d.actual,
            "evidence": {
                evidence_key(e.source.value, e.row): {"key": e.key, **e.fields} for e in d.evidence
            },
        },
    )
