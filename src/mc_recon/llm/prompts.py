"""Versioned prompt templates: ``LLMRequest`` -> chat messages.

Real providers send exactly these messages. The fake doesn't need them to answer, but the
agent still renders them for every call so ``MODEL_EXEC`` spans carry the input messages the
brief asks for, identically on the fake and on a live model.

Changing a template's text means bumping its version in ``PROMPT_VERSIONS``. That changes
``LLMRequest.cache_key``, so stale recordings fail loudly instead of replaying silently.
"""

from __future__ import annotations

from .provider import LLMRequest, LLMTask, canonical_json

SYSTEM_PROMPTS: dict[str, str] = {
    "mapping.v1": (
        "You map columns of three mortgage reconciliation files (servicing extract, payments "
        "file, investor report) to a canonical schema. You receive column profiles (name, "
        "inferred type, a few sample values) and the canonical schema. Return a ColumnMapping: "
        "for each source, canonical field -> exact source header, and the strptime date format "
        "if dates are not ISO-8601. Never invent headers. Leave optional fields unmapped if no "
        "column fits. Do not compute or alter any values."
    ),
    "classify.v1": (
        "You classify one reconciliation discrepancy found by a deterministic engine. Choose "
        "exactly one category from allowed_categories and give a one-sentence reason grounded "
        "in the discrepancy. If none fits, choose OTHER_NEEDS_REVIEW. Do not recompute numbers."
    ),
    "rationale.v1": (
        "You write a plain-English rationale and remediation note for one reconciliation "
        "exception for an operations analyst. Use ONLY the supplied evidence rows, expected and "
        "actual values. Quote numbers exactly as given; never estimate, round or introduce new "
        "figures. Mention only this loan. List the evidence keys (source:row) you relied on in "
        "cited_evidence."
    ),
}

_TASK_INSTRUCTION = {
    LLMTask.PROPOSE_MAPPING: "Propose the column mapping for these files.",
    LLMTask.CLASSIFY_EXCEPTION: "Classify this discrepancy.",
    LLMTask.DRAFT_RATIONALE: "Draft the rationale and remediation note.",
}


def render_messages(request: LLMRequest) -> list[dict[str, str]]:
    try:
        system = SYSTEM_PROMPTS[request.prompt_version]
    except KeyError:
        raise KeyError(f"no template for prompt_version {request.prompt_version!r}") from None
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{_TASK_INSTRUCTION[request.task]}\n\n{canonical_json(request.payload)}",
        },
    ]
    if request.feedback:
        messages.append(
            {
                "role": "user",
                "content": f"Your previous answer was rejected: {request.feedback}\nTry again.",
            }
        )
    return messages
