"""Ground-truth ("oracle") LLM recordings for a synthetic pair.

Because each pair's correct mapping and error categories are known, we can record what a
perfect model *should* return for every LLM call the agent will make on that pair. Replayed
through :class:`~mc_recon.llm.fake.RecordedFakeProvider`, these let the full graph run in CI
with no model, and give the live-model drift run something exact to compare against.

Oracle rationales are templated from the engine's own summary — they exist to exercise the
pipeline, not to stand in for real model prose.
"""

from __future__ import annotations

from pathlib import Path

from ..engine import load_inputs, profile_csv, reconcile
from ..engine.models import Discrepancy
from ..engine.taxonomy import ExceptionCategory
from ..llm.fake import Recording, RecordingSet
from ..llm.payloads import (
    classification_request,
    evidence_key,
    mapping_request,
    rationale_request,
)
from ..llm.provider import LLMRequest
from ..llm.schemas import ClassificationOutput, MappingProposal, RationaleOutput
from .synthetic import load_catalog, load_expected_mapping, pair_paths

RECORDINGS_FILE = "oracle_recordings.json"

_REMEDIATION: dict[ExceptionCategory, str] = {
    ExceptionCategory.MISSING_PAYMENT: (
        "Contact the borrower and check the direct debit / standing order status; raise an "
        "arrears case if unpaid after the grace period."
    ),
    ExceptionCategory.DUPLICATE_DIRECT_DEBIT: (
        "Refund the duplicate collection to the borrower and raise a ticket with the DD bureau."
    ),
    ExceptionCategory.RATE_MARGIN_BREACH: (
        "Correct the applied rate to base + contractual margin, recalculate interest and "
        "refund any overcharge; notify the investor of the restated rate."
    ),
    ExceptionCategory.PARTIAL_OR_OVERPAYMENT: (
        "Confirm the shortfall or excess with the borrower and allocate per the payment "
        "allocation policy."
    ),
    ExceptionCategory.INVESTOR_BALANCE_BREAK: (
        "Reissue the investor report line with the servicing closing balance after review."
    ),
    ExceptionCategory.UNIDENTIFIED_PAYMENT: (
        "Move the funds to suspense and trace the payer via the bank reference."
    ),
    ExceptionCategory.INVESTOR_POPULATION_BREAK: (
        "Reconcile the loan population between servicing and the investor tape."
    ),
    ExceptionCategory.OTHER_NEEDS_REVIEW: "Route to an analyst for manual review.",
}


def _rec(request: LLMRequest, output: dict, label: str) -> Recording:
    return Recording(
        task=request.task,
        prompt_version=request.prompt_version,
        cache_key=request.cache_key,
        output=output,
        label=label,
    )


def _category_for(d: Discrepancy, catalog: dict) -> ExceptionCategory:
    matches = [
        e
        for e in catalog["errors"]
        if e["expected_kind"] == d.kind.value and e["loan_id"] == d.loan_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"discrepancy {d.id} ({d.kind}, {d.loan_id}) does not match exactly one seeded "
            f"error (found {len(matches)}) — the engine and the catalog disagree"
        )
    return ExceptionCategory(matches[0]["expected_category"])


def build_oracle_recordings(pair_dir: Path) -> RecordingSet:
    pair_dir = Path(pair_dir)
    paths = pair_paths(pair_dir)
    mapping = load_expected_mapping(pair_dir)
    catalog = load_catalog(pair_dir)

    recordings: list[Recording] = []
    profiles = [profile_csv(path, source) for source, path in paths.items()]
    recordings.append(
        _rec(
            mapping_request(profiles),
            MappingProposal(mapping=mapping).model_dump(mode="json"),
            label=f"{catalog['pair']}:mapping",
        )
    )

    result = reconcile(load_inputs(paths, mapping))
    for d in result.discrepancies:
        category = _category_for(d, catalog)
        recordings.append(
            _rec(
                classification_request(d),
                ClassificationOutput(
                    category=category, reason=f"Engine finding {d.kind.value}: {d.summary}"
                ).model_dump(mode="json"),
                label=f"{catalog['pair']}:{d.id}:classify",
            )
        )
        recordings.append(
            _rec(
                rationale_request(d, category.value),
                RationaleOutput(
                    rationale=d.summary,
                    remediation_note=_REMEDIATION[category],
                    cited_evidence=[evidence_key(e.source.value, e.row) for e in d.evidence],
                ).model_dump(mode="json"),
                label=f"{catalog['pair']}:{d.id}:rationale",
            )
        )
    return RecordingSet(recordings=recordings)


def write_oracle_recordings(pair_dir: Path) -> Path:
    out = Path(pair_dir) / RECORDINGS_FILE
    build_oracle_recordings(pair_dir).save(out)
    return out
