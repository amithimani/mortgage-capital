"""Week 1 baseline: run the current manual process through the same checks as the agent.

The brief asks for the PoC to be measured against a recorded baseline, not an estimate. The
input is a set of analyst-written exception notes per dataset (``manual_process_notes.json``).
Until real anonymised notes are provided, the synthetic generator produces a simulated set
with typical manual-process traits (rounded figures, missing citations, a false positive).

Each note is matched to the engine's finding for the same loan with a compatible category,
then evaluated by the **same policy gate** with the same grounding context. A note that matches
no engine finding is counted as unverifiable (a manual false positive). Run-trajectory
validators (tool-call-accuracy, plan-coherence) don't apply to a manual process and are excluded.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .data.synthetic import load_catalog, load_expected_mapping, pair_paths
from .engine import ReconConfig, load_inputs, reconcile
from .engine.taxonomy import ALLOWED_CATEGORIES, ExceptionCategory
from .graph import policy_subject
from .policy import PolicyGate, evaluate_fail_closed

NOTES_FILE = "manual_process_notes.json"


class ManualNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_id: str
    loan_id: str
    category: ExceptionCategory
    rationale: str
    remediation_note: str
    cited_evidence: list[str]
    minutes_spent: int


class ManualProcessSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_version: str | None = None
    pair: str
    simulated: bool
    notes: list[ManualNote]


class NoteResult(BaseModel):
    dataset: str
    note_id: str
    loan_id: str
    category: str
    matched_exception_id: str | None
    decision: str
    """PASS / BLOCK, or UNVERIFIABLE when the note matches no engine finding."""
    rules: list[dict]
    minutes_spent: int


def has_notes(pair_dir: Path) -> bool:
    return (Path(pair_dir) / NOTES_FILE).is_file()


def evaluate_dataset(
    pair_dir: Path, gate: PolicyGate, config: ReconConfig | None = None
) -> tuple[list[NoteResult], dict]:
    pair_dir = Path(pair_dir)
    sample = ManualProcessSample.model_validate(json.loads((pair_dir / NOTES_FILE).read_text()))
    result = reconcile(load_inputs(pair_paths(pair_dir), load_expected_mapping(pair_dir)), config)
    findings = [d.model_dump(mode="json") for d in result.discrepancies]

    out: list[NoteResult] = []
    matched_ids: set[str] = set()
    for note in sample.notes:
        match = next(
            (
                d
                for d in findings
                if d["loan_id"] == note.loan_id and note.category in ALLOWED_CATEGORIES[d["kind"]]
            ),
            None,
        )
        if match is None:
            out.append(
                NoteResult(
                    dataset=pair_dir.name,
                    note_id=note.note_id,
                    loan_id=note.loan_id,
                    category=note.category.value,
                    matched_exception_id=None,
                    decision="UNVERIFIABLE",
                    rules=[],
                    minutes_spent=note.minutes_spent,
                )
            )
            continue
        matched_ids.add(match["id"])
        exc = {
            **match,
            "category": note.category.value,
            "rationale": note.rationale,
            "remediation_note": note.remediation_note,
            "cited_evidence": note.cited_evidence,
        }
        verdict = evaluate_fail_closed(gate, policy_subject(exc))
        out.append(
            NoteResult(
                dataset=pair_dir.name,
                note_id=note.note_id,
                loan_id=note.loan_id,
                category=note.category.value,
                matched_exception_id=match["id"],
                decision=verdict.decision,
                rules=[r.model_dump(mode="json") for r in verdict.rules],
                minutes_spent=note.minutes_spent,
            )
        )

    seeded = load_catalog(pair_dir)["errors"] if (pair_dir / "seeded_errors.json").exists() else []
    detected = sum(
        1
        for e in seeded
        if any(
            d["id"] in matched_ids
            for d in findings
            if d["loan_id"] == e["loan_id"] and d["kind"] == e["expected_kind"]
        )
    )
    coverage = {
        "dataset": pair_dir.name,
        "simulated": sample.simulated,
        "engine_findings": len(findings),
        "seeded_errors": len(seeded),
        "seeded_detected_by_manual": detected,
    }
    return out, coverage


def summarise(results: list[NoteResult], coverage: list[dict]) -> dict:
    n = len(results)
    decisions: dict[str, int] = defaultdict(int)
    for r in results:
        decisions[r.decision] += 1
    per_validator: dict[str, dict] = {}
    for name in sorted({rule["validator"] for r in results for rule in r.rules}):
        rules = [rule for r in results for rule in r.rules if rule["validator"] == name]
        scores = [rule["score"] for rule in rules if rule["score"] is not None]
        per_validator[name] = {
            "evaluations": len(rules),
            "pass_rate": round(sum(rule["status"] == "PASS" for rule in rules) / len(rules), 4),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        }
    seeded = sum(c["seeded_errors"] for c in coverage)
    minutes = sum(r.minutes_spent for r in results)
    return {
        "datasets": [c["dataset"] for c in coverage],
        "simulated": any(c["simulated"] for c in coverage),
        "notes": n,
        "decisions": dict(sorted(decisions.items())),
        "pass_rate": round(decisions.get("PASS", 0) / n, 4) if n else None,
        "unverifiable_rate": round(decisions.get("UNVERIFIABLE", 0) / n, 4) if n else None,
        "per_validator": per_validator,
        "seeded_errors": seeded,
        "seeded_detected_by_manual": sum(c["seeded_detected_by_manual"] for c in coverage),
        "manual_recall": (
            round(sum(c["seeded_detected_by_manual"] for c in coverage) / seeded, 4)
            if seeded
            else None
        ),
        "minutes_total": minutes,
        "minutes_per_note": round(minutes / n, 2) if n else None,
        "coverage": coverage,
    }


def run_baseline(
    pair_dirs: list[Path], gate: PolicyGate, config: ReconConfig | None = None
) -> dict:
    results: list[NoteResult] = []
    coverage: list[dict] = []
    for pair_dir in pair_dirs:
        r, c = evaluate_dataset(pair_dir, gate, config)
        results.extend(r)
        coverage.append(c)
    return {
        "summary": summarise(results, coverage),
        "results": [r.model_dump(mode="json") for r in results],
    }
