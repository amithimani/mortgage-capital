"""Use Case 2: three EU AI Act controls, evidenced by what the agent already emits.

Flow, as in the brief: agent span -> validator score -> control test -> control status ->
requirement coverage.

Every evaluation appends an evidence record with a freshness SLA. A control counts as met only
while its evidence is current, so an agent that stops running drifts to ``stale`` instead of
staying green. All three controls are block severity: a breach holds the exception for human
review, and is recorded either way.

Status of a control, from the latest record per ``(run, subject)``:

* ``no_evidence``: nothing has ever been recorded
* ``stale``: records exist, none is current
* ``breached``: a current record is a breach (held, or released by a human override)
* ``met``: current records, none in breach

``resolved`` marks a breach whose output a human rejected. The failing text never reached
anyone, so it does not keep the control red, and the breach itself stays on the record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .policy import PolicyVerdict, RuleResult
from .store import RunStore

GROUNDING = "TECH-RAG-GROUNDING-CITE"
HUMAN_LOOP = "TECH-HUMAN-LOOP"
LOG_PIPELINE = "TECH-LOG-PIPELINE"

REQUIRED_MANIFEST = (
    "file_hashes",
    "engine_version",
    "prompt_versions",
    "model_id",
    "mapping_confirmed_by",
)
"""Art. 12: model used, inputs, rule version and approver must be on the record."""

Outcome = Literal["pass", "breach", "resolved"]


class AISystem(BaseModel):
    """Registry metadata for the governed system (brief, UC2 step 1)."""

    model_config = ConfigDict(frozen=True)

    application_id: str
    name: str = "mc-reconciliation-agent"
    risk_level: Literal["unacceptable", "high", "limited", "minimal"] = "limited"
    lifecycle_stage: str = "proof_of_concept"
    provider_role: Literal["provider", "deployer"] = "deployer"
    purpose: str = "Reconcile mortgage servicing, payment and investor data; explain exceptions"
    domain: str = "mortgage servicing operations"
    affected_population: str = "mortgage borrowers and funding investors (indirectly)"
    data_categories: list[str] = Field(
        default_factory=lambda: ["loan identifiers", "payment records", "rates and balances"]
    )
    retention_days: int = 2555
    """Placeholder of seven years. The real Art. 12 period is a decision for compliance."""


class Control(BaseModel):
    model_config = ConfigDict(frozen=True)

    control_id: str
    title: str
    requirement: str
    severity: Literal["block"] = "block"
    test_type: Literal["metric_threshold"] = "metric_threshold"
    evidence_source: Literal["sdk_event"] = "sdk_event"
    metric: str
    threshold: float
    freshness_hours: int = 24


CONTROLS = (
    Control(
        control_id=GROUNDING,
        title="Source Grounding & Verifiable Citations",
        requirement="Art 15 - Accuracy & Robustness",
        metric="min(faithfulness, context_relevance)",
        threshold=0.9,
    ),
    Control(
        control_id=HUMAN_LOOP,
        title="Human-in-the-Loop Checkpoint",
        requirement="Art 14 - Human Oversight",
        metric="oversight_event_recorded",
        threshold=1.0,
    ),
    Control(
        control_id=LOG_PIPELINE,
        title="Audit Logging Pipeline",
        requirement="Art 12 - Record-Keeping",
        metric="record_completeness",
        threshold=1.0,
    ),
)
_BY_ID = {c.control_id: c for c in CONTROLS}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GovernanceLayer:
    def __init__(
        self, store: RunStore, system: AISystem, clock: Callable[[], datetime] = _utcnow
    ) -> None:
        self.store, self.system, self.clock = store, system, clock

    # --- evidence --------------------------------------------------------------------------

    def _record(
        self,
        control_id: str,
        run_id: str,
        subject_id: str,
        attempt: int,
        value: float | None,
        outcome: Outcome,
        detail: str = "",
        policy_version: str | None = None,
    ) -> None:
        control, now = _BY_ID[control_id], self.clock()
        natural = f"{run_id}:{control_id}:{subject_id}:{attempt}:{outcome}"
        self.store.append_evidence(
            {
                "evidence_id": hashlib.sha256(natural.encode()).hexdigest()[:24],
                "application_id": self.system.application_id,
                "control_id": control_id,
                "run_id": run_id,
                "subject_id": subject_id,
                "metric": control.metric,
                "value": value,
                "threshold": control.threshold,
                "outcome": outcome,
                "detail": detail,
                "policy_version": policy_version,
                "source": control.evidence_source,
                "evaluated_at": now.isoformat(timespec="milliseconds"),
                "expires_at": (now + timedelta(hours=control.freshness_hours)).isoformat(
                    timespec="milliseconds"
                ),
            }
        )

    # --- runtime hooks (called from the graph) ----------------------------------------------

    def on_mapping_confirmed(self, run_id: str, analyst: str) -> None:
        ok = bool(analyst and analyst.strip())
        self._record(
            HUMAN_LOOP, run_id, "mapping", 1, 1.0 if ok else 0.0, "pass" if ok else "breach"
        )

    def on_verdict(
        self, run_id: str, exc: dict, verdict: PolicyVerdict, attempt: int, state: dict
    ) -> list[RuleResult]:
        """Run the control tests for one verdict. Returns a failing rule per block-worthy breach."""
        version = f"{verdict.policy_id}@{verdict.policy_version}"
        breaches: list[RuleResult] = []

        def test(control_id: str, value: float | None, detail: str) -> None:
            passed = value is not None and value >= _BY_ID[control_id].threshold
            self._record(
                control_id,
                run_id,
                exc["id"],
                attempt,
                value,
                "pass" if passed else "breach",
                "" if passed else detail,
                version,
            )
            if not passed:
                breaches.append(
                    RuleResult(
                        validator=f"control:{control_id}", status="FAIL", detail=detail, score=value
                    )
                )

        scores = [
            r.score
            for r in verdict.rules
            if r.validator in {"faithfulness", "context_relevance"} and r.score is not None
        ]
        test(
            GROUNDING,
            min(scores) if len(scores) == 2 else None,
            verdict.error or "grounding scores below threshold or missing",
        )

        confirmed = 1.0 if state.get("mapping_confirmed_by") else 0.0
        test(HUMAN_LOOP, confirmed, "no analyst confirmation on record for the mapping")

        run = self.store.get_run(run_id) or {}
        manifest = run.get("manifest", {})
        missing = [k for k in REQUIRED_MANIFEST if not manifest.get(k)]
        if not verdict.policy_id:
            missing.append("policy_id")
        if not exc.get("evidence"):
            missing.append("evidence")
        total = len(REQUIRED_MANIFEST) + 2
        test(LOG_PIPELINE, (total - len(missing)) / total, f"record is missing {missing}")
        return breaches

    def on_review(
        self, run_id: str, exception_id: str, analyst: str, action: str, round_: int
    ) -> None:
        ok = bool(analyst and analyst.strip())
        self._record(
            HUMAN_LOOP,
            run_id,
            exception_id,
            1000 + round_,  # after any on_verdict attempt, so it is the latest for this subject
            1.0 if ok else 0.0,
            "pass" if ok else "breach",
            f"{action} by {analyst}",
        )
        if action == "reject":
            self._record(
                GROUNDING,
                run_id,
                exception_id,
                1000 + round_,
                None,
                "resolved",
                f"output rejected by {analyst}; it was never served",
            )

    # --- reporting --------------------------------------------------------------------------

    def control_statuses(self) -> list[dict[str, Any]]:
        now = self.clock().isoformat(timespec="milliseconds")
        out = []
        for control in CONTROLS:
            records = self.store.evidence(control.control_id)
            latest: dict[tuple[str, str], dict] = {}
            for r in records:  # oldest first, so the last one wins
                latest[(r["run_id"], r["subject_id"])] = r
            current = [r for r in latest.values() if r["expires_at"] > now]
            open_breaches = [r for r in current if r["outcome"] == "breach"]
            if not records:
                status = "no_evidence"
            elif not current:
                status = "stale"
            elif open_breaches:
                status = "breached"
            else:
                status = "met"
            out.append(
                {
                    **control.model_dump(),
                    "status": status,
                    "evidence_records": len(records),
                    "current_subjects": len(current),
                    "open_breaches": [
                        {k: r[k] for k in ("run_id", "subject_id", "detail", "evaluated_at")}
                        for r in open_breaches
                    ],
                    "breaches_recorded": sum(r["outcome"] == "breach" for r in records),
                    "newest_evidence_at": records[-1]["evaluated_at"] if records else None,
                    "evidence_expires_at": (
                        max(r["expires_at"] for r in current) if current else None
                    ),
                }
            )
        return out

    def coverage(self) -> dict[str, Any]:
        statuses = self.control_statuses()
        requirements = [
            {
                "requirement": c["requirement"],
                "control_id": c["control_id"],
                "status": c["status"],
                "covered": c["status"] == "met",
            }
            for c in statuses
        ]
        gaps = [r for r in requirements if not r["covered"]]
        return {
            "requirements": requirements,
            "covered": len(requirements) - len(gaps),
            "total": len(requirements),
            "gaps": gaps,
        }

    def report(self) -> dict[str, Any]:
        return {
            "title": "Compliance, Governance and Assurance Report",
            "generated_at": self.clock().isoformat(timespec="milliseconds"),
            "system": self.system.model_dump(),
            "controls": self.control_statuses(),
            "coverage": self.coverage(),
            "scope_note": "Three controls only, by design. Not a statement of overall compliance.",
        }

    def report_markdown(self) -> str:
        r = self.report()
        lines = [
            f"# {r['title']}",
            "",
            f"System: **{r['system']['name']}** (`{r['system']['application_id']}`), "
            f"risk {r['system']['risk_level']}, {r['system']['provider_role']}, "
            f"{r['system']['lifecycle_stage']}",
            f"Generated: {r['generated_at']}",
            "",
            "| Control | Requirement | Status | Current subjects | Breaches recorded |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {c['control_id']} | {c['requirement']} | {c['status']} | "
            f"{c['current_subjects']} | {c['breaches_recorded']} |"
            for c in r["controls"]
        ]
        cov = r["coverage"]
        lines += ["", f"Coverage: {cov['covered']} of {cov['total']} requirements.", ""]
        if cov["gaps"]:
            lines.append("Open gaps:")
            lines += [f"- {g['requirement']}: {g['status']}" for g in cov["gaps"]]
        else:
            lines.append("No open gaps.")
        return "\n".join(lines + ["", r["scope_note"], ""])
