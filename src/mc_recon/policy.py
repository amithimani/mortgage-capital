"""Policy gate: the synchronous PASS/BLOCK check every drafted exception must clear.

:class:`PolicyGate` is the seam where Disseqt plugs in (Week 3: ``DisseqtPolicyGate`` wrapping
``disseqt_sdk.Client.validate(..., policies=[...])`` + ``disseqt_sdk.policy.parse``). Until then
:class:`FakePolicyGate` runs deterministic stand-ins for every validator the brief names, so
BLOCK paths are exercised with real logic:

| Brief validator | Fake rule | Scope |
|---|---|---|
| faithfulness | every number in the response is in the grounding context | per exception |
| context-relevance | citations exist and are this exception's own evidence rows | per exception |
| answer-relevance | the response names this loan and addresses this category | per exception |
| tool-call-accuracy | engine ran once on confirmed mapping + files; all findings covered | run |
| plan-coherence | steps ran in the expected order | run |
| prompt-injection | known injection phrasings in ingested text | input screen |
| PII | e-mail, UK phone, NI number, sort code, IBAN in ingested text | input screen |
| (output data leakage) | no other loan ids / e-mails / sort codes in the response | per exception |

:func:`evaluate_fail_closed`, :func:`evaluate_run_fail_closed` and :func:`screen_fail_closed`
are what the graph calls. Any exception from the gate, or a non-final (async) verdict, becomes
BLOCK: an outage never lets output through unvalidated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .engine.taxonomy import ExceptionCategory as C

Decision = Literal["PASS", "BLOCK"]

EXPECTED_PLAN = (
    "ingest",
    "propose_mapping",
    "await_mapping_confirm",
    "reconcile",
    "classify_exceptions",
    "draft_rationale",
)
"""The agent's required step order up to the first policy evaluation."""
LOOP_STEPS = frozenset({"validate_policy", "human_review_queue"})


class RuleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    validator: str
    status: Literal["PASS", "FAIL"]
    score: float | None = None
    threshold: float | None = None
    detail: str = ""
    scope: Literal["exception", "run", "input"] = "exception"


class PolicyVerdict(BaseModel):
    """Mirrors the fields of ``disseqt_sdk.policy.PolicyDecision`` that we persist."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    policy_id: str
    policy_version: int | None = None
    enforcement: str = "sync"
    aggregation: str = "all"
    rules: list[RuleResult] = Field(default_factory=list)
    error: str | None = None
    """Set when the verdict is a fail-closed BLOCK (gate unreachable, async, unparseable)."""

    @property
    def blocked(self) -> bool:
        return self.decision != "PASS"


class PolicySubject(BaseModel):
    """One drafted exception, in the prompt/context/response shape validators expect."""

    model_config = ConfigDict(frozen=True)

    exception_id: str
    loan_id: str | None
    category: str | None = None
    prompt: str
    context: str
    """Evidence rows + engine summary + expected/actual: the only grounding allowed."""
    response: str
    """Rationale + remediation note, as the analyst would read it."""
    evidence_keys: list[str]
    cited_evidence: list[str]


class PolicyGate(Protocol):
    policy_id: str

    def evaluate(self, subject: PolicySubject) -> PolicyVerdict: ...

    def evaluate_run(self, trajectory: Sequence[dict[str, Any]]) -> PolicyVerdict: ...

    def screen_input(self, text: str) -> PolicyVerdict: ...


def _fail_closed(gate: Any, exc: Exception) -> PolicyVerdict:
    return PolicyVerdict(
        decision="BLOCK",
        policy_id=getattr(gate, "policy_id", "unknown"),
        enforcement="fail_closed",
        error=f"{type(exc).__name__}: {exc}",
    )


def _final_only(verdict: PolicyVerdict) -> PolicyVerdict:
    if verdict.enforcement != "sync":
        return verdict.model_copy(
            update={"decision": "BLOCK", "error": f"non-final {verdict.enforcement} verdict"}
        )
    return verdict


def evaluate_fail_closed(gate: PolicyGate, subject: PolicySubject) -> PolicyVerdict:
    try:
        return _final_only(gate.evaluate(subject))
    except Exception as exc:  # noqa: BLE001 — any failure must block, never pass through
        return _fail_closed(gate, exc)


def evaluate_run_fail_closed(gate: PolicyGate, trajectory: Sequence[dict]) -> PolicyVerdict:
    try:
        return _final_only(gate.evaluate_run(trajectory))
    except Exception as exc:  # noqa: BLE001
        return _fail_closed(gate, exc)


def screen_fail_closed(gate: PolicyGate, text: str) -> PolicyVerdict:
    try:
        return _final_only(gate.screen_input(text))
    except Exception as exc:  # noqa: BLE001
        return _fail_closed(gate, exc)


def combine(exception_verdict: PolicyVerdict, run_verdict: PolicyVerdict) -> PolicyVerdict:
    """One verdict per exception covering both scopes (``all`` strategy across them)."""
    errors = [e for e in (exception_verdict.error, run_verdict.error) if e]
    blocked = exception_verdict.blocked or run_verdict.blocked
    return exception_verdict.model_copy(
        update={
            "decision": "BLOCK" if blocked else "PASS",
            "rules": [*exception_verdict.rules, *run_verdict.rules],
            "error": "; ".join(errors) or None,
            "enforcement": (
                exception_verdict.enforcement
                if exception_verdict.enforcement != "sync"
                else run_verdict.enforcement
            ),
        }
    )


# --- deterministic fake -------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w\-.])-?\d[\d,]*(?:\.\d+)?(?![\w\-])")
_LOAN_ID = re.compile(r"\bMC-\d+\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_SORT_CODE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
_UK_PHONE = re.compile(r"(?:\+44\s?7\d{3}|\b07\d{3})\s?\d{3}\s?\d{3}\b")
_NI_NUMBER = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")
_IBAN_GB = re.compile(r"\bGB\d{2}\s?[A-Z]{4}(?:\s?\d){14}\b")
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)"
    r"|disregard (the )?(system|previous) (prompt|instructions)"
    r"|you are now\b|system prompt|<\s*/?\s*(system|assistant)\s*>",
    re.IGNORECASE,
)
_PII_PATTERNS = {
    "email": _EMAIL,
    "uk_phone": _UK_PHONE,
    "ni_number": _NI_NUMBER,
    "sort_code": _SORT_CODE,
    "iban": _IBAN_GB,
}

# Terms an on-topic explanation of each category must use (any one suffices).
_CATEGORY_TERMS: dict[str, tuple[str, ...]] = {
    C.MISSING_PAYMENT: ("payment", "paid", "received", "collection"),
    C.DUPLICATE_DIRECT_DEBIT: ("duplicate", "twice", "double"),
    C.RATE_MARGIN_BREACH: ("rate", "margin"),
    C.PARTIAL_OR_OVERPAYMENT: ("received", "short", "partial", "overpa", "underpa", "differs"),
    C.INVESTOR_BALANCE_BREAK: ("balance",),
    C.UNIDENTIFIED_PAYMENT: ("payment", "unallocated", "unidentified", "suspense"),
    C.INVESTOR_POPULATION_BREAK: ("investor", "servicing", "population", "missing"),
    C.OTHER_NEEDS_REVIEW: (),
}


def numbers_in(text: str) -> set[Decimal]:
    out: set[Decimal] = set()
    for token in _NUMBER.findall(text):
        try:
            out.add(abs(Decimal(token.replace(",", ""))))
        except InvalidOperation:
            continue
    return out


def pii_in(text: str) -> dict[str, int]:
    return {name: len(p.findall(text)) for name, p in _PII_PATTERNS.items() if p.search(text)}


def _rule(validator: str, ok: bool, score: float, detail: str, scope: str) -> RuleResult:
    return RuleResult(
        validator=validator,
        status="PASS" if ok else "FAIL",
        score=round(score, 4),
        threshold=1.0,
        detail="" if ok else detail,
        scope=scope,  # type: ignore[arg-type]
    )


class FakePolicyGate:
    """Deterministic stand-ins for the brief's validators (``all`` strategy); see module doc."""

    policy_id = "fake-exception-policy"
    policy_version = 2

    def __init__(self, force_block: set[str] | None = None):
        self.force_block = set(force_block or ())
        """Exception ids to BLOCK unconditionally (for tests of the review loop)."""
        self.calls: list[PolicySubject] = []
        self.run_calls: list[list[dict]] = []

    def _verdict(self, rules: list[RuleResult]) -> PolicyVerdict:
        return PolicyVerdict(
            decision="PASS" if all(r.status == "PASS" for r in rules) else "BLOCK",
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            rules=rules,
        )

    # per exception ----------------------------------------------------------------------

    def evaluate(self, subject: PolicySubject) -> PolicyVerdict:
        self.calls.append(subject)
        text = subject.response
        grounded, claimed = numbers_in(subject.context), numbers_in(text)
        ungrounded = sorted(claimed - grounded)
        faith = 1.0 if not claimed else (len(claimed) - len(ungrounded)) / len(claimed)

        bad_cites = sorted(set(subject.cited_evidence) - set(subject.evidence_keys))
        cited_ok = bool(subject.cited_evidence) and not bad_cites
        cite_score = (
            0.0
            if not subject.cited_evidence
            else 1 - len(bad_cites) / len(set(subject.cited_evidence))
        )

        lowered = text.lower()
        names_loan = subject.loan_id is None or subject.loan_id.lower() in lowered
        terms = _CATEGORY_TERMS.get(subject.category or "", ())
        on_topic = not terms or any(t in lowered for t in terms)
        relevance = (names_loan + on_topic) / 2

        other_loans = sorted(
            {m.upper() for m in _LOAN_ID.findall(text)}
            - ({subject.loan_id.upper()} if subject.loan_id else set())
        )
        leaks = other_loans + _EMAIL.findall(text) + _SORT_CODE.findall(text)

        rules = [
            _rule(
                "faithfulness",
                not ungrounded,
                faith,
                f"ungrounded numbers: {[str(n) for n in ungrounded]}",
                "exception",
            ),
            _rule(
                "context_relevance",
                cited_ok,
                cite_score,
                f"no citations or citations outside evidence: {bad_cites}",
                "exception",
            ),
            _rule(
                "answer_relevance",
                relevance == 1.0,
                relevance,
                f"names loan: {names_loan}; addresses {subject.category}: {on_topic}",
                "exception",
            ),
            _rule(
                "output_data_leakage",
                not leaks,
                0.0 if leaks else 1.0,
                f"leaked: {leaks}",
                "exception",
            ),
        ]
        if subject.exception_id in self.force_block:
            rules.append(RuleResult(validator="forced_block", status="FAIL", detail="test hook"))
        return self._verdict(rules)

    # per run (trajectory) ----------------------------------------------------------------

    def evaluate_run(self, trajectory: Sequence[dict[str, Any]]) -> PolicyVerdict:
        self.run_calls.append(list(trajectory))
        steps = [t["step"] for t in trajectory]
        head, tail = steps[: len(EXPECTED_PLAN)], steps[len(EXPECTED_PLAN) :]
        in_order = sum(a == b for a, b in zip(head, EXPECTED_PLAN, strict=False))
        coherence = in_order / len(EXPECTED_PLAN)
        tail_ok = all(s in LOOP_STEPS for s in tail)
        coherent = coherence == 1.0 and tail_ok

        by_step: dict[str, list[dict]] = {}
        for t in trajectory:
            by_step.setdefault(t["step"], []).append(t)
        checks: dict[str, bool] = {}
        engine = by_step.get("reconcile", [])
        checks["engine called exactly once"] = len(engine) == 1
        confirm = (by_step.get("await_mapping_confirm") or [{}])[-1]
        ingest = (by_step.get("ingest") or [{}])[-1]
        if engine:
            call = engine[0]
            checks["engine used the confirmed mapping"] = call.get(
                "mapping_sha"
            ) is not None and call.get("mapping_sha") == confirm.get("mapping_sha")
            checks["engine read the ingested files"] = call.get("file_hashes") == ingest.get(
                "file_hashes"
            )
            classify = (by_step.get("classify_exceptions") or [{}])[-1]
            draft = (by_step.get("draft_rationale") or [{}])[-1]
            checks["every finding classified"] = classify.get("count") == call.get("discrepancies")
            checks["every exception drafted"] = draft.get("count") == classify.get("count")
        accuracy = sum(checks.values()) / len(checks)
        failed = [name for name, ok in checks.items() if not ok]

        return self._verdict(
            [
                _rule("tool_call_accuracy", not failed, accuracy, f"failed: {failed}", "run"),
                _rule(
                    "plan_coherence",
                    coherent,
                    coherence if tail_ok else min(coherence, 0.5),
                    f"steps: {steps}",
                    "run",
                ),
            ]
        )

    # input screen ------------------------------------------------------------------------

    def screen_input(self, text: str) -> PolicyVerdict:
        hit = _INJECTION.search(text)
        pii = pii_in(text)
        return PolicyVerdict(
            decision="BLOCK" if hit or pii else "PASS",
            policy_id="fake-input-screen",
            policy_version=2,
            rules=[
                _rule(
                    "input_prompt_injection",
                    not hit,
                    0.0 if hit else 1.0,
                    f"matched {hit.group(0)!r}" if hit else "",
                    "input",
                ),
                _rule("input_pii", not pii, 0.0 if pii else 1.0, f"found: {pii}", "input"),
            ],
        )
