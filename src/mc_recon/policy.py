"""Policy gate: the synchronous PASS/BLOCK check every drafted exception must clear.

:class:`PolicyGate` is the seam where Disseqt plugs in (Week 3: ``DisseqtPolicyGate`` wrapping
``disseqt_sdk.Client.validate(..., policies=[...])`` + ``disseqt_sdk.policy.parse``). Until then
:class:`FakePolicyGate` runs deterministic stand-ins for the validators in the exception policy,
so BLOCK paths are exercised with real logic rather than coin flips.

:func:`evaluate_fail_closed` is what the graph calls. Any exception from the gate becomes a
BLOCK verdict carrying the error — an outage never lets output through unvalidated.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["PASS", "BLOCK"]


class RuleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    validator: str
    status: Literal["PASS", "FAIL"]
    score: float | None = None
    threshold: float | None = None
    detail: str = ""


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
    prompt: str
    context: str
    """Evidence rows + engine summary + expected/actual — the only grounding allowed."""
    response: str
    """Rationale + remediation note, as the analyst would read it."""
    evidence_keys: list[str]
    cited_evidence: list[str]


class PolicyGate(Protocol):
    policy_id: str

    def evaluate(self, subject: PolicySubject) -> PolicyVerdict: ...

    def screen_input(self, text: str) -> PolicyVerdict: ...


def evaluate_fail_closed(gate: PolicyGate, subject: PolicySubject) -> PolicyVerdict:
    try:
        verdict = gate.evaluate(subject)
    except Exception as exc:  # noqa: BLE001 — any failure must block, never pass through
        return PolicyVerdict(
            decision="BLOCK",
            policy_id=getattr(gate, "policy_id", "unknown"),
            enforcement="fail_closed",
            error=f"{type(exc).__name__}: {exc}",
        )
    if verdict.enforcement != "sync":
        return verdict.model_copy(
            update={"decision": "BLOCK", "error": f"non-final {verdict.enforcement} verdict"}
        )
    return verdict


def screen_fail_closed(gate: PolicyGate, text: str) -> PolicyVerdict:
    try:
        return gate.screen_input(text)
    except Exception as exc:  # noqa: BLE001
        return PolicyVerdict(
            decision="BLOCK",
            policy_id=getattr(gate, "policy_id", "unknown"),
            enforcement="fail_closed",
            error=f"{type(exc).__name__}: {exc}",
        )


# --- deterministic fake -------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w\-.])-?\d[\d,]*(?:\.\d+)?(?![\w\-])")
_LOAN_ID = re.compile(r"\bMC-\d+\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_SORT_CODE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)"
    r"|disregard (the )?(system|previous) (prompt|instructions)"
    r"|you are now\b|system prompt|<\s*/?\s*(system|assistant)\s*>",
    re.IGNORECASE,
)


def numbers_in(text: str) -> set[Decimal]:
    out: set[Decimal] = set()
    for token in _NUMBER.findall(text):
        try:
            out.add(abs(Decimal(token.replace(",", ""))))
        except InvalidOperation:
            continue
    return out


class FakePolicyGate:
    """Deterministic stand-ins for the exception policy's validators (``all`` strategy).

    * ``faithfulness`` — every number in the response appears in the grounding context.
    * ``context_relevance`` — at least one citation, and every citation is one of the
      exception's own evidence rows.
    * ``output_data_leakage`` — no other loan ids, e-mail addresses or sort codes.
    * ``input_prompt_injection`` (``screen_input``) — common injection phrasings.
    """

    policy_id = "fake-exception-policy"
    policy_version = 1

    def __init__(self, force_block: set[str] | None = None):
        self.force_block = set(force_block or ())
        """Exception ids to BLOCK unconditionally (for tests of the review loop)."""
        self.calls: list[PolicySubject] = []

    def evaluate(self, subject: PolicySubject) -> PolicyVerdict:
        self.calls.append(subject)
        grounded = numbers_in(subject.context)
        claimed = numbers_in(subject.response)
        ungrounded = sorted(claimed - grounded)
        faith_score = 1.0 if not claimed else (len(claimed) - len(ungrounded)) / len(claimed)

        bad_cites = sorted(set(subject.cited_evidence) - set(subject.evidence_keys))
        relevant = bool(subject.cited_evidence) and not bad_cites

        other_loans = sorted(
            {m.upper() for m in _LOAN_ID.findall(subject.response)}
            - ({subject.loan_id.upper()} if subject.loan_id else set())
        )
        leaks = (
            other_loans + _EMAIL.findall(subject.response) + _SORT_CODE.findall(subject.response)
        )

        rules = [
            RuleResult(
                validator="faithfulness",
                status="PASS" if not ungrounded else "FAIL",
                score=faith_score,
                threshold=1.0,
                detail=f"ungrounded numbers: {[str(n) for n in ungrounded]}" if ungrounded else "",
            ),
            RuleResult(
                validator="context_relevance",
                status="PASS" if relevant else "FAIL",
                score=1.0 if relevant else 0.0,
                threshold=1.0,
                detail=f"citations outside evidence: {bad_cites}" if bad_cites else "",
            ),
            RuleResult(
                validator="output_data_leakage",
                status="PASS" if not leaks else "FAIL",
                score=0.0 if leaks else 1.0,
                threshold=1.0,
                detail=f"leaked: {leaks}" if leaks else "",
            ),
        ]
        if subject.exception_id in self.force_block:
            rules.append(RuleResult(validator="forced_block", status="FAIL", detail="test hook"))
        decision: Decision = "PASS" if all(r.status == "PASS" for r in rules) else "BLOCK"
        return PolicyVerdict(
            decision=decision,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            rules=rules,
        )

    def screen_input(self, text: str) -> PolicyVerdict:
        hit = _INJECTION.search(text)
        return PolicyVerdict(
            decision="BLOCK" if hit else "PASS",
            policy_id="fake-input-screen",
            policy_version=1,
            rules=[
                RuleResult(
                    validator="input_prompt_injection",
                    status="FAIL" if hit else "PASS",
                    detail=f"matched {hit.group(0)!r}" if hit else "",
                )
            ],
        )
