"""The reconciliation agent as a LangGraph state machine.

    ingest → propose_mapping → await_mapping_confirm (interrupt) → reconcile
      → classify_exceptions → draft_rationale → validate_policy
      → [all PASS] emit_log
      → [any BLOCK] human_review_queue (interrupt)
            → [any edited] validate_policy  (re-check, plan D1)
            → [else]       emit_log

State holds only JSON-native values (dicts/lists/str), so checkpoints are portable and never
depend on pickling our classes. Dependencies (LLM, policy gate, store, tracer) come in through
:class:`AgentDeps`; nothing in here knows whether they are fakes.

``trajectory`` is an append-only list (LangGraph reducer) with one entry per step: what ran and
with which inputs. The tool-call-accuracy and plan-coherence validators score it, and it is
recorded in the run's audit trail.

Interrupt payloads (what the API shows a human) and resume values (what it sends back):

* ``mapping_confirmation`` → ``{"analyst": str, "mapping": ColumnMapping-json | None}``
* ``exception_review`` → ``{exception_id: {"analyst", "action": approve|edit|reject,
  "rationale"?, "remediation_note"?, "note"?}}`` covering every pending id.
"""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .engine import (
    ENGINE_VERSION,
    ColumnMapping,
    Discrepancy,
    EngineInputError,
    ExceptionCategory,
    FileProfile,
    ReconConfig,
    Source,
    constrain_category,
    load_inputs,
    profile_csv,
    reconcile,
)
from .engine.loader import read_csv
from .governance import GovernanceLayer
from .limits import AgentLimits, LimitExceeded, RunBudget, estimate_tokens
from .llm import (
    PROMPT_VERSIONS,
    ClassificationOutput,
    LLMOutputValidationError,
    LLMProvider,
    LLMRequest,
    LLMUnavailable,
    MappingProposal,
    RationaleOutput,
    classification_request,
    mapping_request,
    rationale_request,
)
from .llm.prompts import render_messages
from .llm.provider import T, canonical_json
from .memory import header_fingerprint, recalled_mapping_fits
from .observability import NullTracer, Tracer
from .policy import (
    PolicyGate,
    PolicySubject,
    PolicyVerdict,
    RuleResult,
    combine,
    evaluate_fail_closed,
    evaluate_run_fail_closed,
    screen_fail_closed,
)
from .remediation import assign
from .store import RunStore

SYSTEM_ACTOR = "agent"
DEGRADABLE = (LimitExceeded, LLMUnavailable, LLMOutputValidationError)
"""Per-finding LLM failures that hold the finding for a human instead of failing the run."""
_PROMPT_TEXT_FIELDS = ("mandate_ref", "channel", "loan_id")
WITHHELD = "[withheld by input screen]"

# exception lifecycle
DRAFTED, EDITED = "drafted", "edited"
PASSED, BLOCKED = "passed", "blocked"
APPROVED_OVERRIDE, REJECTED, LOGGED = "approved_override", "rejected", "logged"
NEEDS_VALIDATION = frozenset({DRAFTED, EDITED})
LOGGABLE = frozenset({PASSED, APPROVED_OVERRIDE})


class InputScreenError(EngineInputError):
    """A column *header* failed the input screen. Headers can't be withheld, so stop."""


class RunState(TypedDict, total=False):
    run_id: str
    dataset: str
    input_paths: dict[str, str]
    file_hashes: dict[str, str]
    profiles: list[dict]
    input_screen: list[dict]
    input_screen_full: list[dict]
    usage: dict[str, Any]
    mapping_proposal: dict
    mapping_problems: list[str]
    mapping: dict
    mapping_confirmed_by: str
    recon_stats: dict
    discrepancies: list[dict]
    exceptions: dict[str, dict]
    review_round: int
    trajectory: Annotated[list[dict], operator.add]
    status: str


@dataclass
class AgentDeps:
    llm: LLMProvider
    gate: PolicyGate
    store: RunStore
    tracer: Tracer = field(default_factory=NullTracer)
    recon_config: ReconConfig = field(default_factory=ReconConfig)
    limits: AgentLimits = field(default_factory=AgentLimits)
    mapping_memory_days: int = 0
    """Days a confirmed mapping can be recalled for. ``0`` turns long-term memory off."""
    governance: GovernanceLayer | None = None
    """Use Case 2 controls. ``None`` runs Use Case 1 alone."""


# --- helpers ------------------------------------------------------------------------------


def _paths(state: RunState) -> dict[Source, Path]:
    return {Source(k): Path(v) for k, v in state["input_paths"].items()}


def _headers(state: RunState) -> dict[Source, list[str]]:
    return {Source(p["source"]): [c["name"] for c in p["columns"]] for p in state["profiles"]}


def mapping_sha(mapping: dict) -> str:
    return hashlib.sha256(canonical_json(mapping).encode()).hexdigest()[:16]


def evidence_keys(exc: dict) -> list[str]:
    return [f"{e['source']}:{e['row']}" for e in exc["evidence"]]


def policy_subject(exc: dict) -> PolicySubject:
    context = canonical_json(
        {
            "summary": exc["summary"],
            "expected": exc["expected"],
            "actual": exc["actual"],
            "evidence": {
                f"{e['source']}:{e['row']}": {"key": e["key"], **e["fields"]}
                for e in exc["evidence"]
            },
        }
    )
    return PolicySubject(
        exception_id=exc["id"],
        loan_id=exc["loan_id"],
        category=exc.get("category"),
        prompt=(
            f"Explain reconciliation exception {exc['id']} ({exc.get('category')}) and propose "
            "remediation using only the supplied evidence."
        ),
        context=context,
        response=f"{exc['rationale']}\n\nRemediation: {exc['remediation_note']}",
        evidence_keys=evidence_keys(exc),
        cited_evidence=exc["cited_evidence"],
    )


def _discrepancy(exc: dict) -> Discrepancy:
    return Discrepancy.model_validate(
        {k: exc[k] for k in ("id", "kind", "loan_id", "summary", "expected", "actual", "evidence")}
    )


def build_graph(deps: AgentDeps) -> StateGraph:
    store, tracer, llm, gate = deps.store, deps.tracer, deps.llm, deps.gate
    limits, governance = deps.limits, deps.governance

    def call_llm(
        run_id: str, node: str, request: LLMRequest, schema: type[T], budget: RunBudget
    ) -> T:
        """One structured call: admitted by the budget, with a bounded re-ask on invalid output."""
        attempts = limits.max_reasks + 1
        for attempt in range(1, attempts + 1):
            messages = render_messages(request)
            budget.admit(estimate_tokens(messages))
            with tracer.span(
                run_id,
                node,
                "MODEL_EXEC",
                task=request.task.value,
                prompt_version=request.prompt_version,
                cache_key=request.cache_key,
                model_id=llm.model_id,
                temperature=request.temperature,
                attempt=attempt,
                input_messages=messages,
            ) as span:
                try:
                    result = llm.generate(request, schema)
                except LLMOutputValidationError as exc:
                    budget.charge_failure()
                    span.set(outcome="invalid_output", error=str(exc)[:500])
                    if attempt == attempts:
                        raise
                    request = request.model_copy(update={"feedback": str(exc)[:500]})
                    continue
                except LLMUnavailable as exc:
                    budget.charge_failure()
                    span.set(outcome="unavailable", error=str(exc)[:500])
                    raise
                budget.charge(result.usage, result.model_id)
                span.set(
                    outcome="ok",
                    model_id=result.model_id,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    tokens_estimated=result.usage.estimated,
                    output_message=result.output.model_dump(mode="json"),
                )
                return result.output
        raise AssertionError("unreachable")

    def record_degraded(run_id: str, exc_id: str, node: str, error: Exception) -> str:
        reason = f"{node} skipped: {error}"[:500]
        store.record_event(
            run_id,
            "limit_reached" if isinstance(error, LimitExceeded) else "agent_degraded",
            SYSTEM_ACTOR,
            {"node": node, "error_type": type(error).__name__, "reason": reason},
            subject_id=exc_id,
            dedupe_key=f"{run_id}:degraded:{exc_id}:{node}",
        )
        return reason

    def screen_prompt_text(run_id: str, raw: dict) -> str | None:
        """Screen the file-derived text this finding would put in a prompt.

        The profile samples only cover the first few values per column, and the evidence rows
        can come from anywhere in the file. Returns why the finding must not reach the LLM.
        """
        values = {e["key"] for e in raw["evidence"]}
        for e in raw["evidence"]:
            values |= {e["fields"][f] for f in _PROMPT_TEXT_FIELDS if e["fields"].get(f)}
        verdict = screen_fail_closed(gate, "\n".join(sorted(values)))
        if not verdict.blocked:
            return None
        failed = [r.validator for r in verdict.rules if r.status == "FAIL"] or ["screen_error"]
        store.record_event(
            run_id,
            "input_screen_block",
            SYSTEM_ACTOR,
            {"scope": "finding", "failed_rules": failed},
            subject_id=raw["id"],
            dedupe_key=f"{run_id}:screen:finding:{raw['id']}",
        )
        return f"evidence failed the input screen ({', '.join(failed)}); not sent to the LLM"

    # --- nodes ----------------------------------------------------------------------------

    def ingest(state: RunState) -> RunState:
        run_id = state["run_id"]
        with tracer.span(run_id, "ingest", "TOOL_EXEC", tool="parse_and_profile") as span:
            profiles, findings, full_scan = [], [], []
            for source, path in _paths(state).items():
                profile = profile_csv(path, source).model_dump(mode="json")
                if profile["row_count"] > limits.max_rows:
                    raise LimitExceeded(
                        "max_rows",
                        f"{source}: {profile['row_count']} rows > {limits.max_rows}",
                    )
                _, rows = read_csv(path)
                for col in profile["columns"]:
                    header_verdict = screen_fail_closed(gate, col["name"])
                    if header_verdict.blocked:
                        cause = f" ({header_verdict.error})" if header_verdict.error else ""
                        raise InputScreenError(
                            f"{source}: header {col['name']!r} failed the input screen{cause}"
                        )
                    verdict = screen_fail_closed(gate, "\n".join(col["samples"]))
                    if verdict.blocked:
                        col["samples"] = [WITHHELD] * len(col["samples"])
                        findings.append(
                            {
                                "source": source.value,
                                "column": col["name"],
                                "failed_rules": [
                                    r.validator for r in verdict.rules if r.status == "FAIL"
                                ],
                                "verdict": verdict.model_dump(mode="json"),
                            }
                        )
                    elif col["inferred_type"] == "string":
                        # The brief asks for checks on the ingested *files*, not just the samples.
                        distinct = sorted({r[col["name"]].strip() for r in rows} - {""})
                        whole = screen_fail_closed(gate, "\n".join(distinct))
                        if whole.blocked:
                            full_scan.append(
                                {
                                    "source": source.value,
                                    "column": col["name"],
                                    "failed_rules": [
                                        r.validator for r in whole.rules if r.status == "FAIL"
                                    ],
                                }
                            )
                profiles.append(profile)
            for f in full_scan:
                store.record_event(
                    run_id,
                    "input_screen_finding",
                    SYSTEM_ACTOR,
                    f,
                    subject_id=f"{f['source']}.{f['column']}",
                    dedupe_key=f"{run_id}:fullscan:{f['source']}:{f['column']}",
                )
            hashes = {p["source"]: p["sha256"] for p in profiles}
            for f in findings:
                store.record_event(
                    run_id,
                    "input_screen_block",
                    SYSTEM_ACTOR,
                    f,
                    subject_id=f"{f['source']}.{f['column']}",
                    dedupe_key=f"{run_id}:screen:{f['source']}:{f['column']}",
                )
            store.update_run(run_id, status="ingested", manifest={"file_hashes": hashes})
            span.set(files=len(profiles), screen_blocks=len(findings))
        return {
            "profiles": profiles,
            "file_hashes": hashes,
            "input_screen": findings,
            "input_screen_full": full_scan,
            "trajectory": [
                {
                    "step": "ingest",
                    "tool": "parse_and_profile",
                    "file_hashes": hashes,
                    "screen_blocks": len(findings),
                }
            ],
        }

    def propose_mapping(state: RunState) -> RunState:
        run_id = state["run_id"]
        headers = _headers(state)
        recalled = None
        if deps.mapping_memory_days:
            fingerprint = header_fingerprint(headers)
            memory = store.recall_mapping(fingerprint, deps.mapping_memory_days)
            if memory and recalled_mapping_fits(
                ColumnMapping.model_validate(memory["mapping"]), headers, state["profiles"]
            ):
                recalled = memory
        budget = RunBudget(limits, state.get("usage"))
        if recalled:
            # Still goes to the analyst: memory replaces the LLM's proposal, not the human.
            proposal = MappingProposal(
                mapping=ColumnMapping.model_validate(recalled["mapping"]),
                notes=f"Recalled: confirmed by {recalled['confirmed_by']}.",
            )
            store.record_event(
                run_id,
                "mapping_recalled",
                SYSTEM_ACTOR,
                {k: recalled[k] for k in ("fingerprint", "confirmed_by", "source_run")},
                dedupe_key=f"{run_id}:recalled",
            )
            tool, cache_key = "memory", None
        else:
            request = mapping_request([FileProfile.model_validate(p) for p in state["profiles"]])
            proposal = call_llm(run_id, "propose_mapping", request, MappingProposal, budget)
            tool, cache_key = "llm", request.cache_key
        problems = proposal.mapping.problems(headers)
        return {
            "mapping_proposal": proposal.model_dump(mode="json"),
            "mapping_problems": problems,
            "usage": budget.snapshot(),
            "status": "awaiting_mapping_confirmation",
            "trajectory": [
                {
                    "step": "propose_mapping",
                    "tool": tool,
                    "cache_key": cache_key,
                    "problems": len(problems),
                }
            ],
        }

    def await_mapping_confirm(state: RunState) -> RunState:
        run_id = state["run_id"]
        answer = interrupt(
            {
                "type": "mapping_confirmation",
                "run_id": run_id,
                "proposal": state["mapping_proposal"],
                "problems": state["mapping_problems"],
            }
        )
        analyst = answer["analyst"]
        edited = answer.get("mapping")
        mapping = ColumnMapping.model_validate(edited or state["mapping_proposal"]["mapping"])
        mapping.validate_against(_headers(state))  # API pre-validates; this is the backstop
        mapping_json = mapping.model_dump(mode="json")
        with tracer.span(
            run_id,
            "await_mapping_confirm",
            "OVERSIGHT",
            analyst=analyst,
            decision="edited" if edited else "confirmed",
        ):
            store.record_event(
                run_id,
                "mapping_edited" if edited else "mapping_confirmed",
                analyst,
                {"mapping": mapping_json, "proposal_problems": state["mapping_problems"]},
                dedupe_key=f"{run_id}:mapping",
            )
        if deps.mapping_memory_days:  # only a human-confirmed mapping is ever remembered
            store.remember_mapping(
                header_fingerprint(_headers(state)), mapping_json, analyst, run_id
            )
        if governance:
            governance.on_mapping_confirmed(run_id, analyst)
        return {
            "mapping": mapping_json,
            "mapping_confirmed_by": analyst,
            "status": "reconciling",
            "trajectory": [
                {
                    "step": "await_mapping_confirm",
                    "tool": "human",
                    "analyst": analyst,
                    "edited": bool(edited),
                    "mapping_sha": mapping_sha(mapping_json),
                }
            ],
        }

    def reconcile_node(state: RunState) -> RunState:
        run_id = state["run_id"]
        mapping = ColumnMapping.model_validate(state["mapping"])
        used_sha = mapping_sha(mapping.model_dump(mode="json"))
        with tracer.span(
            run_id,
            "reconcile",
            "TOOL_EXEC",
            tool="reconciliation_engine",
            engine_version=ENGINE_VERSION,
            mapping_sha=used_sha,
        ) as span:
            inputs = load_inputs(_paths(state), mapping)
            hashes = {s.value: h for s, h in inputs.file_hashes.items()}
            if hashes != state["file_hashes"]:
                raise EngineInputError("input files changed between ingest and reconcile")
            result = reconcile(inputs, deps.recon_config)
            span.set(discrepancies=len(result.discrepancies))
            if len(result.discrepancies) > limits.max_findings:
                raise LimitExceeded(
                    "max_findings",
                    f"{len(result.discrepancies)} findings > {limits.max_findings}; "
                    "this usually means the confirmed mapping is wrong",
                )
        store.update_run(
            run_id,
            status="classifying",
            manifest={
                "engine_version": result.engine_version,
                "recon_config": result.config.model_dump(mode="json"),
                "mapping_confirmed_by": state["mapping_confirmed_by"],
                "recon_stats": result.stats,
            },
        )
        return {
            "discrepancies": [d.model_dump(mode="json") for d in result.discrepancies],
            "recon_stats": result.stats,
            "trajectory": [
                {
                    "step": "reconcile",
                    "tool": "reconciliation_engine",
                    "engine_version": result.engine_version,
                    "mapping_sha": used_sha,
                    "file_hashes": hashes,
                    "discrepancies": len(result.discrepancies),
                }
            ],
        }

    def classify_exceptions(state: RunState) -> RunState:
        run_id = state["run_id"]
        budget = RunBudget(limits, state.get("usage"))
        exceptions: dict[str, dict] = {}
        for raw in state["discrepancies"]:
            d = Discrepancy.model_validate(raw)
            held = screen_prompt_text(run_id, raw)
            out = None
            if held is None:
                try:
                    out = call_llm(
                        run_id,
                        "classify_exception",
                        classification_request(d),
                        ClassificationOutput,
                        budget,
                    )
                except DEGRADABLE as exc:
                    held = record_degraded(run_id, d.id, "classify_exception", exc)
            if out is not None:
                proposed, reason = out.category.value, out.reason
                category = constrain_category(d.kind, proposed)
            else:
                category = ExceptionCategory.OTHER_NEEDS_REVIEW
                proposed, reason = category.value, held
            exceptions[d.id] = {
                **raw,
                "proposed_category": proposed,
                "category": category.value,
                "classification_reason": reason,
                **assign(raw, category.value),
                "status": DRAFTED,
                "verdicts": [],
                "reviews": [],
                "degraded": held,
            }
        return {
            "exceptions": exceptions,
            "discrepancies": [],  # now held in `exceptions`; don't checkpoint it twice
            "usage": budget.snapshot(),
            "trajectory": [
                {"step": "classify_exceptions", "tool": "llm", "count": len(exceptions)}
            ],
        }

    def draft_rationale(state: RunState) -> RunState:
        run_id = state["run_id"]
        budget = RunBudget(limits, state.get("usage"))
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        for exc_id in sorted(exceptions):
            exc = exceptions[exc_id]
            if not exc["degraded"]:
                try:
                    out = call_llm(
                        run_id,
                        "draft_rationale",
                        rationale_request(_discrepancy(exc), exc["category"]),
                        RationaleOutput,
                        budget,
                    )
                except DEGRADABLE as err:
                    exc["degraded"] = record_degraded(run_id, exc_id, "draft_rationale", err)
                else:
                    exc.update(
                        rationale=out.rationale,
                        remediation_note=out.remediation_note,
                        cited_evidence=out.cited_evidence,
                    )
                    continue
            exc.update(
                rationale=exc["summary"],
                remediation_note=(
                    "Automated drafting did not run for this finding. Review the evidence rows "
                    "and decide the remediation manually."
                ),
                cited_evidence=evidence_keys(exc),
            )
        return {
            "exceptions": exceptions,
            "usage": budget.snapshot(),
            "status": "validating",
            "trajectory": [{"step": "draft_rationale", "tool": "llm", "count": len(exceptions)}],
        }

    def validate_policy(state: RunState) -> RunState:
        run_id = state["run_id"]
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        trajectory = state.get("trajectory", [])
        with tracer.span(run_id, "validate_run_trajectory", "TOOL_EXEC") as span:
            run_verdict = evaluate_run_fail_closed(gate, trajectory)
            span.set(decision=run_verdict.decision, steps=[t["step"] for t in trajectory])
        store.record_event(
            run_id,
            "trajectory_verdict",
            SYSTEM_ACTOR,
            {"steps": [t["step"] for t in trajectory], **run_verdict.model_dump(mode="json")},
            dedupe_key=f"{run_id}:trajectory:{len(trajectory)}",
        )
        evaluated = 0
        for exc_id in sorted(exceptions):
            exc = exceptions[exc_id]
            if exc["status"] not in NEEDS_VALIDATION:
                continue
            attempt = len(exc["verdicts"]) + 1
            with tracer.span(run_id, "validate_policy", "TOOL_EXEC", exception_id=exc_id) as span:
                if exc.get("degraded"):  # template text: hold it, there's nothing to score
                    verdict = PolicyVerdict(
                        decision="BLOCK",
                        policy_id=gate.policy_id,
                        enforcement="harness_hold",
                        error=exc["degraded"],
                        rules=[
                            RuleResult(
                                validator="harness_hold", status="FAIL", detail=exc["degraded"]
                            )
                        ],
                    )
                else:
                    verdict = combine(evaluate_fail_closed(gate, policy_subject(exc)), run_verdict)
                if governance:
                    breaches = governance.on_verdict(run_id, exc, verdict, attempt, dict(state))
                    if breaches:
                        verdict = verdict.model_copy(
                            update={"decision": "BLOCK", "rules": [*verdict.rules, *breaches]}
                        )
                span.set(
                    decision=verdict.decision,
                    policy_id=verdict.policy_id,
                    policy_version=verdict.policy_version,
                    scores={r.validator: r.score for r in verdict.rules},
                )
            evaluated += 1
            exc["verdicts"] = [*exc["verdicts"], verdict.model_dump(mode="json")]
            exc["status"] = BLOCKED if verdict.blocked else PASSED
            store.record_event(
                run_id,
                "policy_verdict",
                SYSTEM_ACTOR,
                {"attempt": attempt, **verdict.model_dump(mode="json")},
                subject_id=exc_id,
                dedupe_key=f"{run_id}:verdict:{exc_id}:{attempt}",
            )
        all_verdicts = [v for e in exceptions.values() for v in e["verdicts"]]
        store.update_run(
            run_id,
            manifest={
                "policy_ids": sorted({v["policy_id"] for v in all_verdicts}),
                "policy_versions": sorted(
                    {
                        f"{v['policy_id']}@{v['policy_version']}"
                        for v in all_verdicts
                        if v["policy_version"] is not None
                    }
                ),
            },
        )
        return {
            "exceptions": exceptions,
            "trajectory": [
                {
                    "step": "validate_policy",
                    "tool": "policy_gate",
                    "evaluated": evaluated,
                    "run_decision": run_verdict.decision,
                }
            ],
        }

    def human_review_queue(state: RunState) -> RunState:
        run_id = state["run_id"]
        round_ = state.get("review_round", 0) + 1
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        pending = sorted(k for k, v in exceptions.items() if v["status"] == BLOCKED)
        final_round = round_ >= limits.max_review_rounds
        if final_round:
            store.record_event(
                run_id,
                "limit_reached",
                SYSTEM_ACTOR,
                {"limit": "max_review_rounds", "round": round_, "pending": pending},
                dedupe_key=f"{run_id}:review_cap",
            )
        decisions: dict[str, dict] = interrupt(
            {
                "type": "exception_review",
                "run_id": run_id,
                "round": round_,
                "max_rounds": limits.max_review_rounds,
                "allowed_actions": (
                    ["approve", "reject"] if final_round else ["approve", "edit", "reject"]
                ),
                "pending": pending,
            }
        )
        missing = set(pending) - set(decisions)
        if missing:
            raise ValueError(f"review round {round_} is missing decisions for {sorted(missing)}")
        actions: dict[str, int] = {}
        for exc_id in pending:
            exc, dec = exceptions[exc_id], decisions[exc_id]
            action, analyst = dec["action"], dec["analyst"]
            actions[action] = actions.get(action, 0) + 1
            with tracer.span(
                run_id,
                "human_review",
                "OVERSIGHT",
                exception_id=exc_id,
                action=action,
                analyst=analyst,
            ):
                if action == "approve":
                    exc["status"] = APPROVED_OVERRIDE
                elif action == "reject":
                    exc["status"] = REJECTED
                elif action == "edit":
                    if final_round:  # the service refuses this first; this is the backstop
                        raise ValueError(f"edit is not allowed in final round {round_}")
                    exc["rationale"] = dec.get("rationale") or exc["rationale"]
                    exc["remediation_note"] = dec.get("remediation_note") or exc["remediation_note"]
                    exc["degraded"] = None  # human text now, so the gate can judge it
                    exc["status"] = EDITED
                else:
                    raise ValueError(f"unknown review action {action!r}")
                exc["reviews"] = [*exc["reviews"], {"round": round_, **dec}]
                store.record_event(
                    run_id,
                    "exception_reviewed",
                    analyst,
                    {"round": round_, **dec},
                    subject_id=exc_id,
                    dedupe_key=f"{run_id}:review:{exc_id}:{round_}",
                )
                if governance:
                    governance.on_review(run_id, exc_id, analyst, action, round_)
        return {
            "exceptions": exceptions,
            "review_round": round_,
            "trajectory": [
                {"step": "human_review_queue", "tool": "human", "round": round_, **actions}
            ],
        }

    def emit_log(state: RunState) -> RunState:
        run_id = state["run_id"]
        exceptions = {k: dict(v) for k, v in state["exceptions"].items()}
        written = 0
        with tracer.span(run_id, "emit_log", "TOOL_EXEC", tool="exception_writer") as span:
            for exc_id in sorted(exceptions):
                exc = exceptions[exc_id]
                if exc["status"] not in LOGGABLE:
                    continue
                entry = {
                    "run_id": run_id,
                    "exception_id": exc_id,
                    "loan_id": exc["loan_id"],
                    "kind": exc["kind"],
                    "category": exc["category"],
                    "owner": exc["owner"],
                    "priority": exc["priority"],
                    "sla_business_days": exc["sla_business_days"],
                    "amount_at_risk": exc["amount_at_risk"],
                    "amount_basis": exc["amount_basis"],
                    "priority_reason": exc["priority_reason"],
                    "summary": exc["summary"],
                    "rationale": exc["rationale"],
                    "remediation_note": exc["remediation_note"],
                    "evidence": exc["evidence"],
                    "cited_evidence": exc["cited_evidence"],
                    "final_verdict": exc["verdicts"][-1],
                    "human_override": exc["status"] == APPROVED_OVERRIDE,
                    "degraded": exc.get("degraded"),
                    "reviews": exc["reviews"],
                }
                written += store.append_log(run_id, exc_id, entry)
                exc["status"] = LOGGED
            span.set(written=written)
        store.update_run(run_id, status="completed")
        return {"exceptions": exceptions, "status": "completed"}

    # --- routing --------------------------------------------------------------------------

    def after_validate(state: RunState) -> str:
        blocked = any(e["status"] == BLOCKED for e in state["exceptions"].values())
        return "human_review_queue" if blocked else "emit_log"

    def after_review(state: RunState) -> str:
        edited = any(e["status"] == EDITED for e in state["exceptions"].values())
        return "validate_policy" if edited else "emit_log"

    g = StateGraph(RunState)
    g.add_node("ingest", ingest)
    g.add_node("propose_mapping", propose_mapping)
    g.add_node("await_mapping_confirm", await_mapping_confirm)
    g.add_node("reconcile", reconcile_node)
    g.add_node("classify_exceptions", classify_exceptions)
    g.add_node("draft_rationale", draft_rationale)
    g.add_node("validate_policy", validate_policy)
    g.add_node("human_review_queue", human_review_queue)
    g.add_node("emit_log", emit_log)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "propose_mapping")
    g.add_edge("propose_mapping", "await_mapping_confirm")
    g.add_edge("await_mapping_confirm", "reconcile")
    g.add_edge("reconcile", "classify_exceptions")
    g.add_edge("classify_exceptions", "draft_rationale")
    g.add_edge("draft_rationale", "validate_policy")
    g.add_conditional_edges("validate_policy", after_validate, ["human_review_queue", "emit_log"])
    g.add_conditional_edges("human_review_queue", after_review, ["validate_policy", "emit_log"])
    g.add_edge("emit_log", END)
    return g


def manifest_base(dataset: str, llm: LLMProvider) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "engine_version": ENGINE_VERSION,
        "prompt_versions": {t.value: v for t, v in PROMPT_VERSIONS.items()},
        "model_id": llm.model_id,
    }


__all__ = [
    "AgentDeps",
    "InputScreenError",
    "RunState",
    "build_graph",
    "evidence_keys",
    "manifest_base",
    "mapping_sha",
    "policy_subject",
]
