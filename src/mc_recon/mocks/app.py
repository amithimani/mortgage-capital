"""One FastAPI app that stands in for everything external (``mc-recon mock``).

* **Disseqt** (shapes assumed, see ``disseqt_http.py``): application registry, policy publishing,
  synchronous validate, trace ingest, Decisions ledger and dashboard analytics.
* **LLM**: OpenAI-compatible chat completions replaying the oracle recordings by ``cache_key``.
  An unknown key is a ``422``. Like the in-process fake, the mock never improvises.
* **Fault injection**: ``POST /__mock/faults`` queues failures so outage, rate-limit and
  bad-output paths can be driven over the wire.

Everything is held in bounded in-memory collections. Restarting the mock clears it.

No ``from __future__ import annotations`` here: the routes close over local dependencies, which
FastAPI can only see if annotations are evaluated when the function is defined.
"""

import json
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from ..llm.fake import Recording
from ..llm.provider import canonical_json
from ..policy import FakePolicyGate, PolicySubject, PolicyVerdict

DEFAULT_POLICY = "exception-policy"
LEDGER_SIZE = 10_000
MOCK_MODEL = "mock-llm"


class Fault(BaseModel):
    target: Literal["llm", "validate", "traces"]
    mode: Literal["http_500", "http_429", "http_503", "invalid_json"]
    count: int = 1


class PolicyDefinition(BaseModel):
    policy_id: str = DEFAULT_POLICY
    thresholds: dict[str, float] = {}
    """validator -> minimum score. Validators not listed keep the mock's own PASS/FAIL."""
    strategy: Literal["all"] = "all"


def _apply(verdict: PolicyVerdict, policy: dict) -> PolicyVerdict:
    rules = []
    for rule in verdict.rules:
        threshold = policy["thresholds"].get(rule.validator)
        if threshold is not None and rule.score is not None:
            status = "PASS" if rule.score >= threshold else "FAIL"
            rule = rule.model_copy(update={"status": status, "threshold": threshold})
        rules.append(rule)
    return verdict.model_copy(
        update={
            "decision": "PASS" if all(r.status == "PASS" for r in rules) else "BLOCK",
            "policy_id": policy["policy_id"],
            "policy_version": policy["version"],
            "rules": rules,
        }
    )


def create_mock_app(recordings: Iterable[Recording], api_key: str = "mock-key") -> FastAPI:
    app = FastAPI(title="Mock Disseqt + LLM", version="0.1.0")
    index = {r.cache_key: r for r in recordings}
    gate = FakePolicyGate()
    applications: dict[str, dict] = {}
    policies: dict[str, dict] = {
        DEFAULT_POLICY: {
            "policy_id": DEFAULT_POLICY,
            "thresholds": {},
            "strategy": "all",
            "version": 1,
        }
    }
    traces: deque[dict] = deque(maxlen=LEDGER_SIZE)
    decisions: deque[dict] = deque(maxlen=LEDGER_SIZE)
    faults: dict[str, deque[str]] = defaultdict(deque)
    stats: Counter[str] = Counter()

    def need_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key != api_key:
            raise HTTPException(401, "invalid X-API-Key")

    def need_app(x_application_id: Annotated[str | None, Header()] = None) -> str:
        if x_application_id not in applications:
            raise HTTPException(403, "unknown X-Application-Id")
        return x_application_id

    def inject(target: str) -> JSONResponse | str | None:
        stats[f"{target}_requests"] += 1
        if not faults[target]:
            return None
        stats[f"{target}_faults"] += 1
        mode = faults[target].popleft()
        if mode == "invalid_json":
            return mode
        return JSONResponse(status_code=int(mode.split("_")[1]), content={"error": mode})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    # --- test control ----------------------------------------------------------------------

    @app.post("/__mock/faults")
    def add_fault(fault: Fault) -> dict:
        faults[fault.target].extend([fault.mode] * fault.count)
        return {"queued": len(faults[fault.target])}

    @app.get("/__mock/stats")
    def get_stats() -> dict:
        return dict(stats)

    # --- Disseqt: registry + policies ------------------------------------------------------

    @app.post("/api/v1/applications", status_code=201, dependencies=[Depends(need_key)])
    def register(body: dict[str, Any]) -> dict:
        application_id = f"app-{uuid.uuid4().hex[:12]}"
        applications[application_id] = {"application_id": application_id, **body}
        return applications[application_id]

    @app.get("/api/v1/applications/{application_id}", dependencies=[Depends(need_key)])
    def get_application(application_id: str) -> dict:
        if application_id not in applications:
            raise HTTPException(404, "unknown application")
        return applications[application_id]

    @app.post("/api/v1/policies", status_code=201, dependencies=[Depends(need_key)])
    def publish(body: PolicyDefinition) -> dict:
        version = policies.get(body.policy_id, {}).get("version", 0) + 1
        policies[body.policy_id] = {**body.model_dump(), "version": version}
        return policies[body.policy_id]

    # --- Disseqt: validate + ledger --------------------------------------------------------

    @app.post("/api/v1/validate", dependencies=[Depends(need_key)])
    def validate(body: dict[str, Any], application_id: Annotated[str, Depends(need_app)]) -> Any:
        fault = inject("validate")
        if isinstance(fault, JSONResponse):
            return fault
        if fault == "invalid_json":
            return {"not": "a verdict"}
        policy = policies.get(body.get("policy_id", ""))
        if policy is None:
            raise HTTPException(404, "unknown policy")
        match body.get("scope"):
            case "exception":
                raw = gate.evaluate(PolicySubject.model_validate(body["subject"]))
            case "run":
                raw = gate.evaluate_run(body["trajectory"])
            case "input":
                raw = gate.screen_input(body["text"])
            case other:
                raise HTTPException(422, f"unknown scope {other!r}")
        verdict = _apply(raw, policy)
        decisions.append(
            {
                "application_id": application_id,
                "scope": body["scope"],
                **verdict.model_dump(mode="json"),
            }
        )
        return verdict.model_dump(mode="json")

    @app.get("/api/v1/decisions", dependencies=[Depends(need_key)])
    def list_decisions() -> list[dict]:
        return list(decisions)

    # --- Disseqt: traces + dashboard -------------------------------------------------------

    @app.post("/api/v1/traces", status_code=202, dependencies=[Depends(need_key)])
    def ingest(body: dict[str, Any], application_id: Annotated[str, Depends(need_app)]) -> Any:
        fault = inject("traces")
        if isinstance(fault, JSONResponse):
            return fault
        for span in body["spans"]:
            traces.append({"application_id": application_id, **span})
        return {"accepted": len(body["spans"])}

    @app.get("/api/v1/traces/{run_id}", dependencies=[Depends(need_key)])
    def trace_timeline(run_id: str) -> list[dict]:
        return sorted((s for s in traces if s["run_id"] == run_id), key=lambda s: s["started_at"])

    @app.get("/api/v1/dashboard/{application_id}", dependencies=[Depends(need_key)])
    def dashboard(application_id: str) -> dict:
        spans = [s for s in traces if s["application_id"] == application_id]
        model = [s for s in spans if s["kind"] == "MODEL_EXEC"]
        mine = [d for d in decisions if d["application_id"] == application_id]
        exception_scope = [d for d in mine if d["scope"] == "exception"]
        scores: dict[str, list[float]] = defaultdict(list)
        for d in exception_scope:
            for rule in d["rules"]:
                if rule["score"] is not None:
                    scores[rule["validator"]].append(rule["score"])
        return {
            "application_id": application_id,
            "runs": len({s["run_id"] for s in spans}),
            "spans": len(spans),
            "spans_by_kind": dict(Counter(s["kind"] for s in spans)),
            "tokens": sum(
                s["attributes"].get("input_tokens", 0) + s["attributes"].get("output_tokens", 0)
                for s in model
            ),
            "decisions": dict(Counter(d["decision"] for d in exception_scope)),
            "mean_validator_scores": {
                k: round(sum(v) / len(v), 4) for k, v in sorted(scores.items())
            },
        }

    # --- OpenAI-compatible LLM -------------------------------------------------------------

    @app.post("/v1/chat/completions")
    def chat(body: dict[str, Any], authorization: Annotated[str | None, Header()] = None) -> Any:
        if authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid bearer token")
        fault = inject("llm")
        if isinstance(fault, JSONResponse):
            return fault
        key = body.get("metadata", {}).get("cache_key", "")
        recording = index.get(key)
        if recording is None:
            return PlainTextResponse(f"no recording for cache_key {key[:16]}", status_code=422)
        content = "{not json" if fault == "invalid_json" else canonical_json(recording.output)
        prompt_chars = len(json.dumps(body.get("messages", [])))
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": MOCK_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": max(1, prompt_chars // 4),
                "completion_tokens": max(1, len(content) // 4),
            },
        }

    return app
