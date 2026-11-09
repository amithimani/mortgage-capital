"""FastAPI surface (plan §4). Thin: auth header, idempotency, error mapping, then RunService.

With ``MC_ANALYST_TOKENS`` set, every route except the probes needs a bearer token and the
analyst identity comes from it. Without it (local PoC mode) identity is the unverified
``X-Analyst-Id`` header and reads are open. Either way the identity is recorded on every
oversight event. Mutating endpoints honour ``Idempotency-Key``: a retry with the same key and
body replays the original response; the same key with a different body is ``409``.

Run with the fakes::

    mc-recon serve                    # reads PORT / MC_* env vars, see mc_recon.config
    uvicorn mc_recon.api:app --reload # dev
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from .config import ConfigError, Settings
from .llm.provider import canonical_json
from .service import Conflict, ReviewDecision, RunService, ServiceError, Unauthorized
from .store import IN_PROGRESS, RunStore

PUBLIC_PATHS = frozenset({"/healthz", "/readyz"})


class StartRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str


class StartBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = "week-1 baseline"
    datasets: list[str] | None = None


class ConfirmMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping: dict | None = None
    """Omit to accept the proposal as-is; supply a full ColumnMapping to edit it."""


def _token_analyst(request: Request, authorization: str | None) -> str:
    """Resolve a bearer token. Compares against every token so timing reveals nothing."""
    scheme, _, presented = (authorization or "").partition(" ")
    found = None
    for token, analyst in request.app.state.analyst_tokens.items():
        if secrets.compare_digest(presented.encode(), token.encode()):
            found = analyst
    if scheme.lower() != "bearer" or found is None:
        raise Unauthorized("a valid bearer token is required")
    return found


def analyst_id(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_analyst_id: Annotated[str | None, Header()] = None,
) -> str:
    if request.app.state.analyst_tokens:
        return _token_analyst(request, authorization)
    if not x_analyst_id or not x_analyst_id.strip():
        raise Unauthorized("X-Analyst-Id header is required")
    return x_analyst_id.strip()


def require_reader(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    """With tokens configured every route needs one, reads included. Probes stay open."""
    if request.app.state.analyst_tokens and request.url.path not in PUBLIC_PATHS:
        _token_analyst(request, authorization)


Analyst = Annotated[str, Depends(analyst_id)]
IdemKey = Annotated[str | None, Header(alias="Idempotency-Key")]


def _is_ready(dependency: Any) -> bool:
    """Ask a dependency that can answer (the HTTP clients); in-process ones are always up."""
    check = getattr(dependency, "ready", None)
    return dependency is not None and (check is None or bool(check()))


def create_app(service: RunService, analyst_tokens: dict[str, str] | None = None) -> FastAPI:
    app = FastAPI(
        title="Mortgage Capital Reconciliation Agent",
        version="0.1.0",
        dependencies=[Depends(require_reader)],
    )
    app.state.analyst_tokens = dict(analyst_tokens or {})
    store: RunStore = service.store

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.message, "detail": exc.detail}
        )

    def idempotent(
        key: str | None, scope: str, analyst: str, body: Any, status: int, fn: Callable[[], Any]
    ) -> JSONResponse:
        if not key:
            return JSONResponse(status_code=status, content=fn())
        digest = hashlib.sha256(canonical_json([scope, analyst, body]).encode()).hexdigest()
        if not store.reserve_idempotent(key, digest):  # atomic: only one request does the work
            prior = store.get_idempotent(key)
            if prior is None or prior["status_code"] == IN_PROGRESS:
                if prior is not None and prior["request_hash"] != digest:
                    raise Conflict("Idempotency-Key was already used with a different request")
                raise Conflict("a request with this Idempotency-Key is still in progress")
            if prior["request_hash"] != digest:
                raise Conflict("Idempotency-Key was already used with a different request")
            return JSONResponse(status_code=prior["status_code"], content=prior["response"])
        try:
            content = fn()
        except BaseException:
            store.release_idempotent(key)
            raise
        store.complete_idempotent(key, status, content)
        return JSONResponse(status_code=status, content=content)

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "model_id": service.deps.llm.model_id,
            "policy_id": service.deps.gate.policy_id,
        }

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        """Readiness: the run store answers and the LLM and policy gate are reachable."""
        checks = {
            "store": store.ping(),
            "llm": _is_ready(service.deps.llm),
            "gate": _is_ready(service.deps.gate),
        }
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **checks})

    @app.post("/runs", status_code=201)
    def start_run(body: StartRun, analyst: Analyst, idem: IdemKey = None) -> JSONResponse:
        return idempotent(
            idem,
            "start_run",
            analyst,
            body.model_dump(),
            201,
            lambda: service.start_run(body.dataset, analyst),
        )

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return service.get_run(run_id)

    @app.get("/runs/{run_id}/mapping")
    def get_mapping(run_id: str) -> dict:
        return service.get_mapping(run_id)

    @app.post("/runs/{run_id}/mapping/confirm")
    def confirm_mapping(
        run_id: str, body: ConfirmMapping, analyst: Analyst, idem: IdemKey = None
    ) -> JSONResponse:
        return idempotent(
            idem,
            f"confirm:{run_id}",
            analyst,
            body.model_dump(),
            200,
            lambda: service.confirm_mapping(run_id, analyst, body.mapping),
        )

    @app.get("/runs/{run_id}/exceptions")
    def list_exceptions(run_id: str) -> list[dict]:
        return service.list_exceptions(run_id)

    @app.post("/runs/{run_id}/exceptions/{exception_id}/review")
    def review(
        run_id: str,
        exception_id: str,
        body: ReviewDecision,
        analyst: Analyst,
        idem: IdemKey = None,
    ) -> JSONResponse:
        return idempotent(
            idem,
            f"review:{run_id}:{exception_id}",
            analyst,
            body.model_dump(),
            200,
            lambda: service.review_exception(run_id, exception_id, analyst, body),
        )

    @app.get("/runs/{run_id}/remediation-log")
    def remediation_log(run_id: str) -> list[dict]:
        return service.remediation_log(run_id)

    @app.get("/runs/{run_id}/metrics")
    def run_metrics(run_id: str) -> dict:
        return service.metrics(run_id)

    @app.get("/runs/{run_id}/trajectory")
    def trajectory(run_id: str) -> list[dict]:
        return service.trajectory(run_id)

    @app.get("/metrics")
    def metrics_summary() -> dict:
        return service.metrics_summary()

    @app.post("/baselines", status_code=201)
    def start_baseline(body: StartBaseline, analyst: Analyst, idem: IdemKey = None) -> JSONResponse:
        return idempotent(
            idem,
            "baseline",
            analyst,
            body.model_dump(),
            201,
            lambda: service.run_baseline(analyst, body.label, body.datasets),
        )

    @app.get("/baselines/latest")
    def latest_baseline() -> dict:
        return service.latest_baseline()

    @app.get("/runs/{run_id}/events")
    def events(run_id: str) -> list[dict]:
        return service.events(run_id)

    @app.get("/runs/{run_id}/harness")
    def harness(run_id: str) -> dict:
        return service.harness(run_id)

    @app.get("/memory/mappings")
    def list_memory() -> list[dict]:
        return service.list_memory()

    @app.delete("/memory/mappings/{fingerprint}")
    def forget_memory(fingerprint: str, analyst: Analyst) -> dict:
        return service.forget_memory(fingerprint, analyst)

    @app.get("/governance/system")
    def governance_system() -> dict:
        return service.governance.system.model_dump()

    @app.get("/governance/controls")
    def governance_controls() -> list[dict]:
        return service.governance.control_statuses()

    @app.get("/governance/coverage")
    def governance_coverage() -> dict:
        return service.governance.coverage()

    @app.get("/governance/evidence")
    def governance_evidence(control_id: str | None = None, run_id: str | None = None) -> list[dict]:
        service.governance  # noqa: B018 — 404 when the layer is off
        return store.evidence(control_id, run_id)

    @app.get("/governance/report", response_model=None)
    def governance_report(format: str = "json") -> dict | PlainTextResponse:
        if format == "markdown":
            return PlainTextResponse(
                service.governance.report_markdown(), media_type="text/markdown"
            )
        return service.governance.report()

    return app


def build_service(settings: Settings) -> RunService:
    """Wire the agent from settings: in-process fakes, or HTTP clients (mock or real)."""
    from .data.oracle import RECORDINGS_FILE
    from .disseqt_http import HttpPolicyGate, HttpTraceExporter, register_application
    from .governance import AISystem, GovernanceLayer
    from .graph import AgentDeps
    from .llm import RecordedFakeProvider
    from .llm.http import OpenAICompatibleProvider
    from .observability import NullTracer
    from .policy import FakePolicyGate
    from .service import sqlite_checkpointer

    if settings.llm_provider == "http":
        llm = OpenAICompatibleProvider(
            httpx.Client(base_url=settings.llm_base_url, timeout=settings.http_timeout),
            settings.llm_model,
            settings.llm_api_key,
        )
    else:
        recordings = sorted(settings.data_root.glob(f"*/{RECORDINGS_FILE}"))
        if not recordings:
            raise ConfigError(
                f"MC_LLM_PROVIDER=fake needs */{RECORDINGS_FILE} under {settings.data_root}"
            )
        llm = RecordedFakeProvider.from_files(*recordings)

    tracer = NullTracer()
    application_id = settings.disseqt_application_id or "local-mc-recon"
    gate: Any = FakePolicyGate()
    if settings.disseqt_url:
        client = httpx.Client(base_url=settings.disseqt_url, timeout=settings.http_timeout)
        if not settings.disseqt_application_id:
            application_id = register_application(
                client, settings.disseqt_api_key, "mc-reconciliation-agent"
            )
        tracer.add_exporter(HttpTraceExporter(client, settings.disseqt_api_key, application_id))
        if settings.policy_gate == "http":
            gate = HttpPolicyGate(
                client, settings.disseqt_api_key, application_id, settings.disseqt_policy_id
            )

    store = RunStore(settings.db_path)
    deps = AgentDeps(
        llm=llm,
        gate=gate,
        store=store,
        tracer=tracer,
        limits=settings.limits,
        mapping_memory_days=settings.mapping_memory_days,
        governance=(
            GovernanceLayer(store, AISystem(application_id=application_id))
            if settings.governance
            else None
        ),
    )
    return RunService(deps, settings.data_root, checkpointer=sqlite_checkpointer(settings.db_path))


def __getattr__(name: str) -> Any:  # lazy module-level `app` for `uvicorn mc_recon.api:app`
    if name == "app":
        global app
        settings = Settings.from_env()
        app = create_app(build_service(settings), settings.analyst_tokens)
        return app
    raise AttributeError(name)
