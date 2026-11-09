"""FastAPI surface. Thin: auth header, idempotency, error mapping, then RunService.

With ``MC_ANALYST_TOKENS`` set, every route except the probes needs a token, the API docs
included, and the analyst identity comes from it. Without it (local PoC mode) identity is the
unverified ``X-Analyst-Id`` header and reads are open. Either way the identity is recorded on
every oversight event. Mutating endpoints honour ``Idempotency-Key``: a retry with the same key
and body replays the original response; the same key with a different body is ``409``.

Run with the fakes::

    mc-recon serve                    # reads PORT / MC_* env vars, see mc_recon.config
    uvicorn mc_recon.api:app --reload # dev
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

from . import api_models as m
from .config import ConfigError, Settings
from .governance import AISystem
from .llm.provider import canonical_json
from .service import Conflict, ReviewDecision, RunService, ServiceError, Unauthorized
from .store import IN_PROGRESS, RunStore

API_DESCRIPTION = """
AI-assisted reconciliation of mortgage servicing, payment and investor data, with EU AI Act
controls.

**Authentication.** When the server has `MC_ANALYST_TOKENS` set, every operation except the
health probes needs a bearer token, and the analyst identity comes from it. In local mode the
server instead trusts the `X-Analyst-Id` header on write operations and leaves reads open.

**Retries.** Write operations accept `Idempotency-Key`. A repeat with the same key and body replays
the first response. The same key with a different body, or while the first is running, is `409`.
"""

TAGS = [
    {"name": "Runs", "description": "Start and inspect reconciliation runs"},
    {"name": "Human review", "description": "The two checkpoints that need an analyst"},
    {"name": "Results", "description": "Findings, the remediation log and the audit trail"},
    {"name": "Analytics", "description": "Latency, tokens, cost, scores and baseline drift"},
    {"name": "Harness", "description": "Run limits and long-term memory"},
    {"name": "Governance", "description": "EU AI Act controls, evidence and coverage"},
    {"name": "Health", "description": "Liveness and readiness probes. No authentication"},
]


def _error(description: str) -> dict[str, Any]:
    return {"model": m.ErrorBody, "description": description}


AUTH_ERROR = {401: _error("Missing or invalid credentials")}
NOT_FOUND = {404: _error("No such resource")}
CONFLICT = {409: _error("Not allowed in the current state, or an Idempotency-Key conflict")}
INVALID = {422: _error("The request failed validation")}
RUN_FAILED = {500: _error("The agent failed; the run is marked failed")}
NO_GOVERNANCE = {404: _error("The governance layer is not enabled")}

bearer_scheme = HTTPBearer(
    auto_error=False, description="Analyst token, when the server has MC_ANALYST_TOKENS set"
)
analyst_header_scheme = APIKeyHeader(
    name="X-Analyst-Id",
    auto_error=False,
    description="Analyst identity in local mode. Unverified, and ignored when tokens are on",
)
BearerCredentials = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]
AnalystHeader = Annotated[str | None, Depends(analyst_header_scheme)]


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


def _lookup_token(request: Request, presented: str) -> str | None:
    """Find the analyst for a token. Compares against every token so timing reveals nothing."""
    found = None
    for token, analyst in request.app.state.analyst_tokens.items():
        if secrets.compare_digest(presented.encode(), token.encode()):
            found = analyst
    return found


def _token_analyst(request: Request, credentials: HTTPAuthorizationCredentials | None) -> str:
    found = _lookup_token(request, credentials.credentials) if credentials else None
    if found is None:
        raise Unauthorized("a valid bearer token is required")
    return found


def analyst_id(
    request: Request, credentials: BearerCredentials, x_analyst_id: AnalystHeader
) -> str:
    if request.app.state.analyst_tokens:
        return _token_analyst(request, credentials)
    if not x_analyst_id or not x_analyst_id.strip():
        raise Unauthorized("X-Analyst-Id header is required")
    return x_analyst_id.strip()


def require_reader(request: Request, credentials: BearerCredentials) -> None:
    """With tokens configured, reads need one too."""
    if request.app.state.analyst_tokens:
        _token_analyst(request, credentials)


def require_docs_access(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> None:
    """Guard the docs when tokens are on. Accepts a bearer token or HTTP Basic.

    Basic exists because a browser can't attach a bearer header when opening a page, but it does
    prompt for Basic credentials. Any username works; the password is the token.
    """
    if not request.app.state.analyst_tokens:
        return
    scheme, _, value = (authorization or "").partition(" ")
    presented = value if scheme.lower() == "bearer" else ""
    if scheme.lower() == "basic":
        try:
            presented = base64.b64decode(value, validate=True).decode().partition(":")[2]
        except (binascii.Error, UnicodeDecodeError):
            presented = ""
    if not presented or _lookup_token(request, presented) is None:
        raise HTTPException(
            status_code=401,
            detail="a valid token is required to view the API docs",
            headers={"WWW-Authenticate": 'Basic realm="mc-recon docs"'},
        )


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
        description=API_DESCRIPTION,
        openapi_tags=TAGS,
        docs_url=None,  # served below, so they can sit behind the token
        redoc_url=None,
        openapi_url=None,
    )
    app.state.analyst_tokens = dict(analyst_tokens or {})
    store: RunStore = service.store
    api = APIRouter(dependencies=[Depends(require_reader)], responses={**AUTH_ERROR, **INVALID})
    probes = APIRouter(tags=["Health"])
    docs = APIRouter(include_in_schema=False, dependencies=[Depends(require_docs_access)])

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.message, "detail": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid request", "detail": jsonable_encoder(exc.errors())},
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

    @probes.get("/healthz", response_model=m.Health, summary="Liveness")
    def healthz() -> dict:
        return {
            "status": "ok",
            "model_id": service.deps.llm.model_id,
            "policy_id": service.deps.gate.policy_id,
        }

    @probes.get(
        "/readyz",
        response_model=m.Readiness,
        responses={503: {"model": m.Readiness, "description": "A dependency is unavailable"}},
        summary="Readiness",
    )
    def readyz() -> JSONResponse:
        """The run store answers, and the LLM and policy gate are reachable."""
        checks = {
            "store": store.ping(),
            "llm": _is_ready(service.deps.llm),
            "gate": _is_ready(service.deps.gate),
        }
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **checks})

    @api.post(
        "/runs",
        status_code=201,
        response_model=m.RunView,
        responses={
            **CONFLICT,
            413: _error("An input file is over the size limit"),
            **RUN_FAILED,
        },
        tags=["Runs"],
        summary="Start a run",
    )
    def start_run(body: StartRun, analyst: Analyst, idem: IdemKey = None) -> JSONResponse:
        """Ingests the dataset and proposes a mapping, then pauses for an analyst."""
        return idempotent(
            idem,
            "start_run",
            analyst,
            body.model_dump(),
            201,
            lambda: service.start_run(body.dataset, analyst),
        )

    @api.get(
        "/runs/{run_id}",
        response_model=m.RunView,
        responses=NOT_FOUND,
        tags=["Runs"],
        summary="Get a run",
    )
    def get_run(run_id: str) -> dict:
        return service.get_run(run_id)

    @api.get(
        "/runs/{run_id}/mapping",
        response_model=m.MappingView,
        responses={**NOT_FOUND, **CONFLICT},
        tags=["Human review"],
        summary="Get the proposed mapping",
    )
    def get_mapping(run_id: str) -> dict:
        """The proposal, any problems with it, column profiles and screening results."""
        return service.get_mapping(run_id)

    @api.post(
        "/runs/{run_id}/mapping/confirm",
        response_model=m.RunView,
        responses={**NOT_FOUND, **CONFLICT, **RUN_FAILED},
        tags=["Human review"],
        summary="Confirm or edit the mapping",
    )
    def confirm_mapping(
        run_id: str, body: ConfirmMapping, analyst: Analyst, idem: IdemKey = None
    ) -> JSONResponse:
        """Send `{}` to accept the proposal, or a full `mapping` to replace it. Runs the rest
        of the agent, so the response reflects where it stopped."""
        return idempotent(
            idem,
            f"confirm:{run_id}",
            analyst,
            body.model_dump(),
            200,
            lambda: service.confirm_mapping(run_id, analyst, body.mapping),
        )

    @api.get(
        "/runs/{run_id}/exceptions",
        response_model=list[m.ExceptionView],
        responses=NOT_FOUND,
        tags=["Results"],
        summary="List exceptions",
    )
    def list_exceptions(run_id: str) -> list[dict]:
        return service.list_exceptions(run_id)

    @api.post(
        "/runs/{run_id}/exceptions/{exception_id}/review",
        response_model=m.ReviewResult,
        responses={**NOT_FOUND, **CONFLICT, **RUN_FAILED},
        tags=["Human review"],
        summary="Review a held exception",
    )
    def review(
        run_id: str,
        exception_id: str,
        body: ReviewDecision,
        analyst: Analyst,
        idem: IdemKey = None,
    ) -> JSONResponse:
        """`approve` is an override and needs a `note`. `edit` sends new text back through the
        policy, and is refused in the last round. `reject` keeps the finding out of the log."""
        return idempotent(
            idem,
            f"review:{run_id}:{exception_id}",
            analyst,
            body.model_dump(),
            200,
            lambda: service.review_exception(run_id, exception_id, analyst, body),
        )

    @api.get(
        "/runs/{run_id}/remediation-log",
        response_model=list[m.RemediationEntry],
        responses=NOT_FOUND,
        tags=["Results"],
        summary="Get the remediation log",
    )
    def remediation_log(run_id: str) -> list[dict]:
        return service.remediation_log(run_id)

    @api.get(
        "/runs/{run_id}/events",
        response_model=list[m.EventView],
        responses=NOT_FOUND,
        tags=["Results"],
        summary="Get the audit trail",
    )
    def events(run_id: str) -> list[dict]:
        return service.events(run_id)

    @api.get(
        "/runs/{run_id}/trajectory",
        response_model=list[m.TrajectoryStep],
        responses=NOT_FOUND,
        tags=["Results"],
        summary="Get the agent's ordered steps",
    )
    def trajectory(run_id: str) -> list[dict]:
        return service.trajectory(run_id)

    @api.get(
        "/runs/{run_id}/metrics",
        response_model=m.RunMetrics,
        responses=NOT_FOUND,
        tags=["Analytics"],
        summary="Get run metrics",
    )
    def run_metrics(run_id: str) -> dict:
        return service.metrics(run_id)

    @api.get(
        "/metrics",
        response_model=m.MetricsSummary,
        tags=["Analytics"],
        summary="Get metrics across runs",
    )
    def metrics_summary() -> dict:
        return service.metrics_summary()

    @api.post(
        "/baselines",
        status_code=201,
        response_model=m.Baseline,
        responses=CONFLICT,
        tags=["Analytics"],
        summary="Run the manual-process baseline",
    )
    def start_baseline(body: StartBaseline, analyst: Analyst, idem: IdemKey = None) -> JSONResponse:
        return idempotent(
            idem,
            "baseline",
            analyst,
            body.model_dump(),
            201,
            lambda: service.run_baseline(analyst, body.label, body.datasets),
        )

    @api.get(
        "/baselines/latest",
        response_model=m.Baseline,
        responses=NOT_FOUND,
        tags=["Analytics"],
        summary="Get the latest baseline",
    )
    def latest_baseline() -> dict:
        return service.latest_baseline()

    @api.get(
        "/runs/{run_id}/harness",
        response_model=m.HarnessView,
        responses=NOT_FOUND,
        tags=["Harness"],
        summary="Get usage against limits",
    )
    def harness(run_id: str) -> dict:
        return service.harness(run_id)

    @api.get(
        "/memory/mappings",
        response_model=list[m.MappingMemoryEntry],
        tags=["Harness"],
        summary="List remembered mappings",
    )
    def list_memory() -> list[dict]:
        return service.list_memory()

    @api.delete(
        "/memory/mappings/{fingerprint}",
        response_model=m.ForgetResult,
        responses=NOT_FOUND,
        tags=["Harness"],
        summary="Forget a remembered mapping",
    )
    def forget_memory(fingerprint: str, analyst: Analyst) -> dict:
        return service.forget_memory(fingerprint, analyst)

    @api.get(
        "/governance/system",
        response_model=AISystem,
        responses=NO_GOVERNANCE,
        tags=["Governance"],
        summary="Get the AI system record",
    )
    def governance_system() -> dict:
        return service.governance.system.model_dump()

    @api.get(
        "/governance/controls",
        response_model=list[m.ControlStatus],
        responses=NO_GOVERNANCE,
        tags=["Governance"],
        summary="Get control status",
    )
    def governance_controls() -> list[dict]:
        return service.governance.control_statuses()

    @api.get(
        "/governance/coverage",
        response_model=m.Coverage,
        responses=NO_GOVERNANCE,
        tags=["Governance"],
        summary="Get requirement coverage and gaps",
    )
    def governance_coverage() -> dict:
        return service.governance.coverage()

    @api.get(
        "/governance/evidence",
        response_model=list[m.EvidenceRecord],
        responses=NO_GOVERNANCE,
        tags=["Governance"],
        summary="List evidence records",
    )
    def governance_evidence(control_id: str | None = None, run_id: str | None = None) -> list[dict]:
        service.governance  # noqa: B018 — 404 when the layer is off
        return store.evidence(control_id, run_id)

    @api.get(
        "/governance/report",
        response_model=m.GovernanceReport,
        responses={
            **NO_GOVERNANCE,
            200: {
                "description": "The report as JSON, or as Markdown with `format=markdown`",
                "content": {"text/markdown": {"schema": {"type": "string"}}},
            },
        },
        tags=["Governance"],
        summary="Get the compliance report",
    )
    def governance_report(format: Literal["json", "markdown"] = "json") -> Any:
        if format == "markdown":
            return PlainTextResponse(
                service.governance.report_markdown(), media_type="text/markdown"
            )
        return service.governance.report()

    @docs.get("/openapi.json")
    def openapi_json() -> JSONResponse:
        return JSONResponse(app.openapi())

    @docs.get("/docs")
    def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger")

    @docs.get("/redoc")
    def redoc() -> HTMLResponse:
        return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

    app.include_router(api)
    app.include_router(probes)
    app.include_router(docs)
    return app


def openapi_spec() -> dict[str, Any]:
    """The spec, without configuration, data or a running server (for ``mc-recon openapi``)."""
    from .graph import AgentDeps
    from .llm import RecordedFakeProvider
    from .policy import FakePolicyGate

    deps = AgentDeps(llm=RecordedFakeProvider([]), gate=FakePolicyGate(), store=RunStore())
    return create_app(RunService(deps, Path("."))).openapi()


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
