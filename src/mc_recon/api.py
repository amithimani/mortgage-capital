"""FastAPI surface (plan §4). Thin: auth header, idempotency, error mapping, then RunService.

Analyst identity comes from the ``X-Analyst-Id`` header — a PoC stand-in for real auth
(SSO/JWT), recorded on every oversight event. Mutating endpoints honour ``Idempotency-Key``:
a retry with the same key and body replays the original response; the same key with a
different body is ``409``.

Run with the fakes::

    uvicorn mc_recon.api:app --reload
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .llm.provider import canonical_json
from .service import Conflict, ReviewDecision, RunService, ServiceError, Unauthorized
from .store import RunStore


class StartRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str


class ConfirmMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mapping: dict | None = None
    """Omit to accept the proposal as-is; supply a full ColumnMapping to edit it."""


def analyst_id(x_analyst_id: Annotated[str | None, Header()] = None) -> str:
    if not x_analyst_id or not x_analyst_id.strip():
        raise Unauthorized("X-Analyst-Id header is required")
    return x_analyst_id.strip()


Analyst = Annotated[str, Depends(analyst_id)]
IdemKey = Annotated[str | None, Header(alias="Idempotency-Key")]


def create_app(service: RunService) -> FastAPI:
    app = FastAPI(title="Mortgage Capital Reconciliation Agent", version="0.1.0")
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
        prior = store.get_idempotent(key)
        if prior is not None:
            if prior["request_hash"] != digest:
                raise Conflict("Idempotency-Key was already used with a different request")
            return JSONResponse(status_code=prior["status_code"], content=prior["response"])
        content = fn()
        store.put_idempotent(key, digest, status, content)
        return JSONResponse(status_code=status, content=content)

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "model_id": service.deps.llm.model_id,
            "policy_id": service.deps.gate.policy_id,
        }

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

    @app.get("/runs/{run_id}/events")
    def events(run_id: str) -> list[dict]:
        return service.events(run_id)

    return app


def build_default_service() -> RunService:
    """Fakes end to end: recorded LLM over every pair's oracle recordings + FakePolicyGate."""
    from .data.oracle import RECORDINGS_FILE
    from .graph import AgentDeps
    from .llm import RecordedFakeProvider
    from .policy import FakePolicyGate
    from .service import sqlite_checkpointer

    data_root = Path(os.environ.get("MC_DATA_ROOT", "data/synthetic"))
    db_path = os.environ.get("MC_DB_PATH", "var/mc_recon.sqlite")
    llm = RecordedFakeProvider.from_files(*sorted(data_root.glob(f"*/{RECORDINGS_FILE}")))
    deps = AgentDeps(llm=llm, gate=FakePolicyGate(), store=RunStore(db_path))
    return RunService(deps, data_root, checkpointer=sqlite_checkpointer(db_path))


def __getattr__(name: str) -> Any:  # lazy module-level `app` for uvicorn
    if name == "app":
        global app
        app = create_app(build_default_service())
        return app
    raise AttributeError(name)
