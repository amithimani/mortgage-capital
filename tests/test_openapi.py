"""The API contract: the spec is complete, committed, protected, and true in every run state."""

from __future__ import annotations

import base64
import json

import pytest
from agent_helpers import make_service, poisoned_rationale_provider
from conftest import REPO_ROOT, SYNTHETIC_DIR
from fastapi.testclient import TestClient
from pydantic import BaseModel

from mc_recon import api_models as m
from mc_recon.api import create_app, openapi_spec
from mc_recon.evaluation import Env
from mc_recon.governance import AISystem, GovernanceLayer
from mc_recon.limits import AgentLimits
from mc_recon.llm import LLMTask, LLMUnavailable
from mc_recon.store import RunStore

ALICE = {"X-Analyst-Id": "alice"}
METHODS = {"get", "post", "delete", "put", "patch"}
PROBES = {"/healthz", "/readyz"}


@pytest.fixture(scope="module")
def spec() -> dict:
    return openapi_spec()


def operations(spec: dict):
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if method in METHODS:
                yield path, method, op


# --- the spec is complete -----------------------------------------------------------------


def test_committed_spec_is_up_to_date(spec):
    committed = (REPO_ROOT / "docs" / "openapi.json").read_text()
    expected = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    assert committed == expected, "docs/openapi.json is out of date: run `mc-recon openapi`"


def test_every_operation_has_a_typed_success_response(spec):
    for path, method, op in operations(spec):
        code = next(c for c in op["responses"] if c.startswith("2"))
        schema = op["responses"][code]["content"]["application/json"]["schema"]
        assert schema.get("$ref") or schema.get("items", {}).get("$ref"), (method, path)


def test_every_operation_is_tagged_and_summarised(spec):
    known = {t["name"] for t in spec["tags"]}
    for path, method, op in operations(spec):
        assert set(op["tags"]) <= known and op["tags"], (method, path)
        assert op["summary"], (method, path)


def test_errors_use_one_schema(spec):
    for path, method, op in operations(spec):
        for code, response in op["responses"].items():
            if code.startswith("4") or code == "500":
                schema = response["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/ErrorBody"}, (method, path, code)
    assert "HTTPValidationError" not in spec["components"]["schemas"]


def test_auth_is_declared_on_everything_but_the_probes(spec):
    schemes = spec["components"]["securitySchemes"]
    assert schemes["HTTPBearer"]["scheme"] == "bearer"
    assert schemes["APIKeyHeader"]["name"] == "X-Analyst-Id"
    for path, method, op in operations(spec):
        if path in PROBES:
            assert "security" not in op and "401" not in op["responses"], path
            continue
        assert {"HTTPBearer": []} in op["security"], (method, path)
        assert "401" in op["responses"], (method, path)
        if method != "get":
            assert {"APIKeyHeader": []} in op["security"], (method, path)


def test_auth_is_not_a_plain_parameter(spec):
    for path, method, op in operations(spec):
        names = {p["name"].lower() for p in op.get("parameters", [])}
        assert not names & {"authorization", "x-analyst-id"}, (method, path)


def test_docs_routes_are_not_part_of_the_api(spec):
    assert not {"/docs", "/redoc", "/openapi.json"} & set(spec["paths"])


# --- the docs are guarded -----------------------------------------------------------------


DOCS = ("/docs", "/redoc", "/openapi.json")


@pytest.mark.parametrize("path", DOCS)
def test_docs_are_open_in_local_mode(path):
    svc, _ = make_service()
    assert TestClient(create_app(svc)).get(path).status_code == 200


@pytest.mark.parametrize("path", DOCS)
def test_docs_need_a_token_when_auth_is_on(path):
    svc, _ = make_service()
    client = TestClient(create_app(svc, {"secret": "alice"}))

    denied = client.get(path)
    assert denied.status_code == 401 and denied.headers["www-authenticate"].startswith("Basic")
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer secret"}).status_code == 200

    basic = base64.b64encode(b"anyone:secret").decode()
    assert client.get(path, headers={"Authorization": f"Basic {basic}"}).status_code == 200


@pytest.mark.parametrize("value", ["Basic !!!notbase64", "Basic ", "Digest secret", "secret"])
def test_malformed_docs_credentials_are_refused(value):
    svc, _ = make_service()
    client = TestClient(create_app(svc, {"secret": "alice"}))
    assert client.get("/docs", headers={"Authorization": value}).status_code == 401


def test_probes_stay_open_when_auth_is_on():
    svc, _ = make_service()
    client = TestClient(create_app(svc, {"secret": "alice"}))
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200


# --- the spec is true: real responses, every state, nothing undeclared --------------------


def undeclared(value, path="$"):
    """Fields the response carried that its ``ApiModel`` doesn't declare."""
    if isinstance(value, m.ApiModel):
        yield from (f"{path}.{k}" for k in value.model_extra or {})
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from undeclared(getattr(value, name), f"{path}.{name}")
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            yield from undeclared(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for k, item in value.items():
            yield from undeclared(item, f"{path}.{k}")


def assert_contract(response, model, status=200):
    assert response.status_code == status, response.text
    body = response.json()
    parsed = [model.model_validate(b) for b in body] if isinstance(body, list) else None
    parsed = parsed if parsed is not None else model.model_validate(body)
    assert list(undeclared(parsed)) == []
    return body


def assert_run_views(client, rid, run_model=m.RunView):
    assert_contract(client.get(f"/runs/{rid}"), run_model)
    assert_contract(client.get(f"/runs/{rid}/exceptions"), m.ExceptionView)
    assert_contract(client.get(f"/runs/{rid}/remediation-log"), m.RemediationEntry)
    assert_contract(client.get(f"/runs/{rid}/events"), m.EventView)
    assert_contract(client.get(f"/runs/{rid}/trajectory"), m.TrajectoryStep)
    assert_contract(client.get(f"/runs/{rid}/metrics"), m.RunMetrics)
    assert_contract(client.get(f"/runs/{rid}/harness"), m.HarnessView)


def _governed(llm=None, **kwargs):
    store = RunStore()
    governance = GovernanceLayer(store, AISystem(application_id="app"))
    svc, _ = make_service(llm=llm, store=store, governance=governance, **kwargs)
    return svc, TestClient(create_app(svc), raise_server_exceptions=False)


def _start(client):
    body = assert_contract(
        client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE), m.RunView, 201
    )
    return body["run_id"]


def test_contract_on_a_completed_run():
    _, client = _governed()
    rid = _start(client)
    assert_run_views(client, rid)
    assert_contract(client.get(f"/runs/{rid}/mapping"), m.MappingView)
    body = assert_contract(
        client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE), m.RunView
    )
    assert body["status"] == "completed"
    assert_run_views(client, rid)
    assert_contract(client.get(f"/runs/{rid}/mapping"), m.MappingView)
    assert_contract(client.get("/metrics"), m.MetricsSummary)


def test_contract_through_the_review_loop():
    svc, client = _governed(llm=poisoned_rationale_provider("pair_01"))
    rid = _start(client)
    body = assert_contract(
        client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE), m.RunView
    )
    assert body["pending_action"]["type"] == "exception_review"
    assert_run_views(client, rid)

    first, second, third = body["pending_action"]["pending"]
    url = f"/runs/{rid}/exceptions/{{}}/review"
    edit = {"action": "edit", "rationale": "Still 7.77."}
    assert_contract(client.post(url.format(first), json=edit, headers=ALICE), m.ReviewResult)
    reject = {"action": "reject", "note": "noise"}
    assert_contract(client.post(url.format(second), json=reject, headers=ALICE), m.ReviewResult)
    approve = {"action": "approve", "note": "verified"}
    assert_contract(client.post(url.format(third), json=approve, headers=ALICE), m.ReviewResult)
    assert_run_views(client, rid)

    final = assert_contract(
        client.post(url.format(first), json=approve, headers=ALICE), m.ReviewResult
    )  # noqa: E501
    assert final["status"] == "completed"
    assert_run_views(client, rid)


def test_contract_for_findings_held_by_the_harness():
    class Down:
        model_id = "down"

        def __init__(self, inner):
            self.inner = inner

        def generate(self, request, schema):
            if request.task is LLMTask.PROPOSE_MAPPING:
                return self.inner.generate(request, schema)
            raise LLMUnavailable("503")

    svc, client = _governed(
        llm=Down(Env(SYNTHETIC_DIR).oracle()), limits=AgentLimits(max_consecutive_llm_failures=1)
    )
    rid = _start(client)
    client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE)
    excs = assert_contract(client.get(f"/runs/{rid}/exceptions"), m.ExceptionView)
    assert all(e["degraded"] for e in excs)
    harness = assert_contract(client.get(f"/runs/{rid}/harness"), m.HarnessView)
    assert harness["limit_events"]
    assert_run_views(client, rid)


def test_contract_on_a_failed_run():
    class Broken:
        model_id = "broken"

        def generate(self, request, schema):
            raise RuntimeError("boom")

    svc, client = _governed(llm=Broken())
    r = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE)
    assert_contract(r, m.ErrorBody, 500)
    rid = svc.store.list_runs()[0]["run_id"]
    body = assert_contract(client.get(f"/runs/{rid}"), m.RunView)
    assert body["status"] == "failed"
    assert_run_views(client, rid)


def test_contract_for_other_resources():
    svc, client = _governed()
    assert_contract(client.get("/healthz"), m.Health)
    assert_contract(client.get("/readyz"), m.Readiness)
    assert_contract(client.get("/governance/controls"), m.ControlStatus)  # before any evidence
    assert_contract(client.get("/governance/coverage"), m.Coverage)

    rid = _start(client)
    client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE)
    assert_contract(client.get("/governance/system"), AISystem)
    assert_contract(client.get("/governance/controls"), m.ControlStatus)
    assert_contract(client.get("/governance/coverage"), m.Coverage)
    assert_contract(client.get("/governance/evidence"), m.EvidenceRecord)
    assert_contract(client.get("/governance/report"), m.GovernanceReport)
    assert_contract(client.post("/baselines", json={}, headers=ALICE), m.Baseline, 201)
    assert_contract(client.get("/baselines/latest"), m.Baseline)


def test_contract_for_mapping_memory():
    svc, _ = make_service()
    svc.store.remember_mapping(
        "fp",
        json.loads((REPO_ROOT / "data/synthetic/pair_01/expected_mapping.json").read_text()),
        "alice",
        "run-1",
    )
    client = TestClient(create_app(svc))
    assert_contract(client.get("/memory/mappings"), m.MappingMemoryEntry)
    assert_contract(client.delete("/memory/mappings/fp", headers=ALICE), m.ForgetResult)


@pytest.mark.parametrize(
    ("method", "path", "kwargs", "status"),
    [
        ("get", "/runs/nope", {}, 404),
        ("post", "/runs", {"json": {}, "headers": ALICE}, 422),
        ("post", "/runs", {"json": {"dataset": "../x"}, "headers": ALICE}, 422),
        ("post", "/runs", {"json": {"dataset": "pair_01"}}, 401),
        ("get", "/governance/report?format=xml", {}, 422),
    ],
)
def test_every_error_has_the_documented_body(method, path, kwargs, status):
    svc, _ = make_service()
    client = TestClient(create_app(svc))
    r = getattr(client, method)(path, **kwargs)
    assert_contract(r, m.ErrorBody, status)
    assert r.json()["error"]


def test_invalid_review_returns_the_documented_body():
    svc, client = _governed(llm=poisoned_rationale_provider("pair_01"))
    rid = _start(client)
    exc = client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE).json()[
        "pending_action"
    ]["pending"][0]
    r = client.post(
        f"/runs/{rid}/exceptions/{exc}/review", json={"action": "approve"}, headers=ALICE
    )
    assert_contract(r, m.ErrorBody, 422)


def test_markdown_report_is_documented_and_served():
    _, client = _governed()
    r = client.get("/governance/report", params={"format": "markdown"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    op = openapi_spec()["paths"]["/governance/report"]["get"]
    assert "text/markdown" in op["responses"]["200"]["content"]
