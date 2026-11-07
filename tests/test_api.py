from __future__ import annotations

import pytest
from agent_helpers import make_service, poisoned_rationale_provider
from fastapi.testclient import TestClient

from mc_recon.api import create_app

ALICE = {"X-Analyst-Id": "alice"}


@pytest.fixture
def client():
    svc, _ = make_service()
    return TestClient(create_app(svc))


def test_full_run_over_http(client):
    r = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE)
    assert r.status_code == 201, r.text
    rid = r.json()["run_id"]
    assert r.json()["status"] == "awaiting_mapping_confirmation"

    m = client.get(f"/runs/{rid}/mapping").json()
    assert m["problems"] == [] and m["confirmed_mapping"] is None

    r = client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE)
    assert r.status_code == 200 and r.json()["status"] == "completed"

    assert len(client.get(f"/runs/{rid}/exceptions").json()) == 3
    log = client.get(f"/runs/{rid}/remediation-log").json()
    assert {e["category"] for e in log} == {
        "MISSING_PAYMENT",
        "DUPLICATE_DIRECT_DEBIT",
        "RATE_MARGIN_BREACH",
    }
    events = client.get(f"/runs/{rid}/events").json()
    assert events[0]["event_type"] == "mapping_confirmed" and events[0]["actor"] == "alice"


def test_analyst_header_required(client):
    r = client.post("/runs", json={"dataset": "pair_01"})
    assert r.status_code == 401


def test_idempotency_key_replays_and_rejects_mismatch(client):
    h = {**ALICE, "Idempotency-Key": "k-1"}
    a = client.post("/runs", json={"dataset": "pair_01"}, headers=h)
    b = client.post("/runs", json={"dataset": "pair_01"}, headers=h)
    assert a.status_code == b.status_code == 201
    assert a.json()["run_id"] == b.json()["run_id"]
    c = client.post("/runs", json={"dataset": "pair_02"}, headers=h)
    assert c.status_code == 409


def test_errors_map_to_status_codes(client):
    assert client.get("/runs/run-missing").status_code == 404
    assert client.post("/runs", json={"dataset": "../x"}, headers=ALICE).status_code == 422
    rid = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE).json()["run_id"]
    bad = client.post(
        f"/runs/{rid}/mapping/confirm", json={"mapping": {"servicing": {}}}, headers=ALICE
    )
    assert bad.status_code == 422


def test_review_over_http():
    svc, _ = make_service(llm=poisoned_rationale_provider("pair_01"))
    client = TestClient(create_app(svc))
    rid = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE).json()["run_id"]
    run = client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE).json()
    assert run["status"] == "awaiting_review"

    bob = {"X-Analyst-Id": "bob"}
    pending = run["pending_action"]["pending"]
    bad = client.post(
        f"/runs/{rid}/exceptions/{pending[0]}/review", json={"action": "edit"}, headers=bob
    )
    assert bad.status_code == 422
    for exc_id in pending:
        r = client.post(
            f"/runs/{rid}/exceptions/{exc_id}/review",
            json={"action": "reject", "note": "false positive"},
            headers={**bob, "Idempotency-Key": f"rev-{exc_id}"},
        )
        assert r.status_code == 200, r.text
    assert client.get(f"/runs/{rid}").json()["status"] == "completed"
    assert client.get(f"/runs/{rid}/remediation-log").json() == []
