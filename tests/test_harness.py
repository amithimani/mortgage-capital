"""Agent harness: every evaluation scenario, plus the units and API contracts behind them."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from agent_helpers import make_service
from conftest import SYNTHETIC_DIR
from fastapi.testclient import TestClient

from mc_recon.api import create_app
from mc_recon.config import ConfigError, Settings
from mc_recon.evaluation import SCENARIOS, Env, run_scenario
from mc_recon.limits import AgentLimits, LimitExceeded, RunBudget
from mc_recon.llm import LLMRequest, LLMTask, TokenUsage
from mc_recon.llm.prompts import render_messages
from mc_recon.mocks import create_mock_app
from mc_recon.policy import FakePolicyGate
from mc_recon.service import ReviewDecision
from mc_recon.store import RunStore
from mc_recon.transport import DependencyRejected, DependencyUnavailable, RetryPolicy, post_json

ALICE = {"X-Analyst-Id": "alice"}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name):
    result = run_scenario(name, SYNTHETIC_DIR)
    assert result.passed, [c for c in result.checks if not c["ok"]]


# --- budgets ------------------------------------------------------------------------------


def _spent(budget: RunBudget, n: int = 1) -> None:
    for _ in range(n):
        budget.admit(1)
        budget.charge(TokenUsage(input_tokens=10, output_tokens=10), "mock-llm")


@pytest.mark.parametrize(
    ("limits", "limit"),
    [
        (AgentLimits(max_llm_calls=1), "max_llm_calls"),
        (AgentLimits(max_total_tokens=20), "max_total_tokens"),
        (AgentLimits(max_cost_usd=0.000001), "max_cost_usd"),
    ],
)
def test_budget_refuses_the_call_past_each_limit(limits, limit):
    budget = RunBudget(limits)
    _spent(budget)
    with pytest.raises(LimitExceeded) as exc:
        budget.admit(1)
    assert exc.value.limit == limit


def test_oversized_prompt_is_refused_not_truncated():
    budget = RunBudget(AgentLimits(max_input_tokens_per_call=10))
    with pytest.raises(LimitExceeded, match="max_input_tokens_per_call"):
        budget.admit(11)
    assert budget.llm_calls == 0


def test_llm_time_limit_uses_only_time_in_nodes():
    now = [100.0]
    budget = RunBudget(AgentLimits(max_llm_seconds=5), clock=lambda: now[0])
    now[0] += 6
    with pytest.raises(LimitExceeded, match="max_llm_seconds"):
        budget.admit(1)


def test_budget_survives_pause_and_restart():
    first = RunBudget(AgentLimits(max_llm_calls=2))
    _spent(first)
    second = RunBudget(AgentLimits(max_llm_calls=2), first.snapshot())
    _spent(second)
    with pytest.raises(LimitExceeded):
        second.admit(1)


def test_success_resets_the_circuit_breaker():
    budget = RunBudget(AgentLimits(max_consecutive_llm_failures=2))
    budget.charge_failure()
    _spent(budget)
    budget.charge_failure()
    budget.admit(1)  # still closed: the failures weren't consecutive
    budget.charge_failure()
    with pytest.raises(LimitExceeded, match="circuit open"):
        budget.admit(1)


def test_recursion_limit_follows_review_rounds():
    assert AgentLimits(max_review_rounds=1).recursion_limit == 12
    assert AgentLimits(max_review_rounds=5).recursion_limit == 20
    with pytest.raises(ValueError):
        AgentLimits(max_review_rounds=0)


def test_feedback_reaches_the_prompt_but_not_the_cache_key():
    base = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="classify.v1", payload={})
    retry = base.model_copy(update={"feedback": "bad enum"})
    assert retry.cache_key == base.cache_key
    assert "bad enum" in render_messages(retry)[-1]["content"]


# --- transport ----------------------------------------------------------------------------


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://x")


def test_transient_failures_retry_then_succeed_and_honour_retry_after():
    responses = [
        httpx.Response(429, headers={"retry-after": "1.5"}),
        httpx.Response(200, json={"ok": True}),
    ]
    naps: list[float] = []
    out = post_json(_client(lambda _r: responses.pop(0)), "/p", {}, sleep=naps.append)
    assert out == {"ok": True} and naps == [1.5]


def test_retries_are_bounded():
    calls: list[int] = []

    def handler(_r):
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(DependencyUnavailable):
        post_json(_client(handler), "/p", {}, retry=RetryPolicy(attempts=3), sleep=lambda _s: None)
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    calls: list[int] = []

    def handler(_r):
        calls.append(1)
        return httpx.Response(400, text="bad")

    with pytest.raises(DependencyRejected):
        post_json(_client(handler), "/p", {}, sleep=lambda _s: None)
    assert len(calls) == 1


# --- review-finding fixes -----------------------------------------------------------------


def test_concurrent_requests_with_one_key_do_the_work_once():
    svc, _ = make_service()
    original = svc.start_run

    def slow(*args, **kwargs):
        time.sleep(0.2)
        return original(*args, **kwargs)

    svc.start_run = slow
    client = TestClient(create_app(svc), raise_server_exceptions=False)
    codes: list[int] = []

    def go():
        headers = {**ALICE, "Idempotency-Key": "race"}
        codes.append(client.post("/runs", json={"dataset": "pair_01"}, headers=headers).status_code)

    threads = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes) == [201, 409, 409] and len(svc.store.list_runs()) == 1


def test_failed_request_frees_its_idempotency_key():
    svc, _ = make_service()
    client = TestClient(create_app(svc))
    h = {**ALICE, "Idempotency-Key": "retry-me"}
    assert client.post("/runs", json={"dataset": "nope"}, headers=h).status_code == 422
    assert client.post("/runs", json={"dataset": "pair_01"}, headers=h).status_code == 201


def test_crashed_claim_can_be_reclaimed():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = RunStore(clock=lambda: now[0].isoformat(timespec="milliseconds"))
    assert store.reserve_idempotent("k", "h")
    assert not store.reserve_idempotent("k", "h")
    now[0] += timedelta(minutes=16)
    assert store.reserve_idempotent("k", "h")


def test_agent_failure_is_a_5xx():
    class Broken:
        model_id = "broken"

        def generate(self, request, schema):
            raise RuntimeError("boom")

    svc, _ = make_service(llm=Broken())
    client = TestClient(create_app(svc), raise_server_exceptions=False)
    r = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE)
    assert r.status_code == 500


def test_override_needs_a_reason():
    with pytest.raises(ValueError, match="override"):
        ReviewDecision(action="approve")
    with pytest.raises(ValueError, match="override"):
        ReviewDecision(action="approve", note="   ")
    assert ReviewDecision(action="approve", note="verified against bank statement")


def test_bearer_tokens_protect_reads_and_set_identity():
    svc, _ = make_service()
    client = TestClient(create_app(svc, {"secret": "alice"}))
    assert client.get("/healthz").status_code == 200  # probes stay open
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    # a forged identity header is ignored once tokens are on
    assert client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE).status_code == 401
    auth = {"Authorization": "Bearer secret"}
    r = client.post("/runs", json={"dataset": "pair_01"}, headers=auth)
    assert r.status_code == 201 and r.json()["created_by"] == "alice"


def test_fake_call_history_is_bounded():
    gate = FakePolicyGate()
    assert gate.calls.maxlen and gate.run_calls.maxlen


def test_run_locks_are_released_when_a_run_finishes():
    svc, _ = make_service()
    rid = svc.start_run("pair_01", "alice")["run_id"]
    assert rid in svc._locks
    svc.confirm_mapping(rid, "alice", None)
    assert rid not in svc._locks


# --- memory -------------------------------------------------------------------------------


def test_mapping_memory_expires():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    store = RunStore(clock=lambda: now[0].isoformat(timespec="milliseconds"))
    store.remember_mapping("fp", {"a": 1}, "alice", "run-1")
    assert store.recall_mapping("fp", 30)["confirmed_by"] == "alice"
    now[0] += timedelta(days=31)
    assert store.recall_mapping("fp", 30) is None


def test_governance_evidence_is_append_only():
    import sqlite3

    store = RunStore()
    record = {"evidence_id": "e1", "control_id": "c", "run_id": "r", "subject_id": "s"}
    assert store.append_evidence(record)
    assert not store.append_evidence(record)  # a replayed node writes nothing new
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._conn.execute("DELETE FROM governance_evidence")


# --- API + mock + config ------------------------------------------------------------------


def test_harness_and_governance_endpoints():
    from mc_recon.governance import AISystem, GovernanceLayer

    off, _ = make_service()
    assert TestClient(create_app(off)).get("/governance/coverage").status_code == 404

    store = RunStore()
    svc, _ = make_service(
        store=store, governance=GovernanceLayer(store, AISystem(application_id="a"))
    )
    client = TestClient(create_app(svc))
    rid = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE).json()["run_id"]
    client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE)

    h = client.get(f"/runs/{rid}/harness").json()
    assert h["usage"]["llm_calls"] == 7 and h["limits"]["max_review_rounds"] == 3
    assert h["state_bytes"] > 0
    cov = client.get("/governance/coverage").json()
    assert cov["covered"] == 3 and cov["gaps"] == []
    assert client.get("/governance/system").json()["provider_role"] == "deployer"
    assert client.get("/governance/evidence", params={"run_id": rid}).json()
    md = client.get("/governance/report", params={"format": "markdown"})
    assert md.headers["content-type"].startswith("text/markdown") and "No open gaps" in md.text


def test_mock_requires_credentials_and_never_improvises():
    mock = TestClient(create_mock_app(Env(SYNTHETIC_DIR).recordings))
    assert mock.post("/api/v1/applications", json={"name": "x"}).status_code == 401
    bad = mock.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    unknown = mock.post(
        "/v1/chat/completions",
        json={"metadata": {"cache_key": "nope"}},
        headers={"Authorization": "Bearer mock-key"},
    )
    assert unknown.status_code == 422


def test_mock_policy_threshold_changes_the_verdict_without_code():
    key = {"X-API-Key": "mock-key"}
    mock = TestClient(create_mock_app([]))
    app_id = mock.post("/api/v1/applications", json={"name": "x"}, headers=key).json()[
        "application_id"
    ]
    published = mock.post(
        "/api/v1/policies",
        json={"policy_id": "exception-policy", "thresholds": {"input_pii": 2.0}},
        headers=key,
    ).json()
    assert published["version"] == 2
    verdict = mock.post(
        "/api/v1/validate",
        json={"policy_id": "exception-policy", "scope": "input", "text": "clean"},
        headers={**key, "X-Application-Id": app_id},
    ).json()
    assert verdict["decision"] == "BLOCK" and verdict["policy_version"] == 2


def test_limits_come_from_env_and_are_validated():
    env = {"MC_DATA_ROOT": str(SYNTHETIC_DIR), "MC_LIMIT_MAX_REVIEW_ROUNDS": "2"}
    assert Settings.from_env(env).limits.max_review_rounds == 2
    with pytest.raises(ConfigError, match="MC_LIMIT_MAX_LLM_CALLS"):
        Settings.from_env({"MC_DATA_ROOT": str(SYNTHETIC_DIR), "MC_LIMIT_MAX_LLM_CALLS": "lots"})
    with pytest.raises(ConfigError, match="max_review_rounds"):
        Settings.from_env({"MC_DATA_ROOT": str(SYNTHETIC_DIR), "MC_LIMIT_MAX_REVIEW_ROUNDS": "0"})


def test_http_providers_need_urls():
    with pytest.raises(ConfigError) as exc:
        Settings.from_env(
            {
                "MC_DATA_ROOT": str(SYNTHETIC_DIR),
                "MC_LLM_PROVIDER": "http",
                "MC_POLICY_GATE": "http",
            }
        )
    assert "MC_LLM_BASE_URL" in str(exc.value) and "MC_DISSEQT_URL" in str(exc.value)
