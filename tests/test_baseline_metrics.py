"""G2/G3/G5/G6: trajectory validators in the run, baseline harness, metrics endpoints."""

from __future__ import annotations

from agent_helpers import make_service, poisoned_rationale_provider
from conftest import SYNTHETIC_DIR
from fastapi.testclient import TestClient

from mc_recon.api import create_app
from mc_recon.baseline import run_baseline
from mc_recon.policy import EXPECTED_PLAN, FakePolicyGate

ALICE = {"X-Analyst-Id": "alice"}


def _complete(svc, dataset="pair_01"):
    rid = svc.start_run(dataset, "alice")["run_id"]
    svc.confirm_mapping(rid, "alice", None)
    return rid


# --- G3 in the running agent ---------------------------------------------------------------


def test_every_verdict_carries_exception_and_trajectory_validators():
    svc, _ = make_service()
    rid = _complete(svc)
    for exc in svc.list_exceptions(rid):
        names = {r["validator"] for r in exc["verdicts"][0]["rules"]}
        assert names == {
            "faithfulness",
            "context_relevance",
            "answer_relevance",
            "output_data_leakage",
            "tool_call_accuracy",
            "plan_coherence",
        }
    steps = [t["step"] for t in svc.trajectory(rid)]
    assert tuple(steps[: len(EXPECTED_PLAN)]) == EXPECTED_PLAN
    reconcile_call = next(t for t in svc.trajectory(rid) if t["step"] == "reconcile")
    confirm = next(t for t in svc.trajectory(rid) if t["step"] == "await_mapping_confirm")
    assert reconcile_call["mapping_sha"] == confirm["mapping_sha"]
    assert svc.store.events(rid, "trajectory_verdict")[0]["payload"]["decision"] == "PASS"


def test_bad_trajectory_blocks_every_exception():
    class Incoherent(FakePolicyGate):
        def evaluate_run(self, trajectory):
            return super().evaluate_run(list(reversed(trajectory)))

    svc, _ = make_service(gate=Incoherent())
    rid = _complete(svc)
    excs = svc.list_exceptions(rid)
    assert excs and all(e["status"] == "blocked" for e in excs)
    failed = {r["validator"] for r in excs[0]["verdicts"][0]["rules"] if r["status"] == "FAIL"}
    assert "plan_coherence" in failed


def test_trajectory_recorded_through_review_loop():
    svc, _ = make_service(llm=poisoned_rationale_provider("pair_01"))
    rid = _complete(svc)
    for exc_id in svc.get_run(rid)["pending_action"]["pending"]:
        from mc_recon.service import ReviewDecision

        svc.review_exception(rid, exc_id, "bob", ReviewDecision(action="approve", note="ok"))
    steps = [t["step"] for t in svc.trajectory(rid)]
    assert steps[len(EXPECTED_PLAN) :] == ["validate_policy", "human_review_queue"]


# --- G5 baseline ---------------------------------------------------------------------------


def test_baseline_scores_manual_notes_with_the_same_gate():
    report = run_baseline(sorted(SYNTHETIC_DIR.glob("pair_*")), FakePolicyGate())
    s = report["summary"]
    assert s["simulated"] is True and s["notes"] == len(report["results"]) > 0
    assert s["decisions"].get("UNVERIFIABLE", 0) >= 1  # the false-positive split payment
    assert s["decisions"].get("BLOCK", 0) >= 1  # rounded figures / missing citations
    assert 0 < s["pass_rate"] < 1
    assert {"faithfulness", "context_relevance", "answer_relevance"} <= set(s["per_validator"])
    assert "tool_call_accuracy" not in s["per_validator"]  # no trajectory for a manual process
    assert s["seeded_errors"] == 14 and 0 < s["manual_recall"] <= 1


def test_baseline_rounded_figure_fails_faithfulness():
    report = run_baseline([SYNTHETIC_DIR / "pair_01"], FakePolicyGate())
    by_loan = {r["loan_id"]: r for r in report["results"]}
    dup = by_loan["MC-1010008"]  # "about £1,500 each time"
    assert dup["decision"] == "BLOCK"
    assert any(r["validator"] == "faithfulness" and r["status"] == "FAIL" for r in dup["rules"])


# --- G6 metrics over HTTP ------------------------------------------------------------------


def test_metrics_endpoints_with_baseline_drift():
    svc, _ = make_service()
    client = TestClient(create_app(svc))
    assert client.get("/baselines/latest").status_code == 404
    r = client.post("/baselines", json={"label": "week-1"}, headers=ALICE)
    assert r.status_code == 201, r.text
    baseline = r.json()
    assert baseline["label"] == "week-1" and baseline["summary"]["notes"] > 0

    rid = client.post("/runs", json={"dataset": "pair_01"}, headers=ALICE).json()["run_id"]
    client.post(f"/runs/{rid}/mapping/confirm", json={}, headers=ALICE)
    m = client.get(f"/runs/{rid}/metrics").json()

    assert m["latency"]["segments"] == 2 and m["latency"]["human_wait_ms"] >= 0
    assert {"ingest", "reconcile", "emit_log"} <= set(m["latency"]["by_step"])
    assert m["llm"]["calls"] == 7 and m["llm"]["by_task"]["classify_exception"] == 3
    assert m["llm"]["input_tokens"] > 0 and m["llm"]["tokens_estimated"] is True
    assert m["llm"]["cost_usd"] == 0.0 and m["llm"]["unpriced_models"] == []
    assert m["policy"]["first_pass"] == {"PASS": 3, "BLOCK": 0, "pass_rate": 1.0}
    assert m["policy"]["per_validator"]["plan_coherence"]["pass_rate"] == 1.0
    assert m["exceptions"]["by_priority"] == {"P1": 2, "P2": 1}
    assert set(m["exceptions"]["by_owner"]) == {
        "collections",
        "payments-operations",
        "pricing-and-product",
    }
    drift = m["vs_baseline"]
    assert drift["baseline_id"] == baseline["baseline_id"]
    assert drift["pass_rate"]["delta"] > 0  # agent beats the simulated manual process
    assert drift["per_validator_pass_rate_delta"]["faithfulness"] > 0

    summary = client.get("/metrics").json()
    assert summary["runs"] == 1 and summary["by_status"] == {"completed": 1}
    assert summary["policy"]["first_pass_rate"] == 1.0
    assert summary["vs_baseline"]["pass_rate"]["delta"] > 0
    assert client.get(f"/runs/{rid}/trajectory").json()[0]["step"] == "ingest"


def test_remediation_log_has_owner_and_priority():
    svc, _ = make_service()
    rid = _complete(svc, "pair_03")
    log = svc.remediation_log(rid)
    assert len(log) == 6
    for e in log:
        assert e["owner"] and e["priority"] in {"P1", "P2", "P3"} and e["priority_reason"]
        assert e["sla_business_days"] in {1, 2, 5}
    by_cat = {e["category"]: e for e in log}
    assert by_cat["UNIDENTIFIED_PAYMENT"]["priority"] == "P3"
    assert by_cat["RATE_MARGIN_BREACH"]["owner"] == "pricing-and-product"
