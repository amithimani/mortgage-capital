"""Run analytics: the brief's dashboard view (step 8), computed from our own records.

Sources: persisted spans (latency, tokens, model), exception state (verdicts, reviews, owner,
priority) and the latest Week 1 baseline. The Disseqt dashboard shows the same things once
traces flow there; this keeps the numbers available, and comparable, without it.

* **latency**: agent compute time per step (span durations) and human wait time (the gaps
  between ``AGENT_EXEC`` segments, i.e. time spent waiting on an analyst).
* **llm**: calls, tokens, cost. Fake-provider tokens are chars/4 estimates and flagged as such.
  Cost uses ``MODEL_PRICES_PER_MTOK``; add real models' prices there when a provider is wired.
* **policy**: first-pass PASS/BLOCK (the agent's raw quality), final outcomes after review, and
  per-validator pass rate and mean score.
* **vs_baseline**: drift against the Week 1 baseline on the validators both share.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from statistics import median

MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "recorded-fake": (0.0, 0.0),
}
"""model_id -> (USD per 1M input tokens, USD per 1M output tokens)."""


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp() * 1000


def _per_validator(verdicts: list[dict]) -> dict[str, dict]:
    rules_by: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        for r in v["rules"]:
            rules_by[r["validator"]].append(r)
    out = {}
    for name in sorted(rules_by):
        rules = rules_by[name]
        scores = [r["score"] for r in rules if r.get("score") is not None]
        out[name] = {
            "evaluations": len(rules),
            "pass_rate": round(sum(r["status"] == "PASS" for r in rules) / len(rules), 4),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        }
    return out


def _vs_baseline(first_pass_rate: float | None, per_validator: dict, baseline: dict | None):
    if not baseline:
        return None
    b = baseline["summary"]
    shared = sorted(set(per_validator) & set(b["per_validator"]))
    return {
        "baseline_id": baseline["baseline_id"],
        "baseline_label": baseline["label"],
        "baseline_simulated": b.get("simulated"),
        "pass_rate": {
            "agent_first_pass": first_pass_rate,
            "baseline": b["pass_rate"],
            "delta": (
                round(first_pass_rate - b["pass_rate"], 4)
                if first_pass_rate is not None and b["pass_rate"] is not None
                else None
            ),
        },
        "per_validator_pass_rate_delta": {
            name: round(per_validator[name]["pass_rate"] - b["per_validator"][name]["pass_rate"], 4)
            for name in shared
        },
        "baseline_unverifiable_rate": b.get("unverifiable_rate"),
        "baseline_minutes_per_note": b.get("minutes_per_note"),
    }


def run_metrics(run: dict, spans: list[dict], exceptions: list[dict], baseline: dict | None):
    agent = sorted((s for s in spans if s["kind"] == "AGENT_EXEC"), key=lambda s: s["started_at"])
    steps: dict[str, list[float]] = defaultdict(list)
    for s in spans:
        if s["kind"] != "AGENT_EXEC" and s["duration_ms"] is not None:
            steps[s["name"]].append(s["duration_ms"])
    wait_ms = 0.0
    for prev, nxt in zip(agent, agent[1:], strict=False):
        wait_ms += max(
            0.0, _ts(nxt["started_at"]) - (_ts(prev["started_at"]) + prev["duration_ms"])
        )

    model = [s for s in spans if s["kind"] == "MODEL_EXEC"]
    tokens_in = sum(s["attributes"].get("input_tokens", 0) for s in model)
    tokens_out = sum(s["attributes"].get("output_tokens", 0) for s in model)
    cost = 0.0
    for s in model:
        p_in, p_out = MODEL_PRICES_PER_MTOK.get(s["attributes"].get("model_id", ""), (0.0, 0.0))
        cost += (
            s["attributes"].get("input_tokens", 0) * p_in
            + s["attributes"].get("output_tokens", 0) * p_out
        ) / 1e6
    unpriced = sorted({s["attributes"].get("model_id") for s in model} - set(MODEL_PRICES_PER_MTOK))

    first = [e["verdicts"][0] for e in exceptions if e.get("verdicts")]
    first_counts = Counter(v["decision"] for v in first)
    first_pass_rate = round(first_counts["PASS"] / len(first), 4) if first else None
    per_validator = _per_validator(first)
    review_actions = Counter(r["action"] for e in exceptions for r in e.get("reviews", []))
    amounts = [Decimal(e["amount_at_risk"]) for e in exceptions if e.get("amount_at_risk")]

    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "dataset": run["dataset"],
        "model_id": run["manifest"].get("model_id"),
        "latency": {
            "agent_compute_ms": round(sum(s["duration_ms"] or 0 for s in agent), 3),
            "human_wait_ms": round(wait_ms, 3),
            "segments": len(agent),
            "by_step": {
                name: {
                    "count": len(d),
                    "total_ms": round(sum(d), 3),
                    "mean_ms": round(sum(d) / len(d), 3),
                }
                for name, d in sorted(steps.items())
            },
        },
        "llm": {
            "calls": len(model),
            "by_task": dict(sorted(Counter(s["attributes"].get("task") for s in model).items())),
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "tokens_estimated": any(s["attributes"].get("tokens_estimated") for s in model),
            "cost_usd": round(cost, 6),
            "unpriced_models": unpriced,
        },
        "policy": {
            "exceptions_evaluated": len(first),
            "first_pass": {
                "PASS": first_counts["PASS"],
                "BLOCK": first_counts["BLOCK"],
                "pass_rate": first_pass_rate,
            },
            "final_status": dict(sorted(Counter(e["status"] for e in exceptions).items())),
            "review_actions": dict(sorted(review_actions.items())),
            "per_validator": per_validator,
        },
        "exceptions": {
            "total": len(exceptions),
            "by_category": dict(sorted(Counter(e["category"] for e in exceptions).items())),
            "by_priority": dict(sorted(Counter(e.get("priority") for e in exceptions).items())),
            "by_owner": dict(sorted(Counter(e.get("owner") for e in exceptions).items())),
            "amount_at_risk_total": str(sum(amounts, Decimal(0))),
        },
        "vs_baseline": _vs_baseline(first_pass_rate, per_validator, baseline),
    }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 3)


def summary_metrics(per_run: list[dict], first_verdicts: list[dict], baseline: dict | None):
    completed = [m for m in per_run if m["status"] == "completed"]
    compute = [m["latency"]["agent_compute_ms"] for m in completed]
    first_counts = Counter(v["decision"] for v in first_verdicts)
    first_pass_rate = (
        round(first_counts["PASS"] / len(first_verdicts), 4) if first_verdicts else None
    )
    per_validator = _per_validator(first_verdicts)
    return {
        "runs": len(per_run),
        "by_status": dict(sorted(Counter(m["status"] for m in per_run).items())),
        "latency": {
            "agent_compute_ms_p50": round(median(compute), 3) if compute else None,
            "agent_compute_ms_p95": _pct(compute, 0.95),
        },
        "llm": {
            "calls": sum(m["llm"]["calls"] for m in per_run),
            "input_tokens": sum(m["llm"]["input_tokens"] for m in per_run),
            "output_tokens": sum(m["llm"]["output_tokens"] for m in per_run),
            "cost_usd": round(sum(m["llm"]["cost_usd"] for m in per_run), 6),
        },
        "policy": {
            "exceptions_evaluated": len(first_verdicts),
            "first_pass_rate": first_pass_rate,
            "per_validator": per_validator,
        },
        "vs_baseline": _vs_baseline(first_pass_rate, per_validator, baseline),
    }
