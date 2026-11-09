# Agent harness: limits, memory, mocks and evaluation

The agent's only non-deterministic parts are three LLM calls and two human checkpoints. This page
covers what bounds them. Run `mc-recon eval` to see all of it exercised.

## Limits

One object, `AgentLimits` in `src/mc_recon/limits.py`. Override any field with
`MC_LIMIT_<NAME>`. `GET /runs/{id}/harness` shows a run's usage against them.

| Limit | Default | Bounds | At the limit |
|---|---|---|---|
| `max_reasks` | 1 | Retries after schema-invalid output, per call | Finding held for a human |
| `max_review_rounds` | 3 | The review → re-check loop | Last round refuses `edit`, so approve or reject ends it |
| `recursion_limit` (derived) | 14 | LangGraph supersteps | Run fails; only a routing bug can reach it |
| `max_llm_calls` | 500 | Calls per run, re-asks included | Remaining findings held |
| `max_total_tokens` | 500,000 | Tokens per run | Remaining findings held |
| `max_cost_usd` | 5.00 | Spend per run | Remaining findings held |
| `max_llm_seconds` | 600 | Time in LLM nodes (human wait excluded) | Remaining findings held |
| `max_input_tokens_per_call` | 16,000 | One prompt's size. Refused, never truncated | That finding held |
| `max_consecutive_llm_failures` | 3 | Circuit breaker | Remaining findings held |
| `max_findings` | 500 | Fan-out; a flood usually means a wrong mapping | Run fails loudly |
| `max_file_bytes`, `max_rows` | 10 MB, 50,000 | Input size, checked before reading into memory | Request refused |

**The rule:** a limit can cost automation, never a finding. The deterministic engine has already
found the discrepancy by the time any per-finding limit applies. A held finding keeps its engine
summary and evidence, is categorised `OTHER_NEEDS_REVIEW`, skips the LLM, and goes to review with
`enforcement: harness_hold`. Every limit reached is an event on the audit trail.

External calls have their own bound: three attempts with capped, jittered backoff, on `429`,
`5xx` and connection errors only (`transport.py`).

## Memory

| Layer | What it holds | What keeps it bounded |
|---|---|---|
| Graph state | JSON only | Raw discrepancies dropped once classified; findings and rounds are capped |
| Checkpoints | Whole state per step | Pruned to the latest after every segment. A paused run needs only that |
| LLM context | One finding per call, no conversation | Nothing accumulates; oversized calls refused |
| Process | Per-run locks, fake call history, trace buffer | Locks freed on finish; the others are bounded deques |
| Long-term | Analyst-confirmed mappings, keyed by exact headers | TTL (`MC_MAPPING_MEMORY_DAYS`), `DELETE /memory/mappings/{fingerprint}` |

Pruning loses no history, because the audit record is the append-only events, spans, remediation
log and governance evidence, none of which are pruned.

Long-term memory saves one LLM call and nothing else. Only a human-confirmed mapping is written,
so a bad model output can't poison later runs. A recalled mapping is re-checked against the
current headers and date formats, is logged as `mapping_recalled`, and still needs an analyst to
confirm it.

## Input screening

Three layers, because the original screen only saw the first three values in each column:

1. **Prompt samples**: flagged samples are withheld from the mapping prompt.
2. **Whole file**: every distinct value of every text column, recorded as `input_screen_finding`.
3. **Prompt boundary**: each finding's file-derived text is screened before any prompt. A hit
   holds that finding and it never reaches the LLM.

## Mock endpoints

`mc-recon mock` serves Disseqt (registry, policy publishing, validate, traces, ledger, dashboard)
and an OpenAI-compatible LLM. The agent reaches them over real HTTP:

```bash
mc-recon mock &
MC_LLM_PROVIDER=http MC_LLM_BASE_URL=http://localhost:8100 MC_LLM_API_KEY=mock-key \
MC_POLICY_GATE=http MC_DISSEQT_URL=http://localhost:8100 MC_DISSEQT_API_KEY=mock-key \
mc-recon serve
```

`POST /__mock/faults` queues failures, for example
`{"target": "llm", "mode": "http_429", "count": 2}`.

**The Disseqt routes and payloads are assumptions**, since the real API docs are behind a login.
All of that guesswork is confined to `disseqt_http.py` and `mocks/app.py`. The LLM side follows
the OpenAI chat completions shape, so `MC_LLM_BASE_URL` can point at a real endpoint.

## Evaluation

`mc-recon eval` (`--json` for machines) runs 16 scenarios and exits non-zero on any failure. CI
runs it. They cover golden accuracy on all three datasets, both loop limits, budgets, the circuit
breaker, fail-closed outages, injection in a late row, checkpoint compaction with restart,
mapping memory, the full path through the mocks, and the governance lifecycle.

## Authentication

Set `MC_ANALYST_TOKENS=token:analyst,...` and every route except the probes needs a bearer token,
with the identity taken from it. Without it the service runs in local PoC mode, prints a warning
at startup, and trusts the `X-Analyst-Id` header. Static tokens are a stand-in for SSO.

## What this doesn't cover

- No real LLM or Disseqt call has been made. Token counts and costs come from the mock.
- SQLite and in-process locks mean one replica (`docs/DEPLOYMENT.md`).
- Graph calls are synchronous. A live model needs a background worker and `202` responses.
- Retention is recorded as a figure on the AI system record. Nothing purges on a schedule yet, and
  the real Art. 12 period is a decision for compliance.
