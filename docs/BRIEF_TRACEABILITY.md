# Brief traceability — Use Case 1

**Source:** "Phase One Proposal — Mortgage Capital", Disseqt AI Limited, August 2026 (FDE
exercise, pick one of two). The brief itself is confidential and is not stored in this repo.
This page summarises its requirements in our own words and maps each one to the
implementation.

**Scope chosen:** Use Case 1, Data Reconciliation & Anomaly Detection. Use Case 2 (EU AI Act
controls) is out of scope for this exercise. UC1 is built to emit the evidence UC2 would consume
(oversight events, validator scores, an append-only audit trail), so UC2 can follow without
changing UC1's design.

Status key: ✅ done · 🟡 partial / on fakes · ⛔ blocked on access · ❌ not started

## Step-by-step

| # | Brief requirement | Status | Where / what's missing |
|---|---|---|---|
| 1 | 2–3 anonymised reconciliation pairs (servicing, payments, investor) with known seeded errors; small and repeatable; no production book | ✅ | `data/synthetic/pair_01..03`, `seeded_errors.json`, byte-identical regeneration (`mc-recon generate`). Open question in the brief (real vs mock data) → mock-first; real samples drop in behind the loader |
| 2 | Register the agent in the Disseqt Applications Registry and use its `application_id` | ⛔ | Needs a Disseqt account. Code path is ready: `application_id` goes to the tracer client (plan §2) |
| 3a | Ingest + profile files in code | ✅ | `engine/loader.py`, `engine/profile.py`, `graph.ingest` |
| 3b | LLM proposes field mapping | 🟡 | `propose_mapping` node + strict `MappingProposal` schema. Runs on the recorded fake; a real provider is pending a key ("OpenAI or any LLM") |
| 3c | Analyst confirms the mapping before anything runs | ✅ | Durable interrupt `await_mapping_confirm`; `POST /runs/{id}/mapping/confirm`; analyst identity recorded |
| 3d | Reconcile, match, net, compare in deterministic code, not the LLM | ✅ | `engine/reconcile.py`, `Decimal`-exact; golden tests show zero misses and zero false positives |
| 3e | Classify into a fixed exception set: missing payment, duplicate DD, rate vs contractual margin breach | ✅ | `ExceptionCategory` + per-kind `constrain_category` guard. Superset: also partial payment, balance break, unidentified payment |
| 3f | Plain-English rationale + remediation note per exception | 🟡 | `draft_rationale` node, grounded in evidence rows only. Oracle text on the fake |
| 3g | Remediation log with **owner, priority**, evidence and audit trail | ✅ | `remediation.py`: deterministic owner routing, priority P1–P3 with SLA, amount at risk and the reason for the priority; append-only log; `/events` (G1 closed) |
| 4 | `DisseqtAgenticClient(project_id, service_name, application_id)`; spans: `AGENT_EXEC` orchestration, `MODEL_EXEC` per LLM call (model, tokens, input/output messages), `TOOL_EXEC` for parse / engine / writer | ✅ locally / ⛔ Disseqt | `observability.py`: `AGENT_EXEC` per graph invocation with nested, timed child spans; `MODEL_EXEC` carries model, tokens, rendered input messages (`llm/prompts.py`) and output; `TOOL_EXEC` for parse, engine, policy and writer. Spans are persisted; a Disseqt exporter plugs in once there is access (G2 closed) |
| 5a | Faithfulness + context-relevance per drafted exception | 🟡 | `FakePolicyGate` stand-ins (number grounding, citation scope) |
| 5b | Answer-relevance | ✅ on fake | Names this loan and addresses this category (G3 closed) |
| 5c | Tool-call-accuracy + plan-coherence (engine called correctly and in order) | ✅ on fake | Each step appends to the run `trajectory`. The gate checks the engine ran once on the confirmed mapping and ingested files, every finding was classified and drafted, and the step order. Merged into every exception's verdict; `GET /runs/{id}/trajectory` (G3 closed) |
| 5d | Prompt-injection **and PII** checks on ingested files | ✅ on fake | Injection + PII (e-mail, UK phone, NI number, sort code, IBAN) screens on everything that reaches a prompt. Flagged samples are withheld and logged with the failed rule (G4 closed) |
| 6 | Realtime policy bundling validators with thresholds → one PASS/BLOCK; BLOCK holds for human review | ✅ on fake / ⛔ real | `PolicyGate`, fail-closed, review queue with approve / edit (re-checked) / reject. The real policy is authored in the Disseqt dashboard |
| 7 | Week 1 baseline: run samples of the current manual process through the same checks | ✅ simulated | `baseline.py`: manual-process notes go through the **same gate** with the same grounding; results stored (`POST /baselines`, `mc-recon baseline`). Notes are simulated until real anonymised ones are provided (same schema) (G5 closed) |
| 8 | Dashboard analytics: trace timeline, per-validator scores, PASS/BLOCK rates, tokens and cost per run, latency per step, drift vs baseline | ✅ locally / ⛔ Disseqt | `metrics.py`: `GET /runs/{id}/metrics` and `GET /metrics` cover all of them, plus human wait time. The Disseqt dashboard needs access (G6 closed) |

## Gaps closed (no Disseqt access needed)

All six gaps (G1–G6) found when the brief was first reviewed are now closed on the fakes. On the
three synthetic datasets, the simulated manual process passes the checks on 35% of its notes. It
uses rounded figures, misses citations, and raises false positives. The agent passes 100% on
its first attempt, and `vs_baseline` in `/metrics` reports the difference.

What remains needs access: a real LLM provider (step 3), and Disseqt registration, tracing
export, published policies and the dashboard (steps 2, 4, 6, 8).

## Access to request

| Item | Why | Brief step |
|---|---|---|
| Disseqt sandbox org: API key + `project_id` | Both clients need them (the SDK still requires `project_id` in 0.11.2) | 2, 4 |
| Applications Registry access → `application_id` | The server rejects traces without it | 2 |
| Dashboard rights to author and publish realtime policies | Policies (validators + thresholds + strategy) are created there; code references the policy id | 6 |
| Platform docs (behind login) | The exact request shape for policy evaluation; the right validator for PII on input text; the input format for tool-call-accuracy and plan-coherence | 5, 6 |
| Confirmation that mock data is acceptable (or anonymised samples) | The brief's own open question | 1 |
| LLM: a provided OpenAI key, or OK to use our own (Claude / Gemini)? | The brief says "OpenAI (or any LLM)" | 3 |
| Network allow-listing of the Disseqt API host for this dev environment | Needed to send traces and policy calls from the cloud session | 4–6 |
| Expected deliverable and demo format; timebox | Shapes what we prioritise | — |
