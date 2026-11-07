# Mortgage Capital — Reconciliation & Anomaly Detection PoC

An AI-assisted reconciliation agent for mortgage servicing data, with EU AI Act governance
(Arts. 12, 14, 15) built in. It reconciles three files per period:

- the **servicing extract** (scheduled payments, balances, rates)
- the **payments file** (cash actually received)
- the **investor report** (what was reported to funders)

It surfaces exceptions such as missing payments, duplicate direct debits and rate-vs-margin
breaches, each with row-level evidence, a drafted rationale and a remediation note.

See **[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)** for the full design,
timeline and the verified Disseqt SDK surface.

## Design in one paragraph

The LLM never touches a number. Deterministic, unit-tested code does all matching, netting and
comparison (`mc_recon.engine`). The LLM only (1) proposes how raw columns map to the canonical
schema, which an analyst confirms, (2) picks an exception category from a closed list that is
constrained per finding type, and (3) drafts prose grounded only in the evidence rows. Every
drafted exception passes a Disseqt realtime policy. A BLOCK goes to a human and is never dropped.

## Status

| Component | State |
|---|---|
| Deterministic engine (`engine/`) | ✅ loader, mapping validation, profiling, reconciliation rules |
| Synthetic data (`data/`) | ✅ 3 seeded pairs + error catalog + oracle LLM recordings |
| LLM adapter (`llm/`) | ✅ provider protocol, strict output schemas, payload builders, recorded fake |
| LangGraph agent (`graph.py`) | ✅ all 9 nodes, SQLite checkpointer, durable interrupts, BLOCK → review → re-check loop |
| Policy gate (`policy.py`) | ✅ interface + deterministic `FakePolicyGate`, fail-closed wrapper, input screen |
| Run store (`store.py`) | ✅ manifests, append-only event trail + remediation log, reviews, idempotency keys |
| FastAPI service (`api.py`, `service.py`) | ✅ plan §4 endpoints + `/events` audit trail |
| Real LLM provider (Claude / Gemini) | Pending key; drops in behind `LLMProvider` |
| Disseqt tracing + `DisseqtPolicyGate` | Week 1–3 (needs credentials); drops in behind `Tracer` / `PolicyGate` |
| UC2 governance adapter | Week 2–3, after the vendor-confirmation gate |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -e ".[dev]"

pytest                                                 # 95 tests, ~1s, no network, no LLM
mc-recon reconcile data/synthetic/pair_01 --check      # run the engine vs ground truth
mc-recon profile   data/synthetic/pair_01              # what the mapping LLM will see
mc-recon generate                                      # regenerate fixtures (byte-identical)
uvicorn mc_recon.api:app                               # the agent API, on fakes
```

`mc-recon reconcile` prints the full `ReconciliationResult` as JSON. With `--check` it compares
the result against `seeded_errors.json` and exits non-zero on any miss or false positive.

## Running the agent

`uvicorn mc_recon.api:app` starts the service with the recorded fake LLM (every pair's oracle
recordings) and `FakePolicyGate`. State lives in `var/mc_recon.sqlite`; change it with
`MC_DB_PATH`. Datasets come from `MC_DATA_ROOT` (default `data/synthetic`). Mutating calls need
an `X-Analyst-Id` header (a PoC stand-in for real auth) and accept `Idempotency-Key`.

```bash
H='-H X-Analyst-Id:alice -H content-type:application/json'
RID=$(curl -s -X POST localhost:8000/runs $H -d '{"dataset":"pair_01"}' | jq -r .run_id)
curl -s localhost:8000/runs/$RID/mapping | jq '.proposal, .problems'     # review the proposal
curl -s -X POST localhost:8000/runs/$RID/mapping/confirm $H -d '{}'     # confirm (or send {"mapping": {...}})
curl -s localhost:8000/runs/$RID/exceptions | jq '.[] | {id, category, status}'
curl -s localhost:8000/runs/$RID/remediation-log | jq
curl -s localhost:8000/runs/$RID/events | jq '.[] | {event_type, actor, subject_id}'
```

| Endpoint | Purpose |
|---|---|
| `POST /runs` | Start a run on a dataset. It stops at mapping confirmation |
| `GET /runs/{id}` | Status, pending human action, exception counts, manifest |
| `GET /runs/{id}/mapping` | Proposed mapping, problems with it, column profiles, input-screen findings |
| `POST /runs/{id}/mapping/confirm` | Accept the proposal (`{}`) or supply an edited `mapping`. Validated against the headers before resuming |
| `GET /runs/{id}/exceptions` | Exceptions with category, rationale, every policy verdict and reviews |
| `POST /runs/{id}/exceptions/{exc}/review` | `approve` (human override) / `edit` (re-checked against the policy) / `reject` |
| `GET /runs/{id}/remediation-log` | Final append-only log with evidence and final verdict |
| `GET /runs/{id}/events` | Audit trail: mapping confirmation, policy verdicts, reviews, input-screen blocks |

**Exception lifecycle:** `drafted` → `passed` or `blocked`. A blocked exception goes to review:
`approve` makes it `approved_override` (logged and flagged as a human override), `edit` makes it
`edited` and sends it back through the policy, and `reject` makes it `rejected` (not logged).
Logged exceptions end up `logged`. When the policy gate is down, or returns a non-final async
verdict, the result is BLOCK.

## Repository layout

```
src/mc_recon/
  engine/            # deterministic core — no LLM, no network
    schema.py        #   canonical fields per source, ColumnMapping + validation, errors
    loader.py        #   CSV -> typed records through a confirmed mapping; fails loudly
    profile.py       #   column profiles for the ingest node / mapping prompt
    reconcile.py     #   match, net, compare -> Discrepancy with row evidence
    models.py        #   records, Discrepancy, EvidenceRef, ReconConfig, ReconciliationResult
    taxonomy.py      #   DiscrepancyKind (engine) / ExceptionCategory (business) + guard
  llm/
    provider.py      #   LLMProvider protocol, LLMRequest (+cache_key), prompt versions
    schemas.py       #   MappingProposal, ClassificationOutput, RationaleOutput (strict)
    payloads.py      #   request builders shared by graph nodes and the oracle
    fake.py          #   RecordedFakeProvider (replay) + RecordingProvider (capture)
  data/
    synthetic.py     #   seeded generator for reconciliation pairs + error catalog
    oracle.py        #   ground-truth LLM recordings for each pair
  graph.py           # LangGraph agent: nodes, routing, interrupts
  service.py         # run lifecycle; validates human actions before resuming the graph
  api.py             # FastAPI app (thin HTTP shell over RunService)
  policy.py          # PolicyGate protocol, FakePolicyGate, fail-closed wrappers
  store.py           # SQLite run store; append-only events + remediation log
  observability.py   # Tracer seam (Null / InMemory; Disseqt later)
  cli.py             # `mc-recon` entry point
data/synthetic/pair_0{1,2,3}/   # committed fixtures (regenerated byte-for-byte by tests)
docs/IMPLEMENTATION_PLAN.md
tests/
```

## Reconciliation rules

| Discrepancy kind | Rule | Default tolerance |
|---|---|---|
| `UNMATCHED_EXPECTED_PAYMENT` | No payment for the loan within ± window of the due date | window 3 days |
| `DUPLICATE_PAYMENT` | Two or more DDs with the same mandate and amount within the duplicate window; later ones are excluded from netting | 5 days |
| `AMOUNT_MISMATCH` | Net received ≠ scheduled amount (split payments that net correctly are fine) | 0.01 |
| `RATE_COMPONENT_MISMATCH` | applied ≠ base + margin, or the investor's rate/margin ≠ servicing's | 0.005 pp |
| `BALANCE_MISMATCH` | Investor closing balance ≠ servicing closing balance | 0.05 |
| `UNMATCHED_PAYMENT` | Payment for an unknown loan, or outside the match window | — |
| `POPULATION_MISMATCH` | Loan is in servicing or investor data but not both | — |

All amounts are `Decimal`. Output is fully deterministic: the same inputs and config always give
byte-identical JSON. Discrepancy ids (`D-<sha256[:12]>`) are derived from content, so
remediation-log writes can be idempotent.

## Synthetic data

`data/synthetic/pair_NN/` contains:

| File | Purpose |
|---|---|
| `servicing_extract.csv`, `payments.csv`, `investor_report.csv` | Raw inputs, deliberately with non-canonical headers (two header variants) and mixed date and number formats |
| `expected_mapping.json` | The correct `ColumnMapping`, which is the ground truth for the mapping step |
| `seeded_errors.json` | Every planted error, with its expected kind and category, plus every piece of benign noise |
| `oracle_recordings.json` | What a perfect LLM would return for each call on this pair, replayed by `RecordedFakeProvider` |

Benign noise that must **not** raise exceptions: loan refs with odd case or whitespace, split
Faster Payments, ±0.01 balance rounding, 4dp rate drift within tolerance, late value dates
inside the window, and weekend rolls. The golden tests enforce zero misses and zero false
positives on every pair.

## Fake LLM provider

```python
from mc_recon.llm import RecordedFakeProvider, mapping_request, MappingProposal

fake = RecordedFakeProvider.from_files("data/synthetic/pair_01/oracle_recordings.json")
proposal = fake.generate(mapping_request(profiles), MappingProposal).output
```

Recordings are keyed by `sha256(task, prompt_version, payload)`. A changed payload or prompt
version raises `MissingRecordingError` rather than returning something plausible, and a
recording that no longer fits its output schema raises `LLMOutputValidationError`. To freeze a
live model run into fixtures, wrap the real provider in `RecordingProvider` and call `.save()`.

## Disseqt SDK

The plan targets `disseqt-ai-sdk==0.11.2`, pinned in the `agent` extra. Several README-level
claims about the SDK don't match its code (for example, `project_id` is still required, policy
evaluation goes through `disseqt_sdk.Client.validate(..., policies=[...])`, and a span-level
`realtime_policy_id` does not block anything). See the change log in the implementation plan.
