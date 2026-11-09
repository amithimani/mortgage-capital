"""Shared builders for graph / service / API tests."""

from __future__ import annotations

import shutil
from pathlib import Path

from conftest import SYNTHETIC_DIR

from mc_recon.data.oracle import RECORDINGS_FILE
from mc_recon.graph import AgentDeps
from mc_recon.llm import LLMTask, RecordedFakeProvider, RecordingSet
from mc_recon.observability import InMemoryTracer
from mc_recon.policy import FakePolicyGate
from mc_recon.service import RunService
from mc_recon.store import RunStore


def oracle_recordings(pairs=("pair_01", "pair_02", "pair_03"), data_root=SYNTHETIC_DIR):
    recs = []
    for p in pairs:
        recs.extend(RecordingSet.load(Path(data_root) / p / RECORDINGS_FILE).recordings)
    return recs


def poisoned_rationale_provider(pair: str, poison: str = " Estimated arrears 9,999.99."):
    """Oracle fake, but every rationale for ``pair`` cites a number not in the evidence."""
    recs = []
    for r in oracle_recordings():
        if r.task is LLMTask.DRAFT_RATIONALE and (r.label or "").startswith(f"{pair}:"):
            out = dict(r.output)
            out["rationale"] = out["rationale"] + poison
            r = r.model_copy(update={"output": out})
        recs.append(r)
    return RecordedFakeProvider(recs)


def make_service(
    llm=None,
    gate=None,
    data_root: Path = SYNTHETIC_DIR,
    store: RunStore | None = None,
    checkpointer=None,
    governance=None,
) -> tuple[RunService, InMemoryTracer]:
    tracer = InMemoryTracer()
    deps = AgentDeps(
        llm=llm or RecordedFakeProvider(oracle_recordings()),
        gate=gate or FakePolicyGate(),
        store=store or RunStore(),
        tracer=tracer,
        governance=governance,
    )
    return RunService(deps, data_root, checkpointer=checkpointer), tracer


def copy_pair(dst_root: Path, pair: str = "pair_01") -> Path:
    dst = Path(dst_root) / pair
    shutil.copytree(SYNTHETIC_DIR / pair, dst)
    return dst
