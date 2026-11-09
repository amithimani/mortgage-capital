"""Agent memory, and what keeps it from growing or going wrong.

**Working memory** is the graph state. It holds JSON only, drops the raw discrepancy list once
classified, and is bounded because the findings and the review rounds are bounded.

**Checkpoints** are the expensive part: LangGraph writes the whole state on every step, and a
paused run only ever needs the latest one. :func:`prune_checkpoints` keeps just that. History is
not lost, because the audit trail lives in the append-only events, spans and remediation log.

**LLM context** does not accumulate. Each call is a fresh, single-finding payload with no
conversation, and an oversized call is refused rather than truncated (``AgentLimits``).

**Long-term memory** is one thing: mappings an analyst has confirmed, keyed by the exact header
set. It only saves an LLM call. Three rules keep it from becoming a way to skip oversight or
poison a later run: only human-confirmed mappings are written, a recalled mapping is re-checked
against the current file and still goes to an analyst, and entries expire and can be deleted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from .engine.schema import ColumnMapping, FieldType, Source, field_specs
from .llm.provider import canonical_json


def header_fingerprint(headers: Mapping[Source, Sequence[str]]) -> str:
    body = {s.value: sorted(cols) for s, cols in sorted(headers.items(), key=lambda kv: kv[0])}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()[:24]


def recalled_mapping_fits(
    mapping: ColumnMapping, headers: Mapping[Source, Sequence[str]], profiles: Sequence[dict]
) -> bool:
    """Same headers can still hide a changed date format, so check that against the profile."""
    if mapping.problems(headers):
        return False
    for profile in profiles:
        source = Source(profile["source"])
        sm = mapping.for_source(source)
        specs = field_specs(source)
        columns = {c["name"]: c for c in profile["columns"]}
        wanted = sm.date_format or "ISO-8601"
        for name, column in sm.columns.items():
            if specs[name].type is FieldType.DATE and wanted not in columns[column]["date_formats"]:
                return False
    return True


def prune_checkpoints(saver: Any, thread_id: str) -> int:
    """Delete all but the newest checkpoint of a run. Returns how many were removed.

    Non-SQLite savers are left alone (the in-memory saver is only used by tests).
    """
    if not isinstance(saver, SqliteSaver):
        return 0
    with saver.cursor() as cur:
        cur.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id=? "
            "ORDER BY checkpoint_id DESC LIMIT 1",
            (thread_id,),
        )
        row = cur.fetchone()
        if row is None:
            return 0
        cur.execute(
            "DELETE FROM checkpoints WHERE thread_id=? AND checkpoint_id<>?", (thread_id, row[0])
        )
        removed = cur.rowcount
        cur.execute(
            "DELETE FROM writes WHERE thread_id=? AND checkpoint_id<>?", (thread_id, row[0])
        )
    return removed
