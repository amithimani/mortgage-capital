from __future__ import annotations

import sqlite3

import pytest

from mc_recon.store import ConflictError, RunStore


@pytest.fixture
def store():
    s = RunStore()
    s.create_run("r1", "pair_01", "alice", {"a": 1})
    return s


def test_manifest_merges(store):
    store.update_run("r1", status="ingested", manifest={"b": 2})
    run = store.get_run("r1")
    assert run["status"] == "ingested" and run["manifest"] == {"a": 1, "b": 2}


def test_events_dedupe_and_are_append_only(store):
    assert store.record_event("r1", "x", "alice", {"k": 1}, dedupe_key="r1:x")
    assert not store.record_event("r1", "x", "alice", {"k": 1}, dedupe_key="r1:x")
    assert len(store.events("r1")) == 1
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._conn.execute("UPDATE events SET actor='mallory'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._conn.execute("DELETE FROM events")


def test_remediation_log_is_idempotent_and_append_only(store):
    assert store.append_log("r1", "D-1", {"v": 1})
    assert not store.append_log("r1", "D-1", {"v": 1})
    with pytest.raises(ConflictError):
        store.append_log("r1", "D-1", {"v": 2})
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store._conn.execute("DELETE FROM remediation_log")
    assert [e["v"] for e in store.remediation_log("r1")] == [1]


def test_review_conflict(store):
    assert store.put_review("r1", "D-1", 1, "alice", {"action": "approve"})
    assert not store.put_review("r1", "D-1", 1, "alice", {"action": "approve"})
    with pytest.raises(ConflictError):
        store.put_review("r1", "D-1", 1, "alice", {"action": "reject"})
    assert store.reviews("r1", 1) == {"D-1": {"action": "approve"}}
