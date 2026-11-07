"""The generator is deterministic and the committed fixtures match it byte-for-byte."""

from __future__ import annotations

import pytest

from mc_recon.data.oracle import RECORDINGS_FILE, build_oracle_recordings
from mc_recon.data.synthetic import PAIRS, PairSpec, build_pair
from mc_recon.llm.fake import RecordingSet


def test_generation_is_deterministic():
    spec = PAIRS["pair_01"]
    assert build_pair(spec) == build_pair(spec)


def test_different_seed_changes_output():
    a = PAIRS["pair_01"]
    b = PairSpec(**{**a.__dict__, "seed": a.seed + 1})
    assert build_pair(a)["payments.csv"] != build_pair(b)["payments.csv"]


def test_committed_fixtures_match_generator(pair_dir):
    expected = build_pair(PAIRS[pair_dir.name])
    for name, content in expected.items():
        assert (pair_dir / name).read_text() == content, (
            f"{pair_dir.name}/{name} is stale — run `mc-recon generate`"
        )


def test_committed_oracle_recordings_are_current(pair_dir):
    committed = RecordingSet.load(pair_dir / RECORDINGS_FILE)
    rebuilt = build_oracle_recordings(pair_dir)
    assert {r.cache_key for r in committed.recordings} == {r.cache_key for r in rebuilt.recordings}


def test_unknown_error_type_rejected():
    spec = PairSpec("x", 1, 10, errors=("nope",), noise=(), variant=PAIRS["pair_01"].variant)
    with pytest.raises(ValueError, match="unknown"):
        build_pair(spec)
