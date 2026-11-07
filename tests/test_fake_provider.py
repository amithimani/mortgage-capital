"""Recorded fake provider + oracle recordings: the LLM-shaped path runs with no model."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import BaseModel

from mc_recon.data.oracle import RECORDINGS_FILE
from mc_recon.data.synthetic import load_catalog, load_expected_mapping, pair_paths
from mc_recon.engine import (
    DiscrepancyKind,
    ExceptionCategory,
    constrain_category,
    load_inputs,
    profile_csv,
    reconcile,
)
from mc_recon.llm import (
    ClassificationOutput,
    LLMOutputValidationError,
    LLMRequest,
    LLMResult,
    LLMTask,
    MappingProposal,
    MissingRecordingError,
    RationaleOutput,
    RecordedFakeProvider,
    Recording,
    RecordingProvider,
    classification_request,
    mapping_request,
    rationale_request,
)


def _fake(pair_dir) -> RecordedFakeProvider:
    return RecordedFakeProvider.from_files(pair_dir / RECORDINGS_FILE)


def test_cache_key_is_order_independent_and_version_sensitive():
    a = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="v1", payload={"x": 1, "y": 2})
    b = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="v1", payload={"y": 2, "x": 1})
    c = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="v2", payload={"x": 1, "y": 2})
    assert a.cache_key == b.cache_key != c.cache_key


def test_mapping_replay_returns_expected_mapping(pair_dir):
    profiles = [profile_csv(p, s) for s, p in pair_paths(pair_dir).items()]
    result = _fake(pair_dir).generate(mapping_request(profiles), MappingProposal)
    assert result.output.mapping == load_expected_mapping(pair_dir)
    assert result.model_id == "recorded-fake"


def test_end_to_end_classification_matches_catalog(pair_dir):
    """Mapping from the fake -> engine -> classify + rationale from the fake."""
    fake = _fake(pair_dir)
    paths = pair_paths(pair_dir)
    profiles = [profile_csv(p, s) for s, p in paths.items()]
    mapping = fake.generate(mapping_request(profiles), MappingProposal).output.mapping

    result = reconcile(load_inputs(paths, mapping))
    labels = Counter()
    for d in result.discrepancies:
        proposed = fake.generate(classification_request(d), ClassificationOutput).output.category
        category = constrain_category(d.kind, proposed)
        labels[(category.value, d.loan_id)] += 1

        rationale = fake.generate(rationale_request(d, category.value), RationaleOutput).output
        evidence_keys = {f"{e.source.value}:{e.row}" for e in d.evidence}
        assert set(rationale.cited_evidence) <= evidence_keys

    want = Counter((e["expected_category"], e["loan_id"]) for e in load_catalog(pair_dir)["errors"])
    assert labels == want


def test_changed_payload_is_a_loud_miss(pair_dir):
    profiles = [profile_csv(p, s) for s, p in pair_paths(pair_dir).items()]
    req = mapping_request(profiles)
    tampered = req.model_copy(update={"payload": {**req.payload, "extra": True}})
    with pytest.raises(MissingRecordingError, match="propose_mapping"):
        _fake(pair_dir).generate(tampered, MappingProposal)


def test_recording_that_violates_schema_is_rejected():
    req = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="v1", payload={})
    rec = Recording(
        task=req.task,
        prompt_version=req.prompt_version,
        cache_key=req.cache_key,
        output={"category": "NOT_A_CATEGORY", "reason": "x"},
    )
    with pytest.raises(LLMOutputValidationError):
        RecordedFakeProvider([rec]).generate(req, ClassificationOutput)


def test_conflicting_recordings_rejected():
    req = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="v1", payload={})
    mk = lambda reason: Recording(  # noqa: E731
        task=req.task,
        prompt_version="v1",
        cache_key=req.cache_key,
        output={"category": "OTHER_NEEDS_REVIEW", "reason": reason},
    )
    with pytest.raises(ValueError, match="conflicting"):
        RecordedFakeProvider([mk("a"), mk("b")])


def test_recording_provider_round_trips(tmp_path):
    class Echo:
        model_id = "echo-model"

        def generate(self, request, output_schema: type[BaseModel]):
            out = output_schema.model_validate({"category": "MISSING_PAYMENT", "reason": "r"})
            return LLMResult(output=out, model_id=self.model_id, cache_key=request.cache_key)

    req = LLMRequest(task=LLMTask.CLASSIFY_EXCEPTION, prompt_version="v1", payload={"k": 1})
    recorder = RecordingProvider(Echo())
    live = recorder.generate(req, ClassificationOutput).output
    recorder.save(tmp_path / "rec.json")

    replay = RecordedFakeProvider.from_files(tmp_path / "rec.json")
    assert replay.generate(req, ClassificationOutput).output == live
    assert recorder.recordings[0].source_model == "echo-model"


def test_constrain_category_guards_llm_output():
    k = DiscrepancyKind.DUPLICATE_PAYMENT
    assert (
        constrain_category(k, "DUPLICATE_DIRECT_DEBIT") is ExceptionCategory.DUPLICATE_DIRECT_DEBIT
    )
    assert constrain_category(k, "MISSING_PAYMENT") is ExceptionCategory.OTHER_NEEDS_REVIEW
    assert constrain_category(k, "made up") is ExceptionCategory.OTHER_NEEDS_REVIEW
