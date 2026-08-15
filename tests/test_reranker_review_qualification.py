from __future__ import annotations

import pytest

from opennicf.reranker_qualification import (
    CONFIG_SHA256, DeterministicQwenRerankerAdapter, MODEL_ID, MODEL_REVISION, MODEL_SHA256, MODEL_SIZE,
    PREFIX, SUFFIX, validate_artifact_manifest,
    TOKENIZER_SHA256,
)
from opennicf.evaluation import RerankerOutOfMemory, RerankerTimeout
from opennicf.text_evaluation import apply_human_review, build_review_artifact
from opennicf.multimodal_evaluation import (
    Lifecycle, MultimodalQuery, Representation, ReviewStatus,
    apply_multimodal_review, build_multimodal_review_artifact,
)


def _manifest():
    return {"status": "EXECUTABLE", "model": MODEL_ID, "revision": MODEL_REVISION,
            "license": "apache-2.0", "filename": "model.safetensors", "sha256": MODEL_SHA256,
            "size_bytes": MODEL_SIZE, "runtime": "transformers", "runtime_version": "4.57.0",
            "tokenizer_sha256": TOKENIZER_SHA256, "config_sha256": CONFIG_SHA256}


def test_artifact_manifest_is_exact_and_fail_closed():
    assert validate_artifact_manifest(_manifest()) == ()
    for field in ("model", "revision", "sha256", "size_bytes", "runtime_version", "status"):
        broken = _manifest(); broken[field] = None
        assert validate_artifact_manifest(broken)


def test_adapter_score_direction_stable_ties_batching_and_truncation():
    calls = []
    def encode(value):
        if value == PREFIX: return [1, 2]
        if value == SUFFIX: return [3, 4]
        return list(range(len(value.split())))
    def runner(batch, timeout):
        calls.append((len(batch), timeout))
        return [(2.0, 1.0), (0.0, 3.0)][:len(batch)]
    # Four prompt-boundary tokens leave a 20-token body budget: the first and
    # third pairs fit while the deliberately longer second pair is truncated.
    adapter = DeterministicQwenRerankerAdapter(encode, runner, max_length=24, batch_size=2)
    result = adapter.score([("q", "one"), ("q", "one two three four five six"), ("q", "one")])
    assert result.scores[1] > result.scores[0]
    assert result.ordered_indices == (1, 0, 2)  # deterministic index tie-break
    assert result.truncated == (False, True, False)
    assert calls == [(2, 30.0), (1, 30.0)]


@pytest.mark.parametrize(("error", "classification"), [
    (TimeoutError("late"), RerankerTimeout), (MemoryError("oom"), RerankerOutOfMemory),
])
def test_adapter_preserves_timeout_and_oom_classification(error, classification):
    def runner(batch, timeout): raise error
    adapter = DeterministicQwenRerankerAdapter(lambda _: [1], runner, max_length=4)
    with pytest.raises(classification):
        adapter.score([("q", "d")])


def _text_review_artifact():
    corpus = {"chunks": [{"chunk_id": "c1", "text": "alpha", "source_path": "a"},
                         {"chunk_id": "c2", "text": "beta", "source_path": "b"}]}
    gold = {"queries": [{"query_id": "q1", "query_text": "alpha?", "language": "en",
                          "categories": ["test"], "relevant_chunk_ids": ["c1"],
                          "hard_negative_chunk_ids": ["c2"]}]}
    review = {"queries": [{"query_id": "q1", "review_status": "PENDING_REVIEW"}]}
    return build_review_artifact(corpus, gold, review)


def test_text_review_requires_attribution_preserves_ids_and_recalculates_digest():
    artifact = _text_review_artifact()
    decision = {"q1": {"reviewer_decision": "ACCEPT", "reviewer_identity": "reviewer-1",
                       "review_timestamp": "2026-08-12T12:00:00Z", "hard_negative_chunk_ids": ["c2"]}}
    result = apply_human_review(artifact, decision, {"c1", "c2"})
    assert len(result["labels_sha256"]) == 64
    assert result["decisions"][0]["original_query_text"] == result["decisions"][0]["reviewed_query_text"]
    assert result["decisions"][0]["hard_negative_chunk_ids"] == ["c2"]
    bad = {"q1": {**decision["q1"], "hard_negative_chunk_ids": ["missing"]}}
    with pytest.raises(ValueError, match="invalid hard-negative"):
        apply_human_review(artifact, bad, {"c1", "c2"})
    bad = {"q1": {**decision["q1"], "relevant_chunk_ids": ["c2"]}}
    with pytest.raises(ValueError, match="silently change"):
        apply_human_review(artifact, bad, {"c1", "c2"})


def test_multimodal_review_uses_safe_refs_and_validates_rewrite():
    rep = Representation("r1", "s1", "image", 1, "renderer-1", "a" * 64, 64, 64,
                         "test", "fixture", ("review",), Lifecycle.READY)
    negative = Representation("r2", "s2", "image", 1, "renderer-1", "b" * 64, 64, 64,
                              "test", "fixture", ("review",), Lifecycle.READY)
    query = MultimodalQuery("q1", "What is shown?", "en", "mixed", "test", "screen",
                            ("r1",), ("r1",), ("r2",), False, ReviewStatus.PENDING_REVIEW)
    artifact = build_multimodal_review_artifact([query], [rep, negative])
    assert artifact["label"] == "NON_DECISION_GRADE_REVIEW_ASSISTANCE"
    assert all(item["bounded_preview_ref"].startswith("fixture://") for item in artifact["items"][0]["previews"])
    decisions = {"q1": {"reviewer_decision": "REWRITE", "reviewer_identity": "reviewer-1",
                         "review_timestamp": "2026-08-12T12:00:00Z",
                         "reviewed_query_text": "Which control is visible?", "hard_negative_ids": ["r2"]}}
    applied = apply_multimodal_review(artifact, decisions, {"r1", "r2"})
    assert applied["decisions"][0]["reviewed_query_text"] == "Which control is visible?"
    with pytest.raises(ValueError, match="invalid hard-negative"):
        apply_multimodal_review(artifact, {"q1": {**decisions["q1"], "hard_negative_ids": ["bad"]}}, {"r1", "r2"})
