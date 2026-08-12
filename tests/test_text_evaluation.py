import pytest

from opennicf.text_evaluation import (
    apply_human_review_batches,
    audit_inputs,
    build_review_artifact,
    disagreement_analysis,
    leakage_indicators,
    paired_bootstrap,
    retrieval_metrics,
    validate_executable_manifest,
)


def fixtures():
    corpus = {"chunks": [
        {"chunk_id": "c1", "source_path": "sanitized/a.md", "text": "The alpha service rotates signing keys every seven days."},
        {"chunk_id": "c2", "source_path": "sanitized/b.md", "text": "The beta service uses a separate queue."},
    ]}
    gold = {"corpus_manifest_sha256": "corpus", "queries": [{
        "query_id": "q1", "query_text": "alpha service rotates signing keys every seven days",
        "language": "EN", "categories": ["CODE_RETRIEVAL"], "query_type": "hard_negative",
        "relevant_chunk_ids": ["c1"], "highly_relevant_chunk_ids": ["c1"], "hard_negative_chunk_ids": ["c2"],
    }]}
    review = {"queries": [{"query_id": "q1", "review_status": "NEEDS_HUMAN_REVIEW"}]}
    vectors = {"a": {"space_id": "A", "model": "a", "dimension": 2, "normalized": True,
                       "document_vectors": [[1, 0], [0, 1]], "query_vectors": [[1, 0]]},
               "b": {"space_id": "B", "model": "b", "dimension": 2, "normalized": True,
                       "document_vectors": [[1, 0], [0, 1]], "query_vectors": [[1, 0]]}}
    return corpus, gold, review, vectors


def test_audit_is_fail_closed_and_counts_objective_state():
    corpus, gold, review, vectors = fixtures()
    result = audit_inputs(corpus, gold, review, vectors, corpus_digest="corpus", gold_digest="gold", expected_corpus_digest="corpus")
    assert result.valid
    assert result.summary["corpus"]["chunks"] == 2
    assert result.summary["gold"]["review_status_counts"] == {"NEEDS_HUMAN_REVIEW": 1}
    assert result.summary["gold"]["normalized_decision_counts"] == {
        "ACCEPT": 0, "REWRITE": 0, "REJECT": 0, "PENDING_REVIEW": 1,
    }
    assert result.summary["hard_negative_query_count"] == 1
    assert result.summary["decision_grade_paired_metrics_executable"] is False
    gold["queries"][0]["relevant_chunk_ids"] = ["missing"]
    assert not audit_inputs(corpus, gold, review, vectors, corpus_digest="corpus", gold_digest="gold").valid


def test_review_artifact_is_deterministic_and_never_machine_accepts():
    corpus, gold, review, _ = fixtures()
    first = build_review_artifact(corpus, gold, review, batch_size=1)
    assert first == build_review_artifact(corpus, gold, review, batch_size=1)
    item = first["batches"][0]["items"][0]
    assert item["review_status"] == "PENDING_REVIEW"
    assert item["machine_recommendation"] == "REWRITE"
    assert item["reviewer_decision"] is None
    assert leakage_indicators(item["query_text"], [corpus["chunks"][0]["text"]])
    assert item["hard_negative_candidates"]
    assert item["hard_negative_candidates"][0]["review_status"] == "PENDING_REVIEW"


def test_metrics_bootstrap_and_disagreements_are_deterministic():
    labels = {"q": {"relevant_chunk_ids": ["r"], "highly_relevant_chunk_ids": ["r"]}}
    assert retrieval_metrics({"q": ["x", "r"]}, labels)["mrr_at_10"] == 0.5
    assert retrieval_metrics({"q": ["x", "r"]}, labels)["recall_at_1"] == 0
    one = paired_bootstrap({"a": 0, "b": 1}, {"a": 1, "b": 1}, samples=100, seed=4)
    assert one == paired_bootstrap({"a": 0, "b": 1}, {"a": 1, "b": 1}, samples=100, seed=4)
    assert disagreement_analysis({"q": ["x", "r"]}, {"q": ["r"]}, labels) == [
        {"query_id": "q", "baseline_first_relevant_rank": 2, "candidate_first_relevant_rank": 1}
    ]


def test_executable_manifest_requires_human_labels_vectors_digests_and_reranker():
    assert validate_executable_manifest({"status": "PREPARATION"})
    manifest = {
        "status": "EXECUTABLE", "safety": {"production_untouched": True, "deployment": False},
        "inputs": {"human_accepted_queries": 150, "unresolved_references": 0,
                   "reviewed_hard_negative_queries": 10, "corpus_sha256": "a", "labels_sha256": "b"},
        "embedding_spaces": [{"document_vector_count": 715}, {"document_vector_count": 715}],
        "reranker": {key: key for key in ("model", "revision", "license", "artifact", "artifact_sha256", "runtime", "runtime_version")},
    }
    assert validate_executable_manifest(manifest) == ()


def test_metrics_reject_unpaired_inputs():
    with pytest.raises(ValueError):
        retrieval_metrics({"q": []}, {})


def test_completed_review_ingestion_preserves_ids_and_recalculates_digest():
    corpus, gold, review, _ = fixtures()
    artifact = build_review_artifact(corpus, gold, review, batch_size=1)
    item = artifact["batches"][0]["items"][0]
    item.update({"reviewer_decision": "REWRITE", "reviewer_notes": "Distinct wording reviewed.",
                 "rewritten_query_text": "How often does the alpha service rotate signing keys?",
                 "selected_hard_negative_ids": ["c2"]})
    applied = apply_human_review_batches(gold, artifact, reviewer="declared-human", reviewed_at="2026-08-12T19:00:00Z")
    assert applied["queries"][0]["query_id"] == "q1"
    assert applied["queries"][0]["original_query_text"] == gold["queries"][0]["query_text"]
    assert applied["queries"][0]["hard_negative_chunk_ids"] == ["c2"]
    assert len(applied["reviewed_label_digest"]) == 64


def test_review_ingestion_rejects_silent_relevance_or_negative_changes():
    corpus, gold, review, _ = fixtures()
    artifact = build_review_artifact(corpus, gold, review, batch_size=1)
    item = artifact["batches"][0]["items"][0]
    item.update({"reviewer_decision": "ACCEPT", "reviewer_notes": "Reviewed.",
                 "selected_hard_negative_ids": ["not-a-candidate"]})
    with pytest.raises(ValueError, match="hard-negative"):
        apply_human_review_batches(gold, artifact, reviewer="human", reviewed_at="now")
    item["selected_hard_negative_ids"] = ["c2"]
    item["intended_relevant"] = [{"chunk_id": "c2"}]
    with pytest.raises(ValueError, match="relevant IDs"):
        apply_human_review_batches(gold, artifact, reviewer="human", reviewed_at="now")
