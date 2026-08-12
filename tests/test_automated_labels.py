from copy import deepcopy

from opennicf.automated_labels import (
    WARNING, build_audit, build_multimodal_automated_labels, build_text_automated_labels,
)


def test_text_labels_are_deterministic_separate_and_preserve_relevant_ids():
    queue = {"batches": [{"items": [{
        "query_id": "q1", "query_text": "How is ACL applied?", "original_query_text": "How is ACL applied?",
        "language": "EN", "categories": ["GENERAL_TECHNICAL"],
        "intended_relevant": [{"chunk_id": "c1", "text": "ACL is applied before retrieval."}],
        "hard_negatives": [], "hard_negative_candidates": [{"chunk_id": "c2", "text": "ACL audit after retrieval",
                                                               "mechanical_overlap_score": 4}],
        "leakage_indicators": [], "item_digest": "source",
    }]}]}
    before = deepcopy(queue)
    first = build_text_automated_labels(queue)
    assert first == build_text_automated_labels(queue)
    assert queue == before
    assert first["warnings"] == WARNING
    assert first["labels"][0]["original_relevant_ids"] == first["labels"][0]["final_relevant_ids"] == ["c1"]
    assert first["labels"][0]["automated_status"] == "AUTO_ACCEPT"


def test_multimodal_rewrites_leaky_query_without_claiming_human_review():
    queue = {"representations": [{"representation_id": "r1"}, {"representation_id": "r2"}], "queries": [{
        "query_id": "m1", "text": "Find evidence 001", "language": "en", "modality": "mixed",
        "domain": "d", "category": "diagram", "relevant_representation_ids": ["r1"],
        "hard_negative_ids": ["r2"], "hard_negative_category": "same_diagram_labels_wrong_flow",
        "leakage_indicators": ["synthetic_numeric_identifier"], "text_only_evidence_sufficient": False,
    }]}
    manifest = build_multimodal_automated_labels(queue)
    row = manifest["labels"][0]
    assert row["automated_status"] == "AUTO_HARD_NEGATIVE"
    assert row["original_query_text"] != row["automated_query_text"]
    assert manifest["automated_decisions_are_human_review"] is False


def test_audit_enforces_hard_negative_minimums():
    base = {"labels": [], "counts": {"explicit_hard_negative_queries": 0}, "labels_sha256": "x",
            "source_manifest_sha256": "y"}
    audit = build_audit(base, base)
    assert not audit["valid"]
    assert audit["human_reviewer_identity"] is None
