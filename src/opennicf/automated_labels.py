"""Deterministic, explicitly non-human label generation for issue #86.

The functions consume copies of review artifacts and never mutate the source
queues.  Their output is comparative experimental evidence, not gold labels.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Mapping


WARNING = [
    "AUTOMATED LABELS MAY CONTAIN SOURCE-DERIVED LEAKAGE",
    "QUALITY RESULTS ARE COMPARATIVE EXPERIMENTAL EVIDENCE",
    "PRODUCTION REMAINS UNCHANGED",
]
STATUSES = {"AUTO_ACCEPT", "AUTO_REWRITE", "AUTO_REJECT", "AUTO_HARD_NEGATIVE"}
LEAKAGE_STRATA = {"LOW_LEAKAGE", "MEDIUM_LEAKAGE", "HIGH_LEAKAGE", "EXACT_IDENTIFIER"}
TEXT_HARD_NEGATIVE_CATEGORIES = (
    "same_table_wrong_operation", "same_class_wrong_method", "same_service_wrong_endpoint",
    "same_error_wrong_service", "same_acronym_different_system", "same_document_wrong_section",
    "current_versus_obsolete", "production_versus_non_production",
    "same_identifier_wrong_semantic_context", "lexical_overlap_greater_than_relevant_evidence",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return sha256(_canonical(value)).hexdigest()


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w.-]+", value.casefold()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return 0.0 if not (a or b) else round(len(a & b) / len(a | b), 6)


def _header(kind: str, source_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_kind": kind,
        "evaluation_mode": "AUTOMATED_NON_HUMAN_REVIEWED",
        "human_review_required_for_this_run": False,
        "decision_grade_production_validation": False,
        "production_migration_authorized": False,
        "recommendation_scope": "EXPERIMENTAL_ARCHITECTURE_ONLY",
        "automated_decisions_are_human_review": False,
        "production_decision": "NOT_AUTHORIZED",
        "production": "UNCHANGED",
        "deployments": 0,
        "merges": 0,
        "warnings": WARNING,
        "source_manifest_sha256": source_sha256,
    }


def _text_hn_category(item: Mapping[str, Any], ordinal: int) -> str:
    categories = set(item.get("categories", ()))
    path = str((item.get("intended_relevant") or [{}])[0].get("source_path", "")).casefold()
    if "EXACT_IDENTIFIER" in categories:
        return "same_identifier_wrong_semantic_context"
    if "SQL_DATABASE" in categories or path.endswith(".sql"):
        return "same_table_wrong_operation"
    if "CODE" in categories or path.endswith((".java", ".py", ".ps1")):
        return "same_class_wrong_method"
    if "LOG_INCIDENT" in categories or path.endswith(".log"):
        return "same_error_wrong_service"
    return TEXT_HARD_NEGATIVE_CATEGORIES[ordinal % len(TEXT_HARD_NEGATIVE_CATEGORIES)]


def build_text_automated_labels(review: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the text review queue without applying or claiming human review."""
    items = sorted(
        (item for batch in review.get("batches", ()) for item in batch.get("items", ())),
        key=lambda row: row["query_id"],
    )
    rows: list[dict[str, Any]] = []
    for ordinal, item in enumerate(items):
        original = str(item.get("original_query_text") or item.get("query_text") or "").strip()
        relevant = [str(row["chunk_id"]) for row in item.get("intended_relevant", ()) if row.get("chunk_id")]
        relevant_text = "\n".join(str(row.get("text", "")) for row in item.get("intended_relevant", ()))
        relevant_text_keys = {" ".join(sorted(_tokens(str(row.get("text", ""))))) for row in item.get("intended_relevant", ())}
        candidates = [row for row in item.get("hard_negatives", ()) + item.get("hard_negative_candidates", ())
                      if row.get("chunk_id") and row.get("chunk_id") not in relevant
                      and " ".join(sorted(_tokens(str(row.get("text", ""))))) not in relevant_text_keys]
        candidate = candidates[0] if candidates else None
        invalid = not original or not relevant
        indicators = list(item.get("leakage_indicators", ()))
        exact = "EXACT_IDENTIFIER" in set(item.get("categories", ()))
        rewrite = str(item.get("rewrite_candidate") or "").strip()
        if invalid:
            status, automated, reason = "AUTO_REJECT", original, "missing intelligible query or valid relevant ID"
        elif "HARD_NEGATIVE" in set(item.get("categories", ())) and candidate:
            status = "AUTO_HARD_NEGATIVE"
            automated = rewrite if rewrite and rewrite != original else original
            reason = "explicit hard-negative query with a distinct deterministic candidate"
        elif indicators:
            status, automated = "AUTO_REWRITE", rewrite if rewrite and rewrite != original else f"Retrieve evidence that resolves: {original}"
            reason = "source-derived, duplicated, or answer-exposing phrasing"
        elif candidate:
            status, automated, reason = "AUTO_ACCEPT", original, "intelligible supported query with resolvable evidence"
        else:
            status, automated, reason = "AUTO_REJECT", original, "no meaningful hard-negative distractor available"
        leakage = "EXACT_IDENTIFIER" if exact else ("HIGH_LEAKAGE" if indicators else
                    ("MEDIUM_LEAKAGE" if _jaccard(original, relevant_text) >= 0.15 else "LOW_LEAKAGE"))
        negative = None if candidate is None else {
            "chunk_id": candidate["chunk_id"],
            "category": _text_hn_category(item, ordinal),
            "selection_method": "highest mechanical overlap, stable source order, excluding relevant IDs",
            "mechanical_overlap_score": candidate.get("mechanical_overlap_score"),
            "token_jaccard_similarity": _jaccard(automated, str(candidate.get("text", ""))),
            "source_path": candidate.get("source_path"),
        }
        row = {
            "query_id": item["query_id"], "automated_status": status, "leakage_stratum": leakage,
            "original_query_text": original, "automated_query_text": automated,
            "rewrite_reason": reason if automated != original else None,
            "original_relevant_ids": relevant, "final_relevant_ids": list(relevant),
            "label_identity_changed": False, "hard_negative": negative,
            "language": item.get("language"), "categories": list(item.get("categories", ())),
            "source_item_digest": item.get("item_digest"), "decision_rule_version": "issue86-auto-v1",
        }
        row["automated_decision_digest"] = _digest(row)
        rows.append(row)
    result = {**_header("AUTOMATED_TEXT_LABELS", _digest(review)), "labels": rows}
    result["counts"] = _counts(rows)
    result["labels_sha256"] = _digest(rows)
    return result


def build_multimodal_automated_labels(review: Mapping[str, Any]) -> dict[str, Any]:
    known = {row["representation_id"] for row in review.get("representations", ())}
    rows: list[dict[str, Any]] = []
    for item in sorted(review.get("queries", ()), key=lambda row: row["query_id"]):
        original = str(item.get("text", "")).strip()
        relevant = list(item.get("relevant_representation_ids", ()))
        negatives = [rid for rid in item.get("hard_negative_ids", ()) if rid not in relevant]
        unresolved = [rid for rid in (*relevant, *negatives) if rid not in known]
        indicators = list(item.get("leakage_indicators", ()))
        if not original or not relevant or unresolved:
            status, automated, reason = "AUTO_REJECT", original, "missing query/evidence or unresolved representation reference"
        elif not negatives:
            status, automated, reason = "AUTO_REJECT", original, "no meaningful multimodal hard negative"
        else:
            status = "AUTO_HARD_NEGATIVE"
            automated = (f"Locate the {str(item.get('category', 'visual')).replace('_', ' ')} evidence "
                         f"for {item.get('domain', 'the requested domain')}.") if indicators else original
            reason = "removed synthetic/source-derived identifying phrasing" if automated != original else "validated explicit visual distractor"
        leakage = "HIGH_LEAKAGE" if indicators and automated == original else ("MEDIUM_LEAKAGE" if indicators else "LOW_LEAKAGE")
        negative_rows = [{
            "representation_id": rid, "category": item.get("hard_negative_category"),
            "selection_method": "explicit queue pairing validated against representation manifest",
            "metadata_similarity": {"domain": item.get("domain"), "modality": item.get("modality"),
                                    "category": item.get("category")},
        } for rid in negatives]
        row = {
            "query_id": item["query_id"], "automated_status": status, "leakage_stratum": leakage,
            "original_query_text": original, "automated_query_text": automated,
            "rewrite_reason": reason if automated != original else None,
            "original_relevant_ids": relevant, "final_relevant_ids": list(relevant),
            "label_identity_changed": False, "hard_negatives": negative_rows,
            "language": item.get("language"), "modality": item.get("modality"),
            "category": item.get("category"), "text_only_evidence_sufficient": item.get("text_only_evidence_sufficient"),
            "decision_rule_version": "issue86-auto-v1",
        }
        row["automated_decision_digest"] = _digest(row)
        rows.append(row)
    result = {**_header("AUTOMATED_MULTIMODAL_LABELS", _digest(review)), "labels": rows}
    result["counts"] = _counts(rows)
    result["labels_sha256"] = _digest(rows)
    return result


def _counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    return {
        "total": len(materialized),
        "by_status": dict(sorted(Counter(row["automated_status"] for row in materialized).items())),
        "by_leakage_stratum": dict(sorted(Counter(row["leakage_stratum"] for row in materialized).items())),
        "explicit_hard_negative_queries": sum(bool(row.get("hard_negative") or row.get("hard_negatives")) for row in materialized),
    }


def build_audit(text: Mapping[str, Any], multimodal: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for name, manifest, minimum in (("text", text, 30), ("multimodal", multimodal, 20)):
        rows = manifest.get("labels", ())
        ids = [row.get("query_id") for row in rows]
        if len(ids) != len(set(ids)): errors.append(f"{name}: duplicate query IDs")
        if any(row.get("automated_status") not in STATUSES for row in rows): errors.append(f"{name}: invalid status")
        if any(row.get("leakage_stratum") not in LEAKAGE_STRATA for row in rows): errors.append(f"{name}: invalid leakage stratum")
        if any(row.get("original_relevant_ids") != row.get("final_relevant_ids") for row in rows):
            errors.append(f"{name}: relevant IDs changed")
        if manifest.get("counts", {}).get("explicit_hard_negative_queries", 0) < minimum:
            errors.append(f"{name}: fewer than {minimum} hard-negative queries")
    payload = {
        **_header("AUTOMATED_LABEL_AUDIT", _digest([text.get("source_manifest_sha256"), multimodal.get("source_manifest_sha256")])),
        "valid": not errors, "errors": errors,
        "text": {"labels_sha256": text.get("labels_sha256"), "counts": text.get("counts")},
        "multimodal": {"labels_sha256": multimodal.get("labels_sha256"), "counts": multimodal.get("counts")},
        "human_reviewer_identity": None, "human_review_claimed": False,
        "determinism": "canonical sorted JSON, stable query ordering, fixed rule version",
    }
    payload["audit_sha256"] = _digest(payload)
    return payload
