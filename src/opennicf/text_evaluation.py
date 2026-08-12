"""Fail-closed preparation helpers for the issue #85/#89 text evaluation.

The functions in this module are model independent.  They audit immutable
evaluation inputs, prepare deterministic human-review queues and calculate
paired retrieval evidence without treating machine suggestions as labels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence


FINAL_REVIEW_STATUSES = {"ACCEPT", "ACCEPT_LEGACY", "REWRITE", "REJECT"}
HUMAN_ACCEPT_STATUSES = {"ACCEPT", "ACCEPT_LEGACY"}
REVIEW_DECISIONS = {"ACCEPT", "REWRITE", "REJECT"}
REQUIRED_METRICS = (
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "recall_at_20", "mrr_at_10", "ndcg_at_5", "ndcg_at_10", "zero_hit_rate",
)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.casefold())


def leakage_indicators(query: str, relevant_texts: Iterable[str]) -> list[dict[str, Any]]:
    """Return conservative mechanical indicators, never a review decision."""
    q_tokens = _normal_tokens(query)
    indicators: list[dict[str, Any]] = []
    if not q_tokens:
        return indicators
    normalized_query = " ".join(q_tokens)
    for text in relevant_texts:
        text_tokens = _normal_tokens(text)
        normalized_text = " ".join(text_tokens)
        if len(normalized_query) >= 24 and normalized_query in normalized_text:
            indicators.append({"kind": "exact_query_in_relevant_text", "token_count": len(q_tokens)})
            continue
        best: list[str] = []
        max_n = min(12, len(q_tokens))
        for width in range(max_n, 4, -1):
            match = next(
                (q_tokens[start:start + width] for start in range(len(q_tokens) - width + 1)
                 if " ".join(q_tokens[start:start + width]) in normalized_text),
                None,
            )
            if match:
                best = match
                break
        if best:
            indicators.append({"kind": "copied_distinctive_phrase", "token_count": len(best), "phrase": " ".join(best)})
    unique = {json.dumps(item, sort_keys=True): item for item in indicators}
    return [unique[key] for key in sorted(unique)]


def suggested_rewrite(query: str, indicators: Sequence[Mapping[str, Any]]) -> str | None:
    """Supply a review aid while leaving its status PENDING_REVIEW."""
    if not indicators:
        return None
    tokens = _normal_tokens(query)
    if len(tokens) < 5:
        return None
    return "Explain the evidence needed for: " + " ".join(tokens[: min(14, len(tokens))])


def hard_negative_candidates(
    query: str,
    relevant_ids: Sequence[str],
    chunks: Mapping[str, Mapping[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Find deterministic lexical confounders; candidates still need review."""
    query_tokens = set(_normal_tokens(query))
    relevant = set(relevant_ids)
    scored: list[tuple[int, str, Mapping[str, Any]]] = []
    for chunk_id, chunk in chunks.items():
        if chunk_id in relevant:
            continue
        overlap = len(query_tokens & set(_normal_tokens(str(chunk.get("text", "")))))
        same_source = int(any(
            chunks[item].get("source_path") == chunk.get("source_path")
            for item in relevant if item in chunks
        ))
        score = overlap * 2 + same_source
        if score:
            scored.append((score, chunk_id, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"chunk_id": chunk_id, "source_path": chunk.get("source_path"), "text": chunk.get("text"),
         "mechanical_overlap_score": score, "review_status": "PENDING_REVIEW"}
        for score, chunk_id, chunk in scored[:limit]
    ]


@dataclass(frozen=True)
class AuditResult:
    summary: dict[str, Any]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def audit_inputs(
    corpus: Mapping[str, Any],
    gold: Mapping[str, Any],
    review: Mapping[str, Any],
    vectors: Mapping[str, Mapping[str, Any]],
    *,
    corpus_digest: str,
    gold_digest: str,
    expected_corpus_digest: str | None = None,
) -> AuditResult:
    chunks = corpus.get("chunks", [])
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    chunk_id_set = set(chunk_ids)
    queries = gold.get("queries", [])
    review_by_id = {item.get("query_id"): item for item in review.get("queries", [])}
    statuses = Counter(
        str(review_by_id.get(query.get("query_id"), {}).get("review_status", "PENDING_REVIEW"))
        for query in queries
    )
    missing: dict[str, list[str]] = {}
    for query in queries:
        refs = set(query.get("relevant_chunk_ids", ()))
        refs.update(query.get("highly_relevant_chunk_ids", ()))
        refs.update(query.get("hard_negative_chunk_ids", ()))
        unresolved = sorted(refs - chunk_id_set)
        if unresolved:
            missing[str(query.get("query_id"))] = unresolved

    errors: list[str] = []
    if len(chunk_ids) != len(chunk_id_set):
        errors.append("corpus chunk IDs are not unique")
    if expected_corpus_digest and corpus_digest != expected_corpus_digest:
        errors.append("corpus digest does not match the owning manifest")
    embedded_digest = gold.get("corpus_manifest_sha256")
    if embedded_digest and embedded_digest != corpus_digest:
        errors.append("gold-set corpus digest does not match the corpus")
    if missing:
        errors.append("one or more query references do not resolve")
    if len(review_by_id) != len(queries):
        errors.append("review queue and gold-set query IDs differ")

    vector_summary: dict[str, Any] = {}
    for name, artifact in sorted(vectors.items()):
        document_vectors = artifact.get("document_vectors", [])
        query_vectors = artifact.get("query_vectors", [])
        dimension = artifact.get("dimension")
        dimensions_valid = all(len(vector) == dimension for vector in document_vectors + query_vectors)
        vector_summary[name] = {
            "space_id": artifact.get("space_id"),
            "model": artifact.get("model"),
            "dimension": dimension,
            "normalized": artifact.get("normalized"),
            "document_vector_count": len(document_vectors),
            "query_vector_count": len(query_vectors),
            "complete_for_corpus": len(document_vectors) == len(chunks),
            "complete_for_queries": len(query_vectors) == len(queries),
            "dimensions_valid": dimensions_valid,
        }
        if not dimensions_valid:
            errors.append(f"{name} contains vectors with the wrong dimension")

    typed_hard_negatives = sum(query.get("query_type") == "hard_negative" for query in queries)
    labelled_hard_negatives = sum(bool(query.get("hard_negative_chunk_ids")) for query in queries)
    explicit_hard_negatives = sum(
        query.get("query_type") == "hard_negative" or bool(query.get("hard_negative_chunk_ids")) for query in queries
    )
    accepted = sum(statuses[status] for status in HUMAN_ACCEPT_STATUSES)
    normalized_statuses = {
        "ACCEPT": accepted,
        "REWRITE": statuses["REWRITE"],
        "REJECT": statuses["REJECT"],
        "PENDING_REVIEW": len(queries) - accepted - statuses["REWRITE"] - statuses["REJECT"],
    }
    summary = {
        "corpus": {"chunks": len(chunks), "unique_chunk_ids": len(chunk_id_set), "sha256": corpus_digest},
        "gold": {"queries": len(queries), "sha256": gold_digest, "review_status_counts": dict(sorted(statuses.items())),
                 "normalized_decision_counts": normalized_statuses},
        "references": {"unresolved_query_count": len(missing), "unresolved": missing},
        "hard_negative_query_count": explicit_hard_negatives,
        "hard_negative_typed_query_count": typed_hard_negatives,
        "hard_negative_labelled_query_count": labelled_hard_negatives,
        "human_accepted_query_count": accepted,
        "vectors": vector_summary,
        "decision_grade_paired_metrics_executable": (
            accepted >= 150
            and labelled_hard_negatives > 0
            and not missing
            and len(vector_summary) >= 2
            and all(item["complete_for_corpus"] and item["complete_for_queries"] for item in vector_summary.values())
        ),
    }
    return AuditResult(summary, tuple(errors))


def build_review_artifact(
    corpus: Mapping[str, Any],
    gold: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    batch_size: int = 25,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunks = {chunk["chunk_id"]: chunk for chunk in corpus.get("chunks", [])}
    review_by_id = {item["query_id"]: item for item in review.get("queries", [])}
    query_text_counts = Counter(
        " ".join(_normal_tokens(query.get("query_text", ""))) for query in gold.get("queries", [])
    )
    items: list[dict[str, Any]] = []
    for query in sorted(gold.get("queries", []), key=lambda item: item["query_id"]):
        source_review = review_by_id.get(query["query_id"], {})
        current_status = source_review.get("review_status", "PENDING_REVIEW")
        relevant_ids = query.get("relevant_chunk_ids", [])
        hard_negative_ids = query.get("hard_negative_chunk_ids", [])
        relevant_texts = [chunks[item]["text"] for item in relevant_ids if item in chunks]
        indicators = leakage_indicators(query.get("query_text", ""), relevant_texts)
        normalized_query = " ".join(_normal_tokens(query.get("query_text", "")))
        if query_text_counts[normalized_query] > 1:
            indicators.append({"kind": "duplicate_query_text", "occurrences": query_text_counts[normalized_query]})
        explicit_negatives = [
            {"chunk_id": item, "source_path": chunks.get(item, {}).get("source_path"), "text": chunks.get(item, {}).get("text"),
             "review_status": "PENDING_REVIEW"}
            for item in hard_negative_ids
        ]
        candidate_negatives = hard_negative_candidates(query.get("query_text", ""), relevant_ids, chunks)
        items.append({
            "query_id": query["query_id"],
            "query_text": query.get("query_text"),
            "language": query.get("language"),
            "categories": query.get("categories", []),
            "intended_relevant": [
                {"chunk_id": item, "source_path": chunks.get(item, {}).get("source_path"), "text": chunks.get(item, {}).get("text")}
                for item in relevant_ids
            ],
            "hard_negatives": explicit_negatives,
            "hard_negative_candidates": candidate_negatives,
            "leakage_indicators": indicators,
            "machine_recommendation": "REWRITE" if indicators else "ACCEPT",
            "rewrite_candidate": suggested_rewrite(query.get("query_text", ""), indicators),
            "review_status": current_status if current_status in HUMAN_ACCEPT_STATUSES else "PENDING_REVIEW",
            "reviewer_decision": None,
            "reviewer_notes": None,
        })
    return {
        "schema_version": 1,
        "machine_review_is_not_human_review": True,
        "instructions": "A human reviewer must set reviewer_decision to ACCEPT, REWRITE, or REJECT. Machine recommendations remain PENDING_REVIEW.",
        "batch_size": batch_size,
        "batches": [
            {"batch_id": f"review-{offset // batch_size + 1:03d}", "items": items[offset:offset + batch_size]}
            for offset in range(0, len(items), batch_size)
        ],
    }


def apply_human_review_batches(
    gold: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    reviewer: str,
    reviewed_at: str,
) -> dict[str, Any]:
    """Validate and apply completed text review batches without changing IDs silently."""
    if not reviewer.strip() or not reviewed_at.strip():
        raise ValueError("reviewer and reviewed_at are required")
    source = {item["query_id"]: item for item in gold.get("queries", [])}
    if len(source) != len(gold.get("queries", [])):
        raise ValueError("gold query IDs must be unique")
    completed: dict[str, Mapping[str, Any]] = {}
    for batch in artifact.get("batches", []):
        for item in batch.get("items", []):
            query_id = item.get("query_id")
            if query_id in completed:
                raise ValueError(f"duplicate reviewed query ID: {query_id}")
            if query_id not in source:
                raise ValueError(f"unknown reviewed query ID: {query_id}")
            completed[str(query_id)] = item

    output: list[dict[str, Any]] = []
    for query_id in sorted(source):
        original = dict(source[query_id])
        review = completed.get(query_id)
        if review is None or review.get("reviewer_decision") is None:
            output.append(original)
            continue
        decision = str(review["reviewer_decision"])
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"invalid reviewer decision for {query_id}")
        notes = str(review.get("reviewer_notes") or "").strip()
        if not notes:
            raise ValueError(f"reviewer notes are required for {query_id}")
        expected_relevant = tuple(original.get("relevant_chunk_ids", ()))
        submitted_relevant = tuple(item.get("chunk_id") for item in review.get("intended_relevant", ()))
        if submitted_relevant != expected_relevant:
            raise ValueError(f"relevant IDs changed for {query_id}")
        allowed_negatives = {
            item.get("chunk_id") for key in ("hard_negatives", "hard_negative_candidates")
            for item in review.get(key, ())
        }
        selected_negatives = tuple(review.get("selected_hard_negative_ids", original.get("hard_negative_chunk_ids", ())))
        if any(item not in allowed_negatives for item in selected_negatives):
            raise ValueError(f"invalid hard-negative ID for {query_id}")
        rewritten = review.get("rewritten_query_text")
        if decision == "REWRITE" and not str(rewritten or "").strip():
            raise ValueError(f"rewritten query text is required for {query_id}")
        original.update({
            "original_query_text": original.get("query_text"),
            "query_text": str(rewritten).strip() if decision == "REWRITE" else original.get("query_text"),
            "hard_negative_chunk_ids": list(selected_negatives),
            "review_status": decision,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "review_notes": notes,
        })
        output.append(original)
    payload = {**gold, "queries": output, "review_source": reviewer, "reviewed_at": reviewed_at}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["reviewed_label_digest"] = sha256(canonical).hexdigest()
    return payload


def retrieval_metrics(rankings: Mapping[str, Sequence[str]], labels: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, float]:
    if set(rankings) != set(labels):
        raise ValueError("rankings and labels must contain identical query IDs")
    per_query = [_query_metrics(rankings[qid], labels[qid]) for qid in sorted(labels)]
    if not per_query:
        raise ValueError("at least one query is required")
    return {metric: sum(item[metric] for item in per_query) / len(per_query) for metric in REQUIRED_METRICS}


def _query_metrics(ranking: Sequence[str], label: Mapping[str, Sequence[str]]) -> dict[str, float]:
    relevant = set(label.get("relevant_chunk_ids", ()))
    highly = set(label.get("highly_relevant_chunk_ids", ()))
    if not relevant:
        raise ValueError("each query needs a relevant chunk")
    output: dict[str, float] = {}
    for cutoff in (1, 3, 5, 10, 20):
        output[f"recall_at_{cutoff}"] = float(bool(set(ranking[:cutoff]) & relevant))
    first = next((index for index, item in enumerate(ranking[:10], 1) if item in relevant), None)
    output["mrr_at_10"] = 0.0 if first is None else 1.0 / first
    for cutoff in (5, 10):
        gains = [2 if item in highly else 1 if item in relevant else 0 for item in ranking[:cutoff]]
        dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sorted(([2] * len(highly) + [1] * len(relevant - highly)), reverse=True)[:cutoff]
        idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        output[f"ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0
    output["zero_hit_rate"] = float(not bool(set(ranking[:20]) & relevant))
    return output


def paired_bootstrap(
    baseline: Mapping[str, float], candidate: Mapping[str, float], *, samples: int = 2000, seed: int = 85,
) -> dict[str, float]:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired inputs must have identical non-empty query IDs")
    if samples <= 0:
        raise ValueError("samples must be positive")
    query_ids = sorted(baseline)
    deltas = [candidate[item] - baseline[item] for item in query_ids]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    return {
        "mean_delta": sum(deltas) / len(deltas),
        "ci95_low": means[int(samples * 0.025)],
        "ci95_high": means[min(samples - 1, int(samples * 0.975))],
        "samples": samples,
        "seed": seed,
    }


def disagreement_analysis(
    baseline: Mapping[str, Sequence[str]], candidate: Mapping[str, Sequence[str]], labels: Mapping[str, Mapping[str, Sequence[str]]], *, cutoff: int = 10,
) -> list[dict[str, Any]]:
    if set(baseline) != set(candidate) or set(baseline) != set(labels):
        raise ValueError("all disagreement inputs must contain identical query IDs")
    rows = []
    for query_id in sorted(labels):
        relevant = set(labels[query_id].get("relevant_chunk_ids", ()))
        left = next((i for i, item in enumerate(baseline[query_id][:cutoff], 1) if item in relevant), None)
        right = next((i for i, item in enumerate(candidate[query_id][:cutoff], 1) if item in relevant), None)
        if left != right:
            rows.append({"query_id": query_id, "baseline_first_relevant_rank": left, "candidate_first_relevant_rank": right})
    return rows


def validate_executable_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if manifest.get("status") != "EXECUTABLE":
        errors.append("status must be EXECUTABLE")
    safety = manifest.get("safety", {})
    if safety.get("production_untouched") is not True or safety.get("deployment") is not False:
        errors.append("evaluation must be isolated and production-untouched")
    inputs = manifest.get("inputs", {})
    if inputs.get("human_accepted_queries", 0) < 150:
        errors.append("at least 150 genuinely human-accepted queries are required")
    if inputs.get("unresolved_references") != 0 or inputs.get("reviewed_hard_negative_queries", 0) <= 0:
        errors.append("complete reviewed hard negatives and resolved references are required")
    if not inputs.get("corpus_sha256") or not inputs.get("labels_sha256"):
        errors.append("corpus and label digests are required")
    spaces = manifest.get("embedding_spaces", [])
    if len(spaces) != 2 or any(space.get("document_vector_count") != 715 for space in spaces):
        errors.append("exactly two complete 715-document vector spaces are required")
    reranker = manifest.get("reranker", {})
    for key in ("model", "revision", "license", "artifact", "artifact_sha256", "runtime", "runtime_version"):
        if not reranker.get(key):
            errors.append(f"reranker.{key} is required")
    return tuple(errors)
