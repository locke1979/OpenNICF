#!/usr/bin/env python3
"""Build deterministic retrieval-only RAG prechecks from retained rankings.

This does not call a generator or judge and must never be reported as RAG answer
quality.  It records whether fixed context depths contain expected evidence and
explicit hard negatives for the already executed automated benchmark arms.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
LABELS = ROOT / "docs/automated-text-labels-v1.json"
RANKINGS = ROOT / "evaluation/results/issue86-cuda-2026-08-12/text-report.json"
OUTPUT = ROOT / "evaluation/results/issue86-cuda-2026-08-12/retrieval-rag-prechecks.json"
DEPTHS = (5, 8, 10)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    labels_doc = json.loads(LABELS.read_text())
    report = json.loads(RANKINGS.read_text())
    labels = {item["query_id"]: item for item in labels_doc["labels"]}
    arms = {}
    for arm_id in ("T00", "T01", "T04", "T05"):
        rankings = report["arms"][arm_id]["rankings"]
        depths = {}
        for depth in DEPTHS:
            contexts = []
            hits = negative_hits = 0
            for query_id in sorted(labels):
                label = labels[query_id]
                context_ids = rankings[query_id][:depth]
                relevant = set(label["final_relevant_ids"])
                negative = label.get("hard_negative", {}).get("chunk_id")
                evidence_hit = bool(relevant.intersection(context_ids))
                hard_negative_hit = bool(negative and negative in context_ids)
                hits += evidence_hit
                negative_hits += hard_negative_hit
                contexts.append({
                    "query_id": query_id,
                    "context_ids": context_ids,
                    "expected_evidence_present": evidence_hit,
                    "explicit_hard_negative_present": hard_negative_hit,
                })
            count = len(contexts)
            depths[str(depth)] = {
                "query_count": count,
                "correct_evidence_retrieved_rate": hits / count,
                "explicit_hard_negative_context_rate": negative_hits / count,
                "contexts": contexts,
            }
        arms[arm_id] = depths
    output = {
        "evaluation_mode": "AUTOMATED_NON_HUMAN_REVIEWED",
        "result_kind": "RETRIEVAL_ONLY_RAG_PRECHECK_NOT_GENERATED_ANSWERS",
        "human_review_required_for_this_run": False,
        "decision_grade_production_validation": False,
        "production_migration_authorized": False,
        "recommendation_scope": "EXPERIMENTAL_ARCHITECTURE_ONLY",
        "source_digests": {"labels_sha256": digest(LABELS), "rankings_sha256": digest(RANKINGS)},
        "generator_used": False,
        "judge_used": False,
        "rag_answer_quality_status": "BLOCKED_GENERATOR_JUDGE_BINDING_AND_INCOMPLETE_COMPARISON_SET",
        "arms": arms,
        "warnings": [
            "AUTOMATED LABELS MAY CONTAIN SOURCE-DERIVED LEAKAGE",
            "QUALITY RESULTS ARE COMPARATIVE EXPERIMENTAL EVIDENCE",
            "PRODUCTION REMAINS UNCHANGED",
        ],
        "production_decision": "NOT_AUTHORIZED",
        "production": "UNCHANGED",
        "deployments": 0,
        "merges": 0,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
