"""Model-free reporting contracts for issue #86 automated experiments.

This module neither executes models nor mutates retrieval state.  It makes the
requested experiment matrix, metric aggregation, confidence intervals and
non-production claim boundaries deterministic and testable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Mapping, Sequence


EVALUATION_MODE = "AUTOMATED_NON_HUMAN_REVIEWED"
REQUIRED_WARNINGS = (
    "AUTOMATED LABELS MAY CONTAIN SOURCE-DERIVED LEAKAGE",
    "QUALITY RESULTS ARE COMPARATIVE EXPERIMENTAL EVIDENCE",
    "PRODUCTION REMAINS UNCHANGED",
)
LEAKAGE_STRATA = ("LOW_LEAKAGE", "MEDIUM_LEAKAGE", "HIGH_LEAKAGE", "EXACT_IDENTIFIER")
CANDIDATE_DEPTHS = (10, 20, 30, 50)
CONTEXT_DEPTHS = (5, 8, 10)
CONCURRENCY_LEVELS = (1, 2, 4)


@dataclass(frozen=True)
class ExperimentArm:
    arm_id: str
    modalities: tuple[str, ...]
    dense_space: str | None
    lexical_symbol_aware: bool = False
    fusion: str | None = None
    reranker: str | None = None
    modality_aware_reranking: bool = False


TEXT_ARMS = (
    ExperimentArm("T00", ("text",), "TEXT_QWEN3_06B_768_AUTO_V1"),
    ExperimentArm("T01", ("text",), "TEXT_QWEN3_4B_Q4_768_AUTO_V1"),
    ExperimentArm("T02", ("text",), "TEXT_QWEN3_06B_768_AUTO_V1", reranker="QWEN3_RERANKER_06B"),
    ExperimentArm("T03", ("text",), "TEXT_QWEN3_4B_Q4_768_AUTO_V1", reranker="QWEN3_RERANKER_06B"),
    ExperimentArm("T04", ("text",), "TEXT_QWEN3_06B_768_AUTO_V1", True, "RRF"),
    ExperimentArm("T05", ("text",), "TEXT_QWEN3_4B_Q4_768_AUTO_V1", True, "RRF"),
    ExperimentArm("T06", ("text",), "TEXT_QWEN3_06B_768_AUTO_V1", True, "RRF", "QWEN3_RERANKER_06B"),
    ExperimentArm("T07", ("text",), "TEXT_QWEN3_4B_Q4_768_AUTO_V1", True, "RRF", "QWEN3_RERANKER_06B"),
)
MULTIMODAL_ARMS = (
    ExperimentArm("M00", ("text", "image", "mixed"), "QWEN3_VL_EMBEDDING_2B"),
    ExperimentArm("M01", ("text", "image", "mixed"), "QWEN3_VL_EMBEDDING_2B", reranker="QWEN3_VL_RERANKER_2B", modality_aware_reranking=True),
    ExperimentArm("M02", ("text", "image", "mixed"), "BEST_TEXT_PLUS_QWEN3_VL_2B", fusion="RRF"),
    ExperimentArm("M03", ("text", "image", "mixed"), "BEST_TEXT_PLUS_QWEN3_VL_2B", fusion="RRF", reranker="MODALITY_AWARE", modality_aware_reranking=True),
    ExperimentArm("M04", ("text", "image", "mixed"), "BEST_06B_TEXT_PLUS_QWEN3_VL_2B", fusion="RRF", reranker="MODALITY_AWARE", modality_aware_reranking=True),
    ExperimentArm("M05", ("text", "image", "mixed"), "BEST_4B_TEXT_PLUS_QWEN3_VL_2B", fusion="RRF", reranker="MODALITY_AWARE", modality_aware_reranking=True),
)


def per_query_retrieval_metrics(
    ranking: Sequence[str], relevant_ids: Iterable[str], highly_relevant_ids: Iterable[str] = (),
    hard_negative_ids: Iterable[str] = (), *, recall_depths: Sequence[int] = (1, 3, 5, 10, 20, 30, 50),
) -> dict[str, float]:
    """Calculate one query's complete retrieval contract."""
    relevant, highly, negatives = set(relevant_ids), set(highly_relevant_ids), set(hard_negative_ids)
    if not relevant:
        raise ValueError("at least one relevant ID is required")
    metrics: dict[str, float] = {}
    for depth in recall_depths:
        if depth <= 0:
            raise ValueError("metric depths must be positive")
        metrics[f"recall_at_{depth}"] = float(bool(relevant.intersection(ranking[:depth])))
    first = next((rank for rank, item in enumerate(ranking[:10], 1) if item in relevant), None)
    metrics["mrr_at_10"] = 0.0 if first is None else 1.0 / first
    metrics["relevant_at_rank_1"] = float(bool(ranking and ranking[0] in relevant))
    metrics["zero_hit_rate"] = float(not relevant.intersection(ranking[:50]))
    for depth in (1, 3, 5):
        metrics[f"hard_negative_fp_at_{depth}"] = float(bool(negatives.intersection(ranking[:depth])))
    for depth in (5, 10):
        gains = [2 if item in highly else 1 if item in relevant else 0 for item in ranking[:depth]]
        dcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        ideal = sorted([2] * len(highly) + [1] * len(relevant - highly), reverse=True)[:depth]
        idcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
        metrics[f"ndcg_at_{depth}"] = dcg / idcg if idcg else 0.0
    return metrics


def aggregate_metrics(per_query: Mapping[str, Mapping[str, float]], strata: Mapping[str, Iterable[str]]) -> dict[str, dict[str, Any]]:
    """Aggregate metrics with an explicit query count for every output row."""
    if not per_query:
        raise ValueError("at least one query is required")
    output: dict[str, dict[str, Any]] = {}
    requested = {"OVERALL": tuple(sorted(per_query)), **{name: tuple(sorted(ids)) for name, ids in strata.items()}}
    for name, query_ids in requested.items():
        if any(query_id not in per_query for query_id in query_ids):
            raise ValueError(f"stratum {name} contains an unknown query ID")
        if not query_ids:
            output[name] = {"query_count": 0, "metrics": {}}
            continue
        keys = set(per_query[query_ids[0]])
        if any(set(per_query[item]) != keys for item in query_ids):
            raise ValueError("per-query metric keys must match")
        output[name] = {"query_count": len(query_ids), "metrics": {
            key: sum(float(per_query[item][key]) for item in query_ids) / len(query_ids) for key in sorted(keys)
        }}
    return output


def paired_bootstrap_interval(
    baseline: Mapping[str, float], candidate: Mapping[str, float], *, samples: int = 2000, seed: int = 86,
) -> dict[str, float | int]:
    """Return a deterministic percentile interval over paired query deltas."""
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("paired inputs must have identical non-empty query IDs")
    if samples <= 0:
        raise ValueError("samples must be positive")
    deltas = [float(candidate[key]) - float(baseline[key]) for key in sorted(baseline)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    return {"mean_delta": sum(deltas) / len(deltas), "ci95_low": means[int(samples * .025)],
            "ci95_high": means[min(samples - 1, int(samples * .975))], "query_count": len(deltas),
            "samples": samples, "seed": seed}


def validate_experimental_report(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Fail closed on missing disclaimers, counts, safety or evaluation stages."""
    errors: list[str] = []
    if report.get("evaluation_mode") != EVALUATION_MODE:
        errors.append("evaluation mode must be automated non-human-reviewed")
    if report.get("human_review_required_for_this_run") is not False:
        errors.append("human review must be declared false for this run")
    if report.get("decision_grade_production_validation") is not False:
        errors.append("automated results cannot be decision-grade production validation")
    if report.get("production_migration_authorized") is not False:
        errors.append("production migration must remain unauthorized")
    if report.get("production") != "UNCHANGED" or report.get("deployments") != 0 or report.get("merges") != 0:
        errors.append("production, deployment and merge safety declarations are invalid")
    warnings = set(report.get("warnings", ()))
    if not set(REQUIRED_WARNINGS).issubset(warnings):
        errors.append("all automated-evidence warnings are required")
    for table in report.get("metric_tables", ()):
        if not isinstance(table.get("query_count"), int) or table["query_count"] < 0:
            errors.append("every metric table requires a non-negative automated-label query count")
    rag = report.get("rag")
    if rag is not None and rag.get("query_count", 0) < 50:
        errors.append("RAG evaluation requires at least 50 automated-label questions")
    if rag is not None and rag.get("judge_used") and rag.get("judge_label") != "AUTOMATED_JUDGE_NON_HUMAN_VALIDATED":
        errors.append("judge-based RAG scores require the automated-judge label")
    for sample in report.get("operational_samples", ()):
        if sample.get("concurrency") not in CONCURRENCY_LEVELS:
            errors.append("operational concurrency must be 1, 2 or 4")
    return tuple(errors)
