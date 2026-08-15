#!/usr/bin/env python3
"""Derive and evaluate Issue #86 reranked text arms from immutable scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics


EXPECTED_PAIRS = 16_948
ARM_BASELINES = {"T02": "T00", "T03": "T01", "T06": "T04", "T07": "T05"}
DEPTHS = (5, 8, 10, 20, 30, 50)
METRIC_DEPTHS = (1, 3, 5, 8, 10)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def stable_pair_id(query_id: str, chunk_id: str) -> str:
    return sha256(canonical([query_id, chunk_id])).hexdigest()


def load_scores(path: Path) -> tuple[dict, dict[tuple[str, str], dict]]:
    with path.open() as stream:
        manifest = json.loads(stream.readline())
        rows = [json.loads(line) for line in stream]
    if manifest.get("record_type") != "manifest" or manifest.get("expected_pairs") != EXPECTED_PAIRS:
        raise ValueError("score manifest contract mismatch")
    pairs: dict[tuple[str, str], dict] = {}
    pair_ids: set[str] = set()
    for row in rows:
        key = (row.get("query_id"), row.get("chunk_id"))
        if row.get("record_type") != "score" or key in pairs or row.get("pair_id") in pair_ids:
            raise ValueError("malformed or duplicate score record")
        if not isinstance(row.get("score"), (int, float)) or not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("nonfinite score")
        if row.get("pair_id") != stable_pair_id(*key):
            raise ValueError("unstable pair ID")
        if row.get("ordered_input_sha256") != manifest.get("ordered_pairs_sha256"):
            raise ValueError("ordered input digest mismatch")
        if row.get("model_revision") != manifest.get("revision"):
            raise ValueError("model revision mismatch")
        if row.get("runtime") != manifest.get("runtime") or row.get("artifact_hashes") != manifest.get("artifact_hashes"):
            raise ValueError("runtime or artifact provenance mismatch")
        pairs[key] = row
        pair_ids.add(row["pair_id"])
    if len(rows) != EXPECTED_PAIRS or len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"score coverage mismatch: {len(rows)}/{len(pairs)}")
    return manifest, pairs


def rerank(first_stage: list[str], query_id: str, scores: dict[tuple[str, str], dict], depth: int) -> list[str]:
    candidates = first_stage[:depth]
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"duplicate first-stage candidate for {query_id}")
    missing = [chunk_id for chunk_id in candidates if (query_id, chunk_id) not in scores]
    if missing:
        raise ValueError(f"missing reranker score for {query_id}: {missing[0]}")
    original = {chunk_id: index for index, chunk_id in enumerate(candidates)}
    ordered = sorted(candidates, key=lambda chunk_id: (-scores[(query_id, chunk_id)]["score"], original[chunk_id], chunk_id))
    return ordered + first_stage[depth:]


def query_metrics(ranking: list[str], relevant_ids: list[str], negative_id: str | None) -> dict[str, float]:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("query has no relevant IDs")
    output: dict[str, float] = {}
    for depth in METRIC_DEPTHS:
        selected = ranking[:depth]
        output[f"recall_at_{depth}"] = len(relevant.intersection(selected)) / len(relevant)
        output[f"precision_at_{depth}"] = len(relevant.intersection(selected)) / depth
    for depth in (5, 10):
        first = next((rank for rank, item in enumerate(ranking[:depth], 1) if item in relevant), None)
        output[f"mrr_at_{depth}"] = 0.0 if first is None else 1.0 / first
        dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranking[:depth], 1) if item in relevant)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(depth, len(relevant)) + 1))
        output[f"ndcg_at_{depth}"] = dcg / idcg
    output["query_coverage"] = float(bool(ranking))
    output["zero_hit_rate"] = float(not relevant.intersection(ranking[:10]))
    for depth in (1, 3, 5):
        output[f"hard_negative_fp_at_{depth}"] = float(bool(negative_id and negative_id in ranking[:depth]))
    return output


def aggregate(per_query: dict[str, dict[str, float]], query_ids: list[str]) -> dict:
    if not query_ids:
        return {"query_count": 0, "metrics": {}}
    keys = sorted(next(iter(per_query.values())))
    return {"query_count": len(query_ids), "metrics": {
        key: sum(per_query[qid][key] for qid in query_ids) / len(query_ids) for key in keys
    }}


def bootstrap(baseline: dict[str, float], candidate: dict[str, float], samples: int = 2000, seed: int = 86) -> dict:
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired bootstrap inputs differ")
    deltas = [candidate[qid] - baseline[qid] for qid in sorted(baseline)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    return {"mean_delta": statistics.fmean(deltas), "ci95_low": means[int(samples * .025)],
            "ci95_high": means[min(samples - 1, int(samples * .975))], "query_count": len(deltas),
            "samples": samples, "seed": seed,
            "interpretation": "SUPPORTED" if means[int(samples * .025)] > 0 or means[min(samples - 1, int(samples * .975))] < 0 else "INCONCLUSIVE"}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def reconstruct_batch_telemetry(rows: list[dict]) -> dict:
    """Use one repeated batch latency per contiguous emitted batch."""
    batches: list[tuple[int, float]] = []
    index = 0
    while index < len(rows):
        size = int(rows[index]["effective_batch_size"])
        selected = rows[index:index + size]
        latency = float(rows[index]["batch_elapsed_seconds"])
        if len(selected) != size or any(int(row["effective_batch_size"]) != size or
                                        float(row["batch_elapsed_seconds"]) != latency for row in selected):
            raise ValueError(f"cannot reconstruct emitted batch at record {index}")
        batches.append((size, latency))
        index += size
    output = {}
    for size in (1, 2, 4):
        latencies = [latency for batch_size, latency in batches if batch_size == size]
        output[str(size)] = {"batch_count": len(latencies), "pair_count": len(latencies) * size,
                             "pairs_per_second_service_time": (len(latencies) * size / sum(latencies)) if latencies else None,
                             "batch_latency_seconds_p50": percentile(latencies, .50) if latencies else None,
                             "batch_latency_seconds_p95": percentile(latencies, .95) if latencies else None,
                             "batch_latency_seconds_p99": percentile(latencies, .99) if latencies else None}
    return {"by_effective_batch_size": output, "batch_count": len(batches),
            "limitations": "GPU service-time reconstruction only; resumed run_elapsed values do not provide wall-clock queue/request latency or concurrency telemetry."}


def strata_for(labels: list[dict]) -> dict[str, list[str]]:
    strata: dict[str, list[str]] = defaultdict(list)
    for item in labels:
        qid = item["query_id"]
        strata[item["leakage_stratum"]].append(qid)
        for category in item.get("categories", []):
            strata[category].append(qid)
        if item.get("hard_negative"):
            strata["EXPLICIT_HARD_NEGATIVE"].append(qid)
        language = item.get("language")
        if language:
            strata[language].append(qid)
    return {key: sorted(set(value)) for key, value in sorted(strata.items())}


def rag_metrics(ranking: list[str], label: dict, chunks: dict[str, dict], depth: int) -> dict:
    context = ranking[:depth]
    relevant = set(label["final_relevant_ids"])
    negative = label.get("hard_negative", {}).get("chunk_id")
    sources = [chunks[item]["source_id"] for item in context]
    tokens = sum(len(chunks[item]["text"].split()) for item in context)
    hits = len(relevant.intersection(context))
    return {"context_recall": hits / len(relevant), "context_precision": hits / depth,
            "answerable_with_context": float(hits > 0), "relevant_evidence_retained": float(hits > 0),
            "explicit_hard_negative_present": float(bool(negative and negative in context)),
            "duplicate_evidence_rate": 1 - len(set(context)) / len(context),
            "source_diversity": len(set(sources)) / len(sources), "context_token_estimate": tokens,
            "context_token_utilization_vs_8192_estimate": min(tokens / 8192, 1.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rag-output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    labels_doc = json.loads(args.labels.read_text())
    corpus = json.loads(args.corpus.read_text())
    manifest, scores = load_scores(args.scores)
    labels = labels_doc["labels"]
    label_map = {item["query_id"]: item for item in labels}
    query_ids = sorted(label_map)
    chunks = {item["chunk_id"]: item for item in corpus["chunks"]}
    if set(query_ids) != set(next(iter(report["arms"].values()))["rankings"]):
        raise ValueError("query coverage mismatch")
    expected_keys = {(qid, chunk_id) for arm in ("T00", "T01", "T04", "T05")
                     for qid, ranking in report["arms"][arm]["rankings"].items() for chunk_id in ranking}
    if expected_keys != set(scores):
        raise ValueError("complete query/candidate score coverage mismatch")

    rankings_by_depth: dict[str, dict[str, dict[str, list[str]]]] = {}
    rankings: dict[str, dict[str, list[str]]] = {arm: report["arms"][arm]["rankings"] for arm in ("T00", "T01", "T04", "T05")}
    for arm, baseline in ARM_BASELINES.items():
        rankings_by_depth[arm] = {}
        for depth in DEPTHS:
            rankings_by_depth[arm][str(depth)] = {
                qid: rerank(rankings[baseline][qid], qid, scores, depth) for qid in query_ids
            }
        rankings[arm] = rankings_by_depth[arm]["50"]

    per_arm = {}
    strata = strata_for(labels)
    arms = {}
    for arm in sorted(rankings):
        per = {qid: query_metrics(rankings[arm][qid], label_map[qid]["final_relevant_ids"],
                                  label_map[qid].get("hard_negative", {}).get("chunk_id")) for qid in query_ids}
        per_arm[arm] = per
        arms[arm] = {"status": "EXECUTED", "baseline": ARM_BASELINES.get(arm),
                     "metrics": {"OVERALL": aggregate(per, query_ids), **{name: aggregate(per, ids) for name, ids in strata.items()}},
                     "rankings": rankings[arm]}

    comparisons = {}
    for candidate, baseline in ARM_BASELINES.items():
        key = f"{candidate}_minus_{baseline}"
        comparisons[key] = {}
        for metric in ("recall_at_5", "recall_at_10", "mrr_at_5", "mrr_at_10", "ndcg_at_5", "ndcg_at_10", "hard_negative_fp_at_1", "hard_negative_fp_at_5"):
            comparisons[key][metric] = bootstrap({q: per_arm[baseline][q][metric] for q in query_ids},
                                                  {q: per_arm[candidate][q][metric] for q in query_ids})

    depth_analysis = {}
    depth_paired_bootstrap = {}
    disagreements = {}
    for arm, baseline in ARM_BASELINES.items():
        depth_analysis[arm] = {}
        depth_per_all = {}
        for depth in DEPTHS:
            depth_per = {qid: query_metrics(rankings_by_depth[arm][str(depth)][qid], label_map[qid]["final_relevant_ids"],
                                             label_map[qid].get("hard_negative", {}).get("chunk_id")) for qid in query_ids}
            depth_per_all[depth] = depth_per
            depth_analysis[arm][str(depth)] = aggregate(depth_per, query_ids)
        depth_paired_bootstrap[arm] = {}
        for depth in DEPTHS[:-1]:
            depth_paired_bootstrap[arm][f"{depth}_minus_50"] = {
                metric: bootstrap({q: depth_per_all[50][q][metric] for q in query_ids},
                                  {q: depth_per_all[depth][q][metric] for q in query_ids})
                for metric in ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
            }
        rows = []
        counts = Counter()
        for qid in query_ids:
            relevant = set(label_map[qid]["final_relevant_ids"])
            left = next((i for i, x in enumerate(rankings[baseline][qid][:10], 1) if x in relevant), None)
            right = next((i for i, x in enumerate(rankings[arm][qid][:10], 1) if x in relevant), None)
            outcome = "tie" if left == right else "win" if right is not None and (left is None or right < left) else "loss"
            counts[outcome] += 1
            rows.append({"query_id": qid, "baseline_rank": left, "reranked_rank": right, "outcome": outcome,
                         "rank_displacement": None if left is None or right is None else left - right,
                         "leakage_stratum": label_map[qid]["leakage_stratum"], "categories": label_map[qid].get("categories", [])})
        disagreements[arm] = {"counts": dict(sorted(counts.items())), "queries": rows}

    score_rows = list(scores.values())
    result = {"schema_version": 1, "evaluation_mode": "AUTOMATED_NON_HUMAN_REVIEWED",
              "decision_grade_production_validation": False, "canonical_arm_mapping": ARM_BASELINES,
              "arm_mapping_resolution": "Preserved repository and immutable-score canonical IDs: T02=T00+r, T03=T01+r, T06=T04+r, T07=T05+r; the latest phase prose transposed T03 and T06 descriptions.",
              "score_integrity": {"expected_pairs": EXPECTED_PAIRS, "completed_pairs": len(scores), "duplicates": 0,
                                  "missing": 0, "nonfinite": 0, "score_file_sha256": digest(args.scores),
                                  "ordered_pairs_sha256": manifest["ordered_pairs_sha256"], "manifest": manifest,
                                  "oom_retries": max(row["oom_count"] for row in score_rows),
                                  "batch_distribution": dict(sorted(Counter(str(row["effective_batch_size"]) for row in score_rows).items()))},
              "scoring_telemetry": reconstruct_batch_telemetry(score_rows),
              "source_digests": {"report_sha256": digest(args.report), "labels_sha256": digest(args.labels), "corpus_sha256": digest(args.corpus)},
              "arms": arms, "paired_bootstrap": comparisons, "depth_analysis": depth_analysis,
              "depth_paired_bootstrap": depth_paired_bootstrap,
              "disagreement_analysis": disagreements, "production_decision": "NOT_AUTHORIZED",
              "production": "UNCHANGED", "deployments": 0, "merges": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")

    rag_arms = {}
    for arm in sorted(rankings):
        rag_arms[arm] = {}
        for depth in (5, 8, 10):
            per = {qid: rag_metrics(rankings[arm][qid], label_map[qid], chunks, depth) for qid in query_ids}
            rag_arms[arm][str(depth)] = aggregate(per, query_ids)
    rag = {"evaluation_mode": "AUTOMATED_NON_HUMAN_REVIEWED", "decision_grade_production_validation": False,
           "result_kind": "RETRIEVAL_ONLY_RAG_PRECHECK_NOT_GENERATED_ANSWERS", "generator_used": False,
           "judge_used": False, "generated_answer_metrics": "NOT_EXECUTED_NO_PINNED_GENERATOR_JUDGE_CONTRACT",
           "source_text_report_sha256": sha256(args.output.read_bytes()).hexdigest(), "arms": rag_arms,
           "production_decision": "NOT_AUTHORIZED", "production": "UNCHANGED", "deployments": 0, "merges": 0}
    args.rag_output.write_bytes(json.dumps(rag, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")


if __name__ == "__main__":
    main()
