import pytest

from opennicf.experimental_reporting import (
    MULTIMODAL_ARMS, REQUIRED_WARNINGS, TEXT_ARMS, aggregate_metrics,
    paired_bootstrap_interval, per_query_retrieval_metrics, validate_experimental_report,
)


def test_requested_arm_matrix_is_exact_and_unique():
    assert [arm.arm_id for arm in TEXT_ARMS] == [f"T0{i}" for i in range(8)]
    assert [arm.arm_id for arm in MULTIMODAL_ARMS] == [f"M0{i}" for i in range(6)]
    assert all(arm.fusion == "RRF" for arm in TEXT_ARMS[4:])
    assert all(arm.modality_aware_reranking for arm in MULTIMODAL_ARMS if arm.arm_id in {"M01", "M03", "M04", "M05"})


def test_complete_metrics_include_long_depths_and_hard_negative_false_positives():
    metrics = per_query_retrieval_metrics(["n", "r"], ["r"], ["r"], ["n"])
    assert metrics["recall_at_1"] == 0 and metrics["recall_at_30"] == metrics["recall_at_50"] == 1
    assert metrics["mrr_at_10"] == .5 and metrics["hard_negative_fp_at_1"] == 1
    assert metrics["relevant_at_rank_1"] == 0 and metrics["zero_hit_rate"] == 0


def test_aggregation_always_carries_counts_and_rejects_unknown_stratum_members():
    per_query = {"q1": {"mrr": 1.0}, "q2": {"mrr": 0.0}}
    output = aggregate_metrics(per_query, {"LOW_LEAKAGE": ["q1"], "HARD_NEGATIVE": ["q2"]})
    assert output["OVERALL"] == {"query_count": 2, "metrics": {"mrr": .5}}
    assert output["LOW_LEAKAGE"]["query_count"] == 1
    with pytest.raises(ValueError, match="unknown query"):
        aggregate_metrics(per_query, {"LOW_LEAKAGE": ["missing"]})


def test_paired_bootstrap_is_deterministic_and_reports_pair_count():
    first = paired_bootstrap_interval({"a": 0, "b": 1}, {"a": 1, "b": 1}, samples=100, seed=4)
    assert first == paired_bootstrap_interval({"a": 0, "b": 1}, {"a": 1, "b": 1}, samples=100, seed=4)
    assert first["query_count"] == 2 and first["mean_delta"] == .5


def test_report_contract_enforces_non_production_claims_rag_and_operational_shape():
    report = {
        "evaluation_mode": "AUTOMATED_NON_HUMAN_REVIEWED",
        "human_review_required_for_this_run": False,
        "decision_grade_production_validation": False,
        "production_migration_authorized": False,
        "production": "UNCHANGED", "deployments": 0, "merges": 0,
        "warnings": list(REQUIRED_WARNINGS),
        "metric_tables": [{"query_count": 12}],
        "rag": {"query_count": 50, "judge_used": True,
                "judge_label": "AUTOMATED_JUDGE_NON_HUMAN_VALIDATED"},
        "operational_samples": [{"concurrency": 1}, {"concurrency": 2}, {"concurrency": 4}],
    }
    assert validate_experimental_report(report) == ()
    report["rag"]["query_count"] = 49
    report["production"] = "CHANGED"
    assert len(validate_experimental_report(report)) == 2
