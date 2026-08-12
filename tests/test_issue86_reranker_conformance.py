import json
from pathlib import Path


def test_reranker_conformance_is_not_misreported_as_completed_quality():
    path = Path(__file__).parents[1] / "docs/issue86-reranker-cuda-conformance-2026-08-12.json"
    report = json.loads(path.read_text())
    assert report["status"] == "CUDA_CONFORMANCE_PASS_FULL_EVALUATION_PENDING"
    assert report["synthetic_forward"]["status"] == "PASS"
    assert report["synthetic_forward"]["positive_score"] > report["synthetic_forward"]["negative_score"]
    assert report["cpu_offload_used"] is False
    assert all(report["cuda_residency"].values())
    assert report["capacity"]["batch_16"] == "BLOCKED_CUDA_OOM"
    assert set(report["arms"].values()) == {"PENDING_FULL_SCORING"}
