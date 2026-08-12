import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def test_retrieval_only_rag_prechecks_are_deterministic_and_non_generative():
    script = ROOT / "tools/build_issue86_retrieval_rag_prechecks.py"
    output = ROOT / "evaluation/results/issue86-cuda-2026-08-12/retrieval-rag-prechecks.json"
    subprocess.run([sys.executable, str(script)], check=True)
    first = output.read_bytes()
    subprocess.run([sys.executable, str(script)], check=True)
    assert output.read_bytes() == first
    report = json.loads(first)
    assert report["result_kind"] == "RETRIEVAL_ONLY_RAG_PRECHECK_NOT_GENERATED_ANSWERS"
    assert report["generator_used"] is report["judge_used"] is False
    assert report["decision_grade_production_validation"] is False
    assert sorted(report["arms"]) == ["T00", "T01", "T04", "T05"]
    for arm in report["arms"].values():
        assert sorted(arm, key=int) == ["5", "8", "10"]
        assert all(row["query_count"] == 190 for row in arm.values())
