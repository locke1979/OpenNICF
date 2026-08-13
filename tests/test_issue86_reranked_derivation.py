import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from derive_issue86_reranked_text import bootstrap, load_scores, query_metrics, rerank, stable_pair_id


def test_rerank_is_stable_and_candidate_bounded():
    ranking = ["a", "b", "c", "d"]
    scores = {("q", "a"): {"score": .2}, ("q", "b"): {"score": .9}, ("q", "c"): {"score": .9}}
    assert rerank(ranking, "q", scores, 3) == ["b", "c", "a", "d"]
    assert rerank(ranking, "q", scores, 2) == ["b", "a", "c", "d"]


def test_metrics_and_bootstrap_are_deterministic():
    metrics = query_metrics(["x", "r", "n"], ["r"], "n")
    assert metrics["recall_at_1"] == 0
    assert metrics["recall_at_3"] == 1
    assert metrics["precision_at_5"] == .2
    assert metrics["mrr_at_5"] == .5
    first = bootstrap({"a": 0, "b": 1}, {"a": 1, "b": 1}, samples=100, seed=4)
    assert first == bootstrap({"a": 0, "b": 1}, {"a": 1, "b": 1}, samples=100, seed=4)


def test_score_loader_fails_closed_on_nonfinite(tmp_path):
    path = tmp_path / "scores.jsonl"
    manifest = {"record_type": "manifest", "expected_pairs": 16948, "ordered_pairs_sha256": "x",
                "revision": "r", "runtime": {}, "artifact_hashes": {}}
    row = {"record_type": "score", "pair_id": stable_pair_id("q", "c"), "query_id": "q", "chunk_id": "c", "score": math.nan,
           "ordered_input_sha256": "x", "model_revision": "r", "runtime": {}, "artifact_hashes": {}}
    path.write_text(json.dumps(manifest) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="nonfinite"):
        load_scores(path)
