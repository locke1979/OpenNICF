import math
from pathlib import Path

import pytest

from opennicf.issue86_runtime import mrl_normalize, query_text, require_artifact


def test_artifact_gate_rejects_wrong_digest_and_accepts_exact(tmp_path: Path):
    artifact = tmp_path / "weight"
    artifact.write_bytes(b"trusted")
    with pytest.raises(ValueError, match="digest"):
        require_artifact(artifact, "0" * 64, 7)
    from hashlib import sha256
    assert require_artifact(artifact, sha256(b"trusted").hexdigest(), 7) == artifact


def test_mrl_is_prefix_then_normalization():
    vector = mrl_normalize((3, 4, 100), 2)
    assert vector == pytest.approx((0.6, 0.8))
    assert math.isclose(sum(value * value for value in vector), 1.0)


def test_nonfinite_and_zero_vectors_fail_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        mrl_normalize((float("nan"), 1), 2)
    with pytest.raises(ValueError, match="zero norm"):
        mrl_normalize((0, 0), 2)


def test_official_query_template_is_stable():
    assert query_text("Where is the runbook?") == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:Where is the runbook?"
    )
