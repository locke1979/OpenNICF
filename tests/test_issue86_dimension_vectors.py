import importlib.util
import math
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "tools" / "run_issue86_dimension_vectors.py"
SPEC = importlib.util.spec_from_file_location("dimension_vectors", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_supported_dimensions_preserve_native_contracts():
    assert MODULE.SUPPORTED == {"D06": {768, 1024}, "D4": {768, 1024, 1536, 2560}}
    assert 2560 not in MODULE.SUPPORTED["D06"]


def test_prefix_is_truncated_then_normalized():
    result = MODULE.normalized_prefix([3.0, 4.0, 100.0], 2)
    assert result == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(x*x for x in result)) == pytest.approx(1.0)


def test_prefix_rejects_padding_nonfinite_and_zero():
    with pytest.raises(RuntimeError, match="shorter"):
        MODULE.normalized_prefix([1.0], 2)
    with pytest.raises(RuntimeError, match="nonfinite"):
        MODULE.normalized_prefix([float("nan")], 1)
    with pytest.raises(RuntimeError, match="zero"):
        MODULE.normalized_prefix([0.0], 1)


def test_stable_digest_binds_order():
    assert MODULE.stable_digest(["a", "b"]) != MODULE.stable_digest(["b", "a"])
