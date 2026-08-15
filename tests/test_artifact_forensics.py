import math
import struct

import pytest

from opennicf.artifact_forensics import GGUFError, inspect_gguf, validate_vector_artifact


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def test_inspect_gguf_reads_metadata_and_tensor_directory_without_execution(tmp_path):
    metadata = (
        _string("general.name") + struct.pack("<I", 8) + _string("Synthetic Embedder")
        + _string("embedding_length") + struct.pack("<I", 4) + struct.pack("<I", 768)
    )
    tensor = _string("weight") + struct.pack("<IQQIQ", 2, 8, 8, 12, 0)
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 2) + metadata + tensor + b"payload")
    result = inspect_gguf(path)
    assert result["metadata"]["general.name"] == "Synthetic Embedder"
    assert result["metadata"]["embedding_length"] == 768
    assert result["tensor_count"] == 1
    assert result["tensor_types"] == {"Q4_K": 1}


def test_inspect_gguf_rejects_truncation(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
    with pytest.raises(GGUFError, match="truncated"):
        inspect_gguf(path)


def test_vector_validation_fails_closed_on_missing_identity_and_ids():
    artifact = {
        "space_id": "S", "model": "M", "dimension": 2, "normalized": True,
        "document_vectors": [[1.0, 0.0]], "query_vectors": [[math.nan, 0.0]],
    }
    result = validate_vector_artifact(artifact, ["c1"], ["q1"])
    assert result["safe_to_append"] is False
    assert result["nonfinite_count"] == 1
    assert "model_revision" in result["missing_provenance"]
    assert "document IDs are not embedded" in result["failures"]
