"""Fail-closed GGUF and vector-artifact provenance inspection.

This module intentionally does not load or execute model tensors.  It reads the
GGUF directory and validates retained vector metadata so an unresolved artifact
cannot accidentally be treated as an approved evaluation arm.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Any, BinaryIO, Mapping, Sequence


GGUF_VALUE_TYPES = {
    0: ("uint8", "<B"), 1: ("int8", "<b"), 2: ("uint16", "<H"),
    3: ("int16", "<h"), 4: ("uint32", "<I"), 5: ("int32", "<i"),
    6: ("float32", "<f"), 7: ("bool", "<?"), 8: ("string", None),
    9: ("array", None), 10: ("uint64", "<Q"), 11: ("int64", "<q"),
    12: ("float64", "<d"),
}

# GGML tensor type identifiers used by current GGUF files. Unknown identifiers
# remain visible as TYPE_<n>, rather than being guessed.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8",
    33: "Q4_0_8_8", 34: "TQ1_0", 35: "TQ2_0",
}


class GGUFError(ValueError):
    pass


def _read_exact(source: BinaryIO, size: int) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise GGUFError("truncated GGUF directory")
    return value


def _unpack(source: BinaryIO, fmt: str) -> Any:
    return struct.unpack(fmt, _read_exact(source, struct.calcsize(fmt)))[0]


def _string(source: BinaryIO) -> str:
    size = _unpack(source, "<Q")
    if size > 64 * 1024 * 1024:
        raise GGUFError("unreasonably large GGUF string")
    return _read_exact(source, size).decode("utf-8", errors="replace")


def _value(source: BinaryIO, value_type: int) -> Any:
    if value_type not in GGUF_VALUE_TYPES:
        raise GGUFError(f"unknown GGUF metadata type {value_type}")
    name, fmt = GGUF_VALUE_TYPES[value_type]
    if fmt:
        return _unpack(source, fmt)
    if name == "string":
        return _string(source)
    element_type = _unpack(source, "<I")
    count = _unpack(source, "<Q")
    if count > 10_000_000:
        raise GGUFError("unreasonably large GGUF metadata array")
    return [_value(source, element_type) for _ in range(count)]


def inspect_gguf(path: str | Path) -> dict[str, Any]:
    """Read metadata and tensor directory without touching tensor payloads."""
    artifact = Path(path)
    digest = sha256()
    with artifact.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    with artifact.open("rb") as source:
        if _read_exact(source, 4) != b"GGUF":
            raise GGUFError("not a GGUF artifact")
        version = _unpack(source, "<I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = _unpack(source, "<Q")
        metadata_count = _unpack(source, "<Q")
        metadata: dict[str, Any] = {}
        for _ in range(metadata_count):
            key = _string(source)
            metadata[key] = _value(source, _unpack(source, "<I"))
        tensor_types: Counter[str] = Counter()
        for _ in range(tensor_count):
            _string(source)
            dimensions = _unpack(source, "<I")
            if dimensions > 16:
                raise GGUFError("unreasonable tensor rank")
            for _ in range(dimensions):
                _unpack(source, "<Q")
            tensor_type = _unpack(source, "<I")
            tensor_types[GGML_TYPES.get(tensor_type, f"TYPE_{tensor_type}")] += 1
            _unpack(source, "<Q")
    stat = artifact.stat()
    return {
        "path": str(artifact), "sha256": digest.hexdigest(), "size_bytes": stat.st_size,
        "mtime_utc": stat.st_mtime, "gguf_version": version,
        "metadata_count": metadata_count, "tensor_count": tensor_count,
        "tensor_types": dict(sorted(tensor_types.items())), "metadata": metadata,
    }


def validate_vector_artifact(
    artifact: Mapping[str, Any], ordered_document_ids: Sequence[str], ordered_query_ids: Sequence[str]
) -> dict[str, Any]:
    """Return objective vector diagnostics and provenance omissions."""
    dimension = artifact.get("dimension")
    documents = artifact.get("document_vectors", [])
    queries = artifact.get("query_vectors", [])
    failures: list[str] = []
    if not isinstance(dimension, int) or dimension <= 0:
        failures.append("invalid dimension")
    vectors = list(documents) + list(queries)
    if isinstance(dimension, int) and any(len(vector) != dimension for vector in vectors):
        failures.append("vector dimension mismatch")
    nonfinite = sum(not math.isfinite(float(value)) for vector in vectors for value in vector)
    if nonfinite:
        failures.append("nonfinite vectors present")
    embedded_document_ids = artifact.get("document_ids")
    embedded_query_ids = artifact.get("query_ids")
    if embedded_document_ids is None:
        failures.append("document IDs are not embedded")
    elif list(embedded_document_ids) != list(ordered_document_ids):
        failures.append("document ID ordering mismatch")
    if embedded_query_ids is None:
        failures.append("query IDs are not embedded")
    elif list(embedded_query_ids) != list(ordered_query_ids):
        failures.append("query ID ordering mismatch")
    required_provenance = (
        "model_revision", "artifact_sha256", "runtime", "runtime_revision",
        "preprocessing", "pooling", "projection",
    )
    missing = [key for key in required_provenance if not artifact.get(key)]
    if missing:
        failures.append("incomplete semantic-space provenance")
    payload = json.dumps(list(ordered_document_ids), ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "space_id": artifact.get("space_id"), "model": artifact.get("model"),
        "dimension": dimension, "normalized_declared": artifact.get("normalized"),
        "document_vector_count": len(documents), "query_vector_count": len(queries),
        "ordered_document_id_sha256": sha256(payload).hexdigest(),
        "nonfinite_count": nonfinite, "missing_provenance": missing,
        "failures": failures, "safe_to_append": not failures,
    }
