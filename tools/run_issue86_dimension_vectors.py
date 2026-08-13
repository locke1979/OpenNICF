#!/usr/bin/env python3
"""Build immutable Issue #86 dimension-diagnostic spaces from native embeddings."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile


EXPECTED_DOCUMENTS = 715
EXPECTED_QUERIES = 190
SUPPORTED = {"D06": {768, 1024}, "D4": {768, 1024, 1536, 2560}}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as target:
            json.dump(value, target, separators=(",", ":"))
            target.flush(); os.fsync(target.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def normalized_prefix(values: list[float], dimension: int) -> list[float]:
    if len(values) < dimension:
        raise RuntimeError("native vector shorter than requested dimension")
    vector = [float(item) for item in values[:dimension]]
    if not all(math.isfinite(item) for item in vector):
        raise RuntimeError("nonfinite vector")
    norm = math.sqrt(sum(item * item for item in vector))
    if not norm:
        raise RuntimeError("zero vector")
    return [item / norm for item in vector]


def ordered_rows(path: Path, group: str) -> list[dict]:
    rows = json.loads(path.read_text())[group]
    expected = EXPECTED_DOCUMENTS if group == "documents" else EXPECTED_QUERIES
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise RuntimeError(f"invalid {group} identities")
    return rows


def import_native(args: argparse.Namespace) -> None:
    if args.dimension not in SUPPORTED[args.family]:
        raise RuntimeError("UNSUPPORTED_DIMENSION")
    raw = json.loads(args.raw.read_text())
    output: dict[str, object] = {}
    for group in ("documents", "queries"):
        rows = ordered_rows(args.inputs, group)
        values = raw[group]
        if len(values) != len(rows):
            raise RuntimeError(f"{group} count mismatch")
        built = []
        for expected, item in zip(rows, values):
            if item["id"] != expected["id"]:
                raise RuntimeError(f"{group} ordered ID mismatch")
            built.append({"id": item["id"], "vector": normalized_prefix(item["vector"], args.dimension)})
        output[group] = built
    provenance = dict(raw["provenance"])
    native = 1024 if args.family == "D06" else 2560
    if provenance.get("native_dimension") != native:
        raise RuntimeError("native runtime dimension mismatch")
    if provenance.get("run_id") != args.run_id or provenance.get("raw_output_sha256") not in (None, digest(args.raw)):
        raise RuntimeError("execution provenance mismatch")
    if provenance.get("status") != "EXECUTED_CUDA" or provenance.get("cpu_offload") not in (False, "NONE"):
        raise RuntimeError("CUDA_REQUIRED_CPU_OFFLOAD_DETECTED")
    binding = {
        "family": args.family, "model_revision": provenance["revision"],
        "artifact_sha256": provenance["weight_sha256"], "precision": provenance["precision"],
        "runtime_revision": provenance["runtime"], "pooling": provenance["pooling"],
        "template_digest": provenance["template_digest"], "dimension": args.dimension,
        "truncation": "native_prefix", "normalization": "L2_after_truncation",
        "inputs_sha256": digest(args.inputs),
        "corpus_sha256": digest(args.corpus), "labels_sha256": digest(args.labels),
        "ordered_document_ids_sha256": stable_digest([row["id"] for row in ordered_rows(args.inputs, "documents")]),
        "ordered_query_ids_sha256": stable_digest([row["id"] for row in ordered_rows(args.inputs, "queries")]),
        "run_id": args.run_id, "raw_execution_sha256": digest(args.raw),
    }
    canonical = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    provenance.update(binding)
    provenance["embedding_space_id"] = f"ISSUE86_{args.family}_{args.dimension}_{sha256(canonical).hexdigest()[:16]}"
    output["provenance"] = provenance
    atomic_json(args.output, output)


def validate(args: argparse.Namespace) -> None:
    data = json.loads(args.space.read_text())
    dimension = int(data["provenance"]["dimension"])
    for group, count in (("documents", EXPECTED_DOCUMENTS), ("queries", EXPECTED_QUERIES)):
        rows = data[group]
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise RuntimeError(f"invalid {group} coverage")
        for row in rows:
            vector = row["vector"]
            if len(vector) != dimension or not all(math.isfinite(x) for x in vector):
                raise RuntimeError("invalid vector")
            if abs(math.sqrt(sum(x*x for x in vector)) - 1.0) > 1e-5:
                raise RuntimeError("vector is not normalized")
    print(json.dumps({"status": "PASS", "sha256": digest(args.space), "dimension": dimension}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-native")
    imp.add_argument("--family", choices=sorted(SUPPORTED), required=True)
    imp.add_argument("--dimension", type=int, required=True)
    imp.add_argument("--inputs", type=Path, required=True)
    imp.add_argument("--raw", type=Path, required=True)
    imp.add_argument("--corpus", type=Path, required=True)
    imp.add_argument("--labels", type=Path, required=True)
    imp.add_argument("--run-id", required=True)
    imp.add_argument("--output", type=Path, required=True)
    val = sub.add_parser("validate")
    val.add_argument("--space", type=Path, required=True)
    args = parser.parse_args()
    {"import-native": import_native, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
