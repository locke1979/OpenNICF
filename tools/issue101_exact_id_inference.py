"""Fail-closed, one-input-at-a-time issue 101 4B inference wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import tempfile
from typing import Any


NATIVE_DIMENSION = 2560
TEMPLATE_ID = "template_qwen3_official"
TEMPLATE_DIGEST = "dffc8f23590ad610c08971e79b3bf8245e9ce3895b46f4b097a5fdbce92bfa39"
MODEL_DIGEST = "2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b"
RUNTIME = "llama.cpp@a4a4c51f3d40e086b59b73b631b5c43c8fbf4504"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_digest(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def vector_digest(vector: list[float]) -> str:
    return sha256_bytes(json.dumps(vector, separators=(",", ":")).encode())


def manifest(root: pathlib.Path) -> dict[str, Any]:
    ordered = json.loads((root / "inputs" / "ordered-inputs.json").read_text())
    result: dict[str, Any] = {"schema_version": 1, "template_id": TEMPLATE_ID,
                              "template_digest": TEMPLATE_DIGEST, "model_sha256": MODEL_DIGEST,
                              "runtime": RUNTIME, "native_dimension": NATIVE_DIMENSION,
                              "gpu_layers": 29, "batch_size": 1}
    for group in ("documents", "queries"):
        prompts = (root / "inputs" / f"{group}-4b.txt").read_text(encoding="utf-8").splitlines()
        if len(prompts) != len(ordered[group]):
            raise ValueError(f"{group} prompt/ID count mismatch")
        result[group] = [{"sequence_index": i, "id": item["id"],
                          "input_sha256": source_digest(prompts[i])}
                         for i, item in enumerate(ordered[group])]
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["manifest_sha256"] = sha256_bytes(payload)
    return result


def validate_vector(vector: Any, dimension: int = NATIVE_DIMENSION) -> None:
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError(f"expected vector dimension {dimension}")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector):
        raise ValueError("vector contains nonfinite or nonnumeric values")


def validate_output(payload: Any, expected_id: str | None = None) -> list[float]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("embedding output is not an object with data")
    if len(payload["data"]) != 1:
        raise ValueError(f"expected exactly one embedding, got {len(payload['data'])}")
    record = payload["data"][0]
    if not isinstance(record, dict):
        raise ValueError("embedding record is not an object")
    if expected_id is not None and record.get("id") not in (None, expected_id):
        raise ValueError("embedding output ID does not match input ID")
    vector = record.get("embedding")
    validate_vector(vector)
    return vector


def enrich(record: dict[str, Any], expected: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    validate_vector(record.get("embedding"))
    if record.get("id") != expected["id"]:
        raise ValueError("record ID mismatch")
    return {"id": expected["id"], "sequence_index": expected["sequence_index"],
            "input_sha256": expected["input_sha256"], "template_id": TEMPLATE_ID,
            "template_digest": TEMPLATE_DIGEST, "model_sha256": MODEL_DIGEST,
            "runtime": RUNTIME, "gpu_layers": provenance.get("gpu_layers"),
            "dimension": NATIVE_DIMENSION, "vector_sha256": vector_digest(record["embedding"]),
            "embedding": record["embedding"]}


def atomic_write(path: pathlib.Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    os.replace(temporary, path)


def run_one(binary: pathlib.Path, model: pathlib.Path, text: str, gpu_layers: int) -> list[float]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=model.parent, delete=False) as handle:
        handle.write(text + "\n")
        input_path = pathlib.Path(handle.name)
    try:
        completed = subprocess.run([
            str(binary), "-m", str(model), "--device", "CUDA0", "--gpu-layers", str(gpu_layers),
            "--fit", "off", "--parallel", "1", "--ctx-size", "1024", "--batch-size", "1",
            "--ubatch-size", "1", "--no-escape", "--embd-output-format", "json", "-n", "1", "-f", str(input_path),
        ], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C", "LANG": "C"})
        text_output = completed.stdout if "{" in completed.stdout else completed.stderr
        if "{" not in text_output:
            text_output = completed.stdout + "\n" + completed.stderr
        start, end = text_output.find("{"), text_output.rfind("}")
        if completed.returncode or start < 0 or end < start:
            raise RuntimeError(f"embedding process failed rc={completed.returncode}: {completed.stderr[-4000:]}")
        return validate_output(json.loads(text_output[start:end + 1]))
    finally:
        input_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--group", choices=("documents", "queries"), required=True)
    parser.add_argument("--gpu-layers", type=int, default=29)
    args = parser.parse_args()
    root, group = args.root, args.group
    ordered = json.loads((root / "inputs" / "ordered-inputs.json").read_text())[group]
    prompts = (root / "inputs" / f"{group}-4b.txt").read_text(encoding="utf-8").splitlines()
    manifest_data = manifest(root)[group]
    output = root / "artifacts" / f"D4-OFFICIAL-{group}-raw.json"
    state = json.loads(output.read_text()) if output.exists() else {
        "object": "list", "data": [], "provenance": {"group": group, "gpu_layers": args.gpu_layers,
        "batch_size": 1, "template_id": TEMPLATE_ID, "template_digest": TEMPLATE_DIGEST,
        "model_sha256": MODEL_DIGEST, "runtime": RUNTIME, "native_dimension": NATIVE_DIMENSION}}
    accepted = {record["id"]: record for record in state["data"]}
    for expected in manifest_data:
        current = accepted.get(expected["id"])
        if current is not None:
            validate_vector(current.get("embedding"))
            if current.get("input_sha256") not in (None, expected["input_sha256"]):
                raise ValueError(f"input digest mismatch for {expected['id']}")
            continue
        vector = run_one(args.binary, args.model, prompts[expected["sequence_index"]], args.gpu_layers)
        accepted[expected["id"]] = enrich({"id": expected["id"], "embedding": vector}, expected,
                                           {"gpu_layers": args.gpu_layers})
        state["data"] = [accepted[key] for key in (r["id"] for r in manifest_data) if key in accepted]
        state["provenance"].update({"gpu_layers": args.gpu_layers, "batch_size": 1,
                                     "manifest_sha256": manifest(root)["manifest_sha256"],
                                     "count": len(state["data"])})
        atomic_write(output, state)
        print(f"{group}: {len(state['data'])}/{len(manifest_data)}", flush=True)


if __name__ == "__main__":
    main()
