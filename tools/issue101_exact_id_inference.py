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
EMBEDDING_SEPARATOR = "<#issue101-single-sequence#>"
PREPROCESSING_ID = "issue101-qwen3-official-single-sequence-v1"
EMBEDDING_SPACE_ID = "issue101-d4-official-single-sequence-native-2560-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# Keep this after sha256_bytes is defined for import-time clarity.
EMBEDDING_SEPARATOR_SHA256 = sha256_bytes(EMBEDDING_SEPARATOR.encode("utf-8"))


def source_digest(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def render_prompt(group: str, text: str) -> str:
    """Render the frozen official contract without changing source bytes."""
    if group == "queries":
        return f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text}"
    if group == "documents":
        return text
    raise ValueError(f"unknown input group: {group}")


def freeze_identity_manifest(root: pathlib.Path) -> dict[str, Any]:
    ordered = json.loads((root / "inputs" / "ordered-inputs.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "schema_version": 2,
        "preprocessing_id": PREPROCESSING_ID,
        "embedding_space_id": EMBEDDING_SPACE_ID,
        "template_id": TEMPLATE_ID,
        "template_digest": TEMPLATE_DIGEST,
        "encoding": "UTF-8",
        "source_bytes_preserved": True,
        "runtime": RUNTIME,
        "model_sha256": MODEL_DIGEST,
        "quantization": "Q4_K_M",
        "pooling": "llama.cpp embedding sequence output",
        "native_dimension": NATIVE_DIMENSION,
        "normalization": "runtime output recorded; derived prefixes truncate then L2-normalize",
        "transport": {"no_escape": True, "embedding_separator": EMBEDDING_SEPARATOR,
                      "embedding_separator_sha256": EMBEDDING_SEPARATOR_SHA256},
        "gpu_layers": 27,
        "batch_size": 1,
    }
    for group in ("documents", "queries"):
        rows = ordered.get(group)
        if not isinstance(rows, list) or len(rows) not in (715, 190):
            raise ValueError(f"{group} manifest count invalid")
        output = []
        for index, item in enumerate(rows):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("text"), str):
                raise ValueError(f"{group}[{index}] missing stable id/text")
            text = item["text"]
            prompt = render_prompt(group, text)
            if EMBEDDING_SEPARATOR in text or EMBEDDING_SEPARATOR in prompt:
                raise ValueError(f"separator collision in {group}[{index}]")
            output.append({"sequence_index": index, "id": item["id"],
                          "source_sha256": source_digest(text),
                          "rendered_prompt_sha256": source_digest(prompt),
                          "source_utf8_bytes": len(text.encode("utf-8")),
                          "rendered_prompt_utf8_bytes": len(prompt.encode("utf-8"))})
        ids = [row["id"] for row in output]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {group} IDs")
        result[group] = output
        result[f"{group}_manifest_sha256"] = sha256_bytes(json.dumps(output, sort_keys=True, separators=(",", ":")).encode())
    result["manifest_sha256"] = sha256_bytes(json.dumps(result, sort_keys=True, separators=(",", ":")).encode())
    return result


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


def parse_single_json_stdout(stdout: str) -> Any:
    """Accept exactly one JSON document; logs belong on stderr."""
    decoder = json.JSONDecoder()
    leading = len(stdout) - len(stdout.lstrip())
    if leading == len(stdout):
        raise ValueError("empty JSON stdout")
    payload, end = decoder.raw_decode(stdout, leading)
    if stdout[end:].strip():
        raise ValueError("stdout contains multiple JSON documents or log contamination")
    return payload


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
    with tempfile.TemporaryDirectory(prefix="issue101-one-") as temp:
        stdout_path = pathlib.Path(temp) / "stdout.json"
        stderr_path = pathlib.Path(temp) / "stderr.log"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            completed = subprocess.run([
            str(binary), "-m", str(model), "--device", "CUDA0", "--gpu-layers", str(gpu_layers),
            "--fit", "off", "--parallel", "1", "--ctx-size", "1024", "--batch-size", "1",
            "--ubatch-size", "1", "--no-escape", "--embd-separator", EMBEDDING_SEPARATOR,
            "--embd-output-format", "json", "-n", "1", "-p", text,
            ], stdout=stdout_file, stderr=stderr_file,
                env={**os.environ, "LC_ALL": "C", "LANG": "C"})
        stdout_bytes = stdout_path.read_bytes()
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"embedding process failed rc={completed.returncode}: {stderr_text[-4000:]}")
    try:
        payload = parse_single_json_stdout(stdout_bytes.decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid single-sequence stdout contract: {error}; stderr={stderr_text[-4000:]}") from error
    return validate_output(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path)
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--group", choices=("documents", "queries"), required=True)
    parser.add_argument("--gpu-layers", type=int, default=27)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    root, group = args.root, args.group
    if args.freeze:
        atomic_write(root / "manifests" / "issue101-single-sequence-inputs.json", freeze_identity_manifest(root))
        return
    if args.binary is None or args.model is None:
        parser.error("--binary and --model are required for inference")
    identity = json.loads((root / "manifests" / "issue101-single-sequence-inputs.json").read_text(encoding="utf-8"))
    ordered = json.loads((root / "inputs" / "ordered-inputs.json").read_text(encoding="utf-8"))[group]
    prompts = [render_prompt(group, item["text"]) for item in ordered]
    manifest_data = identity[group]
    output = root / "artifacts" / "d4-official-single-sequence" / f"{group}-native-2560.json"
    state = json.loads(output.read_text()) if output.exists() else {
        "object": "list", "data": [], "provenance": {"group": group, "gpu_layers": args.gpu_layers,
        "batch_size": 1, "template_id": TEMPLATE_ID, "template_digest": TEMPLATE_DIGEST,
        "model_sha256": MODEL_DIGEST, "runtime": RUNTIME, "native_dimension": NATIVE_DIMENSION,
        "preprocessing_id": identity["preprocessing_id"], "embedding_space_id": identity["embedding_space_id"],
        "manifest_sha256": identity["manifest_sha256"], "transport": identity["transport"]}}
    accepted = {record["id"]: record for record in state["data"]}
    for expected in manifest_data:
        current = accepted.get(expected["id"])
        if current is not None:
            validate_vector(current.get("embedding"))
            if current.get("input_sha256", current.get("source_sha256")) not in (expected["source_sha256"],):
                raise ValueError(f"input digest mismatch for {expected['id']}")
            continue
        vector = run_one(args.binary, args.model, prompts[expected["sequence_index"]], args.gpu_layers)
        accepted[expected["id"]] = enrich({"id": expected["id"], "embedding": vector},
                                           {"id": expected["id"], "sequence_index": expected["sequence_index"],
                                            "input_sha256": expected["source_sha256"]},
                                           {"gpu_layers": args.gpu_layers})
        accepted[expected["id"]].update({"source_sha256": expected["source_sha256"],
                                         "rendered_prompt_sha256": expected["rendered_prompt_sha256"],
                                         "preprocessing_id": identity["preprocessing_id"],
                                         "embedding_space_id": identity["embedding_space_id"]})
        state["data"] = [accepted[key] for key in (r["id"] for r in manifest_data) if key in accepted]
        state["provenance"].update({"gpu_layers": args.gpu_layers, "batch_size": 1,
                                     "manifest_sha256": identity["manifest_sha256"],
                                     "count": len(state["data"])})
        atomic_write(output, state)
        print(f"{group}: {len(state['data'])}/{len(manifest_data)}", flush=True)


if __name__ == "__main__":
    main()
