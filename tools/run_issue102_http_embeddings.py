"""Run isolated, checkpointed HTTP embedding probes for issue #102.

This runner never writes canonical issue #101 artifacts. It sends one UTF-8
input per request, records the raw vector digest, and stores only the requested
evaluation prefix after L2 normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile
import urllib.error
import urllib.request
from typing import Any


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_prefix(values: list[Any], dimension: int) -> list[float]:
    if not isinstance(values, list) or len(values) < dimension:
        raise ValueError(f"expected at least {dimension} finite vector values")
    prefix = [float(value) for value in values[:dimension]]
    if not all(math.isfinite(value) for value in prefix):
        raise ValueError("vector contains non-finite values")
    norm = math.sqrt(sum(value * value for value in prefix))
    if norm == 0:
        raise ValueError("vector prefix has zero norm")
    return [value / norm for value in prefix]


def request_embedding(base_url: str, model: str, text: str, timeout: float) -> tuple[list[float], dict[str, Any]]:
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/embeddings",
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"embedding request failed: {type(exc).__name__}") from exc
    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("embedding response was not JSON") from exc
    if not isinstance(result, dict) or result.get("fallback"):
        raise RuntimeError("embedding response indicates fallback or invalid payload")
    response_model = result.get("model")
    if response_model is not None and response_model != model:
        raise RuntimeError("embedding response model mismatch")
    data = result.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise RuntimeError("expected exactly one embedding result")
    if data[0].get("index") not in (None, 0):
        raise RuntimeError("embedding result index mismatch")
    vector = data[0].get("embedding")
    if not isinstance(vector, list):
        raise RuntimeError("embedding result has no vector")
    return vector, {"response_model": response_model, "response_dimension": result.get("dimension", result.get("dimensions")), "raw_vector_sha256": digest(json.dumps(vector, separators=(",", ":")).encode())}


def atomic_write(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def load_inputs(path: pathlib.Path, group: str) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(group)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"missing input group: {group}")
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str) for row in rows):
        raise ValueError(f"invalid input rows in {group}")
    return rows


def run_group(args: argparse.Namespace, config: dict[str, Any], model: str, group: str) -> pathlib.Path:
    model_config = config["models"][model]
    inputs = load_inputs(pathlib.Path(args.input_root) / config["input_manifest"], group)
    expected = config["expected_counts"].get(group)
    if expected is not None and expected != len(inputs):
        raise ValueError(f"{group} count mismatch: expected {expected}, got {len(inputs)}")
    preprocessing = config["preprocessing"]
    results_path = pathlib.Path(args.output_root) / model.replace("/", "_").replace(".", "_") / f"{group}-768.json"
    state = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {
        "schema_version": 1,
        "issue": 102,
        "model": model,
        "native_dimension_expected": model_config["native_dimension"],
        "evaluation_dimension": model_config["evaluation_dimension"],
        "group": group,
        "endpoint": args.base_url,
        "preprocessing": preprocessing,
        "data": [],
    }
    accepted = {row["id"]: row for row in state["data"]}
    for index, item in enumerate(inputs):
        if item["id"] in accepted:
            continue
        text = item["text"]
        prompt = text if group == "documents" else preprocessing["query_template"].format(text=text)
        vector, telemetry = request_embedding(args.base_url, model, prompt, args.timeout)
        if len(vector) != model_config["native_dimension"]:
            raise RuntimeError(f"{model} returned dimension {len(vector)}, expected {model_config['native_dimension']}")
        projected = normalize_prefix(vector, model_config["evaluation_dimension"])
        accepted[item["id"]] = {
            "id": item["id"],
            "sequence_index": index,
            "source_sha256": digest(text.encode("utf-8")),
            "prompt_sha256": digest(prompt.encode("utf-8")),
            "raw_vector_sha256": telemetry["raw_vector_sha256"],
            "dimension": model_config["evaluation_dimension"],
            "embedding": projected,
        }
        state["data"] = [accepted[row["id"]] for row in inputs if row["id"] in accepted]
        atomic_write(results_path, state)
    state["count"] = len(state["data"])
    state["complete"] = state["count"] == len(inputs)
    atomic_write(results_path, state)
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--input-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for model in args.model:
        if model not in config["models"]:
            raise SystemExit(f"model is not configured: {model}")
        for group in config["groups"]:
            print(run_group(args, config, model, group), flush=True)


if __name__ == "__main__":
    main()
