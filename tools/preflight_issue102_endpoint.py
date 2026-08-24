"""Read-only contract preflight for the shared OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any


TEXT_MODEL = "text-embedding-qwen3-embedding-4b"
VL_MODEL = "qwen.qwen3-vl-embedding-2b"
ENDPOINT = "http://192.168.1.137:1234"


def request(base_url: str, path: str, *, payload: dict[str, Any] | None, token: str | None, timeout: float) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"error_type": type(exc).__name__}
        return exc.code, body
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, {"error_type": type(exc).__name__}


def vector(result: dict[str, Any]) -> list[float]:
    data = result.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("expected one embedding result")
    values = data[0].get("embedding")
    if not isinstance(values, list) or not values:
        raise ValueError("missing embedding vector")
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite embedding vector")
    if not any(value != 0.0 for value in values):
        raise ValueError("zero embedding vector")
    return values


def digest(values: list[float]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("OPENNICF_EMBEDDING_BASE_URL", ENDPOINT))
    parser.add_argument("--service-token", default=os.environ.get("OPENNICF_EMBEDDING_SERVICE_TOKEN"))
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    report: dict[str, Any] = {
        "schema_version": 1,
        "endpoint": args.base_url,
        "text_model": TEXT_MODEL,
        "vl_model": VL_MODEL,
        "service_token_used": bool(args.service_token),
        "endpoint_ram": "UNVERIFIED",
        "production_enabled": False,
        "evaluation_vl_enabled": False,
    }

    status, models = request(args.base_url, "/v1/models", payload=None, token=args.service_token, timeout=args.timeout)
    advertised = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)] if isinstance(models, dict) else []
    report["models"] = {"http_status": status, "advertised": advertised, "text_exact": TEXT_MODEL in advertised, "vl_exact": VL_MODEL in advertised}

    probes: dict[str, Any] = {}
    text_payload = {"model": TEXT_MODEL, "input": ["OpenNICF deterministic text probe"]}
    first_status, first = request(args.base_url, "/v1/embeddings", payload=text_payload, token=args.service_token, timeout=args.timeout)
    second_status, second = request(args.base_url, "/v1/embeddings", payload=text_payload, token=args.service_token, timeout=args.timeout)
    try:
        first_vector = vector(first)
        second_vector = vector(second)
        probes["text"] = {"http_status": first_status, "response_model": first.get("model"), "dimension": len(first_vector), "finite": True, "deterministic": first_vector == second_vector, "digest": digest(first_vector), "contract_pass": first.get("model") == TEXT_MODEL and len(first_vector) == 2560 and first_vector == second_vector}
    except (TypeError, ValueError) as exc:
        probes["text"] = {"http_status": first_status, "error": str(exc), "contract_pass": False}

    batch_status, batch = request(args.base_url, "/v1/embeddings", payload={"model": TEXT_MODEL, "input": ["probe A", "probe B"]}, token=args.service_token, timeout=args.timeout)
    probes["text_batch"] = {"http_status": batch_status, "count": len(batch.get("data", [])) if isinstance(batch, dict) and isinstance(batch.get("data"), list) else None, "ordered": [item.get("index") for item in batch.get("data", [])] == [0, 1] if isinstance(batch, dict) and isinstance(batch.get("data"), list) else False}

    invalid_status, invalid = request(args.base_url, "/v1/embeddings", payload={"model": "unconfigured-model", "input": ["probe"]}, token=args.service_token, timeout=args.timeout)
    report["invalid_model_rejected"] = invalid_status >= 400 or invalid.get("model") != "unconfigured-model"

    vl_status, vl = request(args.base_url, "/v1/embeddings", payload={"model": VL_MODEL, "input": ["OpenNICF VL identity probe"]}, token=args.service_token, timeout=args.timeout)
    vl_model = vl.get("model") if isinstance(vl, dict) else None
    probes["vl"] = {"http_status": vl_status, "response_model": vl_model, "status": "VL_UNVERIFIED" if vl_model == VL_MODEL else "VL_ALIASED_OR_UNSUPPORTED", "multimodal_probe": "NOT_RUN"}
    report["probes"] = probes
    report["vl_status"] = probes["vl"]["status"]
    report["text_endpoint_usable"] = bool(probes["text"].get("contract_pass"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
