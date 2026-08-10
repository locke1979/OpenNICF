"""Small provider-neutral HTTP service for the local embedding worker."""

from __future__ import annotations

import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .knowledge.embeddings import (
    HashingEmbeddingBackend,
    LocalFirstEmbeddingService,
    QwenEmbeddingBackend,
)

MODEL = os.environ.get("OPENNICF_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DIMENSION = int(os.environ.get("OPENNICF_EMBEDDING_DIMENSION", "768"))
SERVICE_TOKEN = os.environ.get("OPENNICF_EMBEDDING_SERVICE_TOKEN")


def _service() -> LocalFirstEmbeddingService:
    qwen = QwenEmbeddingBackend(
        dimension=DIMENSION,
        model=MODEL,
        max_input_length=int(
            os.environ.get("OPENNICF_EMBEDDING_MAX_INPUT_TOKENS", "512")
        ),
        batch_size=int(os.environ.get("OPENNICF_EMBEDDING_BATCH_SIZE", "2")),
    )
    cpu = HashingEmbeddingBackend(model="opennicf-cpu-fallback", dimensions=DIMENSION)
    return LocalFirstEmbeddingService(
        preferred_backend=qwen,
        cpu_backend=cpu,
        active_space_id=qwen.info.embedding_space_id,
    )


SERVICE = _service()


class Handler(BaseHTTPRequestHandler):
    server_version = "OpenNICFEmbedding/1"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return (
            not SERVICE_TOKEN
            or self.headers.get("Authorization") == f"Bearer {SERVICE_TOKEN}"
        )

    def do_GET(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path == "/health":
            info = SERVICE.info()
            self._json(
                200,
                {
                    "status": "ok",
                    "provider": "qwen",
                    "model": info.model,
                    "dimension": info.dimensions,
                    "device": info.device,
                    "normalized": True,
                    "fallback": info.fallback,
                    "embedding_space_id": info.embedding_space_id,
                },
            )
        elif self.path == "/v1/models":
            info = SERVICE.info()
            self._json(
                200,
                {
                    "data": [
                        {
                            "id": info.model,
                            "provider": "qwen",
                            "dimension": info.dimensions,
                            "embedding_space_id": info.embedding_space_id,
                        }
                    ]
                },
            )
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path != "/v1/embeddings":
            self._json(404, {"error": "not_found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            texts = request.get("input")
            purpose = request.get("purpose", "retrieval_document")
            dimension = int(request.get("dimension", DIMENSION))
            if not isinstance(texts, list) or not all(
                isinstance(text, str) for text in texts
            ):
                raise ValueError("input must be a list of strings")
            if dimension != DIMENSION:
                raise ValueError("only the configured embedding dimension is supported")
            result = SERVICE.embed(texts, purpose=purpose, dimension=dimension)
            self._json(
                200,
                {
                    "object": "list",
                    "model": result.model,
                    "dimension": result.dimensions,
                    "embedding_space_id": result.embedding_space_id,
                    "normalized": True,
                    "fallback": result.fallback,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": list(vector)}
                        for i, vector in enumerate(result.vectors)
                    ],
                },
            )
        except Exception as exc:  # noqa: BLE001 - API errors must not leak request/runtime data
            self._json(
                400, {"error": "embedding_request_failed", "detail": str(exc)[:160]}
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        # Do not log request text or Authorization headers.
        print(
            f"embedding-service {self.command} {self.path} {args[1] if len(args) > 1 else ''}",
            flush=True,
        )


def main() -> int:
    host = os.environ.get("OPENNICF_EMBEDDING_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENNICF_EMBEDDING_PORT", "8080"))
    if host == "0.0.0.0":
        raise SystemExit(
            "refusing unrestricted bind; set OPENNICF_EMBEDDING_HOST to the Tailscale address"
        )
    server = ThreadingHTTPServer((host, port), Handler)
    print(
        f"embedding worker listening on {host}:{port} hostname={socket.gethostname()}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
