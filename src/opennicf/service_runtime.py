"""Bounded service health/runtime entrypoint for packaged OpenNICF services."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _database_check() -> tuple[bool, str]:
    dsn = (os.environ.get("OPENNICF_POSTGRES_DSN") or os.environ.get("DATABASE_URL", "")).strip()
    if not dsn:
        return False, "database_dsn_missing"
    try:
        import psycopg
        with psycopg.connect(dsn, connect_timeout=int(os.environ.get("DATABASE_CONNECT_TIMEOUT", "5"))) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                if cursor.fetchone() is None:
                    return False, "pgvector_missing"
        return True, "postgresql_pgvector_ok"
    except Exception as exc:  # noqa: BLE001 - readiness must not leak connection details
        return False, type(exc).__name__


def _object_store_check() -> tuple[bool, str]:
    root = os.environ.get("OPENNICF_OBJECT_STORE_ROOT", "/var/lib/opennicf/objects")
    try:
        os.makedirs(root, mode=0o750, exist_ok=True)
        ok = os.access(root, os.W_OK)
        return ok, "filesystem_object_store_ok" if ok else "object_store_not_writable"
    except OSError as exc:
        return False, type(exc).__name__


def checks(service: str) -> dict[str, Any]:
    result: dict[str, Any] = {"service": service, "release": os.environ.get("OPENNICF_RELEASE_VERSION", "unknown"), "checks": {}}
    if service == "knowledge":
        db_ok, db_detail = _database_check()
        object_ok, object_detail = _object_store_check()
        result["checks"] = {"postgresql_pgvector": {"ok": db_ok, "detail": db_detail}, "object_store": {"ok": object_ok, "detail": object_detail}}
    elif service == "ingestion":
        production = os.environ.get("OPENNICF_ENVIRONMENT", "development").lower() in {"prod", "production"}
        root = os.environ.get("OPENNICF_INGESTION_LANDING_ROOT", "/var/lib/opennicf/landing")
        try:
            os.makedirs(root, mode=0o750, exist_ok=True)
            ok = os.access(root, os.W_OK)
            result["checks"] = {"landing_root": {"ok": ok, "detail": "landing_root_ok" if ok else "landing_root_not_writable"}}
            if production:
                db_ok, db_detail = _database_check()
                object_ok, object_detail = _object_store_check()
                result["checks"].update({
                    "postgresql_pgvector": {"ok": db_ok, "detail": db_detail},
                    "object_store": {"ok": object_ok, "detail": object_detail},
                })
        except OSError as exc:
            result["checks"] = {"landing_root": {"ok": False, "detail": type(exc).__name__}}
    result["ready"] = all(item["ok"] for item in result["checks"].values())
    return result


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/health/live", "/health/ready", "/status"}:
            self.send_error(404)
            return
        payload = {"status": "ok", "live": True} if self.path == "/health/live" else checks(self.server.service)
        if self.path == "/health/ready" and not payload.get("ready", False):
            self._write(503, payload)
            return
        self._write(200, payload)

    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=("knowledge", "ingestion"), required=True)
    args = parser.parse_args(argv)
    stop_event = threading.Event()
    worker_thread: threading.Thread | None = None
    if args.service == "knowledge":
        from .knowledge.service import KnowledgeService, serve

        service = KnowledgeService.from_env()
        server = serve(service)
    else:
        from .ingestion import DirectoryWatcher, IngestionQueue, IngestionService
        from .knowledge.embeddings import HashingEmbeddingBackend, HttpEmbeddingBackend, LocalFirstEmbeddingService
        from .knowledge.service import KnowledgeServiceConfig
        from .knowledge.store import KnowledgePlatform

        config = KnowledgeServiceConfig.from_env()
        config.validate()
        if config.production and not config.embedding_base_url:
            raise RuntimeError("Qwen embedding worker URL is required in production")
        preferred = HttpEmbeddingBackend(config.embedding_base_url, service_token=config.embedding_token, timeout=config.embedding_timeout, model=config.embedding_model, dimension=config.embedding_dimension, native_dimension=config.embedding_native_dimension) if config.embedding_base_url else None
        embeddings = LocalFirstEmbeddingService(
            preferred_backend=preferred,
            cpu_backend=HashingEmbeddingBackend(dimensions=config.embedding_dimension),
            allow_cpu_fallback=not config.production,
        )
        platform = KnowledgePlatform.from_dsn(config.postgres_dsn, object_store_root=config.object_store_root, embeddings=embeddings)
        platform.store.migrate()
        queue_path = os.environ.get("OPENNICF_INGESTION_QUEUE_STATE", "/var/lib/opennicf/ingestion/queue.json")
        landing_root = os.environ.get("OPENNICF_INGESTION_LANDING_ROOT", "/var/lib/opennicf/landing")
        queue = IngestionQueue.from_state(queue_path)
        ingestion = IngestionService(platform, queue=queue, landing_root=os.environ.get("OPENNICF_INGESTION_VALIDATED_ROOT", str(Path(queue_path).parent / "landing")))
        watcher = DirectoryWatcher(
            ingestion,
            landing_root,
            channel="filesystem",
            domain=os.environ.get("OPENNICF_INGESTION_DOMAIN", "general"),
            system=os.environ.get("OPENNICF_INGESTION_SYSTEM", "unknown"),
            environment=os.environ.get("OPENNICF_ENVIRONMENT", "production"),
            acl_scope=os.environ.get("OPENNICF_INGESTION_ACL_SCOPE", "internal"),
            local_only=os.environ.get("OPENNICF_INGESTION_LOCAL_ONLY", "false").lower() == "true",
        )

        def process_loop() -> None:
            interval = max(1.0, float(os.environ.get("OPENNICF_INGESTION_POLL_SECONDS", "2")))
            while not stop_event.is_set():
                watcher.scan()
                while ingestion.process_next() is not None:
                    pass
                stop_event.wait(interval)

        worker_thread = threading.Thread(target=process_loop, name="opennicf-ingestion", daemon=True)
        worker_thread.start()
        server = ThreadingHTTPServer((os.environ.get("OPENNICF_SERVICE_HOST", "127.0.0.1"), int(os.environ.get("OPENNICF_SERVICE_PORT", "8090"))), _Handler)
        server.service = args.service

    def stop(_signum: int, _frame: Any) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if worker_thread is not None:
            worker_thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
