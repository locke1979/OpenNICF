"""Long-running, bounded HTTP service for the OpenNICF knowledge platform."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from .admin import KnowledgeAdministration
from .embeddings import EmbeddingMigration, HashingEmbeddingBackend, HttpEmbeddingBackend, LocalFirstEmbeddingService
from .models import EmbeddingRecord, ParsedBlock, RetrievalFilters
from .store import KnowledgePlatform
from ..ingestion import IngestionQueue, _parse_binary_or_text


class KnowledgeServiceError(RuntimeError):
    """An expected service/configuration failure safe to expose to callers."""


@dataclass(frozen=True)
class KnowledgeServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8090
    max_body_bytes: int = 1_048_576
    max_result_items: int = 100
    operation_limit: int = 10
    postgres_dsn: str = ""
    object_store_root: str = ""
    queue_state_path: str = "/var/lib/opennicf/knowledge-queue.json"
    embedding_base_url: str = ""
    embedding_token: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimension: int = 768
    embedding_timeout: float = 30.0
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "KnowledgeServiceConfig":
        def integer(name: str, default: int, minimum: int = 1) -> int:
            value = int(os.environ.get(name, str(default)))
            if value < minimum:
                raise KnowledgeServiceError(f"{name} must be >= {minimum}")
            return value

        return cls(
            host=os.environ.get("OPENNICF_SERVICE_HOST", "127.0.0.1"),
            port=integer("OPENNICF_SERVICE_PORT", 8090),
            max_body_bytes=integer("OPENNICF_KNOWLEDGE_MAX_BODY_BYTES", 1_048_576),
            max_result_items=integer("OPENNICF_KNOWLEDGE_MAX_RESULT_ITEMS", 100),
            operation_limit=integer("OPENNICF_KNOWLEDGE_OPERATION_LIMIT", 10),
            postgres_dsn=(os.environ.get("OPENNICF_POSTGRES_DSN") or os.environ.get("DATABASE_URL", "")).strip(),
            object_store_root=os.environ.get("OPENNICF_OBJECT_STORE_ROOT", "").strip(),
            queue_state_path=os.environ.get("OPENNICF_KNOWLEDGE_QUEUE_STATE", "/var/lib/opennicf/knowledge-queue.json"),
            embedding_base_url=os.environ.get("OPENNICF_EMBEDDING_BASE_URL", "").strip(),
            embedding_token=os.environ.get("OPENNICF_EMBEDDING_SERVICE_TOKEN", ""),
            embedding_model=os.environ.get("OPENNICF_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            embedding_dimension=integer("OPENNICF_EMBEDDING_DIMENSION", 768),
            embedding_timeout=float(os.environ.get("OPENNICF_EMBEDDING_TIMEOUT", "30")),
            environment=os.environ.get("OPENNICF_ENVIRONMENT", "development").lower(),
        )

    @property
    def production(self) -> bool:
        return self.environment in {"prod", "production"}

    def validate(self) -> None:
        if self.production and not self.postgres_dsn:
            raise KnowledgeServiceError("PostgreSQL DSN (OPENNICF_POSTGRES_DSN) is required in production")
        if self.production and not self.object_store_root:
            raise KnowledgeServiceError("object-store root (OPENNICF_OBJECT_STORE_ROOT) is required in production")
        if not self.postgres_dsn:
            raise KnowledgeServiceError("knowledge service requires a PostgreSQL DSN")
        if not self.object_store_root:
            raise KnowledgeServiceError("knowledge service requires an object-store root")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class KnowledgeService:
    """Application boundary that owns one durable platform and bounded admin calls."""

    def __init__(self, platform: KnowledgePlatform, config: KnowledgeServiceConfig, *, queue: IngestionQueue | None = None, reindex: Callable[[str], Any] | None = None):
        self.platform = platform
        self.config = config
        self.queue = queue or IngestionQueue(state_path=config.queue_state_path)
        self.admin = KnowledgeAdministration(platform.store, object_store=platform.object_store, embeddings=platform.embeddings)
        self._reindex = reindex
        self.ready = False

    @classmethod
    def from_env(cls, config: KnowledgeServiceConfig | None = None) -> "KnowledgeService":
        config = config or KnowledgeServiceConfig.from_env()
        config.validate()
        embedding = None
        if config.embedding_base_url:
            preferred = HttpEmbeddingBackend(config.embedding_base_url, service_token=config.embedding_token, timeout=config.embedding_timeout, model=config.embedding_model, dimension=config.embedding_dimension)
            embedding = LocalFirstEmbeddingService(preferred_backend=preferred, cpu_backend=HashingEmbeddingBackend(dimensions=config.embedding_dimension), allow_cpu_fallback=not config.production)
        else:
            if config.production:
                raise KnowledgeServiceError("Qwen embedding worker URL is required in production")
            # This is an explicit, reported CPU backend. It is not a storage fallback.
            embedding = LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=config.embedding_dimension))
        platform = KnowledgePlatform.from_dsn(config.postgres_dsn, object_store_root=config.object_store_root, embeddings=embedding)
        service = cls(platform, config)
        service.startup_validate()
        return service

    def startup_validate(self) -> None:
        if not hasattr(self.platform.store, "migrate"):
            raise KnowledgeServiceError("knowledge store does not support migrations")
        self.platform.store.migrate()
        if not Path(self.config.object_store_root).is_dir() or not os.access(self.config.object_store_root, os.W_OK):
            raise KnowledgeServiceError("object store is not writable")
        if self.config.production:
            try:
                probe = self.platform.embeddings.embed(
                    ["opennicf-readiness-probe"],
                    purpose="retrieval_document",
                    dimension=self.config.embedding_dimension,
                    privacy_policy="local_only",
                )
            except Exception as exc:  # noqa: BLE001 - readiness must fail closed
                raise KnowledgeServiceError("Qwen embedding dependency is not ready") from exc
            expected_space = f"{self.config.embedding_model}:{self.config.embedding_dimension}:v1"
            if (
                probe.fallback
                or probe.provider != "LOCAL"
                or probe.embedding_space_id != expected_space
                or not probe.normalized
            ):
                raise KnowledgeServiceError("Qwen embedding semantic contract is not ready")
        self.ready = True

    def status(self) -> dict[str, Any]:
        return {"service": "knowledge", "ready": self.ready, "storage": "postgresql", "object_store": self.platform.object_store.backend_name, "embedding": self.platform.embeddings.describe(), "limits": {"max_result_items": self.config.max_result_items, "operation_limit": self.config.operation_limit}}

    def filters(self, query: dict[str, list[str]]) -> RetrievalFilters:
        scopes = frozenset(filter(None, query.get("acl_scope", [])))
        domains = frozenset(filter(None, query.get("domain_id", [])))
        systems = frozenset(filter(None, query.get("system_id", [])))
        return RetrievalFilters(principal_acl_scopes=scopes, principal_domain_id=(next(iter(domains), None)), system_ids=tuple(systems))

    def list_sources(self, query: dict[str, list[str]]) -> list[dict[str, Any]]:
        limit = min(int(query.get("limit", [self.config.max_result_items])[0]), self.config.max_result_items)
        return self.admin.list_sources(filters=self.filters(query), include_retired=query.get("include_retired", ["false"])[0].lower() == "true")[:limit]

    def source_status(self, source_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
        return self.admin.source_status(source_id, filters=self.filters(query))

    def provenance(self, source_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
        result = self.admin.provenance(source_id, source_version_id=(query.get("source_version_id") or [None])[0], filters=self.filters(query))
        result["chunks"] = result.get("chunks", [])[: self.config.max_result_items]
        return result

    def artifact(self, artifact_hash: str) -> dict[str, Any]:
        """Return artifact metadata only; original bytes remain in object storage."""
        record = getattr(self.platform.store, "artifacts", {}).get(artifact_hash)
        if record is not None:
            result = _jsonable(record)
            result["object_present"] = self.platform.object_store.exists(record.object_key)
            return result
        connector = getattr(self.platform.store, "_connect", None)
        if connector is None:
            raise KeyError(f"unknown artifact: {artifact_hash}")
        with connector() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT a.artifact_hash, a.source_version_id, v.source_id, a.object_key, a.storage_backend, a.mime_type, a.size_bytes, a.parser_name, a.parser_version, a.created_at, a.metadata FROM knowledge_artifacts a JOIN knowledge_source_versions v ON v.source_version_id = a.source_version_id WHERE a.artifact_hash = %s", (artifact_hash,))
            row = cursor.fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_hash}")
        keys = ("artifact_hash", "source_version_id", "source_id", "object_key", "storage_backend", "mime_type", "size_bytes", "parser_name", "parser_version", "created_at", "metadata")
        result = dict(zip(keys, row))
        result["object_present"] = self.platform.object_store.exists(result["object_key"])
        return result

    def reparse(self, source_id: str, payload: dict[str, Any]) -> Any:
        parser_version = str(payload.get("parser_version", "admin"))[:80]
        provenance = self.admin.provenance(source_id, filters=RetrievalFilters())
        version, artifact = provenance["version"], provenance["artifact"]
        content = self.platform.object_store.get_bytes(version["object_key"])
        blocks, parser_name, detected_version, source_type = _parse_binary_or_text(Path(version["source_uri"]), content)
        operation = self.admin._start("reparse", str(payload.get("actor", "operator"))[:128], source_id, {"parser_version": parser_version})
        try:
            self.platform.ingest(source_id=source_id, source_uri=version["source_uri"], content=content, blocks=blocks, mime_type=artifact["mime_type"], source_kind="document", channel="reparse", domain_id=provenance["source"]["domain_id"], system_id=provenance["source"]["system_id"], component_id=provenance["source"]["component_id"], evidence_type=provenance["source"]["evidence_type"], environment=provenance["source"]["environment"], acl_scope=provenance["source"]["acl_scope"], source_type=source_type, parser_name=parser_name, parser_version=parser_version, metadata={"reparse_of": version["source_version_id"]})
        except Exception as exc:
            return self.admin._finish(operation, "failed", {"error": str(exc)})
        return self.admin._finish(operation, "complete", {"blocks": len(blocks), "detected_parser_version": detected_version})

    def reindex(self, source_id: str, payload: dict[str, Any]) -> Any:
        callback = self._reindex or (lambda target: {"source_id": target, "mode": "store-native", "validated": True})
        return self.admin.reindex(source_id, actor=str(payload.get("actor", "operator"))[:128], reindex=callback)

    def reembed(self, payload: dict[str, Any]) -> Any:
        source_id = str(payload.get("source_id", ""))
        if not source_id:
            raise KnowledgeServiceError("source_id is required")
        target = str(payload.get("embedding_space_id", self.platform.embeddings.active_space_id))
        limit = min(int(payload.get("limit", self.config.operation_limit)), self.config.operation_limit)
        provenance = self.admin.provenance(source_id, filters=RetrievalFilters())
        chunks = [SimpleNamespace(**chunk) for chunk in provenance.get("chunks", [])[:limit]]
        migration = EmbeddingMigration(self.platform.embeddings, target, batch_size=min(limit or 1, self.config.operation_limit))
        def persist(chunk: Any, result: Any) -> None:
            self.platform.store.save_embedding(EmbeddingRecord(chunk_id=chunk.chunk_id, model=result.model, dimensions=result.dimensions, device=result.device, vector=result.vectors[0], metadata=result.metadata, embedding_space_id=result.embedding_space_id, provider=result.provider, model_revision=result.model_revision, normalized=result.normalized, purpose=result.purpose))
        operation = self.admin._start("reembed", str(payload.get("actor", "operator"))[:128], target, {"source_id": source_id, "count": len(chunks)})
        try:
            state = migration.run(chunks, persist, privacy_policy=str(payload.get("privacy_policy", "cloud_allowed")))
        except Exception as exc:
            return self.admin._finish(operation, "failed", {"error": str(exc), "embedding_space_id": target})
        return self.admin._finish(operation, state.get("status", "paused"), state)

    def retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = min(int(payload.get("limit", 1)), self.config.operation_limit)
        job_id = payload.get("job_id")
        if job_id:
            return _jsonable(self.admin.retry_failed_ingestion(self.queue, str(job_id)[:200], actor=str(payload.get("actor", "operator"))[:128]))
        return {"retried": self.queue.retry_dead_letters(limit=limit), "limit": limit}

    def migration_status(self) -> dict[str, Any]:
        return self.admin.embedding_space_status()


class KnowledgeRequestHandler(BaseHTTPRequestHandler):
    service: KnowledgeService

    def setup(self) -> None:
        super().setup()
        self.service = self.server.service

    def _respond(self, code: int, payload: Any) -> None:
        body = json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > self.service.config.max_body_bytes:
            raise KnowledgeServiceError("request body exceeds limit")
        raw = self.rfile.read(length)
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/health/live": return self._respond(200, {"status": "ok", "live": True})
            if parsed.path == "/health/ready": return self._respond(200 if self.service.ready else 503, {"status": "ok" if self.service.ready else "not_ready", "ready": self.service.ready})
            if parsed.path == "/status": return self._respond(200, self.service.status())
            if parsed.path == "/v1/sources": return self._respond(200, {"sources": self.service.list_sources(query)})
            if parsed.path == "/v1/migrations/status": return self._respond(200, self.service.migration_status())
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if len(parts) == 4 and parts[:2] == ["v1", "sources"] and parts[3] in {"status", "provenance"}:
                result = self.service.source_status(parts[2], query) if parts[3] == "status" else self.service.provenance(parts[2], query)
                return self._respond(200, result)
            if len(parts) == 4 and parts[:2] == ["v1", "artifacts"] and parts[3] == "provenance":
                return self._respond(200, self.service.artifact(parts[2]))
            self._respond(404, {"error": "not_found"})
        except KeyError as exc:
            self._respond(404, {"error": "not_found", "detail": str(exc)})
        except PermissionError:
            self._respond(404, {"error": "not_found"})
        except (ValueError, PermissionError, KnowledgeServiceError) as exc:
            self._respond(400, {"error": type(exc).__name__, "detail": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        try:
            payload = self._body()
            if parsed.path == "/v1/retry": result = self.service.retry(payload)
            elif parsed.path == "/v1/re-embed": result = self.service.reembed(payload)
            elif len(parts) == 4 and parts[:2] == ["v1", "sources"] and parts[3] == "reparse": result = self.service.reparse(parts[2], payload)
            elif len(parts) == 4 and parts[:2] == ["v1", "sources"] and parts[3] == "reindex": result = self.service.reindex(parts[2], payload)
            else: return self._respond(404, {"error": "not_found"})
            self._respond(200, result)
        except KeyError as exc:
            self._respond(404, {"error": "not_found", "detail": str(exc)})
        except (ValueError, PermissionError, KnowledgeServiceError) as exc:
            self._respond(400, {"error": type(exc).__name__, "detail": str(exc)})
        except Exception:
            self._respond(500, {"error": "operation_failed"})

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def serve(service: KnowledgeService) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((service.config.host, service.config.port), KnowledgeRequestHandler)
    server.service = service
    return server
