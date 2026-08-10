"""Bounded, auditable administration of OpenNICF-owned knowledge data.

This module deliberately exposes metadata and lifecycle operations only.  It
does not grant access to evidence outside the caller's ACL/domain scope and
never deletes immutable source artifacts as part of retirement.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .embeddings import EmbeddingMigration, LocalFirstEmbeddingService
from .models import ParsedBlock, RetrievalFilters


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RetirementPolicy:
    """Policy guard for a non-destructive source retirement."""

    allow_retirement: bool = True
    require_reason: bool = True
    preserve_provenance: bool = True
    legal_hold_metadata_key: str = "legal_hold"


@dataclass(frozen=True)
class AdminOperation:
    operation_id: str
    operation: str
    status: str
    requested_at: datetime
    actor: str
    target_id: str
    details: dict[str, Any]


class KnowledgeAdministration:
    """Small orchestration boundary for source and index administration.

    ``parser`` and ``reembed`` are injected because parsing and embedding are
    deployment-specific.  The service records an operation before invoking a
    callback, making retries and partial work visible without hiding failures.
    """

    def __init__(
        self,
        store: Any,
        *,
        object_store: Any | None = None,
        embeddings: LocalFirstEmbeddingService | None = None,
        parser: Callable[[bytes, str], Sequence[ParsedBlock]] | None = None,
    ):
        self.store = store
        self.object_store = object_store
        self.embeddings = embeddings
        self.parser = parser

    def list_sources(self, *, filters: RetrievalFilters | None = None, include_retired: bool = False) -> list[dict[str, Any]]:
        return self.store.list_sources(filters or RetrievalFilters(), include_retired=include_retired)

    def source_status(self, source_id: str, *, filters: RetrievalFilters | None = None) -> dict[str, Any]:
        status = self.store.source_status(source_id, filters=filters or RetrievalFilters())
        if self.object_store is not None:
            provenance = self.store.provenance(source_id, source_version_id=None, filters=filters or RetrievalFilters())
            versions = (provenance["version"],)
            status["object_present"] = all(self.object_store.exists(version["object_key"]) for version in versions)
        return status

    def provenance(self, source_id: str, *, source_version_id: str | None = None, filters: RetrievalFilters | None = None) -> dict[str, Any]:
        return self.store.provenance(source_id, source_version_id=source_version_id, filters=filters or RetrievalFilters())

    def retire_source(self, source_id: str, *, actor: str, reason: str, policy: RetirementPolicy | None = None) -> AdminOperation:
        policy = policy or RetirementPolicy()
        if not policy.allow_retirement:
            raise PermissionError("retirement is disabled by policy")
        if policy.require_reason and not reason.strip():
            raise ValueError("retirement reason is required")
        status = self.store.source_status(source_id, filters=RetrievalFilters())
        if status.get("metadata", {}).get(policy.legal_hold_metadata_key) is True:
            raise PermissionError("source is under legal hold")
        operation = self._start("retire", actor, source_id, {"reason": reason})
        try:
            self.store.retire_source(source_id, reason=reason, actor=actor, preserve_provenance=policy.preserve_provenance)
        except Exception as exc:  # noqa: BLE001 - persist operation failure state
            return self._finish(operation, "failed", {"error": str(exc)})
        return self._finish(operation, "complete", {"reason": reason})

    def retry_failed_ingestion(self, queue: Any, job_id: str, *, actor: str = "system") -> AdminOperation:
        operation = self._start("retry_ingestion", actor, job_id, {})
        try:
            retried = bool(queue.retry_dead_letter(job_id))
            if not retried:
                raise KeyError(f"dead-letter job not found: {job_id}")
        except Exception as exc:  # noqa: BLE001 - persist operation failure state
            return self._finish(operation, "failed", {"error": str(exc)})
        return self._finish(operation, "queued", {"job_id": job_id})

    def reparse(self, source_id: str, *, actor: str, parser: Callable[[bytes, str], Sequence[ParsedBlock]] | None = None, parser_version: str = "admin", ingest: Callable[..., Any] | None = None) -> AdminOperation:
        parser = parser or self.parser
        if parser is None or ingest is None:
            raise ValueError("reparse requires parser and ingest callbacks")
        provenance = self.store.provenance(source_id, filters=RetrievalFilters())
        operation = self._start("reparse", actor, source_id, {"parser_version": parser_version})
        try:
            version = provenance["version"]
            if self.object_store is None:
                raise RuntimeError("object store is required for reparse")
            content = self.object_store.get_bytes(version["object_key"])
            blocks = tuple(parser(content, version["source_uri"]))
            ingest(source_id=source_id, source_uri=version["source_uri"], content=content, blocks=blocks, parser_version=parser_version)
        except Exception as exc:  # noqa: BLE001 - persist operation failure state
            return self._finish(operation, "failed", {"error": str(exc)})
        return self._finish(operation, "complete", {"blocks": len(blocks), "source_version_id": version["source_version_id"]})

    def reindex(self, source_id: str, *, actor: str, reindex: Callable[[str], Any]) -> AdminOperation:
        return self._run_callback("reindex", source_id, actor, reindex)

    def reembed(self, chunks: Sequence[Any], *, actor: str, migration: EmbeddingMigration, privacy_policy: str = "cloud_allowed") -> AdminOperation:
        target = migration.target_space_id
        operation = self._start("reembed", actor, target, {"privacy_policy": privacy_policy, "count": len(chunks)})
        try:
            state = migration.run(chunks, self.store.save_embedding, privacy_policy=privacy_policy)
        except Exception as exc:  # noqa: BLE001 - persist operation failure state
            return self._finish(operation, "failed", {"error": str(exc), "embedding_space_id": target})
        return self._finish(operation, state.get("status", "paused"), {"embedding_space_id": target, **state})

    def embedding_space_status(self) -> dict[str, Any]:
        if self.embeddings is None:
            return {"spaces": [], "active_space_id": None, "migrations": {}}
        return {"spaces": self.embeddings.describe()["spaces"], "active_space_id": self.embeddings.active_space_id, "migrations": {key: dict(value) for key, value in self.embeddings.migration_state.items()}}

    def _start(self, operation: str, actor: str, target_id: str, details: Mapping[str, Any]) -> AdminOperation:
        record = AdminOperation(f"admin_{uuid.uuid4().hex}", operation, "running", _now(), actor, target_id, dict(details))
        self.store.record_admin_operation(record)
        return record

    def _finish(self, operation: AdminOperation, status: str, details: Mapping[str, Any]) -> AdminOperation:
        finished = AdminOperation(operation.operation_id, operation.operation, status, operation.requested_at, operation.actor, operation.target_id, {**operation.details, **dict(details), "finished_at": _now().isoformat()})
        self.store.record_admin_operation(finished)
        return finished

    def _run_callback(self, name: str, target_id: str, actor: str, callback: Callable[[str], Any]) -> AdminOperation:
        operation = self._start(name, actor, target_id, {})
        try:
            result = callback(target_id)
        except Exception as exc:  # noqa: BLE001 - persist operation failure state
            return self._finish(operation, "failed", {"error": str(exc)})
        return self._finish(operation, "complete", {"result": result} if isinstance(result, Mapping) else {})
