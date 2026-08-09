"""Memory and PostgreSQL knowledge stores."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib.resources as resources
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence
import uuid

from .embeddings import EmbeddingResult, LocalFirstEmbeddingService
from .namespaces import (
    IntegrationEdgeRecord,
    KnowledgeNamespaceRecord,
    build_integration_records,
    build_namespace_record,
    sanitize_metadata,
    sanitize_text,
)
from .models import (
    ArtifactRecord,
    AuditEvidenceRefRecord,
    AuditFindingRecord,
    AuditReportRecord,
    VerificationRequestRecord,
    ChunkRecord,
    EmbeddingRecord,
    EvidenceHit,
    IngestBundle,
    KnowledgeSource,
    KnowledgeSourceVersion,
    ParsedBlock,
    RetrievalEventRecord,
    RetrievalFilters,
    SearchCandidate,
    SourceKind,
    utcnow,
)
from .object_store import FilesystemObjectStore, MemoryObjectStore, ObjectReference, ObjectStore


def _uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _chunk_locator(prefix: str | None, ordinal: int, line_start: int | None, line_end: int | None) -> str:
    locator = prefix or "chunk"
    if line_start is None and line_end is None:
        return f"{locator}#chunk-{ordinal + 1}"
    if line_start is not None and line_end is not None and line_start != line_end:
        return f"{locator}:L{line_start}-L{line_end}"
    if line_start is not None:
        return f"{locator}:L{line_start}"
    return f"{locator}#chunk-{ordinal + 1}"


def _split_blocks(text: str, *, max_chars: int = 900, overlap: int = 0) -> list[tuple[str, int, int]]:
    lines = text.splitlines()
    blocks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + 1)
        char_count = len(lines[start])
        while end < len(lines) and char_count < max_chars:
            if not lines[end].strip():
                break
            char_count += len(lines[end]) + 1
            if char_count > max_chars:
                break
            end += 1
        if end == start:
            end = start + 1
        block_lines = lines[start:end]
        if not block_lines:
            break
        block_text = "\n".join(block_lines).strip()
        if block_text:
            blocks.append((block_text, start + 1, end))
        start = end + 1 if end < len(lines) and not lines[end].strip() else end
        if overlap and start > overlap:
            start -= overlap
    if not blocks and text.strip():
        blocks.append((text.strip(), 1, len(lines) or 1))
    return blocks


def _coerce_source_record(data: dict[str, Any]) -> KnowledgeSource:
    payload = dict(data)
    payload.setdefault("namespace_id", payload.get("namespace_id") or f"ns_{payload.get('source_id', '')}")
    payload.setdefault("domain_id", payload.get("domain_id") or payload.get("domain") or "general")
    payload.setdefault("system_id", payload.get("system_id") or payload.get("system") or "unknown")
    payload.setdefault("component_id", payload.get("component_id") or payload.get("system_id") or "unknown-component")
    payload.setdefault("evidence_type", payload.get("evidence_type") or payload.get("source_type") or "document")
    payload.setdefault("domain", payload.get("domain") or payload["domain_id"])
    payload.setdefault("system", payload.get("system") or payload["system_id"])
    payload.setdefault("source_type", payload.get("source_type") or payload["evidence_type"])
    return KnowledgeSource(**payload)


def _coerce_chunk_record(data: dict[str, Any]) -> ChunkRecord:
    payload = dict(data)
    payload.setdefault("namespace_id", payload.get("namespace_id") or f"ns_{payload.get('source_id', '')}")
    payload.setdefault("domain_id", payload.get("domain_id") or payload.get("domain") or "general")
    payload.setdefault("system_id", payload.get("system_id") or payload.get("system") or "unknown")
    payload.setdefault("component_id", payload.get("component_id") or payload.get("system_id") or "unknown-component")
    payload.setdefault("evidence_type", payload.get("evidence_type") or payload.get("source_type") or "document")
    payload.setdefault("domain", payload.get("domain") or payload["domain_id"])
    payload.setdefault("system", payload.get("system") or payload["system_id"])
    payload.setdefault("source_type", payload.get("source_type") or payload["evidence_type"])
    return ChunkRecord(**payload)


def _coerce_namespace_record(data: dict[str, Any]) -> KnowledgeNamespaceRecord:
    return KnowledgeNamespaceRecord(**dict(data))


def _coerce_edge_record(data: dict[str, Any]) -> IntegrationEdgeRecord:
    return IntegrationEdgeRecord(**dict(data))


class KnowledgeStore(Protocol):
    def save_bundle(self, bundle: IngestBundle) -> IngestBundle:
        raise NotImplementedError

    def search_candidates(self, filters: RetrievalFilters) -> list[SearchCandidate]:
        raise NotImplementedError

    def record_retrieval_event(self, event: RetrievalEventRecord) -> None:
        raise NotImplementedError

    def record_audit_finding(self, finding: AuditFindingRecord) -> None:
        raise NotImplementedError

    def record_audit_report(self, report: AuditReportRecord) -> None:
        raise NotImplementedError

    def record_verification_request(self, request: VerificationRequestRecord) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError

    def restore(self, snapshot: dict[str, Any]) -> None:
        raise NotImplementedError


class MemoryKnowledgeStore:
    """Deterministic in-memory store used for tests and local development."""

    def __init__(self):
        self.sources: dict[str, KnowledgeSource] = {}
        self.source_versions: dict[str, KnowledgeSourceVersion] = {}
        self.artifacts: dict[str, ArtifactRecord] = {}
        self.namespaces: dict[str, KnowledgeNamespaceRecord] = {}
        self.integration_edges: dict[str, IntegrationEdgeRecord] = {}
        self.chunks: dict[str, ChunkRecord] = {}
        self.embeddings: dict[str, EmbeddingRecord] = {}
        self.retrieval_events: list[RetrievalEventRecord] = []
        self.audit_findings: list[AuditFindingRecord] = []
        self.audit_reports: list[AuditReportRecord] = []
        self.verification_requests: list[VerificationRequestRecord] = []
        self._latest_version_by_source_hash: dict[tuple[str, str], str] = {}
        self._versions_by_source: dict[str, list[str]] = {}
        self._artifact_chunk_ids: dict[str, list[str]] = {}
        self._artifact_namespace_ids: dict[str, list[str]] = {}
        self._artifact_edge_ids: dict[str, list[str]] = {}

    def _store_chunk(self, chunk: ChunkRecord, embedding: EmbeddingRecord | None = None) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self._artifact_chunk_ids.setdefault(chunk.artifact_hash, []).append(chunk.chunk_id)
        if embedding is not None:
            self.embeddings[chunk.chunk_id] = embedding

    def _store_namespace(self, namespace: KnowledgeNamespaceRecord, *, artifact_hash: str) -> None:
        metadata = dict(namespace.metadata)
        metadata.setdefault("artifact_hash", artifact_hash)
        self.namespaces[namespace.namespace_id] = replace(namespace, metadata=metadata)
        self._artifact_namespace_ids.setdefault(artifact_hash, []).append(namespace.namespace_id)

    def _store_edge(self, edge: IntegrationEdgeRecord, *, artifact_hash: str | None = None) -> None:
        metadata = dict(edge.metadata)
        if artifact_hash:
            metadata.setdefault("artifact_hash", artifact_hash)
        self.integration_edges[edge.edge_id] = replace(edge, metadata=metadata)
        key = artifact_hash or edge.source_namespace_id
        self._artifact_edge_ids.setdefault(key, []).append(edge.edge_id)

    def save_bundle(self, bundle: IngestBundle) -> IngestBundle:
        key = (bundle.source.source_id, bundle.source.content_hash)
        if key in self._latest_version_by_source_hash:
            existing_version_id = self._latest_version_by_source_hash[key]
            existing_version = self.source_versions[existing_version_id]
            existing_artifact = self.artifacts[existing_version.artifact_hash]
            existing_chunks = tuple(
                self.chunks[chunk_id]
                for chunk_id in self._artifact_chunk_ids.get(existing_artifact.artifact_hash, [])
            )
            existing_embeddings = tuple(
                self.embeddings[chunk.chunk_id]
                for chunk in existing_chunks
                if chunk.chunk_id in self.embeddings
            )
            existing_namespaces = tuple(
                self.namespaces[namespace_id]
                for namespace_id in self._artifact_namespace_ids.get(existing_artifact.artifact_hash, [])
                if namespace_id in self.namespaces
            )
            existing_edges = tuple(
                self.integration_edges[edge_id]
                for edge_id in self._artifact_edge_ids.get(existing_artifact.artifact_hash, [])
                if edge_id in self.integration_edges
            )
            return IngestBundle(
                source=bundle.source,
                version=existing_version,
                artifact=existing_artifact,
                chunks=existing_chunks,
                embeddings=existing_embeddings,
                object_reference=bundle.object_reference,
                created=False,
                namespaces=existing_namespaces,
                integration_edges=existing_edges,
            )

        self.sources[bundle.source.source_id] = bundle.source
        self.source_versions[bundle.version.source_version_id] = bundle.version
        self.artifacts[bundle.artifact.artifact_hash] = bundle.artifact
        self._latest_version_by_source_hash[key] = bundle.version.source_version_id
        self._versions_by_source.setdefault(bundle.source.source_id, []).append(bundle.version.source_version_id)
        for namespace in bundle.namespaces:
            self._store_namespace(namespace, artifact_hash=bundle.artifact.artifact_hash)
        for edge in bundle.integration_edges:
            self._store_edge(edge, artifact_hash=bundle.artifact.artifact_hash)
        for chunk, embedding in zip(bundle.chunks, bundle.embeddings):
            self._store_chunk(chunk, embedding)
        return bundle

    def next_version_number(self, source_id: str, content_hash: str) -> int:
        if (source_id, content_hash) in self._latest_version_by_source_hash:
            version_id = self._latest_version_by_source_hash[(source_id, content_hash)]
            return self.source_versions[version_id].version_number
        return len(self._versions_by_source.get(source_id, [])) + 1

    def search_candidates(self, filters: RetrievalFilters) -> list[SearchCandidate]:
        filters = filters.normalized()
        allowed_domains = filters.effective_domain_ids()
        allowed_systems = filters.effective_system_ids()
        allowed_components = filters.effective_component_ids()
        allowed_evidence_types = filters.effective_evidence_types()
        allowed_namespaces = filters.effective_namespace_ids()
        candidates: list[SearchCandidate] = []
        for chunk_id, chunk in self.chunks.items():
            if filters.principal_acl_scopes and chunk.acl_scope not in filters.principal_acl_scopes:
                continue
            if allowed_namespaces and chunk.namespace_id not in allowed_namespaces:
                continue
            if allowed_domains and chunk.domain_id not in allowed_domains:
                continue
            if allowed_systems and chunk.system_id not in allowed_systems:
                continue
            if allowed_components and chunk.component_id not in allowed_components:
                continue
            if allowed_evidence_types and chunk.evidence_type not in allowed_evidence_types:
                continue
            if filters.environments and chunk.environment not in filters.environments:
                continue
            if filters.source_types and chunk.source_type not in filters.source_types:
                continue
            if filters.source_ids and chunk.source_id not in filters.source_ids:
                continue
            version = self.source_versions[chunk.source_version_id]
            if filters.since and version.ingest_timestamp < filters.since:
                continue
            if filters.until and version.ingest_timestamp > filters.until:
                continue
            source = self.sources[chunk.source_id]
            artifact = self.artifacts[chunk.artifact_hash]
            candidates.append(
                SearchCandidate(
                    chunk=chunk,
                    source=source,
                    version=version,
                    artifact=artifact,
                    embedding=self.embeddings.get(chunk_id),
                )
            )
        candidates.sort(key=lambda candidate: (candidate.chunk.source_version_id, candidate.chunk.ordinal, candidate.chunk.chunk_id))
        return candidates

    def record_retrieval_event(self, event: RetrievalEventRecord) -> None:
        self.retrieval_events.append(event)

    def record_audit_finding(self, finding: AuditFindingRecord) -> None:
        self.audit_findings.append(finding)

    def record_audit_report(self, report: AuditReportRecord) -> None:
        self.audit_reports.append(report)

    def record_verification_request(self, request: VerificationRequestRecord) -> None:
        self.verification_requests.append(request)

    def snapshot(self) -> dict[str, Any]:
        return {
            "sources": [asdict(item) for item in self.sources.values()],
            "source_versions": [asdict(item) for item in self.source_versions.values()],
            "artifacts": [asdict(item) for item in self.artifacts.values()],
            "namespaces": [asdict(item) for item in self.namespaces.values()],
            "integration_edges": [asdict(item) for item in self.integration_edges.values()],
            "chunks": [asdict(item) for item in self.chunks.values()],
            "embeddings": [asdict(item) for item in self.embeddings.values()],
            "retrieval_events": [asdict(item) for item in self.retrieval_events],
            "audit_findings": [asdict(item) for item in self.audit_findings],
            "audit_reports": [asdict(item) for item in self.audit_reports],
            "verification_requests": [asdict(item) for item in self.verification_requests],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.__init__()
        for source in snapshot.get("sources", []):
            record = _coerce_source_record(source)
            self.sources[record.source_id] = record
        for version in snapshot.get("source_versions", []):
            record = KnowledgeSourceVersion(**version)
            self.source_versions[record.source_version_id] = record
            self._versions_by_source.setdefault(record.source_id, []).append(record.source_version_id)
            self._latest_version_by_source_hash[(record.source_id, record.content_hash)] = record.source_version_id
        for artifact in snapshot.get("artifacts", []):
            self.artifacts[artifact["artifact_hash"]] = ArtifactRecord(**artifact)
        for namespace in snapshot.get("namespaces", []):
            record = _coerce_namespace_record(namespace)
            self.namespaces[record.namespace_id] = record
            artifact_hash = record.metadata.get("artifact_hash")
            if isinstance(artifact_hash, str) and artifact_hash:
                self._artifact_namespace_ids.setdefault(artifact_hash, []).append(record.namespace_id)
        for edge in snapshot.get("integration_edges", []):
            record = _coerce_edge_record(edge)
            self.integration_edges[record.edge_id] = record
            artifact_hash = record.metadata.get("artifact_hash")
            if isinstance(artifact_hash, str) and artifact_hash:
                self._artifact_edge_ids.setdefault(artifact_hash, []).append(record.edge_id)
        for chunk in snapshot.get("chunks", []):
            record = _coerce_chunk_record(chunk)
            self.chunks[record.chunk_id] = record
            self._artifact_chunk_ids.setdefault(record.artifact_hash, []).append(record.chunk_id)
        for embedding in snapshot.get("embeddings", []):
            record = EmbeddingRecord(**embedding)
            self.embeddings[record.chunk_id] = record
        for event in snapshot.get("retrieval_events", []):
            self.retrieval_events.append(RetrievalEventRecord(**event))
        for finding in snapshot.get("audit_findings", []):
            self.audit_findings.append(_audit_finding_from_json(finding))
        for report in snapshot.get("audit_reports", []):
            self.audit_reports.append(_audit_report_from_json(report))
        for request in snapshot.get("verification_requests", []):
            self.verification_requests.append(_verification_request_from_json(request))


class PostgresKnowledgeStore:
    """PostgreSQL + pgvector store with source-controlled migrations."""

    def __init__(self, connection_factory: Callable[[], Any], *, migration_package: str = "opennicf.knowledge.migrations"):
        self._connection_factory = connection_factory
        self._migration_package = migration_package

    @classmethod
    def from_dsn(cls, dsn: str, *, migration_package: str = "opennicf.knowledge.migrations") -> "PostgresKnowledgeStore":
        def factory():
            try:
                import psycopg
            except ImportError as exc:  # pragma: no cover - exercised in deployment, not tests
                raise RuntimeError("psycopg is required for PostgresKnowledgeStore") from exc
            return psycopg.connect(dsn)

        return cls(factory, migration_package=migration_package)

    def _connect(self):
        return self._connection_factory()

    def migrate(self) -> list[int]:
        from .migration_utils import iter_migration_files, migration_checksum, migration_version

        applied: list[int] = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                )
                cur.execute("SELECT version, checksum FROM schema_migrations ORDER BY version")
                existing = {
                    int(version): checksum.decode("utf-8") if isinstance(checksum, bytes) else checksum
                    for version, checksum in cur.fetchall()
                }
                for path in iter_migration_files(self._migration_package):
                    version = migration_version(path.name)
                    checksum = migration_checksum(path)
                    if version in existing and existing[version] != checksum:
                        raise RuntimeError(
                            f"migration checksum drift for version {version}: "
                            "applied migration differs from source"
                        )
                    if version in existing:
                        continue
                    sql = path.read_text(encoding="utf-8")
                    for statement in _split_sql(sql):
                        cur.execute(statement)
                    cur.execute(
                        "INSERT INTO schema_migrations(version, checksum, applied_at) VALUES (%s, %s, now()) ON CONFLICT (version) DO UPDATE SET checksum = EXCLUDED.checksum, applied_at = EXCLUDED.applied_at",
                        (version, checksum),
                    )
                    applied.append(version)
            conn.commit()
        return applied

    def save_bundle(self, bundle: IngestBundle) -> IngestBundle:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM knowledge_source_versions WHERE source_id = %s AND content_hash = %s",
                    (bundle.source.source_id, bundle.source.content_hash),
                )
                if cur.fetchone() is not None:
                    conn.commit()
                    return replace(bundle, created=False)
                _upsert_source(cur, bundle.source)
                _upsert_version(cur, bundle.version)
                _upsert_artifact(cur, bundle.artifact)
                for namespace in bundle.namespaces:
                    _upsert_namespace(cur, namespace)
                for edge in bundle.integration_edges:
                    _upsert_integration_edge(cur, edge)
                for chunk, embedding in zip(bundle.chunks, bundle.embeddings):
                    _upsert_chunk(cur, chunk)
                    _upsert_embedding(cur, embedding)
            conn.commit()
        return bundle

    def next_version_number(self, source_id: str, content_hash: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version_number FROM knowledge_source_versions WHERE source_id = %s AND content_hash = %s",
                    (source_id, content_hash),
                )
                row = cur.fetchone()
                if row:
                    return int(row[0])
                cur.execute(
                    "SELECT COALESCE(MAX(version_number), 0) + 1 FROM knowledge_source_versions WHERE source_id = %s",
                    (source_id,),
                )
                return int(cur.fetchone()[0])

    def search_candidates(self, filters: RetrievalFilters) -> list[SearchCandidate]:
        filters = filters.normalized()
        sql, params = _build_search_sql(filters)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        candidates: list[SearchCandidate] = []
        for row in rows:
            row = tuple(value.decode("utf-8") if isinstance(value, bytes) else value for value in row)
            chunk = ChunkRecord(
                chunk_id=row[0],
                source_id=row[1],
                source_version_id=row[2],
                artifact_hash=row[3],
                namespace_id=row[4],
                domain_id=row[5],
                system_id=row[6],
                component_id=row[7],
                environment=row[8],
                evidence_type=row[9],
                acl_scope=row[10],
                ordinal=row[11],
                text=row[12],
                locator=row[13],
                chunk_hash=row[14],
                parser_version=row[15],
                domain=row[16],
                system=row[17],
                source_type=row[18],
                page=row[19],
                line_start=row[20],
                line_end=row[21],
                metadata=row[22] or {},
            )
            source = KnowledgeSource(
                source_id=row[23],
                source_uri=row[24],
                source_kind=row[25],
                channel=row[26],
                namespace_id=row[27],
                domain_id=row[28],
                system_id=row[29],
                component_id=row[30],
                environment=row[31],
                evidence_type=row[32],
                acl_scope=row[33],
                mime_type=row[34],
                size_bytes=row[35],
                content_hash=row[36],
                created_at=row[37],
                updated_at=row[38],
                domain=row[39],
                system=row[40],
                source_type=row[41],
                metadata=row[42] or {},
            )
            version = KnowledgeSourceVersion(
                source_version_id=row[43],
                source_id=row[44],
                source_uri=row[45],
                version_number=row[46],
                content_hash=row[47],
                artifact_hash=row[48],
                ingest_timestamp=row[49],
                parser_version=row[50],
                storage_backend=row[51],
                object_key=row[52],
                metadata=row[53] or {},
            )
            artifact = ArtifactRecord(
                artifact_hash=row[54],
                source_version_id=row[55],
                object_key=row[56],
                storage_backend=row[57],
                mime_type=row[58],
                size_bytes=row[59],
                parser_name=row[60],
                parser_version=row[61],
                created_at=row[62],
                metadata=row[63] or {},
            )
            embedding = EmbeddingRecord(
                chunk_id=row[0],
                model=row[64],
                dimensions=row[65],
                device=row[66],
                vector=_parse_vector_value(row[67]),
                created_at=row[68],
                metadata=row[69] or {},
            )
            candidates.append(SearchCandidate(chunk=chunk, source=source, version=version, artifact=artifact, embedding=embedding))
        return candidates

    def record_retrieval_event(self, event: RetrievalEventRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_retrieval_events (
                        event_id, query_hash, filters, selected_chunks, scores, reranker, model_route, created_at, metadata
                    ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (event_id) DO UPDATE SET
                        query_hash = EXCLUDED.query_hash,
                        filters = EXCLUDED.filters,
                        selected_chunks = EXCLUDED.selected_chunks,
                        scores = EXCLUDED.scores,
                        reranker = EXCLUDED.reranker,
                        model_route = EXCLUDED.model_route,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        event.event_id,
                        event.query_hash,
                        json.dumps(event.filters, sort_keys=True),
                        json.dumps(list(event.selected_chunks), sort_keys=True),
                        json.dumps(event.scores, sort_keys=True),
                        event.reranker,
                        event.model_route,
                        event.created_at,
                        json.dumps(event.metadata, sort_keys=True),
                    ),
                )
            conn.commit()

    def record_audit_finding(self, finding: AuditFindingRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_audit_findings (
                        finding_id, classification, statement, confidence, evidence_refs, time_range, systems, components,
                        verification_status, recommended_query, diagnostic_action, supporting_evidence,
                        contradicting_evidence, provenance_refs, correlation_ids, source_type_analyzers,
                        created_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb,
                              %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (finding_id) DO UPDATE SET
                        classification = EXCLUDED.classification,
                        statement = EXCLUDED.statement,
                        confidence = EXCLUDED.confidence,
                        evidence_refs = EXCLUDED.evidence_refs,
                        time_range = EXCLUDED.time_range,
                        systems = EXCLUDED.systems,
                        components = EXCLUDED.components,
                        verification_status = EXCLUDED.verification_status,
                        recommended_query = EXCLUDED.recommended_query,
                        diagnostic_action = EXCLUDED.diagnostic_action,
                        supporting_evidence = EXCLUDED.supporting_evidence,
                        contradicting_evidence = EXCLUDED.contradicting_evidence,
                        provenance_refs = EXCLUDED.provenance_refs,
                        correlation_ids = EXCLUDED.correlation_ids,
                        source_type_analyzers = EXCLUDED.source_type_analyzers,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        finding.finding_id,
                        finding.classification,
                        finding.statement,
                        finding.confidence,
                        json.dumps([asdict(ref) for ref in finding.evidence_refs], sort_keys=True),
                        json.dumps(_time_range_to_json(finding.time_range), sort_keys=True),
                        json.dumps(list(finding.systems), sort_keys=True),
                        json.dumps(list(finding.components), sort_keys=True),
                        finding.verification_status,
                        finding.recommended_query,
                        json.dumps(finding.diagnostic_action, sort_keys=True),
                        json.dumps(list(finding.supporting_evidence), sort_keys=True),
                        json.dumps(list(finding.contradicting_evidence), sort_keys=True),
                        json.dumps(list(finding.provenance_refs), sort_keys=True),
                        json.dumps(list(finding.correlation_ids), sort_keys=True),
                        json.dumps(list(finding.source_type_analyzers), sort_keys=True),
                        finding.created_at,
                        json.dumps(finding.metadata, sort_keys=True),
                    ),
                )
            conn.commit()

    def record_audit_report(self, report: AuditReportRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_audit_reports (
                        report_id, request_hash, request_payload, executive_summary, human_report, machine_report,
                        finding_ids, timeline_event_ids, causal_chains, unresolved_hypotheses,
                        recommended_verification_steps, created_at, metadata
                    ) VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (report_id) DO UPDATE SET
                        request_hash = EXCLUDED.request_hash,
                        request_payload = EXCLUDED.request_payload,
                        executive_summary = EXCLUDED.executive_summary,
                        human_report = EXCLUDED.human_report,
                        machine_report = EXCLUDED.machine_report,
                        finding_ids = EXCLUDED.finding_ids,
                        timeline_event_ids = EXCLUDED.timeline_event_ids,
                        causal_chains = EXCLUDED.causal_chains,
                        unresolved_hypotheses = EXCLUDED.unresolved_hypotheses,
                        recommended_verification_steps = EXCLUDED.recommended_verification_steps,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        report.report_id,
                        report.request_hash,
                        json.dumps(report.request_payload, sort_keys=True),
                        report.executive_summary,
                        report.human_report,
                        json.dumps(report.machine_report, sort_keys=True),
                        json.dumps(list(report.finding_ids), sort_keys=True),
                        json.dumps(list(report.timeline_event_ids), sort_keys=True),
                        json.dumps(list(report.causal_chains), sort_keys=True),
                        json.dumps(list(report.unresolved_hypotheses), sort_keys=True),
                        json.dumps(list(report.recommended_verification_steps), sort_keys=True),
                        report.created_at,
                        json.dumps(report.metadata, sort_keys=True),
                    ),
                )
            conn.commit()

    def record_verification_request(self, request: VerificationRequestRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_verification_requests (
                        request_id, audit_id, finding_id, broker_name, request_payload, status, created_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                    ON CONFLICT (request_id) DO UPDATE SET
                        audit_id = EXCLUDED.audit_id,
                        finding_id = EXCLUDED.finding_id,
                        broker_name = EXCLUDED.broker_name,
                        request_payload = EXCLUDED.request_payload,
                        status = EXCLUDED.status,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        request.request_id,
                        request.audit_id,
                        request.finding_id,
                        request.broker_name,
                        json.dumps(request.request_payload, sort_keys=True),
                        request.status,
                        request.created_at,
                        json.dumps(request.metadata, sort_keys=True),
                    ),
                )
            conn.commit()

    def record_audit_report(self, report: AuditReportRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_audit_reports (
                        report_id, request_hash, request_payload, executive_summary, human_report, machine_report,
                        finding_ids, timeline_event_ids, causal_chains, unresolved_hypotheses,
                        recommended_verification_steps, created_at, metadata
                    ) VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                    ON CONFLICT (report_id) DO UPDATE SET
                        request_hash = EXCLUDED.request_hash,
                        request_payload = EXCLUDED.request_payload,
                        executive_summary = EXCLUDED.executive_summary,
                        human_report = EXCLUDED.human_report,
                        machine_report = EXCLUDED.machine_report,
                        finding_ids = EXCLUDED.finding_ids,
                        timeline_event_ids = EXCLUDED.timeline_event_ids,
                        causal_chains = EXCLUDED.causal_chains,
                        unresolved_hypotheses = EXCLUDED.unresolved_hypotheses,
                        recommended_verification_steps = EXCLUDED.recommended_verification_steps,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        report.report_id,
                        report.request_hash,
                        json.dumps(report.request_payload, sort_keys=True),
                        report.executive_summary,
                        report.human_report,
                        json.dumps(report.machine_report, sort_keys=True),
                        json.dumps(list(report.finding_ids), sort_keys=True),
                        json.dumps(list(report.timeline_event_ids), sort_keys=True),
                        json.dumps(list(report.causal_chains), sort_keys=True),
                        json.dumps(list(report.unresolved_hypotheses), sort_keys=True),
                        json.dumps(list(report.recommended_verification_steps), sort_keys=True),
                        report.created_at,
                        json.dumps(report.metadata, sort_keys=True),
                    ),
                )
            conn.commit()

    def record_verification_request(self, request: VerificationRequestRecord) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_verification_requests (
                        request_id, audit_id, finding_id, broker_name, request_payload, status, created_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                    ON CONFLICT (request_id) DO UPDATE SET
                        audit_id = EXCLUDED.audit_id,
                        finding_id = EXCLUDED.finding_id,
                        broker_name = EXCLUDED.broker_name,
                        request_payload = EXCLUDED.request_payload,
                        status = EXCLUDED.status,
                        created_at = EXCLUDED.created_at,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        request.request_id,
                        request.audit_id,
                        request.finding_id,
                        request.broker_name,
                        json.dumps(request.request_payload, sort_keys=True),
                        request.status,
                        request.created_at,
                        json.dumps(request.metadata, sort_keys=True),
                    ),
                )
            conn.commit()

    def snapshot(self) -> dict[str, Any]:
        raise RuntimeError("PostgresKnowledgeStore snapshots are produced by database backups, not in-process export")

    def restore(self, snapshot: dict[str, Any]) -> None:
        raise RuntimeError("PostgresKnowledgeStore restore requires a database restore procedure")


class KnowledgePlatform:
    """High-level pipeline for ingestion, retrieval, and backup/restore helpers."""

    def __init__(
        self,
        store: KnowledgeStore,
        object_store: ObjectStore,
        embeddings: LocalFirstEmbeddingService | None = None,
    ):
        self.store = store
        self.object_store = object_store
        self.embeddings = embeddings or LocalFirstEmbeddingService()
        from .retrieval import HybridRetriever

        self.retriever = HybridRetriever(store, self.embeddings)

    @classmethod
    def in_memory(cls, *, root: str | None = None) -> "KnowledgePlatform":
        object_store = FilesystemObjectStore(root) if root else MemoryObjectStore()
        return cls(MemoryKnowledgeStore(), object_store)

    @classmethod
    def from_dsn(
        cls,
        dsn: str,
        *,
        object_store_root: str,
        embeddings: LocalFirstEmbeddingService | None = None,
        migration_package: str = "opennicf.knowledge.migrations",
    ) -> "KnowledgePlatform":
        """Build the production platform from injected runtime configuration."""
        return cls(
            PostgresKnowledgeStore.from_dsn(dsn, migration_package=migration_package),
            FilesystemObjectStore(object_store_root),
            embeddings,
        )

    @classmethod
    def from_env(
        cls,
        *,
        dsn_var: str = "OPENNICF_POSTGRES_DSN",
        object_store_var: str = "OPENNICF_OBJECT_STORE_ROOT",
        embeddings: LocalFirstEmbeddingService | None = None,
    ) -> "KnowledgePlatform":
        """Build from protected environment variables without logging their values."""
        dsn = os.environ.get(dsn_var)
        object_store_root = os.environ.get(object_store_var)
        if not dsn:
            raise RuntimeError(f"{dsn_var} is required for the PostgreSQL knowledge platform")
        if not object_store_root:
            raise RuntimeError(f"{object_store_var} is required for the PostgreSQL knowledge platform")
        return cls.from_dsn(dsn, object_store_root=object_store_root, embeddings=embeddings)

    @staticmethod
    def _source_version_id(source_id: str, content_hash: str) -> str:
        digest = sha256(f"source-version:{source_id}:{content_hash}".encode("utf-8")).hexdigest()
        return f"srcver_{digest}"

    @staticmethod
    def _artifact_hash(content_hash: str, parser_version: str) -> str:
        return sha256(f"{content_hash}:{parser_version}".encode("utf-8")).hexdigest()

    def ingest(
        self,
        *,
        source_id: str,
        source_uri: str,
        explicit_locator: str | None = None,
        content: str | bytes,
        blocks: Sequence[ParsedBlock] | None = None,
        mime_type: str = "text/plain",
        source_kind: str = "document",
        channel: str = "manual",
        domain: str = "general",
        system: str = "unknown",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "unknown",
        acl_scope: str = "internal",
        source_type: str = "document",
        parser_name: str = "identity",
        parser_version: str = "1",
        metadata: dict[str, Any] | None = None,
        namespace_label: str | None = None,
        namespace_summary: str | None = None,
        integration_edges: Sequence[Mapping[str, Any]] | None = None,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
    ) -> IngestBundle:
        content_bytes = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        content_hash = sha256(content_bytes).hexdigest()
        sanitized_metadata, redacted_values = sanitize_metadata(metadata)
        sanitized_source_uri = sanitize_text(source_uri, redacted_values)
        effective_domain_id = domain_id or domain
        effective_system_id = system_id or system
        effective_source_type = source_type or evidence_type or source_kind
        effective_evidence_type = evidence_type or source_type or source_kind
        namespace = build_namespace_record(
            domain_id=effective_domain_id,
            system_id=effective_system_id,
            component_id=component_id or sanitized_metadata.get("component_id"),
            environment=environment,
            evidence_type=effective_evidence_type,
            acl_scope=acl_scope,
            asset_name=namespace_label or sanitized_metadata.get("asset_name"),
            component_name=sanitized_metadata.get("component_name"),
            summary=namespace_summary,
            metadata={
                **sanitized_metadata,
                "source_id": source_id,
                "source_uri": sanitized_source_uri,
                "source_kind": source_kind,
                "source_type": effective_source_type,
                "domain_id": effective_domain_id,
                "system_id": effective_system_id,
                "component_id": component_id or sanitized_metadata.get("component_id"),
                "environment": environment,
                "evidence_type": effective_evidence_type,
                "acl_scope": acl_scope,
            },
            source_uri=sanitized_source_uri,
            source_kind=source_kind,
            source_type=effective_source_type,
        )
        integration_payload = {"integration_edges": list(integration_edges or sanitized_metadata.get("integration_edges", []) or [])}
        sanitized_integration_payload, _ = sanitize_metadata(integration_payload)
        extra_namespaces, integration_edge_records = build_integration_records(
            namespace,
            {**sanitized_metadata, **sanitized_integration_payload},
        )
        object_reference = self.object_store.put_bytes(
            content_bytes,
            mime_type=mime_type,
            metadata={**sanitized_metadata, "source_uri": sanitized_source_uri, "redacted_fields": list(redacted_values)},
        )
        source = KnowledgeSource(
            source_id=source_id,
            source_uri=sanitized_source_uri,
            source_kind=source_kind,
            channel=channel,
            namespace_id=namespace.namespace_id,
            domain_id=namespace.domain_id,
            system_id=namespace.system_id,
            component_id=namespace.component_id,
            environment=environment,
            evidence_type=namespace.evidence_type,
            acl_scope=acl_scope,
            mime_type=mime_type,
            size_bytes=len(content_bytes),
            content_hash=content_hash,
            domain=namespace.domain_id,
            system=namespace.system_id,
            source_type=effective_source_type,
            metadata={**sanitized_metadata, "source_uri": sanitized_source_uri, "namespace_id": namespace.namespace_id, "redacted_fields": list(redacted_values)},
        )
        version_number = 1
        if hasattr(self.store, "next_version_number"):
            version_number = int(self.store.next_version_number(source_id, content_hash))
        elif hasattr(self.store, "source_versions"):
            version_number = len([version for version in getattr(self.store, "source_versions").values() if version.source_id == source_id]) + 1
        version = KnowledgeSourceVersion(
            source_version_id=self._source_version_id(source_id, content_hash),
            source_id=source_id,
            source_uri=sanitized_source_uri,
            version_number=version_number,
            content_hash=content_hash,
            artifact_hash=self._artifact_hash(content_hash, parser_version),
            ingest_timestamp=utcnow(),
            parser_version=parser_version,
            storage_backend=object_reference.backend,
            object_key=object_reference.object_key,
            metadata={**sanitized_metadata, "source_uri": sanitized_source_uri, "namespace_id": namespace.namespace_id, "redacted_fields": list(redacted_values)},
        )
        artifact = ArtifactRecord(
            artifact_hash=version.artifact_hash,
            source_version_id=version.source_version_id,
            object_key=object_reference.object_key,
            storage_backend=object_reference.backend,
            mime_type=mime_type,
            size_bytes=len(content_bytes),
            parser_name=parser_name,
            parser_version=parser_version,
            metadata={**sanitized_metadata, "source_uri": sanitized_source_uri, "namespace_id": namespace.namespace_id, "redacted_fields": list(redacted_values)},
        )
        parsed_blocks = tuple(blocks) if blocks is not None else tuple(
            ParsedBlock(text=block_text, line_start=line_start, line_end=line_end)
            for block_text, line_start, line_end in _split_blocks(
                content if isinstance(content, str) else content_bytes.decode("utf-8", errors="surrogateescape")
            )
        )
        sanitized_blocks: list[ParsedBlock] = []
        for block in parsed_blocks:
            block_metadata, block_redactions = sanitize_metadata(block.metadata)
            all_redactions = list(dict.fromkeys([*redacted_values, *block_redactions]))
            sanitized_blocks.append(
                replace(
                    block,
                    text=sanitize_text(block.text, all_redactions),
                    metadata={**block_metadata, "redacted_fields": all_redactions} if all_redactions else block_metadata,
                )
            )
        chunk_texts = [block.text for block in sanitized_blocks]
        embedding_result = self.embeddings.embed(
            chunk_texts,
            prefer_gpu=True,
            available_memory_bytes=available_memory_bytes,
            available_vram_bytes=available_vram_bytes,
        ) if chunk_texts else EmbeddingResult(model=self.embeddings.info().model, dimensions=self.embeddings.info().dimensions, device=self.embeddings.info().device, vectors=())
        chunks: list[ChunkRecord] = []
        embeddings: list[EmbeddingRecord] = []
        for ordinal, block in enumerate(sanitized_blocks):
            chunk_digest = sha256(f"chunk:{version.source_version_id}:{ordinal}".encode("utf-8")).hexdigest()
            chunk_id = f"chunk_{chunk_digest}"
            chunk_hash = sha256(f"{version.source_version_id}:{ordinal}:{block.text}".encode("utf-8")).hexdigest()
            if block.locator:
                locator = sanitize_text(block.locator, redacted_values)
            elif explicit_locator:
                locator = sanitize_text(explicit_locator, redacted_values)
            else:
                locator = _chunk_locator(sanitized_source_uri, ordinal, block.line_start, block.line_end)
            chunk_metadata = {
                **sanitized_metadata,
                **block.metadata,
                "source_uri": sanitized_source_uri,
                "namespace_id": namespace.namespace_id,
            }
            chunk = ChunkRecord(
                chunk_id=chunk_id,
                source_id=source_id,
                source_version_id=version.source_version_id,
                artifact_hash=artifact.artifact_hash,
                namespace_id=namespace.namespace_id,
                domain_id=namespace.domain_id,
                system_id=namespace.system_id,
                component_id=namespace.component_id,
                environment=namespace.environment,
                evidence_type=namespace.evidence_type,
                acl_scope=acl_scope,
                ordinal=ordinal,
                text=block.text,
                locator=locator,
                chunk_hash=chunk_hash,
                parser_version=parser_version,
                domain=namespace.domain_id,
                system=namespace.system_id,
                source_type=effective_source_type,
                page=block.page,
                line_start=block.line_start,
                line_end=block.line_end,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
            embeddings.append(
                EmbeddingRecord(
                    chunk_id=chunk_id,
                    model=embedding_result.model,
                    dimensions=embedding_result.dimensions,
                    device=embedding_result.device,
                    vector=embedding_result.vectors[ordinal] if ordinal < len(embedding_result.vectors) else (),
                    metadata={"fallback": embedding_result.fallback, **sanitized_metadata},
                )
            )
        bundle = IngestBundle(
            source=source,
            version=version,
            artifact=artifact,
            chunks=tuple(chunks),
            embeddings=tuple(embeddings),
            object_reference=object_reference,
            created=True,
            namespaces=(namespace, *extra_namespaces),
            integration_edges=integration_edge_records,
        )
        return self.store.save_bundle(bundle)

    def search(self, query: str, *, filters: RetrievalFilters | None = None, route: str | None = None):
        return self.retriever.search(query, filters=filters, route=route)

    def backup(self) -> dict[str, Any]:
        store_snapshot = self.store.snapshot()
        object_snapshot = getattr(self.object_store, "snapshot", lambda: None)()
        return {
            "store": store_snapshot,
            "object_store": object_snapshot,
            "embeddings": self.embeddings.describe(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.store.restore(snapshot["store"])
        if snapshot.get("object_store") and hasattr(self.object_store, "restore"):
            restored = self.object_store.restore(snapshot["object_store"])
            self.object_store = restored


def _split_sql(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for char in sql:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == ";" and not in_single and not in_double:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def _upsert_source(cur, source: KnowledgeSource) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_sources (
            source_id, source_uri, source_kind, channel, namespace_id, domain_id, system_id, component_id,
            environment, evidence_type, acl_scope, mime_type, size_bytes, content_hash, created_at, updated_at,
            domain, system, source_type, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (source_id) DO UPDATE SET
            source_uri = EXCLUDED.source_uri,
            source_kind = EXCLUDED.source_kind,
            channel = EXCLUDED.channel,
            namespace_id = EXCLUDED.namespace_id,
            domain_id = EXCLUDED.domain_id,
            system_id = EXCLUDED.system_id,
            component_id = EXCLUDED.component_id,
            environment = EXCLUDED.environment,
            evidence_type = EXCLUDED.evidence_type,
            acl_scope = EXCLUDED.acl_scope,
            mime_type = EXCLUDED.mime_type,
            size_bytes = EXCLUDED.size_bytes,
            content_hash = EXCLUDED.content_hash,
            updated_at = EXCLUDED.updated_at,
            domain = EXCLUDED.domain,
            system = EXCLUDED.system,
            source_type = EXCLUDED.source_type,
            metadata = EXCLUDED.metadata
        """,
        (
            source.source_id,
            source.source_uri,
            source.source_kind,
            source.channel,
            source.namespace_id,
            source.domain_id,
            source.system_id,
            source.component_id,
            source.environment,
            source.evidence_type,
            source.acl_scope,
            source.mime_type,
            source.size_bytes,
            source.content_hash,
            source.created_at,
            source.updated_at,
            source.domain,
            source.system,
            source.source_type,
            json.dumps(source.metadata, sort_keys=True),
        ),
    )


def _upsert_version(cur, version: KnowledgeSourceVersion) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_source_versions (
            source_version_id, source_id, source_uri, version_number, content_hash,
            artifact_hash, ingest_timestamp, parser_version, storage_backend, object_key, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (source_id, content_hash) DO UPDATE SET
            source_uri = EXCLUDED.source_uri,
            version_number = EXCLUDED.version_number,
            artifact_hash = EXCLUDED.artifact_hash,
            ingest_timestamp = EXCLUDED.ingest_timestamp,
            parser_version = EXCLUDED.parser_version,
            storage_backend = EXCLUDED.storage_backend,
            object_key = EXCLUDED.object_key,
            metadata = EXCLUDED.metadata
        """,
        (
            version.source_version_id,
            version.source_id,
            version.source_uri,
            version.version_number,
            version.content_hash,
            version.artifact_hash,
            version.ingest_timestamp,
            version.parser_version,
            version.storage_backend,
            version.object_key,
            json.dumps(version.metadata, sort_keys=True),
        ),
    )


def _upsert_artifact(cur, artifact: ArtifactRecord) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_artifacts (
            artifact_hash, source_version_id, object_key, storage_backend, mime_type,
            size_bytes, parser_name, parser_version, created_at, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (artifact_hash) DO UPDATE SET
            source_version_id = EXCLUDED.source_version_id,
            object_key = EXCLUDED.object_key,
            storage_backend = EXCLUDED.storage_backend,
            mime_type = EXCLUDED.mime_type,
            size_bytes = EXCLUDED.size_bytes,
            parser_name = EXCLUDED.parser_name,
            parser_version = EXCLUDED.parser_version,
            created_at = EXCLUDED.created_at,
            metadata = EXCLUDED.metadata
        """,
        (
            artifact.artifact_hash,
            artifact.source_version_id,
            artifact.object_key,
            artifact.storage_backend,
            artifact.mime_type,
            artifact.size_bytes,
            artifact.parser_name,
            artifact.parser_version,
            artifact.created_at,
            json.dumps(artifact.metadata, sort_keys=True),
        ),
    )


def _upsert_namespace(cur, namespace: KnowledgeNamespaceRecord) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_namespaces (
            namespace_id, domain_id, system_id, component_id, environment, evidence_type, acl_scope,
            asset_name, component_name, sanitized_summary, metadata, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (namespace_id) DO UPDATE SET
            domain_id = EXCLUDED.domain_id,
            system_id = EXCLUDED.system_id,
            component_id = EXCLUDED.component_id,
            environment = EXCLUDED.environment,
            evidence_type = EXCLUDED.evidence_type,
            acl_scope = EXCLUDED.acl_scope,
            asset_name = EXCLUDED.asset_name,
            component_name = EXCLUDED.component_name,
            sanitized_summary = EXCLUDED.sanitized_summary,
            metadata = EXCLUDED.metadata,
            created_at = EXCLUDED.created_at
        """,
        (
            namespace.namespace_id,
            namespace.domain_id,
            namespace.system_id,
            namespace.component_id,
            namespace.environment,
            namespace.evidence_type,
            namespace.acl_scope,
            namespace.asset_name,
            namespace.component_name,
            namespace.sanitized_summary,
            json.dumps(namespace.metadata, sort_keys=True),
            namespace.created_at,
        ),
    )


def _upsert_integration_edge(cur, edge: IntegrationEdgeRecord) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_integration_edges (
            edge_id, source_namespace_id, target_namespace_id, relation_type, domain_id, system_id,
            component_id, environment, evidence_type, acl_scope, metadata, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (edge_id) DO UPDATE SET
            source_namespace_id = EXCLUDED.source_namespace_id,
            target_namespace_id = EXCLUDED.target_namespace_id,
            relation_type = EXCLUDED.relation_type,
            domain_id = EXCLUDED.domain_id,
            system_id = EXCLUDED.system_id,
            component_id = EXCLUDED.component_id,
            environment = EXCLUDED.environment,
            evidence_type = EXCLUDED.evidence_type,
            acl_scope = EXCLUDED.acl_scope,
            metadata = EXCLUDED.metadata,
            created_at = EXCLUDED.created_at
        """,
        (
            edge.edge_id,
            edge.source_namespace_id,
            edge.target_namespace_id,
            edge.relation_type,
            edge.domain_id,
            edge.system_id,
            edge.component_id,
            edge.environment,
            edge.evidence_type,
            edge.acl_scope,
            json.dumps(edge.metadata, sort_keys=True),
            edge.created_at,
        ),
    )


def _upsert_chunk(cur, chunk: ChunkRecord) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_chunks (
            chunk_id, source_id, source_version_id, artifact_hash, namespace_id, domain_id, system_id,
            component_id, environment, evidence_type, acl_scope, ordinal, text, locator,
            chunk_hash, parser_version, domain, system, source_type, page, line_start, line_end, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (chunk_id) DO UPDATE SET
            source_id = EXCLUDED.source_id,
            source_version_id = EXCLUDED.source_version_id,
            artifact_hash = EXCLUDED.artifact_hash,
            namespace_id = EXCLUDED.namespace_id,
            domain_id = EXCLUDED.domain_id,
            system_id = EXCLUDED.system_id,
            component_id = EXCLUDED.component_id,
            environment = EXCLUDED.environment,
            evidence_type = EXCLUDED.evidence_type,
            ordinal = EXCLUDED.ordinal,
            text = EXCLUDED.text,
            locator = EXCLUDED.locator,
            chunk_hash = EXCLUDED.chunk_hash,
            parser_version = EXCLUDED.parser_version,
            acl_scope = EXCLUDED.acl_scope,
            domain = EXCLUDED.domain,
            system = EXCLUDED.system,
            source_type = EXCLUDED.source_type,
            page = EXCLUDED.page,
            line_start = EXCLUDED.line_start,
            line_end = EXCLUDED.line_end,
            metadata = EXCLUDED.metadata
        """,
        (
            chunk.chunk_id,
            chunk.source_id,
            chunk.source_version_id,
            chunk.artifact_hash,
            chunk.namespace_id,
            chunk.domain_id,
            chunk.system_id,
            chunk.component_id,
            chunk.environment,
            chunk.evidence_type,
            chunk.acl_scope,
            chunk.ordinal,
            chunk.text,
            chunk.locator,
            chunk.chunk_hash,
            chunk.parser_version,
            chunk.domain,
            chunk.system,
            chunk.source_type,
            chunk.page,
            chunk.line_start,
            chunk.line_end,
            json.dumps(chunk.metadata, sort_keys=True),
        ),
    )


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def _parse_vector_value(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, tuple):
        return tuple(float(item) for item in value)
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            return ()
        return tuple(float(item) for item in text.split(","))
    return tuple(float(item) for item in value)


def _upsert_embedding(cur, embedding: EmbeddingRecord) -> None:
    cur.execute(
        """
        INSERT INTO knowledge_embeddings (
            chunk_id, model, dimensions, device, vector, created_at, metadata
        ) VALUES (%s, %s, %s, %s, %s::vector, %s, %s::jsonb)
        ON CONFLICT (chunk_id, model, dimensions) DO UPDATE SET
            device = EXCLUDED.device,
            vector = EXCLUDED.vector,
            created_at = EXCLUDED.created_at,
            metadata = EXCLUDED.metadata
        """,
        (
            embedding.chunk_id,
            embedding.model,
            embedding.dimensions,
            embedding.device,
            _vector_literal(embedding.vector),
            embedding.created_at,
            json.dumps(embedding.metadata, sort_keys=True),
        ),
    )


def _build_search_sql(filters: RetrievalFilters) -> tuple[str, tuple[Any, ...]]:
    clauses = ["1 = 1"]
    params: list[Any] = []
    allowed_namespaces = filters.effective_namespace_ids()
    allowed_domains = filters.effective_domain_ids()
    allowed_systems = filters.effective_system_ids()
    allowed_components = filters.effective_component_ids()
    allowed_evidence_types = filters.effective_evidence_types()
    if filters.principal_acl_scopes:
        clauses.append("c.acl_scope = ANY(%s)")
        params.append(list(filters.principal_acl_scopes))
    if allowed_namespaces:
        clauses.append("c.namespace_id = ANY(%s)")
        params.append(list(allowed_namespaces))
    if allowed_domains:
        clauses.append("c.domain_id = ANY(%s)")
        params.append(list(allowed_domains))
    if allowed_systems:
        clauses.append("c.system_id = ANY(%s)")
        params.append(list(allowed_systems))
    if allowed_components:
        clauses.append("c.component_id = ANY(%s)")
        params.append(list(allowed_components))
    if allowed_evidence_types:
        clauses.append("c.evidence_type = ANY(%s)")
        params.append(list(allowed_evidence_types))
    if filters.environments:
        clauses.append("c.environment = ANY(%s)")
        params.append(list(filters.environments))
    if filters.source_types:
        clauses.append("c.source_type = ANY(%s)")
        params.append(list(filters.source_types))
    if filters.source_ids:
        clauses.append("c.source_id = ANY(%s)")
        params.append(list(filters.source_ids))
    if filters.since:
        clauses.append("v.ingest_timestamp >= %s")
        params.append(filters.since)
    if filters.until:
        clauses.append("v.ingest_timestamp <= %s")
        params.append(filters.until)

    sql = f"""
        SELECT
            c.chunk_id, c.source_id, c.source_version_id, c.artifact_hash, c.namespace_id, c.domain_id,
            c.system_id, c.component_id, c.environment, c.evidence_type, c.acl_scope, c.ordinal, c.text,
            c.locator, c.chunk_hash, c.parser_version, c.domain, c.system, c.source_type, c.page,
            c.line_start, c.line_end, c.metadata,
            s.source_id, s.source_uri, s.source_kind, s.channel, s.namespace_id, s.domain_id, s.system_id,
            s.component_id, s.environment, s.evidence_type, s.acl_scope, s.mime_type, s.size_bytes,
            s.content_hash, s.created_at, s.updated_at, s.domain, s.system, s.source_type, s.metadata,
            v.source_version_id, v.source_id, v.source_uri, v.version_number, v.content_hash, v.artifact_hash,
            v.ingest_timestamp, v.parser_version, v.storage_backend, v.object_key, v.metadata,
            a.artifact_hash, a.source_version_id, a.object_key, a.storage_backend, a.mime_type, a.size_bytes,
            a.parser_name, a.parser_version, a.created_at, a.metadata,
            e.model, e.dimensions, e.device, e.vector, e.created_at, e.metadata
        FROM knowledge_chunks c
        JOIN knowledge_sources s ON s.source_id = c.source_id
        JOIN knowledge_source_versions v ON v.source_version_id = c.source_version_id
        JOIN knowledge_artifacts a ON a.artifact_hash = c.artifact_hash
        JOIN knowledge_embeddings e ON e.chunk_id = c.chunk_id
        WHERE {" AND ".join(clauses)}
        ORDER BY c.source_version_id, c.ordinal
        LIMIT %s
    """
    params.append(filters.limit * max(1, filters.neighbor_window + 1))
    return sql, tuple(params)


def _time_range_to_json(time_range: tuple[datetime | None, datetime | None] | None) -> dict[str, Any] | None:
    if time_range is None:
        return None
    start, end = time_range
    return {
        "start": start.isoformat() if start is not None else None,
        "end": end.isoformat() if end is not None else None,
    }


def _time_range_from_json(data: Any) -> tuple[datetime | None, datetime | None] | None:
    if data in (None, ""):
        return None
    if isinstance(data, dict):
        start = data.get("start")
        end = data.get("end")
    else:
        start, end = data
    return (
        datetime.fromisoformat(start) if start else None,
        datetime.fromisoformat(end) if end else None,
    )


def _audit_evidence_ref_from_json(data: dict[str, Any]) -> AuditEvidenceRefRecord:
    return AuditEvidenceRefRecord(
        reference_id=data["reference_id"],
        source_id=data["source_id"],
        source_version_id=data["source_version_id"],
        source_type=data["source_type"],
        locator=data["locator"],
        role=data["role"],
        provenance_ref=data["provenance_ref"],
        excerpt_hash=data["excerpt_hash"],
        statement=data["statement"],
        metadata=dict(data.get("metadata", {})),
    )


def _audit_finding_from_json(data: dict[str, Any]) -> AuditFindingRecord:
    return AuditFindingRecord(
        finding_id=data["finding_id"],
        classification=data.get("classification", data.get("finding_class", "fact")),
        statement=data["statement"],
        confidence=float(data["confidence"]),
        evidence_refs=tuple(_audit_evidence_ref_from_json(item) for item in data.get("evidence_refs", data.get("evidence_links", []))),
        time_range=_time_range_from_json(data.get("time_range")),
        systems=tuple(data.get("systems", [])),
        components=tuple(data.get("components", [])),
        verification_status=data.get("verification_status", data.get("status", "unverified")),
        recommended_query=data.get("recommended_query"),
        diagnostic_action=dict(data["diagnostic_action"]) if data.get("diagnostic_action") else None,
        supporting_evidence=tuple(data.get("supporting_evidence", [])),
        contradicting_evidence=tuple(data.get("contradicting_evidence", [])),
        provenance_refs=tuple(data.get("provenance_refs", [])),
        correlation_ids=tuple(data.get("correlation_ids", [])),
        source_type_analyzers=tuple(data.get("source_type_analyzers", [])),
        created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", utcnow()),
        metadata=dict(data.get("metadata", {})),
    )


def _audit_report_from_json(data: dict[str, Any]) -> AuditReportRecord:
    return AuditReportRecord(
        report_id=data["report_id"],
        request_hash=data["request_hash"],
        request_payload=dict(data.get("request_payload", {})),
        executive_summary=data["executive_summary"],
        human_report=data["human_report"],
        machine_report=dict(data.get("machine_report", {})),
        finding_ids=tuple(data.get("finding_ids", [])),
        timeline_event_ids=tuple(data.get("timeline_event_ids", [])),
        causal_chains=tuple(data.get("causal_chains", [])),
        unresolved_hypotheses=tuple(data.get("unresolved_hypotheses", [])),
        recommended_verification_steps=tuple(data.get("recommended_verification_steps", [])),
        created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", utcnow()),
        metadata=dict(data.get("metadata", {})),
    )


def _verification_request_from_json(data: dict[str, Any]) -> VerificationRequestRecord:
    return VerificationRequestRecord(
        request_id=data["request_id"],
        audit_id=data["audit_id"],
        finding_id=data.get("finding_id"),
        broker_name=data["broker_name"],
        request_payload=dict(data.get("request_payload", {})),
        status=data["status"],
        created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", utcnow()),
        metadata=dict(data.get("metadata", {})),
    )


def _split_sql(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    for char in sql:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        if char == ";" and not in_single and not in_double:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements
