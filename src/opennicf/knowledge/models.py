"""Dataclasses describing the canonical evidence and retrieval model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SourceKind:
    """Logical source categories observed by the pipeline."""

    kind: str


@dataclass(frozen=True)
class ParsedBlock:
    """Canonical text unit emitted by a parser."""

    text: str
    locator: str | None = None
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    source_uri: str
    source_kind: str
    channel: str
    domain: str
    system: str
    environment: str
    acl_scope: str
    mime_type: str
    size_bytes: int
    content_hash: str
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeSourceVersion:
    source_version_id: str
    source_id: str
    source_uri: str
    version_number: int
    content_hash: str
    artifact_hash: str
    ingest_timestamp: datetime
    parser_version: str
    storage_backend: str
    object_key: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_hash: str
    source_version_id: str
    object_key: str
    storage_backend: str
    mime_type: str
    size_bytes: int
    parser_name: str
    parser_version: str
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    source_version_id: str
    artifact_hash: str
    ordinal: int
    text: str
    locator: str
    chunk_hash: str
    parser_version: str
    acl_scope: str
    domain: str
    system: str
    environment: str
    source_type: str
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    model: str
    dimensions: int
    device: str
    vector: tuple[float, ...]
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditEvidenceRefRecord:
    reference_id: str
    source_id: str
    source_version_id: str
    source_type: str
    locator: str
    role: str
    provenance_ref: str
    excerpt_hash: str
    statement: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditFindingRecord:
    finding_id: str
    classification: str
    statement: str
    confidence: float
    evidence_refs: tuple[AuditEvidenceRefRecord, ...]
    time_range: tuple[datetime | None, datetime | None] | None = None
    systems: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    verification_status: str = "unverified"
    recommended_query: str | None = None
    diagnostic_action: dict[str, Any] | None = None
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    correlation_ids: tuple[str, ...] = ()
    source_type_analyzers: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditTimelineEventRecord:
    event_id: str
    audit_id: str
    timestamp: datetime | None
    statement: str
    source_id: str
    source_type: str
    locator: str
    correlation_ids: tuple[str, ...] = ()
    systems: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    provenance_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditReportRecord:
    report_id: str
    request_hash: str
    request_payload: dict[str, Any]
    executive_summary: str
    human_report: str
    machine_report: dict[str, Any]
    finding_ids: tuple[str, ...]
    timeline_event_ids: tuple[str, ...]
    causal_chains: tuple[str, ...]
    unresolved_hypotheses: tuple[str, ...]
    recommended_verification_steps: tuple[str, ...]
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationRequestRecord:
    request_id: str
    audit_id: str
    finding_id: str | None
    broker_name: str
    request_payload: dict[str, Any]
    status: str
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalEventRecord:
    event_id: str
    query_hash: str
    filters: dict[str, Any]
    selected_chunks: tuple[str, ...]
    scores: dict[str, float]
    reranker: str | None
    model_route: str | None
    created_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalFilters:
    principal_acl_scopes: frozenset[str] = frozenset()
    domains: tuple[str, ...] = ()
    systems: tuple[str, ...] = ()
    environments: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 10
    neighbor_window: int = 1

    def normalized(self) -> "RetrievalFilters":
        return RetrievalFilters(
            principal_acl_scopes=frozenset(scope for scope in self.principal_acl_scopes if scope),
            domains=tuple(domain for domain in self.domains if domain),
            systems=tuple(system for system in self.systems if system),
            environments=tuple(environment for environment in self.environments if environment),
            source_types=tuple(source_type for source_type in self.source_types if source_type),
            source_ids=tuple(source_id for source_id in self.source_ids if source_id),
            since=self.since,
            until=self.until,
            limit=max(1, int(self.limit)),
            neighbor_window=max(0, int(self.neighbor_window)),
        )


@dataclass(frozen=True)
class SearchCandidate:
    chunk: ChunkRecord
    source: KnowledgeSource
    version: KnowledgeSourceVersion
    artifact: ArtifactRecord
    embedding: EmbeddingRecord | None = None


@dataclass(frozen=True)
class EvidenceHit:
    source_id: str
    source_version_id: str
    artifact_hash: str
    locator: str
    text: str
    excerpt_hash: str
    ingest_timestamp: datetime
    parser_version: str
    acl_scope: str
    domain: str
    system: str
    environment: str
    source_type: str
    semantic_score: float
    lexical_score: float
    score: float
    model: str
    dimensions: int
    chunk_id: str
    chunk_ordinal: int
    neighboring_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestBundle:
    source: KnowledgeSource
    version: KnowledgeSourceVersion
    artifact: ArtifactRecord
    chunks: tuple[ChunkRecord, ...]
    embeddings: tuple[EmbeddingRecord, ...]
    object_reference: "ObjectReference"
    created: bool
