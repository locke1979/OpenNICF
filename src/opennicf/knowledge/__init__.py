"""Knowledge platform primitives for OpenNICF."""

from .admin import AdminOperation, KnowledgeAdministration, RetirementPolicy
from .embeddings import (
    EmbeddingBackend,
    EmbeddingBatchSizer,
    EmbeddingError,
    EmbeddingMigration,
    EmbeddingModelInfo,
    EmbeddingResult,
    EmbeddingSpace,
    EmbeddingSpaceMismatch,
    GeminiEmbeddingBackend,
    HashingEmbeddingBackend,
    HttpEmbeddingBackend,
    LocalFirstEmbeddingService,
    PrivacyBoundaryError,
    QwenEmbeddingBackend,
)
from .models import (
    ArtifactRecord,
    AuditEvidenceRefRecord,
    AuditFindingRecord,
    AuditReportRecord,
    AuditTimelineEventRecord,
    ChunkRecord,
    CodeRelationshipRecord,
    CodeSymbolRecord,
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
    VerificationRequestRecord,
)
from .namespaces import (
    IntegrationEdgeRecord,
    KnowledgeNamespaceRecord,
    build_namespace_record,
)
from .object_store import (
    FilesystemObjectStore,
    MemoryObjectStore,
    ObjectReference,
    ObjectStore,
)
from .retrieval import HybridRetriever, cosine_similarity
from .store import KnowledgePlatform, MemoryKnowledgeStore, PostgresKnowledgeStore

_BENCHMARK_EXPORTS = {"BenchmarkManifestError", "RetrievalBenchmark", "compact_hits", "load_manifest"}


def __getattr__(name):
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark
        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdminOperation",
    "ArtifactRecord",
    "AuditEvidenceRefRecord",
    "AuditFindingRecord",
    "AuditReportRecord",
    "AuditTimelineEventRecord",
    "BenchmarkManifestError",
    "ChunkRecord",
    "CodeRelationshipRecord",
    "CodeSymbolRecord",
    "EmbeddingBackend",
    "EmbeddingBatchSizer",
    "EmbeddingError",
    "EmbeddingMigration",
    "EmbeddingModelInfo",
    "EmbeddingRecord",
    "EmbeddingResult",
    "EmbeddingSpace",
    "EmbeddingSpaceMismatch",
    "EvidenceHit",
    "FilesystemObjectStore",
    "GeminiEmbeddingBackend",
    "HashingEmbeddingBackend",
    "HttpEmbeddingBackend",
    "HybridRetriever",
    "IngestBundle",
    "IntegrationEdgeRecord",
    "KnowledgeAdministration",
    "KnowledgeNamespaceRecord",
    "KnowledgePlatform",
    "KnowledgeSource",
    "KnowledgeSourceVersion",
    "LocalFirstEmbeddingService",
    "MemoryKnowledgeStore",
    "MemoryObjectStore",
    "ObjectReference",
    "ObjectStore",
    "ParsedBlock",
    "PostgresKnowledgeStore",
    "PrivacyBoundaryError",
    "QwenEmbeddingBackend",
    "RetirementPolicy",
    "RetrievalBenchmark",
    "RetrievalEventRecord",
    "RetrievalFilters",
    "SearchCandidate",
    "SourceKind",
    "VerificationRequestRecord",
    "build_namespace_record",
    "compact_hits",
    "cosine_similarity",
    "load_manifest",
]
