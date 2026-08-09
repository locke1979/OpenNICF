"""Knowledge platform primitives for OpenNICF."""

from .embeddings import (
    EmbeddingBackend,
    EmbeddingModelInfo,
    EmbeddingResult,
    EmbeddingBatchSizer,
    HashingEmbeddingBackend,
    LocalFirstEmbeddingService,
)
from .models import (
    ArtifactRecord,
    AuditFindingRecord,
    ChunkRecord,
    EvidenceHit,
    EmbeddingRecord,
    IngestBundle,
    KnowledgeSource,
    KnowledgeSourceVersion,
    RetrievalEventRecord,
    RetrievalFilters,
    SearchCandidate,
    SourceKind,
)
from .object_store import (
    FilesystemObjectStore,
    MemoryObjectStore,
    ObjectReference,
    ObjectStore,
)
from .retrieval import HybridRetriever, cosine_similarity
from .store import KnowledgePlatform, MemoryKnowledgeStore, PostgresKnowledgeStore

__all__ = [
    "ArtifactRecord",
    "AuditFindingRecord",
    "ChunkRecord",
    "EmbeddingBackend",
    "EmbeddingBatchSizer",
    "EmbeddingModelInfo",
    "EmbeddingRecord",
    "EmbeddingResult",
    "EvidenceHit",
    "FilesystemObjectStore",
    "HashingEmbeddingBackend",
    "HybridRetriever",
    "IngestBundle",
    "KnowledgePlatform",
    "KnowledgeSource",
    "KnowledgeSourceVersion",
    "LocalFirstEmbeddingService",
    "MemoryKnowledgeStore",
    "MemoryObjectStore",
    "ObjectReference",
    "ObjectStore",
    "PostgresKnowledgeStore",
    "RetrievalEventRecord",
    "RetrievalFilters",
    "SearchCandidate",
    "SourceKind",
    "cosine_similarity",
]
