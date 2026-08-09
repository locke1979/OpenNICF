from __future__ import annotations

import json
from pathlib import Path

from opennicf import (
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    RetrievalFilters,
)
from opennicf.knowledge.embeddings import EmbeddingModelInfo, EmbeddingResult


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge"


def _load_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _load_json(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


class _RecordingEmbeddingBackend:
    def __init__(self):
        self.texts: list[str] = []
        self._info = EmbeddingModelInfo(model="recording-8", dimensions=8, device="cpu", backend="recording", fallback=True)

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def embed(self, texts):
        self.texts.extend(texts)
        vectors = tuple(((0.0,) * self._info.dimensions for _ in texts))
        return EmbeddingResult(
            model=self._info.model,
            dimensions=self._info.dimensions,
            device=self._info.device,
            vectors=vectors,
            fallback=self._info.fallback,
            metadata=dict(self._info.metadata),
        )


def _platform_with_recorder() -> tuple[KnowledgePlatform, _RecordingEmbeddingBackend]:
    backend = _RecordingEmbeddingBackend()
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=backend),
    )
    return platform, backend


def test_namespace_registry_redacts_credentials_before_embedding_and_persists_edges():
    platform, backend = _platform_with_recorder()
    metadata = _load_json("namespace-metadata.json")
    content = _load_text("namespace-evidence.md")

    bundle = platform.ingest(
        source_id="criminal-namespace-1",
        source_uri="https://billing.internal.local/api",
        content=content,
        domain="criminal",
        system="criminal_casework",
        domain_id="criminal",
        system_id="criminal_casework",
        component_id="criminal_evidence",
        evidence_type="document",
        environment="prod",
        acl_scope="internal",
        source_type="document",
        parser_name="identity",
        parser_version="1",
        metadata=metadata,
    )

    assert bundle.namespaces
    assert bundle.integration_edges
    namespace = bundle.namespaces[0]
    edge = bundle.integration_edges[0]

    assert namespace.domain_id == "criminal"
    assert namespace.system_id == "criminal_casework"
    assert namespace.component_id == "criminal_evidence"
    assert namespace.evidence_type == "document"
    assert namespace.acl_scope == "internal"
    assert namespace.sanitized_summary
    assert "secret-token-123" not in namespace.sanitized_summary
    assert "billing.internal.local" not in namespace.sanitized_summary
    assert namespace.metadata["api_key"] == "[redacted]"
    assert namespace.metadata["endpoint"] == "[redacted]"
    assert namespace.metadata["redacted_fields"]

    assert edge.relation_type == "delegates-to"
    assert edge.source_namespace_id == namespace.namespace_id
    assert edge.target_namespace_id != namespace.namespace_id

    assert platform.store.namespaces[namespace.namespace_id].namespace_id == namespace.namespace_id
    assert platform.store.integration_edges[edge.edge_id].edge_id == edge.edge_id
    assert any("[redacted]" in chunk.text for chunk in platform.store.chunks.values())
    assert all("secret-token-123" not in text for text in backend.texts)
    assert all("billing.internal.local" not in text for text in backend.texts)
    assert all("bearer-secret-456" not in text for text in backend.texts)


def test_cross_domain_retrieval_requires_explicit_delegation():
    platform, _backend = _platform_with_recorder()
    platform.ingest(
        source_id="criminal-doc",
        source_uri="tests/fixtures/knowledge/document.md",
        content="criminal only evidence",
        acl_scope="internal",
        domain="criminal",
        system="criminal_casework",
        domain_id="criminal",
        system_id="criminal_casework",
        component_id="criminal_evidence",
        evidence_type="document",
        environment="dev",
        source_type="document",
        parser_version="1",
    )
    platform.ingest(
        source_id="sigef-doc",
        source_uri="tests/fixtures/knowledge/document.md",
        content="sigef only evidence",
        acl_scope="internal",
        domain="encargos_sigef",
        system="encargos_sigef_casework",
        domain_id="encargos_sigef",
        system_id="encargos_sigef_casework",
        component_id="encargos_sigef_evidence",
        evidence_type="document",
        environment="dev",
        source_type="document",
        parser_version="1",
    )

    locked_hits = platform.search(
        "evidence",
        filters=RetrievalFilters(principal_domain_id="criminal", domain_ids=("criminal", "encargos_sigef"), limit=10),
        route="local",
    )
    assert locked_hits
    assert all(hit.domain_id == "criminal" for hit in locked_hits)
    assert all(hit.namespace_id for hit in locked_hits)

    delegated_hits = platform.search(
        "evidence",
        filters=RetrievalFilters(
            principal_domain_id="criminal",
            domain_ids=("criminal", "encargos_sigef"),
            delegated_domain_ids=("encargos_sigef",),
            limit=10,
        ),
        route="local",
    )
    assert delegated_hits
    assert {hit.domain_id for hit in delegated_hits} == {"criminal", "encargos_sigef"}
    assert all(hit.evidence_type == "document" for hit in delegated_hits)
