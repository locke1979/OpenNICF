from __future__ import annotations

import math

import pytest

from opennicf.knowledge import (
    EmbeddingError,
    EmbeddingMigration,
    GeminiEmbeddingBackend,
    HashingEmbeddingBackend,
    LocalFirstEmbeddingService,
    PrivacyBoundaryError,
    QwenEmbeddingBackend,
)
from opennicf.knowledge.models import EmbeddingRecord, RetrievalFilters
from opennicf.knowledge.store import _build_search_sql, _upsert_embedding


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def test_qwen_query_document_contract_and_normalized_768_output():
    calls = []

    def loader(texts, **kwargs):
        calls.append((texts, kwargs))
        return [[1.0] * kwargs["dimension"] for _ in texts]

    backend = QwenEmbeddingBackend(loader=loader)
    query = backend.embed(["database timeout"], purpose="retrieval_query")
    document = backend.embed(["database timeout"], purpose="retrieval_document")
    assert len(query.vectors[0]) == 768
    assert norm(query.vectors[0]) == pytest.approx(1.0)
    assert query.embedding_space_id == "Qwen/Qwen3-Embedding-0.6B:768:v1"
    assert calls[0][0][0].startswith("Instruct:")
    assert calls[1][0][0] == "database timeout"
    assert query.purpose == "retrieval_query"
    assert document.purpose == "retrieval_document"


def test_qwen_oom_retries_and_cpu_fallback():
    attempts = []

    def loader(texts, **kwargs):
        attempts.append(len(texts))
        raise MemoryError("simulated GPU OOM")

    service = LocalFirstEmbeddingService(
        preferred_backend=QwenEmbeddingBackend(loader=loader, batch_size=2),
        cpu_backend=HashingEmbeddingBackend(dimensions=768),
    )
    result = service.embed(["a", "b"], prefer_gpu=True, available_vram_bytes=1_000_000)
    assert result.fallback is True
    assert len(result.vectors) == 2
    assert attempts == [2, 1, 1]


def test_gemini_001_and_2_are_normalized_and_isolated():
    def transport(model, payload):
        assert payload["output_dimensionality"] == 768
        return {"embedding": {"values": [2.0] * 768}}

    one = GeminiEmbeddingBackend("gemini-embedding-001", transport=transport)
    two = GeminiEmbeddingBackend("gemini-embedding-2", transport=transport)
    assert norm(one.embed(["synthetic evidence"]).vectors[0]) == pytest.approx(1.0)
    assert norm(two.embed(["synthetic evidence"]).vectors[0]) == pytest.approx(1.0)
    assert one.info.embedding_space_id != two.info.embedding_space_id


def test_local_only_denies_google_before_transport():
    called = False

    def transport(_model, _payload):
        nonlocal called
        called = True
        return {"embedding": {"values": [1.0] * 768}}

    google = GeminiEmbeddingBackend("gemini-embedding-001", transport=transport)
    service = LocalFirstEmbeddingService(providers=[google], active_space_id=google.info.embedding_space_id)
    with pytest.raises(PrivacyBoundaryError):
        service.embed(["synthetic local-only evidence"], privacy_policy="local_only")
    assert called is False


def test_space_switch_requires_validated_corpus_and_migration_is_idempotent():
    qwen = HashingEmbeddingBackend(model="qwen", dimensions=8)
    gemini = GeminiEmbeddingBackend("gemini-embedding-001", dimension=8,
                                    transport=lambda _m, _p: {"embedding": {"values": [1.0] * 8}})
    service = LocalFirstEmbeddingService(cpu_backend=qwen, providers=[gemini])
    migration = EmbeddingMigration(service, gemini.info.embedding_space_id, batch_size=1)
    class Chunk:
        def __init__(self, chunk_id, text):
            self.chunk_id, self.text = chunk_id, text
    chunks = [Chunk("chunk-1", "same provenance"), Chunk("chunk-2", "more evidence")]
    persisted = {}
    with pytest.raises(EmbeddingError):
        service.switch_active(gemini.info.embedding_space_id)
    state = migration.run(chunks, lambda chunk, result: persisted.setdefault(chunk.chunk_id, result))
    assert state["status"] == "complete"
    migration.run(chunks, lambda chunk, result: persisted.setdefault(chunk.chunk_id, result))
    assert len(persisted) == 2
    migration.activate()
    assert service.active_space_id == gemini.info.embedding_space_id


def test_unavailable_qwen_runtime_uses_consistent_cpu_space_metadata():
    service = LocalFirstEmbeddingService(
        preferred_backend=QwenEmbeddingBackend(loader=lambda *_a, **_k: (_ for _ in ()).throw(EmbeddingError("unavailable"))),
        cpu_backend=HashingEmbeddingBackend(model="cpu-fallback", dimensions=8),
    )
    result = service.embed(["synthetic"], available_vram_bytes=1_000_000)
    assert result.fallback is True
    assert result.model == "cpu-fallback"
    assert result.provider == "LOCAL"
    assert result.model_revision == "v1"
    assert result.embedding_space_id == "cpu-fallback:8:v1"
    assert result.metadata["fallback_space_id"] == result.embedding_space_id


def test_postgres_upsert_and_search_sql_are_space_scoped():
    class Cursor:
        def __init__(self):
            self.statement = ""
            self.params = ()

        def execute(self, statement, params):
            self.statement, self.params = statement, params

    cursor = Cursor()
    record = EmbeddingRecord(
        chunk_id="chunk-1", embedding_space_id="gemini-embedding-001:768:v1", provider="GOOGLE",
        model="gemini-embedding-001", model_revision="v1", dimensions=768, normalized=True,
        purpose="retrieval_document", device="remote", vector=(1.0, 0.0), metadata={"synthetic": True},
    )
    _upsert_embedding(cursor, record)
    assert "embedding_space_id" in cursor.statement
    assert "provider" in cursor.statement
    assert "model_revision" in cursor.statement
    assert "ON CONFLICT (chunk_id, embedding_space_id)" in cursor.statement
    assert record.embedding_space_id in cursor.params

    sql, params = _build_search_sql(RetrievalFilters(embedding_space_id=record.embedding_space_id))
    assert "e.embedding_space_id = %s" in sql
    assert record.embedding_space_id in params
    for field in ("e.provider", "e.model_revision", "e.normalized", "e.purpose"):
        assert field in sql
