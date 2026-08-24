from __future__ import annotations

import math

import pytest

from opennicf.knowledge import (
    EmbeddingError,
    EmbeddingMigration,
    GeminiEmbeddingBackend,
    HashingEmbeddingBackend,
    HttpEmbeddingBackend,
    LocalFirstEmbeddingService,
    PrivacyBoundaryError,
    QwenEmbeddingBackend,
)
from opennicf.knowledge.models import EmbeddingRecord, RetrievalFilters
from opennicf.knowledge.store import _build_search_sql, _upsert_embedding


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


class _HttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        import json

        return json.dumps(self.payload).encode()


def test_http_embedding_backend_selects_model_and_preserves_space(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        import json

        requests.append((json.loads(request.data), timeout))
        return _HttpResponse({
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "data": [{"object": "embedding", "index": 0, "embedding": [3.0, 4.0]}],
        })

    monkeypatch.setattr("opennicf.knowledge.embeddings.urlopen", urlopen)
    backend = HttpEmbeddingBackend("http://embedding", dimension=2)
    result = backend.embed(["evidence"])
    assert requests[0][0]["model"] == backend.info.model
    assert result.model == backend.info.model
    assert result.embedding_space_id == backend.info.embedding_space_id
    assert result.dimensions == 2


@pytest.mark.parametrize(
    "response, message",
    [
        ({"model": "other", "data": []}, "wrong model"),
        ({"data": []}, "wrong model"),
        ({"model": "Qwen/Qwen3-Embedding-0.6B", "dimension": 3, "data": []}, "dimension"),
        ({"model": "Qwen/Qwen3-Embedding-0.6B", "dimensions": 3, "data": []}, "dimension"),
        ({"model": "Qwen/Qwen3-Embedding-0.6B", "data": []}, "shape"),
        ({"model": "Qwen/Qwen3-Embedding-0.6B", "data": [{"index": 1, "embedding": [1.0, 2.0]}]}, "shape"),
    ],
)
def test_http_embedding_backend_fails_closed_on_response_mismatch(monkeypatch, response, message):
    monkeypatch.setattr(
        "opennicf.knowledge.embeddings.urlopen",
        lambda _request, timeout: _HttpResponse(response),
    )
    backend = HttpEmbeddingBackend("http://embedding", dimension=2)
    with pytest.raises(EmbeddingError, match=message):
        backend.embed(["evidence"])


def test_production_style_service_does_not_cpu_fallback(monkeypatch):
    preferred = HttpEmbeddingBackend("http://embedding", dimension=8)
    def unavailable(_request, **_kwargs):
        raise EmbeddingError("synthetic unavailable worker")

    monkeypatch.setattr("opennicf.knowledge.embeddings.urlopen", unavailable)
    service = LocalFirstEmbeddingService(
        preferred_backend=preferred,
        cpu_backend=HashingEmbeddingBackend(dimensions=8),
        allow_cpu_fallback=False,
    )
    with pytest.raises(EmbeddingError):
        service.embed(["evidence"])


def test_http_embedding_backend_projects_verified_native_dimension(monkeypatch):
    def urlopen(_request, timeout):
        return _HttpResponse({
            "model": "text-embedding-qwen3-embedding-4b",
            "data": [{"index": 0, "embedding": [3.0, 4.0, 99.0, 100.0]}],
        })

    monkeypatch.setattr("opennicf.knowledge.embeddings.urlopen", urlopen)
    backend = HttpEmbeddingBackend(
        "http://embedding",
        model="text-embedding-qwen3-embedding-4b",
        dimension=2,
        native_dimension=4,
    )
    result = backend.embed(["evidence"])
    assert result.dimensions == 2
    assert result.vectors[0] == pytest.approx((0.6, 0.8))
    assert result.metadata["native_dimension"] == 4


def test_http_embedding_backend_rejects_alias_model(monkeypatch):
    monkeypatch.setattr(
        "opennicf.knowledge.embeddings.urlopen",
        lambda _request, timeout: _HttpResponse({
            "model": "text-embedding-qwen3-embedding-4b",
            "data": [{"index": 0, "embedding": [1.0, 2.0]}],
        }),
    )
    backend = HttpEmbeddingBackend(
        "http://embedding",
        model="qwen.qwen3-vl-embedding-2b",
        dimension=2,
    )
    with pytest.raises(EmbeddingError, match="wrong model"):
        backend.embed(["evidence"])


@pytest.mark.parametrize("values", ([float("nan"), 1.0], [0.0, 0.0]))
def test_http_embedding_backend_rejects_nonfinite_or_zero_projection(monkeypatch, values):
    monkeypatch.setattr(
        "opennicf.knowledge.embeddings.urlopen",
        lambda _request, timeout: _HttpResponse({
            "model": "text-embedding-qwen3-embedding-4b",
            "data": [{"index": 0, "embedding": values}],
        }),
    )
    backend = HttpEmbeddingBackend(
        "http://embedding",
        model="text-embedding-qwen3-embedding-4b",
        dimension=2,
    )
    with pytest.raises(EmbeddingError, match="non-finite or zero"):
        backend.embed(["evidence"])


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


def test_postgres_search_without_space_fails_closed():
    with pytest.raises(EmbeddingError, match="explicit embedding_space_id"):
        _build_search_sql(RetrievalFilters())
