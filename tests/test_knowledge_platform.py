from __future__ import annotations

from pathlib import Path

import pytest

from opennicf.knowledge import (
    FilesystemObjectStore,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    PostgresKnowledgeStore,
    RetrievalFilters,
)
from opennicf.knowledge.migrations import iter_migration_files, migration_checksum, migration_version


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge"


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def test_filesystem_object_store_is_immutable(tmp_path):
    store = FilesystemObjectStore(tmp_path / "objects")
    first = store.put_bytes(b"hello world", mime_type="text/plain")
    second = store.put_bytes(b"hello world", mime_type="text/plain")
    assert first.content_hash == second.content_hash
    assert first.object_key == second.object_key
    assert store.exists(first.object_key)
    assert store.get_bytes(first.object_key) == b"hello world"


def test_embedding_service_reports_cpu_fallback_metadata():
    service = LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32))
    result = service.embed(["database timeout", "acl denied"], prefer_gpu=False)
    assert result.model == "opennicf-hash-32"
    assert result.dimensions == 32
    assert result.device == "cpu"
    assert len(result.vectors) == 2
    assert result.fallback


def test_memory_platform_import_versioning_acl_and_provenance_roundtrip(tmp_path):
    platform = KnowledgePlatform(MemoryKnowledgeStore(), MemoryObjectStore(), LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=64)))
    original = _load("document.md")
    same = platform.ingest(
        source_id="doc-1",
        source_uri="tests/fixtures/knowledge/document.md",
        content=original,
        acl_scope="team-a",
        domain="knowledge",
        system="audit",
        environment="dev",
        source_type="document",
        parser_version="1",
    )
    duplicate = platform.ingest(
        source_id="doc-1",
        source_uri="tests/fixtures/knowledge/document.md",
        content=original,
        acl_scope="team-a",
        domain="knowledge",
        system="audit",
        environment="dev",
        source_type="document",
        parser_version="1",
    )
    changed = platform.ingest(
        source_id="doc-1",
        source_uri="tests/fixtures/knowledge/document.md",
        content=original + "\nA new evidence line appeared.",
        acl_scope="team-a",
        domain="knowledge",
        system="audit",
        environment="dev",
        source_type="document",
        parser_version="2",
    )

    assert same.created is True
    assert duplicate.created is False
    assert same.version.version_number == 1
    assert changed.version.version_number == 2
    assert len(platform.store.source_versions) == 2

    public = platform.ingest(
        source_id="log-1",
        source_uri="tests/fixtures/knowledge/logs.txt",
        content=_load("logs.txt"),
        acl_scope="team-a",
        domain="operations",
        system="gateway",
        environment="prod",
        source_type="log",
        parser_version="1",
    )
    secret = platform.ingest(
        source_id="query-1",
        source_uri="tests/fixtures/knowledge/query-output.txt",
        content=_load("query-output.txt"),
        acl_scope="team-b",
        domain="operations",
        system="gateway",
        environment="prod",
        source_type="query-output",
        parser_version="1",
    )

    hits = platform.search(
        "database timeout acl denied",
        filters=RetrievalFilters(principal_acl_scopes=frozenset({"team-a"}), domains=("operations",), limit=5),
        route="local",
    )
    assert hits
    assert all(hit.acl_scope == "team-a" for hit in hits)
    assert all(hit.source_id != secret.source.source_id for hit in hits)
    assert platform.store.retrieval_events
    event = platform.store.retrieval_events[-1]
    assert event.selected_chunks
    assert event.query_hash
    assert hits[0].artifact_hash
    assert hits[0].excerpt_hash
    assert hits[0].ingest_timestamp.tzinfo is not None
    assert hits[0].parser_version

    snapshot = platform.backup()
    restored = KnowledgePlatform(MemoryKnowledgeStore(), MemoryObjectStore(), LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=64)))
    restored.restore(snapshot)
    restored_hits = restored.search(
        "database timeout acl denied",
        filters=RetrievalFilters(principal_acl_scopes=frozenset({"team-a"}), domains=("operations",), limit=5),
        route="local",
    )
    assert [hit.chunk_id for hit in restored_hits] == [hit.chunk_id for hit in hits]


def test_fixture_queries_import_and_search_grounded_chunks():
    platform = KnowledgePlatform(MemoryKnowledgeStore(), MemoryObjectStore(), LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=64)))
    platform.ingest(
        source_id="schema-1",
        source_uri="tests/fixtures/knowledge/schema.sql",
        content=_load("schema.sql"),
        acl_scope="internal",
        domain="schema",
        system="db",
        environment="dev",
        source_type="schema",
        parser_version="1",
    )
    platform.ingest(
        source_id="code-1",
        source_uri="tests/fixtures/knowledge/code.py",
        content=_load("code.py"),
        acl_scope="internal",
        domain="code",
        system="router",
        environment="dev",
        source_type="code",
        parser_version="1",
    )
    hits = platform.search(
        "acl_scope and artifact_hash",
        filters=RetrievalFilters(principal_acl_scopes=frozenset({"internal"}), limit=5),
    )
    assert hits
    assert any("artifact_hash" in hit.text for hit in hits)


def test_migrations_are_source_controlled_and_versioned():
    files = iter_migration_files()
    assert [migration_version(path.name) for path in files] == [1, 2]
    assert [path.name for path in files] == ["0001_initial.sql", "0002_hybrid_retrieval.sql"]
    assert all(migration_checksum(path) for path in files)


def test_postgres_migrations_apply_in_order():
    executed: list[str] = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            executed.append(sql.strip())

        def fetchall(self):
            return []

        def fetchone(self):
            return (1,)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return Cursor()

        def commit(self):
            return None

    store = PostgresKnowledgeStore(lambda: Connection())
    applied = store.migrate()
    assert applied == [1, 2]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in statement for statement in executed)
    assert any("knowledge_sources" in statement for statement in executed)
