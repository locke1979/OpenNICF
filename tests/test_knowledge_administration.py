from opennicf.ingestion import IngestionJob, IngestionQueue
from opennicf.knowledge import (
    HashingEmbeddingBackend,
    KnowledgeAdministration,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    RetrievalFilters,
)


def _platform():
    service = LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=8))
    store = MemoryKnowledgeStore()
    platform = KnowledgePlatform(store, MemoryObjectStore(), service)
    return platform, store, service


def test_administration_lists_with_acl_scope_and_preserves_provenance_on_retirement():
    platform, store, service = _platform()
    platform.ingest(source_id="source-public", source_uri="manual://public", content="public evidence", acl_scope="public", domain_id="criminal")
    platform.ingest(source_id="source-internal", source_uri="manual://internal", content="internal evidence", acl_scope="internal", domain_id="criminal")
    admin = KnowledgeAdministration(store, object_store=platform.object_store, embeddings=service)

    visible = admin.list_sources(filters=RetrievalFilters(principal_acl_scopes=frozenset({"public"}), principal_domain_id="criminal"))
    assert [row["source_id"] for row in visible] == ["source-public"]
    operation = admin.retire_source("source-public", actor="operator", reason="superseded")
    assert operation.status == "complete"
    assert admin.list_sources() == [] or [row["source_id"] for row in admin.list_sources()] == ["source-internal"]
    assert admin.source_status("source-public")["status"] == "retired"
    assert admin.provenance("source-public")["source"]["source_id"] == "source-public"


def test_retirement_policy_rejects_legal_hold_without_mutation():
    platform, store, _ = _platform()
    platform.ingest(source_id="held", source_uri="manual://held", content="held", metadata={"legal_hold": True})
    admin = KnowledgeAdministration(store)
    try:
        admin.retire_source("held", actor="operator", reason="cleanup")
    except PermissionError:
        pass
    else:
        raise AssertionError("legal hold must prevent retirement")
    assert admin.source_status("held")["status"] == "active"


def test_failed_ingestion_retry_is_audited_and_embedding_status_is_explicit():
    _, store, service = _platform()
    queue = IngestionQueue()
    job = IngestionJob("job-1", "source-1", "manual://one", "manual", "text/plain", "hash", b"one")
    queue.enqueue(job)
    claimed = queue.claim()
    assert claimed is not None
    queue.fail(claimed, "temporary parser failure")
    admin = KnowledgeAdministration(store, embeddings=service)
    operation = admin.retry_failed_ingestion(queue, "job-1", actor="operator")
    assert operation.status == "queued"
    assert queue.pending
    status = admin.embedding_space_status()
    assert status["active_space_id"] == service.active_space_id
    assert service.active_space_id in status["spaces"]


def test_reindex_callback_failure_is_recorded():
    _, store, _ = _platform()
    admin = KnowledgeAdministration(store)
    operation = admin.reindex("missing", actor="operator", reindex=lambda _: (_ for _ in ()).throw(RuntimeError("index unavailable")))
    assert operation.status == "failed"
    assert store.admin_operations[operation.operation_id].details["error"] == "index unavailable"
