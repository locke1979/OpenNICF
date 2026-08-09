BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_sources (
    source_id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    channel TEXT NOT NULL,
    domain TEXT NOT NULL,
    system TEXT NOT NULL,
    environment TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_source_versions (
    source_version_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    source_uri TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    ingest_timestamp TIMESTAMPTZ NOT NULL,
    parser_version TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    object_key TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (source_id, version_number),
    UNIQUE (source_id, content_hash)
);

CREATE TABLE IF NOT EXISTS knowledge_artifacts (
    artifact_hash TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL REFERENCES knowledge_source_versions(source_version_id) ON DELETE CASCADE,
    object_key TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    source_version_id TEXT NOT NULL REFERENCES knowledge_source_versions(source_version_id) ON DELETE CASCADE,
    artifact_hash TEXT NOT NULL REFERENCES knowledge_artifacts(artifact_hash) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    locator TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    domain TEXT NOT NULL,
    system TEXT NOT NULL,
    environment TEXT NOT NULL,
    source_type TEXT NOT NULL,
    page INTEGER,
    line_start INTEGER,
    line_end INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (artifact_hash, chunk_hash),
    UNIQUE (source_version_id, ordinal)
);

CREATE TABLE IF NOT EXISTS knowledge_embeddings (
    chunk_id TEXT NOT NULL REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    device TEXT NOT NULL,
    vector vector NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (chunk_id, model, dimensions)
);

CREATE INDEX IF NOT EXISTS knowledge_embeddings_model_idx
    ON knowledge_embeddings (model, dimensions);

CREATE INDEX IF NOT EXISTS knowledge_chunks_acl_idx
    ON knowledge_chunks (acl_scope, domain, system, environment, source_type);

CREATE INDEX IF NOT EXISTS knowledge_chunks_hash_idx
    ON knowledge_chunks (chunk_hash);

CREATE INDEX IF NOT EXISTS knowledge_retrieval_scope_idx
    ON knowledge_source_versions (source_id, ingest_timestamp DESC);

CREATE TABLE IF NOT EXISTS knowledge_entities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_tags (
    tag_id TEXT PRIMARY KEY,
    tag_name TEXT NOT NULL UNIQUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_entity_tags (
    entity_id TEXT NOT NULL REFERENCES knowledge_entities(entity_id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES knowledge_tags(tag_id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, tag_id)
);

CREATE TABLE IF NOT EXISTS knowledge_retrieval_events (
    event_id TEXT PRIMARY KEY,
    query_hash TEXT NOT NULL,
    filters JSONB NOT NULL,
    selected_chunks JSONB NOT NULL,
    scores JSONB NOT NULL,
    reranker TEXT,
    model_route TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_audit_findings (
    finding_id TEXT PRIMARY KEY,
    finding_class TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    evidence_links JSONB NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMIT;
