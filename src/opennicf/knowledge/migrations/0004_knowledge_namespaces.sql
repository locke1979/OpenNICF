BEGIN;

CREATE TABLE IF NOT EXISTS knowledge_namespaces (
    namespace_id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL,
    system_id TEXT NOT NULL,
    component_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    asset_name TEXT NOT NULL,
    component_name TEXT NOT NULL,
    sanitized_summary TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_integration_edges (
    edge_id TEXT PRIMARY KEY,
    source_namespace_id TEXT NOT NULL REFERENCES knowledge_namespaces(namespace_id) ON DELETE CASCADE,
    target_namespace_id TEXT NOT NULL REFERENCES knowledge_namespaces(namespace_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    system_id TEXT NOT NULL,
    component_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE knowledge_sources
    ADD COLUMN IF NOT EXISTS namespace_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS domain_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS system_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS component_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS evidence_type TEXT NOT NULL DEFAULT '';

ALTER TABLE knowledge_chunks
    ADD COLUMN IF NOT EXISTS namespace_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS domain_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS system_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS component_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS evidence_type TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS knowledge_sources_namespace_idx
    ON knowledge_sources (namespace_id, domain_id, system_id, component_id, environment, evidence_type, acl_scope);

CREATE INDEX IF NOT EXISTS knowledge_chunks_namespace_idx
    ON knowledge_chunks (namespace_id, domain_id, system_id, component_id, environment, evidence_type, acl_scope);

CREATE INDEX IF NOT EXISTS knowledge_namespaces_lookup_idx
    ON knowledge_namespaces (domain_id, system_id, component_id, environment, evidence_type, acl_scope);

CREATE INDEX IF NOT EXISTS knowledge_integration_edges_lookup_idx
    ON knowledge_integration_edges (source_namespace_id, target_namespace_id, relation_type);

COMMIT;
