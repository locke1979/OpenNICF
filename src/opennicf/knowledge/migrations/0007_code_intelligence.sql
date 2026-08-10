-- Persistent, versioned code intelligence. Rows remain tied to immutable evidence.
CREATE TABLE IF NOT EXISTS knowledge_code_symbols (
    symbol_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    source_version_id TEXT NOT NULL REFERENCES knowledge_source_versions(source_version_id) ON DELETE CASCADE,
    artifact_hash TEXT NOT NULL REFERENCES knowledge_artifacts(artifact_hash) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    signature TEXT NOT NULL,
    locator TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    system_id TEXT NOT NULL,
    component_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_code_relationships (
    relationship_id TEXT PRIMARY KEY,
    source_symbol_id TEXT NOT NULL REFERENCES knowledge_code_symbols(symbol_id) ON DELETE CASCADE,
    target_name TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_version_id TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    locator TEXT NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    namespace_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    system_id TEXT NOT NULL,
    component_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS knowledge_code_symbols_lookup_idx
    ON knowledge_code_symbols (name, qualified_name, kind, domain_id, system_id, acl_scope);
CREATE INDEX IF NOT EXISTS knowledge_code_relationships_lookup_idx
    ON knowledge_code_relationships (target_name, relation_type, domain_id, system_id, acl_scope);
