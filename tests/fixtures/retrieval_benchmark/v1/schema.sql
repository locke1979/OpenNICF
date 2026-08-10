CREATE TABLE knowledge_chunks (
    chunk_id TEXT PRIMARY KEY,
    source_version_id TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    acl_scope TEXT NOT NULL,
    ordinal INTEGER NOT NULL
);

CREATE INDEX knowledge_chunks_artifact_idx ON knowledge_chunks (artifact_hash);
