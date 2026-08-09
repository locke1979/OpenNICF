BEGIN;

CREATE INDEX IF NOT EXISTS knowledge_chunks_source_locator_idx
    ON knowledge_chunks (source_id, locator, ordinal);

CREATE INDEX IF NOT EXISTS knowledge_chunks_source_version_idx
    ON knowledge_chunks (source_version_id, ordinal);

CREATE INDEX IF NOT EXISTS knowledge_source_versions_content_hash_idx
    ON knowledge_source_versions (source_id, content_hash);

COMMIT;
