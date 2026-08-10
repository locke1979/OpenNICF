-- Durable, bounded lifecycle and operational audit records.
CREATE TABLE IF NOT EXISTS knowledge_admin_operations (
    operation_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL CHECK (operation IN ('retire', 'reparse', 'reindex', 'reembed', 'retry_ingestion')),
    status TEXT NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL,
    actor TEXT NOT NULL,
    target_id TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS knowledge_admin_operations_target_idx
    ON knowledge_admin_operations (target_id, requested_at DESC);

CREATE INDEX IF NOT EXISTS knowledge_sources_status_idx
    ON knowledge_sources ((COALESCE(metadata->>'status', 'active')));
