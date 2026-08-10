BEGIN;

-- Space identity is the compatibility boundary; dimensions alone are not.
ALTER TABLE knowledge_embeddings
    ADD COLUMN IF NOT EXISTS embedding_space_id TEXT,
    ADD COLUMN IF NOT EXISTS provider TEXT NOT NULL DEFAULT 'LOCAL',
    ADD COLUMN IF NOT EXISTS model_revision TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS normalized BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'retrieval_document';

UPDATE knowledge_embeddings
SET embedding_space_id = model || ':' || dimensions::text || ':v1'
WHERE embedding_space_id IS NULL;

ALTER TABLE knowledge_embeddings
    ALTER COLUMN embedding_space_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS knowledge_embeddings_chunk_space_uidx
    ON knowledge_embeddings (chunk_id, embedding_space_id);

-- The legacy vector column is intentionally unbounded. Do not cast or build
-- a vector(768) ANN index in-place: existing rows may have other dimensions.
-- Space-scoped ANN tables/indexes are registered by the deployment migration
-- after validating the target dimension; this B-tree keeps legacy lookups
-- safe and makes the space predicate explicit on the compatibility path.
CREATE INDEX IF NOT EXISTS knowledge_embeddings_space_idx
    ON knowledge_embeddings (embedding_space_id, chunk_id);

CREATE TABLE IF NOT EXISTS knowledge_embedding_spaces (
    embedding_space_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    normalized BOOLEAN NOT NULL,
    purpose TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text',
    status TEXT NOT NULL DEFAULT 'available',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_embedding_migrations (
    migration_id TEXT PRIMARY KEY,
    source_space_id TEXT,
    target_space_id TEXT NOT NULL REFERENCES knowledge_embedding_spaces(embedding_space_id),
    cursor_chunk_id TEXT,
    status TEXT NOT NULL,
    processed_count BIGINT NOT NULL DEFAULT 0,
    total_count BIGINT,
    source_checksum TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Registration is metadata-only and is safe for legacy rows of any vector
-- length. Dimension-specific ANN materialization happens only after a target
-- space has been validated by the deployment workflow.
INSERT INTO knowledge_embedding_spaces (
    embedding_space_id, provider, model, model_revision, dimension, normalized, purpose
) VALUES
    ('Qwen/Qwen3-Embedding-0.6B:768:v1', 'LOCAL', 'Qwen/Qwen3-Embedding-0.6B', 'v1', 768, TRUE, 'retrieval_document'),
    ('gemini-embedding-001:768:v1', 'GOOGLE', 'gemini-embedding-001', 'v1', 768, TRUE, 'retrieval_document'),
    ('gemini-embedding-2:768:v1', 'GOOGLE', 'gemini-embedding-2', 'v1', 768, TRUE, 'retrieval_document')
ON CONFLICT (embedding_space_id) DO NOTHING;

COMMIT;
