-- Keep source metadata aligned with the durable ingestion persistence contract.
ALTER TABLE knowledge_sources
    ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'document';
