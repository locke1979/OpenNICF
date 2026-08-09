BEGIN;

ALTER TABLE knowledge_audit_findings
    ADD COLUMN IF NOT EXISTS classification TEXT NOT NULL DEFAULT 'fact',
    ADD COLUMN IF NOT EXISTS evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS time_range JSONB,
    ADD COLUMN IF NOT EXISTS systems JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS components JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS verification_status TEXT NOT NULL DEFAULT 'unverified',
    ADD COLUMN IF NOT EXISTS recommended_query TEXT,
    ADD COLUMN IF NOT EXISTS diagnostic_action JSONB,
    ADD COLUMN IF NOT EXISTS supporting_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS contradicting_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS provenance_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS correlation_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS source_type_analyzers JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS knowledge_audit_reports (
    report_id TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    executive_summary TEXT NOT NULL,
    human_report TEXT NOT NULL,
    machine_report JSONB NOT NULL,
    finding_ids JSONB NOT NULL,
    timeline_event_ids JSONB NOT NULL,
    causal_chains JSONB NOT NULL,
    unresolved_hypotheses JSONB NOT NULL,
    recommended_verification_steps JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS knowledge_verification_requests (
    request_id TEXT PRIMARY KEY,
    audit_id TEXT NOT NULL,
    finding_id TEXT,
    broker_name TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS knowledge_audit_reports_request_hash_idx
    ON knowledge_audit_reports (request_hash);

CREATE INDEX IF NOT EXISTS knowledge_verification_requests_audit_idx
    ON knowledge_verification_requests (audit_id, finding_id);

COMMIT;
