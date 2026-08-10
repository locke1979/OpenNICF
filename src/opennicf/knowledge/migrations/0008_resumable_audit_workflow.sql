BEGIN;

CREATE TABLE IF NOT EXISTS knowledge_audit_workflows (
    audit_id TEXT PRIMARY KEY,
    request_hash TEXT NOT NULL,
    request_payload JSONB NOT NULL,
    scope TEXT NOT NULL,
    domain_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    system_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    component_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    acl_scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL,
    evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    finding_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    missing_evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    diagnostic_request_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    timeline_event_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    report_artifact_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    manifest_artifact_id TEXT,
    manifest_hash TEXT,
    last_error TEXT,
    resume_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT knowledge_audit_workflows_status_check CHECK (
        status IN ('created', 'collecting_evidence', 'analyzing',
                   'waiting_for_diagnostic', 'resumed', 'completed', 'failed')
    )
);

CREATE TABLE IF NOT EXISTS knowledge_audit_artifacts (
    artifact_id TEXT PRIMARY KEY,
    audit_id TEXT NOT NULL REFERENCES knowledge_audit_workflows(audit_id),
    artifact_type TEXT NOT NULL,
    backend TEXT NOT NULL,
    object_key TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    acl_scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
    domain_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS knowledge_audit_workflows_status_idx
    ON knowledge_audit_workflows (status, updated_at);
CREATE INDEX IF NOT EXISTS knowledge_audit_artifacts_audit_idx
    ON knowledge_audit_artifacts (audit_id, artifact_type);

COMMIT;
