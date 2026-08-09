CREATE TABLE invoice_runs (
    invoice_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE invoice_events (
    event_id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL REFERENCES invoice_runs(invoice_id),
    event_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
