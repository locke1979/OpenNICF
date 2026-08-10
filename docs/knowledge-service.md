# OpenNICF knowledge service

`python -m opennicf.service_runtime --service knowledge` runs the long-lived
knowledge administration service. Startup is fail-closed: it requires
`OPENNICF_POSTGRES_DSN` (or the compatibility variable `DATABASE_URL`) and
`OPENNICF_OBJECT_STORE_ROOT`, connects to PostgreSQL with pgvector, applies
source-controlled migrations, and checks that the object store is writable.
It never replaces PostgreSQL with `KnowledgePlatform.in_memory()`.

The service has no unrestricted export or arbitrary SQL endpoint. Responses,
request bodies, source lists, provenance chunks, retry batches, and re-embed
batches are bounded by configuration:

```text
OPENNICF_ENVIRONMENT=production
OPENNICF_POSTGRES_DSN=<protected PostgreSQL DSN>
OPENNICF_OBJECT_STORE_ROOT=/var/lib/opennicf/objects
OPENNICF_KNOWLEDGE_MAX_BODY_BYTES=1048576
OPENNICF_KNOWLEDGE_MAX_RESULT_ITEMS=100
OPENNICF_KNOWLEDGE_OPERATION_LIMIT=10
OPENNICF_EMBEDDING_BASE_URL=<optional local embedding worker URL>
OPENNICF_EMBEDDING_SERVICE_TOKEN=<protected token>
```

Health and status endpoints are `GET /health/live`, `GET /health/ready`, and
`GET /status`. The bounded administrative API is:

- `GET /v1/sources` with optional `limit`, `acl_scope`, `domain_id`,
  `system_id`, and `include_retired` filters.
- `GET /v1/sources/{source_id}/status` and
  `GET /v1/sources/{source_id}/provenance`.
- `POST /v1/sources/{source_id}/reparse` and
  `POST /v1/sources/{source_id}/reindex`.
- `POST /v1/re-embed` with `source_id`, `embedding_space_id`, and `limit`.
- `GET /v1/migrations/status` and `POST /v1/retry` with an optional `job_id`
  or bounded `limit`.

The service records administrative operation state through
`KnowledgeAdministration`. Reparse reads the immutable object and creates a
new parser-versioned bundle; re-embed writes vectors into the selected
embedding space without mutating source, artifact, or chunk identity. A
reindex request is the integration point for the deployment's native index
callback and records a bounded, auditable operation.

SIGTERM and SIGINT stop the HTTP server cleanly. This implementation is not a
deployment or live-acceptance claim: production still requires a reachable
PostgreSQL/pgvector instance, durable object-store filesystem and permissions,
validated migrations, a configured embedding worker if CPU hashing is not
acceptable, service-account/systemd installation, and smoke/rollback
acceptance in the target environment.
