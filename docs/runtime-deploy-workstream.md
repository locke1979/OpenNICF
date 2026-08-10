# RUNTIME:DEPLOY workstream

GitHub write access was not available when this workstream started, so this is
the local issue specification for the focused deployment artifact work. It does
not claim that a GitHub issue or PR exists.

## Objective

Package OpenNICF services as immutable, versioned releases with protected
configuration, systemd service templates, bounded readiness checks, and
automatic rollback on a failed health gate.

## Delivered in this workstream

- deterministic release archive builder with SHA-256 manifest
- immutable `/opt/opennicf/releases/<version>` installation contract
- atomic `/opt/opennicf/current` activation model
- manifest verification and tamper detection
- bounded `/health/live`, `/health/ready`, and `/status` runtime entrypoint
- knowledge and ingestion systemd templates with a dedicated service user
- rollback to the previous release when restart/readiness fails
- runtime state separated from artifacts and secrets kept in environment files

## Acceptance gates

- artifact contains no credentials or Python cache files
- an existing release cannot be overwritten
- a manifest mismatch blocks installation
- a failed readiness check restores the previous `current` symlink
- service units use `NoNewPrivileges`, `PrivateTmp`, and restricted writable paths
- full repository tests and release-specific tests pass

## Remaining deployment work

The framework does not claim live application readiness by itself. The
knowledge and ingestion service adapters must be wired to the production
PostgreSQL/pgvector, object-store, and Qwen endpoints, then installed through a
controlled runner and accepted with real restart, ingestion, retrieval, and
backup evidence.
