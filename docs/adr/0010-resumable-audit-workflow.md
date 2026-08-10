# ADR 0010: Persistent resumable audit workflow

## Status

Accepted

## Decision

Failure audits use the existing `FailureAuditEngine`, knowledge store, and
immutable object-store abstractions. `PersistentAuditWorkflow` adds durable
checkpoints for `created`, `collecting_evidence`, `analyzing`,
`waiting_for_diagnostic`, `resumed`, `completed`, and `failed`.

The persisted request is a typed audit contract. Caller ACL/domain scope is
provided separately and cannot be widened by request metadata or evidence
content. Evidence packages contain bounded provenance references and are
marked untrusted. Diagnostics remain brokered and read-only; resume records
the diagnostic result hash and never interprets its content as instructions.

Reports are written as immutable human, machine, evidence-package, and
manifest artifacts. Each artifact carries ACL/domain metadata, and the
manifest binds report outputs to evidence excerpt hashes and source locators.

## Consequences

Restarts can continue a pending audit without creating a second orchestration
framework or duplicating diagnostic authority. PostgreSQL deployments use an
additive migration, while the memory store provides deterministic tests and
backup/restore coverage. A diagnostic result or changed evidence corpus can
still produce a failed or incomplete audit; the report must retain missing
evidence rather than claim a fact that was not verified.
