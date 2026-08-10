# OpenNICF operations contract

Issue #10 owns the delivery and runtime safety lane. The GitHub-hosted workflow
builds and verifies a versioned artifact only. A persistent, least-privileged
OpenClaw deployment runner pulls that artifact, holds the IaC lock, applies a
reviewed plan, runs migrations, and performs readiness and smoke checks. It is
the only component allowed to reach the Proxmox/LAN control plane.

## Gates and rollback

`deploy/ops-gate RELEASE_ID` is the intentionally small runner boundary. The
runner must provide `MANAGE_INFRASTRUCTURE=true` and `REVIEWED_PLAN=true`, then
execute migration compatibility, readiness, and smoke gates in that order.
`DeploymentController` records each result. Any exception or false result stops
promotion and invokes rollback to the previous known-good release. A successful
synthetic deployment returns its release ID and promotes only after all gates.

Database migrations must be expand/contract: additive changes first, compatible
application deployment second, cleanup only in a later reviewed release. A
maintenance gate is required when this cannot be met.

## Health and observability

Liveness answers whether a process is running. Readiness is false when any
dependency fails and names the failing dependency (for example `postgres` or
`lmstudio`). All ingress, ingestion, parsing/chunking/embedding, retrieval,
model routing, audit, broker, issue-run, and deployment events carry the same
`correlation_id`. Logs are structured JSON. Metrics are named counters and
latency observations; deployment state and release ID are included in health.
`deploy/health` is the synthetic readiness entrypoint used when live targets are
not available.

Audit records are append-only, hash chained, and redact configured secret
values. Prompt and evidence bodies are not accepted as operational fields by
the contract; retain identifiers and hashes instead.

## Secret/config inventory (values intentionally absent)

| Name | Runtime source | Rotation/revocation |
| --- | --- | --- |
| Proxmox API credentials | external secret store | rotate after operator change |
| LM Studio endpoint config | deployment config | review endpoint changes |
| LiteLLM OCI endpoint/token | external secret store | rotate on provider change |
| Webex token/secret | external secret store | revoke and reissue |
| Telegram token | external secret store | revoke and reissue |
| PostgreSQL/object-store credentials | external secret store | rotate on restore |
| Broker signing/mTLS material | external secret store | revoke certificates |
| GitHub/OpenClaw runner credential | runner secret store | revoke runner |

Bootstrap creates these entries in the runtime secret store and injects them
only into the controlled runner. Rotation is create-new, deploy, verify, then
revoke-old. Suspected exposure is immediate revocation and replacement. No
secret values belong in Git, issues, logs, artifacts, plans, or reports.

## Backup, restore, and recovery

PostgreSQL logical backups run daily before deployment and are retained
encrypted outside Git. Object-store data is replicated to a separate location
on its retention schedule. IaC state is locked and copied to a timestamped
recovery location before apply.

A restore drill runs in non-production and follows the same synthetic safety
model as the deployment gates:
1. Create a logical backup and keep the manifest outside Git.
2. Verify checksums and structural evidence before restore.
3. Restore into deterministic new database names rather than overwriting the
   source.
4. Verify row counts, extensions, and release/correlation IDs.
5. Recover configuration by re-provisioning secret-store references and
   non-secret config from Git; secret material is reissued, never recovered
   from Git.

Validate backups monthly and after schema changes. The inventory in
`docs/config-inventory.json` is the runtime bootstrap contract for the secret
set, while `deploy/config-inventory` can re-render it from the code path.
