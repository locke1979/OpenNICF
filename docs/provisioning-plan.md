# Provisioning plan

Status: preparation and contract validation (2026-08-08).

The repository currently contains the bootstrap bridge only. No Proxmox workload,
model gateway, database, object store, channel adapter, broker, or production
deployment has been installed by this plan.

## Execution order

1. Complete the subagent bridge and event/audit ingress (#2).
2. Provision isolated Proxmox workloads and the secure deployment runner (#3).
3. Install and probe LM Studio and LiteLLM routing (#4).
4. Install PostgreSQL/pgvector and object storage with provenance and backups (#5).
5. Add ingestion parsers, idempotency, and dead-letter handling (#6).
6. Add the evidence-grounded audit engine and verification requests (#7).
7. Add Webex and Telegram adapters (#8) and the signed on-prem pull broker (#9) in parallel.
8. Add CI/CD, observability, health checks, secrets handling, and rollback (#10).
9. Run the complete synthetic and channel-to-report acceptance package (#11).

## Required inputs before external provisioning

The deployment lane must receive the Proxmox API endpoint/token, node and storage
pool, network bridge/VLAN, templates, resource limits, DNS/gateway, and SSH key.
The model lane needs LM Studio and LiteLLM endpoints/models and their API keys.
The data lane needs PostgreSQL/object-store credentials, backup targets, and
broker signing/mTLS material. Channel work needs Webex and Telegram credentials.
Secrets must be supplied through the deployment secret store; they must not be
committed to Git or placed in issue comments.

## Acceptance gates

- `python3 -m pytest -q` remains green.
- Model health probes and local-first/complex-escalation/`local_only` routing tests pass.
- Infrastructure passes format, validation, plan, non-production apply, reachability, and rollback checks.
- Ingestion is idempotent and preserves provenance; backup/restore is verified.
- Broker rejects tampered, replayed, and expired jobs and produces hashed read-only results.
- A full end-to-end run correlates channel input, evidence, findings, verification requests, and report output.
