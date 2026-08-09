# OpenNICF architecture

OpenClaw's `subagent-pipeline` is the development and deployment controller. At runtime, QwenAgent is the single application-level agent and tool-orchestration framework.

```text
channel / gateway -> OpenNICF -> QwenAgent -> OpenNICF model router
                                      |             |-> LM Studio / Qwen local
                                      |             `-> LiteLLM OCI
                                      `-> provenance-aware tools -> PostgreSQL + pgvector
```

Application tools include evidence search, ingestion, audit correlation, report generation and diagnostic-job creation. SQLcl/PowerShell and deployment actions remain behind policy-validated services and are never directly available to QwenAgent.

The knowledge layer is split into three explicit pieces: PostgreSQL + pgvector for structured evidence and embeddings, an immutable object-store abstraction for originals and artifacts, and a provenance-aware retrieval service that applies ACL filters before any evidence reaches the model router.

Issue #15 adds a namespace registry on top of that model. Evidence is classified by `domain_id`, `system_id`, `component_id`, `environment`, `evidence_type`, and `acl_scope`, and integration edges between namespaces are stored explicitly so cross-domain relationships stay auditable without collapsing the shared platform into domain-specific copies.

Domain-specific runtime behavior is expressed through `DomainProfile` records and a `DomainAgentFactory`. Every domain agent shares the same model gateway and knowledge platform; the factory only binds profile metadata, domain-scoped tool policy, and the QwenAgent runtime wrapper on demand.

The failure-audit engine is a separate evidence synthesis layer that sits above retrieval. It parses an audit scope, asks source-type analyzers to inspect logs, code, schema, documentation, query outputs and optional integration definitions, then emits findings with explicit classifications, provenance refs, timelines, correlation IDs, and contradiction tracking. When live database verification is required, the engine emits a brokered verification request instead of fabricating runtime state.
