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
