# ADR 0001: QwenAgent is mandatory for application orchestration

## Status

Accepted.

## Decision

OpenNICF uses the official QwenAgent framework as its sole application-level agent and tool-orchestration authority. OpenClaw/subagent-pipeline remains the external development/deployment controller. NullClaw, if retained, is an interface or lightweight execution adapter and does not implement a parallel agent loop.

QwenAgent receives an OpenNICF model adapter/router, never a hard-coded provider. The router enforces local-first policy and `local_only` fail-closed behavior. OpenNICF tools own authorization, provenance and execution boundaries.

## Consequences

The QwenAgent dependency is required in production. Unit tests may use fakes, but integration tests must prove QwenAgent -> model router -> LM Studio and LiteLLM paths, RAG provenance, diagnostic-job policy, and local-only isolation.

