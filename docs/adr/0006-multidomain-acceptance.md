# ADR 0006: Multi-domain acceptance and embedding-safe orchestration

Status: Accepted for Issue #24 synthetic acceptance

## Decision

Issue #24 validates five isolated domain paths and three coordinator paths
using one shared `ModelGateway`, one QwenAgent-backed runtime boundary per
domain agent, and the `IntegrationCorrelationAgent` for cross-domain work.
Domain evidence is filtered by ACL, domain, system/component, and environment
before lexical, symbol, or semantic retrieval. A prompt cannot widen a domain
agent's scope; foreign-domain access requires coordinator delegation.

Every retrieval result exposes the active `embedding_space_id`. A query may
only score vectors from that same space. Equal dimensions are not a
compatibility signal, so Qwen and Gemini vectors are never mixed. Re-embedding
adds a vector for the existing immutable chunk and retains its source,
artifact, version, ACL, and domain provenance.

The coordinator emits bounded evidence packages and preserves fact,
inference, and hypothesis classifications. Missing live data remains an
explicit diagnostic-broker pending path. `local_only` is propagated through
delegation and rejects Google embedding spaces and remote completion routes.

Synthetic acceptance records package/token bounds, active space, correlation
identifiers, health, structured observability, and deployment rollback gates.
The harness does not require Google credentials, private endpoints, or
sensitive evidence. Live local Qwen execution remains an environment-specific
optional gate; the repository acceptance path proves the safe CPU fallback.

## Consequences

- Domain agents remain thin policy facades over the common QwenAgent boundary.
- Cross-domain synthesis is explicit, bounded, and provenance-bearing.
- Embedding provider migration can be resumed without creating a new logical
  document version.
- The acceptance harness is deterministic and safe to run in CI.
