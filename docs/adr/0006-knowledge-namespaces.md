# ADR 0006: Domain-aware knowledge namespaces and sanitized integration registry

Status: accepted

Issue: #15

OpenNICF needs a namespace-aware evidence model that can classify stored evidence by `domain_id`, `system_id`, `component_id`, `environment`, `evidence_type`, and `acl_scope` without splitting the existing shared knowledge platform into domain-specific clones.

We extend the existing PostgreSQL + pgvector knowledge store with a namespace registry and explicit integration edges. Each ingested source and chunk is classified into a namespace record, and namespace metadata is sanitized before it reaches chunking or embedding so credential-bearing fields and private endpoints do not leak into search context.

Retrieval remains ACL-first and domain-aware. Domain agents keep using the shared `KnowledgePlatform`, but their filters now pin a principal domain boundary before semantic or lexical scoring. Cross-domain retrieval is possible only when the caller explicitly delegates additional domain IDs through the filter contract.

The namespace registry is source-controlled and deterministic. Tests cover sanitization, delegation, and namespace edge persistence so the classification contract stays stable as the evidence model evolves.
