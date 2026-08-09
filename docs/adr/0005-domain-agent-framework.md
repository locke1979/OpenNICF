# ADR 0005: Domain agents are profile-driven wrappers over a shared QwenAgent runtime

## Status

Accepted.

## Context

OpenNICF needs one runtime agent per application domain without cloning orchestration code or instantiating separate model stacks. The shared model gateway from Issue #4 and the shared knowledge platform from Issue #5 already provide the reusable runtime services.

## Decision

OpenNICF introduces a generic `DomainProfile` contract, a `DomainAgent` wrapper and a `DomainAgentFactory`.

Each domain agent is built from:

* one declarative `DomainProfile`
* one shared `ModelGateway`
* one shared `KnowledgePlatform`
* one QwenAgent runtime adapter created on demand

The factory owns domain registration, not orchestration logic. Adding a new domain requires registry/configuration changes, not a copy of the agent runtime.

Domain-scoped tools must enforce `domain_id` inside the tool boundary before touching the knowledge platform. Prompt text cannot override the profile boundary.

## Consequences

* All domains share the same routing, ACL and provenance layers.
* Domain-specific behavior stays declarative and testable.
* Future domain-specific retrieval/tooling work can extend the profile and tool registry without changing the agent factory contract.
