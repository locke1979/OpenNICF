"""SCO/Contraordenacional domain agent built on the shared QwenAgent factory."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .audit import FailureAuditEngine
from .domain_agent import DomainAgent, DomainAgentFactory, DomainProfile, _normalize_multi_value
from .knowledge import KnowledgePlatform, RetrievalFilters
from .model_gateway import ModelGateway


CONTRAORDENACIONAL_DOMAIN_ID = "contraordenacional"
SCO_OWNED_SYSTEMS = (
    "sco_application",
    "sco_batches",
    "sco_process_management",
)
SCO_INTEGRATION_ALIASES = ("sinquer", "gff", "istg", "sgrd", "sdrd")
SCO_OWNERSHIP_ALIASES = (
    "sco",
    "sco-app",
    "sco-application",
    "sco-batch",
    "sco-batches",
    "contra-ordenacional",
    "contraordenacional",
)
_COLLABORATION_TARGETS = {"sinquer": "criminal"}


def _sco_profile() -> DomainProfile:
    """Return the public, endpoint-free SCO ownership profile."""
    from .domain_agent import DEFAULT_DOMAIN_PROFILES

    base = DEFAULT_DOMAIN_PROFILES[CONTRAORDENACIONAL_DOMAIN_ID]
    filters = replace(base.default_retrieval_filters, principal_acl_scopes=frozenset({"internal"}))
    return replace(
        base,
        default_retrieval_filters=filters,
        owned_systems=SCO_OWNED_SYSTEMS,
        integration_boundary_aliases=SCO_INTEGRATION_ALIASES,
    ).normalized()


class _DomainScopedPlatform:
    """Read-only platform facade that forces domain and ACL filters."""

    def __init__(self, platform: KnowledgePlatform, domain_id: str, acl_scopes: frozenset[str]) -> None:
        self._platform = platform
        self.store = platform.store
        self.embeddings = platform.embeddings
        self._domain_id = domain_id
        self._acl_scopes = acl_scopes

    def search(self, query: str, *, filters: RetrievalFilters | None = None, route: str = "local"):
        filters = filters or RetrievalFilters()
        scoped = replace(
            filters,
            principal_acl_scopes=self._acl_scopes,
            principal_domain_id=self._domain_id,
            domain_ids=(self._domain_id,),
            delegated_domain_ids=(),
        )
        return self._platform.search(query, filters=scoped, route=route)


class ContraordenacionalDomainAgent(DomainAgent):
    """QwenAgent-backed agent for SCO evidence and failure audits."""

    domain_id = CONTRAORDENACIONAL_DOMAIN_ID
    ownership_aliases = SCO_OWNERSHIP_ALIASES
    integration_aliases = SCO_INTEGRATION_ALIASES

    def audit(self, request: str | Mapping[str, Any]):
        """Run a provenance-preserving SCO-only audit through the safe broker."""
        payload = dict(request) if isinstance(request, Mapping) else {"query": request}
        payload["domain_id"] = self.domain_id
        payload["scope"] = payload.get("scope") or "SCO"
        payload["systems"] = tuple(_normalize_multi_value(payload.get("systems"))) or self.profile.owned_systems
        payload["metadata"] = {**dict(payload.get("metadata") or {}), "domain_id": self.domain_id}
        scoped_platform = _DomainScopedPlatform(
            self.knowledge,
            self.domain_id,
            self.profile.default_retrieval_filters.principal_acl_scopes,
        )
        return FailureAuditEngine(scoped_platform, broker=self.tools.diagnostic_broker).analyze(payload)

    def request_integration_delegation(
        self,
        *,
        target_alias: str,
        query: str,
        correlation_ids: tuple[str, ...] | list[str] = (),
    ) -> dict[str, Any]:
        """Create a coordinator request; never query a foreign corpus directly."""
        alias = target_alias.strip().lower()
        if alias not in self.integration_aliases:
            raise PermissionError("target is not a registered SCO integration boundary")
        target_domain = _COLLABORATION_TARGETS.get(alias, alias)
        return {
            "request_type": "cross_domain_collaboration",
            "source_domain_id": self.domain_id,
            "target_domain_id": target_domain,
            "integration_alias": alias,
            "query": query,
            "correlation_ids": list(correlation_ids),
            "authorization": "explicit-coordinator-delegation-required",
        }


class ContraordenacionalDomainAgentFactory(DomainAgentFactory):
    """Shared-service factory with SCO aliases and one QwenAgent runtime path."""

    def __init__(self, **kwargs: Any) -> None:
        profiles = dict(kwargs.pop("profiles", {}) or {})
        profiles[CONTRAORDENACIONAL_DOMAIN_ID] = _sco_profile()
        super().__init__(profiles=profiles, **kwargs)

    def profile_for(self, domain_id: str) -> DomainProfile:
        normalized = domain_id.strip().lower()
        if normalized in SCO_OWNERSHIP_ALIASES:
            normalized = CONTRAORDENACIONAL_DOMAIN_ID
        return super().profile_for(normalized)

    def create(self, domain_id: str) -> ContraordenacionalDomainAgent:
        profile = self.profile_for(domain_id)
        return ContraordenacionalDomainAgent(
            profile,
            gateway=self.gateway,
            knowledge=self.knowledge,
            diagnostic_broker=self.diagnostic_broker,
            runtime_factory=self._runtime_factory,
        )


def create_contraordenacional_domain_agent_factory(
    *,
    gateway: ModelGateway | None = None,
    knowledge: KnowledgePlatform | None = None,
    diagnostic_broker: Any | None = None,
    runtime_factory: Any | None = None,
) -> ContraordenacionalDomainAgentFactory:
    return ContraordenacionalDomainAgentFactory(
        gateway=gateway,
        knowledge=knowledge,
        diagnostic_broker=diagnostic_broker,
        runtime_factory=runtime_factory,
    )


__all__ = [
    "CONTRAORDENACIONAL_DOMAIN_ID",
    "SCO_OWNED_SYSTEMS",
    "SCO_INTEGRATION_ALIASES",
    "SCO_OWNERSHIP_ALIASES",
    "ContraordenacionalDomainAgent",
    "ContraordenacionalDomainAgentFactory",
    "create_contraordenacional_domain_agent_factory",
]
