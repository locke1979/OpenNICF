"""Domain-agent contracts built on the shared QwenAgent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from .knowledge import KnowledgePlatform, RetrievalFilters
from .model_gateway import ModelGateway, PrivacyPolicy
from .qwen_adapter import OpenNICFChatModel
from .qwen_runtime import QwenAgentRuntime


DEFAULT_DOMAIN_IDS = (
    "criminal",
    "contraordenacional",
    "contencioso_administrativo",
    "contencioso_judicial",
    "encargos_sigef",
)


@dataclass(frozen=True)
class DomainProfile:
    """Declarative configuration for one application domain."""

    domain_id: str
    name: str
    description: str
    owned_systems: tuple[str, ...] = ()
    integration_boundary_aliases: tuple[str, ...] = ()
    permitted_evidence_classes: tuple[str, ...] = ()
    default_retrieval_filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    permitted_tools: tuple[str, ...] = ("search_domain_evidence", "request_domain_diagnostic")
    privacy_policy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED
    task_class: str = "simple_rag"
    system_prompt: str | None = None

    def normalized(self) -> "DomainProfile":
        filters = self.default_retrieval_filters.normalized()
        if self.domain_id:
            filters = replace(filters, domains=(self.domain_id,))
        return replace(self, default_retrieval_filters=filters)


def _profile(
    domain_id: str,
    *,
    name: str,
    description: str,
    owned_systems: tuple[str, ...] = (),
    integration_boundary_aliases: tuple[str, ...] = (),
    permitted_evidence_classes: tuple[str, ...] = (),
    permitted_tools: tuple[str, ...] = ("search_domain_evidence", "request_domain_diagnostic"),
    privacy_policy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED,
    task_class: str = "simple_rag",
    system_prompt: str | None = None,
) -> DomainProfile:
    return DomainProfile(
        domain_id=domain_id,
        name=name,
        description=description,
        owned_systems=owned_systems,
        integration_boundary_aliases=integration_boundary_aliases,
        permitted_evidence_classes=permitted_evidence_classes,
        permitted_tools=permitted_tools,
        privacy_policy=privacy_policy,
        task_class=task_class,
        system_prompt=system_prompt,
    ).normalized()


DEFAULT_DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "criminal": _profile(
        "criminal",
        name="Criminal",
        description="Criminal-law matters and supporting evidence.",
        owned_systems=("criminal_casework", "criminal_evidence"),
        integration_boundary_aliases=("penal", "criminal-law"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the criminal domain agent. Stay inside the criminal evidence namespace.",
    ),
    "contraordenacional": _profile(
        "contraordenacional",
        name="Contraordenação",
        description="Administrative offense matters and supporting evidence.",
        owned_systems=("contraordenacional_casework", "contraordenacional_evidence"),
        integration_boundary_aliases=("administrative-offense", "administrative-penalty"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the contraordenacional domain agent. Stay inside the contraordenacional evidence namespace.",
    ),
    "contencioso_administrativo": _profile(
        "contencioso_administrativo",
        name="Contencioso Administrativo",
        description="Administrative litigation matters and supporting evidence.",
        owned_systems=("contencioso_administrativo_casework", "contencioso_administrativo_evidence"),
        integration_boundary_aliases=("administrative-litigation",),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the contencioso administrativo domain agent. Stay inside the administrative-litigation namespace.",
    ),
    "contencioso_judicial": _profile(
        "contencioso_judicial",
        name="Contencioso Judicial",
        description="Judicial litigation matters and supporting evidence.",
        owned_systems=("contencioso_judicial_casework", "contencioso_judicial_evidence"),
        integration_boundary_aliases=("judicial-litigation",),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the contencioso judicial domain agent. Stay inside the judicial-litigation namespace.",
    ),
    "encargos_sigef": _profile(
        "encargos_sigef",
        name="Encargos SIGEF",
        description="SIGEF assignment operations and supporting evidence.",
        owned_systems=("encargos_sigef_casework", "encargos_sigef_evidence"),
        integration_boundary_aliases=("sigef", "assignment-ops"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the encargos SIGEF domain agent. Stay inside the SIGEF evidence namespace.",
    ),
}


class DomainTools:
    """Domain-scoped tool surface that enforces the profile's domain boundary."""

    def __init__(self, profile: DomainProfile, platform: KnowledgePlatform):
        self.profile = profile
        self.platform = platform

    def _filters(self, *, limit: int | None = None) -> RetrievalFilters:
        filters = self.profile.default_retrieval_filters
        if self.profile.domain_id:
            filters = replace(filters, principal_domain_id=self.profile.domain_id, domain_ids=(self.profile.domain_id,))
        if not filters.source_types and self.profile.permitted_evidence_classes:
            filters = replace(filters, source_types=self.profile.permitted_evidence_classes)
        if limit is not None:
            filters = replace(filters, limit=max(1, int(limit)))
        return filters.normalized()

    def search_domain_evidence(self, query: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        hits = self.platform.search(query, filters=self._filters(limit=limit), route="local")
        return [
            {
                "source_id": hit.source_id,
                "locator": hit.locator,
                "text": hit.text,
                "namespace_id": hit.namespace_id,
                "domain_id": hit.domain_id,
                "system_id": hit.system_id,
                "component_id": hit.component_id,
                "environment": hit.environment,
                "evidence_type": hit.evidence_type,
                "acl_scope": hit.acl_scope,
                "domain": hit.domain,
                "system": hit.system,
                "source_type": hit.source_type,
                "score": hit.score,
            }
            for hit in hits
        ]

    def request_domain_diagnostic(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["domain_id"] = self.profile.domain_id
        if payload.get("operation_class") not in {"read_select", "describe"}:
            raise PermissionError("diagnostic operation is not allow-listed")
        return {
            "status": "pending",
            "job_id": "broker-generated",
            "domain_id": self.profile.domain_id,
            "request": payload,
        }

    def as_qwen_tools(self) -> dict[str, Callable[..., Any]]:
        return {
            "search_domain_evidence": self.search_domain_evidence,
            "request_domain_diagnostic": self.request_domain_diagnostic,
        }

    def permitted_qwen_tools(self) -> dict[str, Callable[..., Any]]:
        registry = self.as_qwen_tools()
        selected: dict[str, Callable[..., Any]] = {}
        missing = [tool_name for tool_name in self.profile.permitted_tools if tool_name not in registry]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"unknown permitted tool(s) for {self.profile.domain_id}: {missing_list}")
        for tool_name in self.profile.permitted_tools:
            selected[tool_name] = registry[tool_name]
        return selected


class DomainAgent:
    """QwenAgent-backed runtime agent scoped to one declarative domain profile."""

    def __init__(
        self,
        profile: DomainProfile,
        *,
        gateway: ModelGateway | None = None,
        knowledge: KnowledgePlatform | None = None,
        runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
    ) -> None:
        self.profile = profile.normalized()
        self.gateway = gateway or ModelGateway.from_env()
        self.knowledge = knowledge or KnowledgePlatform.in_memory()
        self.model_adapter = OpenNICFChatModel(
            gateway=self.gateway,
            task_class=self.profile.task_class,
            privacy=self.profile.privacy_policy,
        )
        self.tools = DomainTools(self.profile, self.knowledge)
        self._runtime_factory = runtime_factory or QwenAgentRuntime
        self._runtime: Any | None = None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_factory(self.model_adapter, self.tools.permitted_qwen_tools())
        return self._runtime

    @property
    def agent(self) -> Any:
        return self.runtime

    def run(self, request: str) -> Any:
        return self.runtime.run(request)


class DomainAgentFactory:
    """Factory that builds isolated domain agents over shared runtime services."""

    def __init__(
        self,
        *,
        gateway: ModelGateway | None = None,
        knowledge: KnowledgePlatform | None = None,
        profiles: Mapping[str, DomainProfile] | None = None,
        runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
    ) -> None:
        self.gateway = gateway or ModelGateway.from_env()
        self.knowledge = knowledge or KnowledgePlatform.in_memory()
        self.profiles = dict(profiles or DEFAULT_DOMAIN_PROFILES)
        self._runtime_factory = runtime_factory or QwenAgentRuntime

    def available_domains(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def profile_for(self, domain_id: str) -> DomainProfile:
        try:
            return self.profiles[domain_id]
        except KeyError as exc:
            raise KeyError(f"unknown domain_id: {domain_id}") from exc

    def create(self, domain_id: str) -> DomainAgent:
        profile = self.profile_for(domain_id)
        return DomainAgent(
            profile,
            gateway=self.gateway,
            knowledge=self.knowledge,
            runtime_factory=self._runtime_factory,
        )


def create_domain_agent_factory(
    *,
    gateway: ModelGateway | None = None,
    knowledge: KnowledgePlatform | None = None,
    profiles: Mapping[str, DomainProfile] | None = None,
    runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
) -> DomainAgentFactory:
    return DomainAgentFactory(
        gateway=gateway,
        knowledge=knowledge,
        profiles=profiles,
        runtime_factory=runtime_factory,
    )
