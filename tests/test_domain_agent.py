from __future__ import annotations

from dataclasses import dataclass
import sys
import types

import pytest

from opennicf import (
    DEFAULT_DOMAIN_IDS,
    DEFAULT_DOMAIN_PROFILES,
    DomainAgentFactory,
    DomainProfile,
    KnowledgePlatform,
    ModelGateway,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    HashingEmbeddingBackend,
    LocalFirstEmbeddingService,
    PrivacyPolicy,
)


@dataclass
class _FakeRuntime:
    model_adapter: object
    tools: dict

    def run(self, request: str):
        return {"request": request, "tools": tuple(self.tools)}


class _FakeAssistant:
    instances: list["_FakeAssistant"] = []

    def __init__(self, llm, function_list):
        self.llm = llm
        self.function_list = function_list
        self.__class__.instances.append(self)

    def run(self, messages):
        return {
            "messages": messages,
            "tool_names": tuple(function.__name__ for function in self.function_list),
        }


def _factory() -> DomainAgentFactory:
    gateway = ModelGateway.from_env({})
    knowledge = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    return DomainAgentFactory(
        gateway=gateway,
        knowledge=knowledge,
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
    )


def _install_fake_qwen_agent(monkeypatch):
    fake_qwen_agent = types.ModuleType("qwen_agent")
    fake_agents = types.ModuleType("qwen_agent.agents")
    fake_agents.Assistant = _FakeAssistant
    fake_qwen_agent.agents = fake_agents
    monkeypatch.setitem(sys.modules, "qwen_agent", fake_qwen_agent)
    monkeypatch.setitem(sys.modules, "qwen_agent.agents", fake_agents)


def test_default_domain_profiles_cover_the_mandatory_set():
    assert DEFAULT_DOMAIN_IDS == tuple(DEFAULT_DOMAIN_PROFILES)
    assert all(isinstance(profile, DomainProfile) for profile in DEFAULT_DOMAIN_PROFILES.values())
    assert all(profile.default_retrieval_filters.domains == (domain_id,) for domain_id, profile in DEFAULT_DOMAIN_PROFILES.items())


def test_factory_reuses_shared_gateway_and_knowledge():
    factory = _factory()
    criminal = factory.create("criminal")
    sigef = factory.create("encargos_sigef")

    assert criminal.gateway is sigef.gateway
    assert criminal.knowledge is sigef.knowledge
    assert criminal.model_adapter.gateway is criminal.gateway
    assert sigef.model_adapter.gateway is sigef.gateway


def test_domain_agent_scopes_retrieval_to_its_profile_domain():
    factory = _factory()
    platform = factory.knowledge
    platform.ingest(
        source_id="criminal-doc",
        source_uri="tests/fixtures/knowledge/document.md",
        content="criminal only evidence",
        acl_scope="internal",
        domain="criminal",
        system="casework",
        environment="dev",
        source_type="document",
        parser_version="1",
    )
    platform.ingest(
        source_id="sigef-doc",
        source_uri="tests/fixtures/knowledge/document.md",
        content="sigef only evidence",
        acl_scope="internal",
        domain="encargos_sigef",
        system="casework",
        environment="dev",
        source_type="document",
        parser_version="1",
    )

    criminal = factory.create("criminal")
    hits = criminal.tools.search_domain_evidence("evidence", limit=5)

    assert hits
    assert all(hit["domain"] == "criminal" for hit in hits)
    assert all(hit["source_id"] != "sigef-doc" for hit in hits)


def test_domain_agent_is_qwen_backed_and_only_exposes_permitted_tools(monkeypatch):
    _FakeAssistant.instances.clear()
    _install_fake_qwen_agent(monkeypatch)
    gateway = ModelGateway.from_env({})
    knowledge = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    profile = DomainProfile(
        domain_id="criminal",
        name="Criminal",
        description="Test profile",
        permitted_tools=("search_domain_evidence",),
    ).normalized()
    agent = DomainAgentFactory(
        gateway=gateway,
        knowledge=knowledge,
        profiles={"criminal": profile},
    ).create("criminal")

    result = agent.run("review the record")

    assert result["messages"] == [{"role": "user", "content": "review the record"}]
    assert result["tool_names"] == ("search_domain_evidence",)
    assert len(_FakeAssistant.instances) == 1
    assert _FakeAssistant.instances[0].llm is agent.model_adapter
    assert tuple(function.__name__ for function in _FakeAssistant.instances[0].function_list) == ("search_domain_evidence",)


def test_domain_diagnostic_enforces_the_domain_boundary_and_allow_list():
    factory = _factory()
    criminal = factory.create("criminal")

    with pytest.raises(PermissionError):
        criminal.tools.request_domain_diagnostic({"operation_class": "drop"})

    result = criminal.tools.request_domain_diagnostic(
        {
            "operation_class": "describe",
            "domain_id": "encargos_sigef",
            "target": "casework",
        }
    )

    assert result["domain_id"] == "criminal"
    assert result["request"]["domain_id"] == "criminal"
    assert result["request"]["target"] == "casework"


def test_factory_adds_new_domains_from_registry_configuration():
    profile = DomainProfile(
        domain_id="civil",
        name="Civil",
        description="Civil matters and supporting evidence.",
        owned_systems=("civil_casework", "civil_evidence"),
        integration_boundary_aliases=("civil-litigation",),
        permitted_evidence_classes=("document", "log"),
        permitted_tools=("search_domain_evidence",),
    ).normalized()
    factory = DomainAgentFactory(
        gateway=ModelGateway.from_env({}),
        knowledge=KnowledgePlatform(
            MemoryKnowledgeStore(),
            MemoryObjectStore(),
            LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
        ),
        profiles={**DEFAULT_DOMAIN_PROFILES, "civil": profile},
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
    )

    civil = factory.create("civil")

    assert factory.available_domains()[-1] == "civil"
    assert civil.profile.domain_id == "civil"
    assert set(civil.tools.permitted_qwen_tools()) == {"search_domain_evidence"}


def test_local_only_preserves_the_profile_policy():
    profile = DomainProfile(
        domain_id="criminal",
        name="Criminal",
        description="Test profile",
        privacy_policy=PrivacyPolicy.LOCAL_ONLY,
    ).normalized()
    factory = DomainAgentFactory(
        profiles={"criminal": profile},
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
    )
    agent = factory.create("criminal")

    assert agent.model_adapter.privacy.value == "local_only"
