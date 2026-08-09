from __future__ import annotations

import pytest

from opennicf import (
    CONTENCIOSO_ADMINISTRATIVO_DELEGATION_TARGETS,
    CONTENCIOSO_ADMINISTRATIVO_SYSTEMS,
    ContenciosoAdministrativoDomainAgent,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    ModelGateway,
    create_domain_agent_factory,
)


class FakeRuntime:
    def __init__(self, model_adapter, tools):
        self.model_adapter = model_adapter
        self.tools = tools

    def run(self, request):
        return {"request": request, "tool_names": tuple(self.tools)}


def platform_with_admin_and_foreign_evidence() -> KnowledgePlatform:
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    for system_id, domain_id in (
        ("SICAT", "contencioso_administrativo"),
        ("SICATPF", "contencioso_administrativo"),
        ("SIGEPRA", "contencioso_administrativo"),
        ("WSCAT", "contencioso_administrativo"),
        ("ISIGEPRAWS", "contencioso_administrativo"),
        ("JTRECLBAT", "contencioso_administrativo"),
        ("SICJUT", "contencioso_judicial"),
        ("SIGEF", "encargos_sigef"),
    ):
        platform.ingest(
            source_id=system_id,
            source_uri=f"evidence/{system_id}.log",
            content=f"{system_id} process correlation-id administrative incident",
            domain=domain_id,
            system_id=system_id,
            component_id=f"{system_id}-component",
            environment="test",
            source_type="log",
        )
    return platform


def test_shared_factory_returns_admin_agent_and_routes_all_sanitized_aliases():
    factory = create_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=platform_with_admin_and_foreign_evidence(),
        runtime_factory=FakeRuntime,
    )

    agent = factory.create("SICAT")

    assert isinstance(agent, ContenciosoAdministrativoDomainAgent)
    assert factory.domain_for_alias("isigepraws") == "contencioso_administrativo"
    assert agent.profile.owned_systems == CONTENCIOSO_ADMINISTRATIVO_SYSTEMS
    assert agent.gateway is factory.gateway
    assert agent.knowledge is factory.knowledge


def test_admin_acl_pins_retrieval_to_owned_systems():
    factory = create_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=platform_with_admin_and_foreign_evidence(),
        runtime_factory=FakeRuntime,
    )
    agent = factory.create("contencioso_administrativo")

    result = agent.tools.search_logs({"query": "incident", "limit": 20})

    assert result["packages"]
    assert {package["system_id"] for package in result["packages"]} <= set(CONTENCIOSO_ADMINISTRATIVO_SYSTEMS)
    with pytest.raises(PermissionError, match="outside the registered domain ACL"):
        agent.tools.search_logs({"query": "incident", "system_ids": ["SICJUT"]})


def test_admin_collaboration_is_explicit_and_limited_to_judicial_or_sigef():
    factory = create_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=platform_with_admin_and_foreign_evidence(),
        runtime_factory=FakeRuntime,
    )
    agent = factory.create("contencioso_administrativo")

    request = agent.tools.request_cross_domain_collaboration(
        {
            "target_domain_id": "contencioso_judicial",
            "integration_edge": "SICAT->SICJUT",
            "evidence_refs": ["SICAT:log:1"],
            "query": "correlate the judicial persistence failure",
        }
    )

    assert request["status"] == "cross_domain_required"
    assert request["target_domain_id"] in CONTENCIOSO_ADMINISTRATIVO_DELEGATION_TARGETS
    assert request["source_domain_id"] == "contencioso_administrativo"
    with pytest.raises(PermissionError):
        agent.tools.request_cross_domain_collaboration(
            {"target_domain_id": "criminal", "evidence_refs": ["SICAT:log:1"]}
        )


def test_qwen_runtime_surface_contains_only_the_explicit_collaboration_tool():
    factory = create_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=platform_with_admin_and_foreign_evidence(),
        runtime_factory=FakeRuntime,
    )
    agent = factory.create("contencioso_administrativo")

    runtime_result = agent.run("audit the administrative incident")

    assert "request_cross_domain_collaboration" in runtime_result["tool_names"]
    assert "search_delegated_domain_evidence" not in runtime_result["tool_names"]
