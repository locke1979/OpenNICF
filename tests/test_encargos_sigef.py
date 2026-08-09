from __future__ import annotations

from dataclasses import dataclass

import pytest

from opennicf import (
    ENCARGOS_SIGEF_PROFILE,
    EncargosSigefDomainAgent,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    ModelGateway,
    create_domain_agent_factory,
)


@dataclass
class FakeRuntime:
    model_adapter: object
    tools: dict

    def run(self, request: str):
        return {"request": request, "tools": tuple(self.tools)}


class Broker:
    def __init__(self):
        self.requests = []

    def request(self, payload):
        self.requests.append(payload)
        return {"status": "pending", "request": payload}


def factory(broker=None):
    return create_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=KnowledgePlatform(
            MemoryKnowledgeStore(),
            MemoryObjectStore(),
            LocalFirstEmbeddingService(),
        ),
        diagnostic_broker=broker,
        runtime_factory=lambda model, tools: FakeRuntime(model, tools),
    )


def test_factory_builds_specialized_agent_on_shared_services():
    shared = factory()
    agent = shared.create("encargos_sigef")

    assert isinstance(agent, EncargosSigefDomainAgent)
    assert agent.profile == ENCARGOS_SIGEF_PROFILE
    assert agent.gateway is shared.gateway
    assert agent.knowledge is shared.knowledge
    result = agent.run("review batch")
    assert "correlate_encargos_batch" in result["tools"]
    assert "request_integration_collaboration" in result["tools"]


def test_batch_correlation_is_domain_acl_scoped():
    agent = factory().create("encargos_sigef")

    with pytest.raises(PermissionError, match="outside the SIGEF/Encargos"):
        agent.tools.correlate_encargos_batch(
            {"batch_id": "EF-42", "system_ids": ["SICJUT"]}
        )


def test_collaboration_requires_explicit_foreign_domain_delegation():
    broker = Broker()
    agent = factory(broker).create("encargos_sigef")

    with pytest.raises(PermissionError, match="explicit single-domain"):
        agent.tools.request_integration_collaboration({"target": "SICJUT"})

    response = agent.tools.request_integration_collaboration(
        {
            "target": "SICJUT",
            "delegated_domain_ids": ["contencioso_judicial"],
            "operation_class": "describe",
        }
    )
    assert response["status"] == "pending"
    assert broker.requests[0]["domain_id"] == "encargos_sigef"
    assert broker.requests[0]["target_system"] == "SICJUT"


def test_shared_boundaries_cannot_be_retrieved_as_owned_systems():
    agent = factory().create("encargos_sigef")
    with pytest.raises(PermissionError):
        agent.tools.correlate_encargos_batch({"batch_id": "B-1", "systems": ["GFF"]})
