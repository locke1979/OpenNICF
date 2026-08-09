from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import types

import pytest

from opennicf import (
    ContraordenacionalDomainAgent,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    ModelGateway,
    create_contraordenacional_domain_agent_factory,
)


FIXTURES = Path(__file__).parent / "fixtures" / "contraordenacional"


@dataclass
class FakeRuntime:
    model_adapter: object
    tools: dict

    def run(self, request: str):
        return {"request": request, "tool_names": tuple(self.tools)}


def platform() -> KnowledgePlatform:
    result = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    for name, source_type, system in (
        ("schedule", "document", "sco_batches"),
        ("batch-log", "log", "sco_batches"),
        ("batch-code", "code", "sco_application"),
        ("integration", "document", "sco_process_management"),
    ):
        suffix = {"schedule": "sco_batch_schedule.json", "batch-log": "sco_batch.log", "batch-code": "sco_batch_worker.py", "integration": "sco_integration.md"}[name]
        result.ingest(
            source_id=f"sco-{name}",
            source_uri=f"tests/fixtures/contraordenacional/{suffix}",
            content=(FIXTURES / suffix).read_text(encoding="utf-8"),
            domain_id="contraordenacional",
            system_id=system,
            component_id="sco_evidence",
            domain="contraordenacional",
            system=system,
            environment="test",
            evidence_type=source_type,
            source_type=source_type,
            acl_scope="internal",
            parser_version="1",
            metadata={"correlation_ids": ["sco-trace-18"], "component_id": "sco_evidence"},
        )
    # A same-corpus record proves that the domain filter, rather than fixture
    # naming, is responsible for ownership isolation.
    result.ingest(
        source_id="criminal-only",
        source_uri="tests/fixtures/knowledge/document.md",
        content="criminal collaboration evidence",
        domain_id="criminal",
        system_id="criminal_casework",
        component_id="criminal_evidence",
        domain="criminal",
        system="criminal_casework",
        environment="test",
        evidence_type="document",
        source_type="document",
        acl_scope="internal",
        parser_version="1",
    )
    return result


def factory() -> object:
    return create_contraordenacional_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=platform(),
        runtime_factory=lambda model, tools: FakeRuntime(model, tools),
    )


def test_sco_aliases_use_shared_gateway_knowledge_and_qwen_runtime():
    agent_factory = factory()
    agent = agent_factory.create("SCO")
    assert isinstance(agent, ContraordenacionalDomainAgent)
    assert agent.domain_id == "contraordenacional"
    assert agent.profile.owned_systems == ("sco_application", "sco_batches", "sco_process_management")
    assert agent_factory.create("sco-batch").gateway is agent.gateway
    assert agent_factory.create("sco-application").knowledge is agent.knowledge
    assert agent.run("review batch")['request'] == "review batch"


def test_sco_retrieval_is_strictly_domain_and_acl_scoped():
    agent = factory().create("sco")
    with pytest.raises(PermissionError):
        agent.tools.search_domain_evidence({"query": "batch", "domain_ids": ["criminal"]})
    with pytest.raises(PermissionError):
        agent.tools.search_domain_evidence({"query": "batch", "principal_acl_scopes": ["public"]})
    result = agent.tools.search_domain_evidence({"query": "batch", "principal_acl_scopes": ["internal"]})
    assert result["packages"]
    assert all(item["domain_id"] == "contraordenacional" for item in result["packages"])
    assert all(item["source_id"] != "criminal-only" for item in result["packages"])
    with pytest.raises(PermissionError):
        agent.tools.search_docs({"query": "criminal", "domains": ["criminal"]})


def test_batch_audit_correlates_schedule_log_and_code_with_provenance():
    report = factory().create("sco").audit(
        {
            "query": "batch execution_id=batch-20260808-0017 correlation_id=sco-trace-18 return code",
            "source_types": ["document", "log", "code"],
            "systems": ["sco_batches", "sco_application"],
        }
    )
    assert report.request.metadata["domain_id"] == "contraordenacional"
    assert any("failure signal" in finding.statement for finding in report.findings)
    assert any("sco_batch_worker.py" in ref.provenance_ref for finding in report.findings for ref in finding.evidence_refs)
    assert any("sco-trace-18" in finding.correlation_ids for finding in report.findings)
    assert all(ref.source_id.startswith("sco-") for finding in report.findings for ref in finding.evidence_refs)


def test_sco_integration_requires_explicit_delegation_and_maps_sinquer_to_criminal():
    agent = factory().create("contraordenacional")
    request = agent.request_integration_delegation(
        target_alias="SINQUER", query="compare producer contract", correlation_ids=["sco-trace-18"]
    )
    assert request["target_domain_id"] == "criminal"
    assert request["authorization"] == "explicit-coordinator-delegation-required"
    with pytest.raises(PermissionError):
        agent.request_integration_delegation(target_alias="unknown-system", query="read foreign corpus")
