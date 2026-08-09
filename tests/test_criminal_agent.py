from __future__ import annotations

from pathlib import Path

from opennicf import (
    CriminalDomainAgent,
    DomainAgentFactory,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    ModelGateway,
)


FIXTURES = Path(__file__).parent / "fixtures" / "criminal"


def _platform() -> KnowledgePlatform:
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    platform.ingest(
        source_id="sinquer-service",
        source_uri="tests/fixtures/criminal/sinquer_service.py",
        content=(FIXTURES / "sinquer_service.py").read_text(),
        acl_scope="internal",
        domain="criminal",
        system="sinquer",
        environment="test",
        source_type="code",
        parser_version="1",
        metadata={"ownership_aliases": ["SINQUER", "sinquer-criminal"]},
    )
    platform.ingest(
        source_id="sinquer-boundary",
        source_uri="tests/fixtures/criminal/sinquer_boundary.md",
        content=(FIXTURES / "sinquer_boundary.md").read_text(),
        acl_scope="internal",
        domain="criminal",
        system="sinquer",
        environment="test",
        source_type="document",
        parser_version="1",
        metadata={
            "integration_edges": [
                {
                    "target_domain_id": "contraordenacional",
                    "relation_type": "case-reference",
                    "component": "SCO workflow",
                }
            ]
        },
    )
    platform.ingest(
        source_id="sco-only",
        source_uri="tests/fixtures/criminal/sco_owned.md",
        content="SCO-only administrative offense evidence.",
        acl_scope="internal",
        domain="contraordenacional",
        system="sco",
        environment="test",
        source_type="document",
        parser_version="1",
    )
    return platform


def _factory() -> DomainAgentFactory:
    return DomainAgentFactory(gateway=ModelGateway.from_env({}), knowledge=_platform(), runtime_factory=lambda model, tools: (model, tools))


def test_sinquer_alias_routes_to_criminal_agent():
    factory = _factory()
    agent = factory.create("SINQUER")
    assert isinstance(agent, CriminalDomainAgent)
    assert factory.resolve_domain_id("criminal-sinquer") == "criminal"


def test_criminal_retrieval_never_returns_other_domain_corpus():
    result = _factory().create("sinquer").tools.search_docs({"query": "evidence", "limit": 20})
    assert result["packages"]
    assert all(package["domain_id"] == "criminal" for package in result["packages"])
    assert all(package["source_id"] != "sco-only" for package in result["packages"])


def test_audit_preserves_provenance_and_classification():
    result = _factory().create("criminal").run_failure_audit("load_case")
    assert result["findings"]
    assert all(finding["classification"] == "fact" for finding in result["findings"])
    assert all(finding["provenance_ref"] for finding in result["findings"])


def test_mixed_sinquer_evidence_requires_cross_domain_handoff():
    result = _factory().create("SINQUER").retrieve("Contra-Ordenacional SCO workflow")
    assert result["cross_domain_required"] is True
    assert result["status"] == "cross_domain_required"
    assert result["requested_domain_id"] == "contraordenacional"
    assert all(ref["provenance_ref"] for ref in result["evidence"])


def test_diagnostic_interface_is_controlled_and_domain_bound():
    agent = _factory().create("criminal")
    pending = agent.request_live_verification({"operation_class": "describe", "query": "case schema"})
    assert pending["domain_id"] == "criminal"
    try:
        agent.request_live_verification({"operation_class": "execute", "query": "DROP TABLE cases"})
    except PermissionError:
        pass
    else:
        raise AssertionError("unsafe diagnostic operation was accepted")
