from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opennicf import (
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    ModelGateway,
    create_domain_agent_factory,
    create_integration_correlation_agent,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "integration"


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


@dataclass
class _FakeRuntime:
    model_adapter: object
    tools: dict

    def run(self, request: str):
        return {"request": request, "tool_names": tuple(self.tools)}


def _platform() -> KnowledgePlatform:
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    platform.ingest(
        source_id="criminal-sinquer",
        source_uri="tests/fixtures/integration/criminal_sinquer.log",
        content=_load("criminal_sinquer.log"),
        acl_scope="internal",
        domain="criminal",
        domain_id="criminal",
        system="SINQUER",
        system_id="SINQUER",
        environment="test",
        source_type="log",
        parser_version="1",
        integration_edges=[
            {
                "target_domain_id": "contraordenacional",
                "relation_type": "case-reference",
                "component_id": "SCO workflow",
                "system_id": "SCO",
            }
        ],
    )
    platform.ingest(
        source_id="contraordenacional-sco",
        source_uri="tests/fixtures/integration/contraordenacional_sco.log",
        content=_load("contraordenacional_sco.log"),
        acl_scope="internal",
        domain="contraordenacional",
        domain_id="contraordenacional",
        system="SCO",
        system_id="SCO",
        environment="test",
        source_type="log",
        parser_version="1",
        integration_edges=[
            {
                "target_domain_id": "criminal",
                "relation_type": "case-reference",
                "component_id": "SINQUER boundary",
                "system_id": "SINQUER",
            }
        ],
    )
    platform.ingest(
        source_id="administrative-sigepra",
        source_uri="tests/fixtures/integration/contencioso_administrativo.log",
        content=_load("contencioso_administrativo.log"),
        acl_scope="internal",
        domain="contencioso_administrativo",
        domain_id="contencioso_administrativo",
        system="SIGEPRA",
        system_id="SIGEPRA",
        environment="test",
        source_type="log",
        parser_version="1",
        integration_edges=[
            {
                "target_domain_id": "contencioso_judicial",
                "relation_type": "handoff",
                "component_id": "SICJUT workflow",
                "system_id": "SICJUT",
            }
        ],
    )
    platform.ingest(
        source_id="judicial-sicjut",
        source_uri="tests/fixtures/integration/contencioso_judicial.log",
        content=_load("contencioso_judicial.log"),
        acl_scope="internal",
        domain="contencioso_judicial",
        domain_id="contencioso_judicial",
        system="SICJUT",
        system_id="SICJUT",
        environment="test",
        source_type="log",
        parser_version="1",
        integration_edges=[
            {
                "target_domain_id": "encargos_sigef",
                "relation_type": "escalation",
                "component_id": "SIGEF batch bridge",
                "system_id": "SIGEF",
            },
            {
                "target_domain_id": "contencioso_administrativo",
                "relation_type": "counterpart",
                "component_id": "SIGEPRA docket",
                "system_id": "SIGEPRA",
            },
        ],
    )
    platform.ingest(
        source_id="sigef-batch",
        source_uri="tests/fixtures/integration/encargos_sigef.log",
        content=_load("encargos_sigef.log"),
        acl_scope="internal",
        domain="encargos_sigef",
        domain_id="encargos_sigef",
        system="SIGEF",
        system_id="SIGEF",
        environment="test",
        source_type="log",
        parser_version="1",
        integration_edges=[
            {
                "target_domain_id": "contencioso_judicial",
                "relation_type": "delegates-to",
                "component_id": "SICJUT integration",
                "system_id": "SICJUT",
            },
            {
                "target_domain_id": "contencioso_administrativo",
                "relation_type": "delegates-to",
                "component_id": "SIGEPRA integration",
                "system_id": "SIGEPRA",
            },
        ],
    )
    return platform


def _factory(platform: KnowledgePlatform | None = None):
    return create_domain_agent_factory(
        gateway=ModelGateway.from_env({}),
        knowledge=platform or _platform(),
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
    )


def _agent(platform: KnowledgePlatform | None = None, **kwargs):
    return create_integration_correlation_agent(
        gateway=ModelGateway.from_env({}),
        knowledge=platform or _platform(),
        domain_factory=_factory(platform),
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
        **kwargs,
    )


def test_runtime_exposes_the_single_integration_correlation_tool():
    agent = _agent()

    runtime_result = agent.run("correlate the cross-domain handoff")

    assert runtime_result["request"] == "correlate the cross-domain handoff"
    assert runtime_result["tool_names"] == ("request_integration_correlation",)


def test_sco_and_sinquer_fan_out_preserves_provenance_and_edges():
    agent = _agent()

    result = agent.correlate(
        {
            "question": "correlate SCO and SINQUER case ownership",
            "systems": ["SCO", "SINQUER"],
            "correlation_ids": ["trace-sco-22"],
            "source_types": ["log"],
            "evidence_budget": 4,
        }
    )

    assert result["status"] == "ok"
    assert set(result["selected_domain_ids"]) == {"criminal", "contraordenacional"}
    assert len(result["delegations"]) == 2
    assert all(delegation["return_code"] == 0 for delegation in result["delegations"])
    assert all(ref["provenance_ref"] for delegation in result["delegations"] for ref in delegation["evidence_refs"])
    assert any(edge["target_domain_id"] == "contraordenacional" for edge in result["integration_graph"]["edges"])
    assert any(edge["target_domain_id"] == "criminal" for edge in result["integration_graph"]["edges"])
    assert any(finding["classification"] == "fact" for finding in result["findings"])
    assert result["estimated_tokens"] > 0
    assert result["estimated_bytes"] > 0


def test_administrative_and_judicial_fan_out_keeps_component_ownership_sanitized():
    agent = _agent()

    result = agent.correlate(
        {
            "question": "correlate administrative and judicial handoff",
            "systems": ["SIGEPRA", "SICJUT"],
            "correlation_ids": ["trace-admin-22"],
            "source_types": ["log"],
            "evidence_budget": 4,
        }
    )

    assert result["status"] == "ok"
    assert set(result["selected_domain_ids"]) == {"contencioso_administrativo", "contencioso_judicial"}
    assert all("http" not in json_value.lower() for delegation in result["delegations"] for json_value in [str(delegation["component_ownership"])])
    assert all(delegation["request"]["question"] == "correlate administrative and judicial handoff" for delegation in result["delegations"])
    assert all(delegation["request"]["time_window"] == {"start": None, "end": None} for delegation in result["delegations"])
    assert any("handoff" in edge["relation_type"] for edge in result["integration_graph"]["edges"])


def test_sigef_sicjut_sigepra_fan_out_is_bounded_and_preserves_timestamps():
    agent = _agent()

    result = agent.correlate(
        {
            "question": "correlate SIGEF SICJUT SIGEPRA batch handoff",
            "systems": ["SIGEF", "SICJUT", "SIGEPRA"],
            "correlation_ids": ["trace-sigef-22"],
            "start_time": "2026-08-08T12:00:00Z",
            "end_time": "2026-08-08T12:10:00Z",
            "source_types": ["log"],
            "evidence_budget": 5,
            "token_budget": 5000,
        }
    )

    assert result["status"] == "ok"
    assert len(result["selected_domain_ids"]) == 3
    assert result["bounded_fan_out"] is False
    assert all(delegation["request"]["time_window"] == {"start": "2026-08-08T12:00:00+00:00", "end": "2026-08-08T12:10:00+00:00"} for delegation in result["delegations"])
    assert all(delegation["retrieval"]["estimated_tokens"] > 0 for delegation in result["delegations"])
    assert any(timestamp.startswith("2026-08-08T12:05") for delegation in result["delegations"] for timestamp in delegation["timestamps"])
    assert any(edge["target_domain_id"] == "encargos_sigef" for edge in result["integration_graph"]["edges"])


def test_partial_failures_are_reported_explicitly_without_losing_successful_delegations():
    class _FailingTools:
        def __init__(self, profile):
            self.profile = profile

        def search_domain_evidence(self, request, *, limit=None):
            raise RuntimeError("simulated delegate failure")

    class _FailingFactory:
        def __init__(self, delegate):
            self.delegate = delegate

        @property
        def gateway(self):
            return self.delegate.gateway

        @property
        def knowledge(self):
            return self.delegate.knowledge

        def resolve_domain_id(self, value):
            return self.delegate.resolve_domain_id(value)

        def profile_for(self, domain_id):
            return self.delegate.profile_for(domain_id)

        def create(self, domain_id):
            agent = self.delegate.create(domain_id)
            if domain_id == "contencioso_judicial":
                return type("BrokenAgent", (), {"profile": agent.profile, "tools": _FailingTools(agent.profile)})()
            return agent

    delegate = _factory()
    agent = create_integration_correlation_agent(
        gateway=delegate.gateway,
        knowledge=delegate.knowledge,
        domain_factory=_FailingFactory(delegate),
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
    )

    result = agent.correlate(
        {
            "question": "correlate SIGEF and SICJUT despite one failing delegate",
            "systems": ["SIGEF", "SICJUT"],
            "source_types": ["log"],
            "evidence_budget": 4,
        }
    )

    assert result["status"] == "partial"
    assert any(delegation["status"] == "failed" for delegation in result["delegations"])
    assert any(delegation["status"] == "ok" for delegation in result["delegations"])
    assert any(finding["classification"] == "missing_evidence" for finding in result["findings"])
    assert any(finding["classification"] == "hypothesis" for finding in result["findings"])
    assert any(item["domain_id"] == "contencioso_judicial" for item in result["contradicting_evidence"])
