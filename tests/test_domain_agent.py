from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import types

import pytest

from opennicf import (
    CONTENCIOSO_JUDICIAL_ALIASES,
    CONTENCIOSO_JUDICIAL_DELEGATED_DOMAINS,
    ContenciosoJudicialDomainAgent,
    DEFAULT_DOMAIN_IDS,
    DEFAULT_DOMAIN_PROFILES,
    DomainAgentFactory,
    DomainProfile,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    ModelGateway,
    PrivacyPolicy,
)
from opennicf.domain_agent import GENERIC_RETRIEVAL_TOOL_NAMES


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "knowledge"


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


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


def _platform() -> KnowledgePlatform:
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=64)),
    )
    platform.ingest(
        source_id="criminal-code",
        source_uri="tests/fixtures/knowledge/service.py",
        content=_load("service.py"),
        acl_scope="internal",
        domain="criminal",
        system="casework",
        environment="dev",
        source_type="code",
        parser_version="1",
    )
    platform.ingest(
        source_id="sigef-code",
        source_uri="tests/fixtures/knowledge/service.py",
        content=_load("service.py"),
        acl_scope="internal",
        domain="encargos_sigef",
        system="casework",
        environment="dev",
        source_type="code",
        parser_version="1",
    )
    platform.ingest(
        source_id="criminal-doc",
        source_uri="tests/fixtures/knowledge/document.md",
        content=_load("document.md"),
        acl_scope="internal",
        domain="criminal",
        system="casework",
        environment="dev",
        source_type="document",
        parser_version="1",
    )
    platform.ingest(
        source_id="criminal-schema",
        source_uri="tests/fixtures/knowledge/schema.sql",
        content=_load("schema.sql"),
        acl_scope="internal",
        domain="criminal",
        system="ledger",
        environment="dev",
        source_type="schema",
        parser_version="1",
    )
    platform.ingest(
        source_id="criminal-query-output",
        source_uri="tests/fixtures/knowledge/query-output.txt",
        content=_load("query-output.txt"),
        acl_scope="internal",
        domain="criminal",
        system="ledger",
        environment="dev",
        source_type="query-output",
        parser_version="1",
    )
    platform.ingest(
        source_id="criminal-log",
        source_uri="tests/fixtures/knowledge/logs.txt",
        content=_load("logs.txt"),
        acl_scope="internal",
        domain="criminal",
        system="gateway",
        environment="prod",
        source_type="log",
        parser_version="1",
        metadata={
            "correlation_ids": ["trace-91"],
            "request_ids": ["req-17"],
            "session_ids": ["sess-4"],
        },
    )
    return platform


def _factory() -> DomainAgentFactory:
    gateway = ModelGateway.from_env({})
    return DomainAgentFactory(
        gateway=gateway,
        knowledge=_platform(),
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


def test_domain_agent_exposes_the_generic_retrieval_tool_surface(monkeypatch):
    _FakeAssistant.instances.clear()
    _install_fake_qwen_agent(monkeypatch)
    factory = DomainAgentFactory(
        gateway=ModelGateway.from_env({}),
        knowledge=_platform(),
    )
    criminal = factory.create("criminal")

    result = criminal.run("review the record")

    assert result["messages"] == [{"role": "user", "content": "review the record"}]
    assert result["tool_names"] == (*GENERIC_RETRIEVAL_TOOL_NAMES, "request_domain_diagnostic")
    assert len(_FakeAssistant.instances) == 1
    assert _FakeAssistant.instances[0].llm is criminal.model_adapter


def test_code_retrieval_succeeds_without_vector_dependence():
    factory = _factory()
    criminal = factory.create("criminal")

    exact = criminal.tools.search_code_exact({"query": "build_query", "limit": 5})
    symbols = criminal.tools.search_code_symbols({"method_name": "route", "class_name": "EvidenceRouter", "limit": 5})
    semantic = criminal.tools.search_code_semantic({"query": "evidence router scope", "limit": 5})

    assert exact["packages"]
    assert exact["packages"][0]["source_id"] == "criminal-code"
    assert all(package["source_id"] != "sigef-code" for package in exact["packages"])
    assert symbols["packages"]
    assert symbols["packages"][0]["source_id"] == "criminal-code"
    assert symbols["packages"][0]["evidence_refs"][0]["metadata"]["match_kind"] == "symbol"
    assert semantic["packages"]
    assert semantic["packages"][0]["source_id"] == "criminal-code"


def test_domain_evidence_stays_scoped_to_the_registered_domain():
    factory = _factory()
    criminal = factory.create("criminal")

    hits = criminal.tools.search_domain_evidence("build_query", limit=5)

    assert hits["packages"]
    assert all(package["domain_id"] == "criminal" for package in hits["packages"])
    assert all(package["source_id"] != "sigef-code" for package in hits["packages"])


def test_log_search_filters_by_time_and_correlation_id():
    factory = _factory()
    criminal = factory.create("criminal")

    hits = criminal.tools.search_logs(
        {
            "query": "gateway timeout",
            "correlation_ids": ["trace-91"],
            "request_ids": ["req-17"],
            "since": datetime(2026, 8, 8, 10, 14, 59, tzinfo=timezone.utc),
            "until": datetime(2026, 8, 8, 10, 15, 2, tzinfo=timezone.utc),
            "limit": 5,
        }
    )

    assert hits["packages"]
    top = hits["packages"][0]
    assert top["source_id"] == "criminal-log"
    assert top["estimated_tokens"] > 0
    assert top["evidence_refs"][0]["metadata"]["correlation_ids"] == ["trace-91"]
    assert top["evidence_refs"][0]["metadata"]["request_ids"] == ["req-17"]


def test_docs_schema_and_query_output_packages_preserve_locators():
    factory = _factory()
    criminal = factory.create("criminal")

    docs = criminal.tools.search_docs({"query": "immutable originals", "limit": 5})
    schema = criminal.tools.search_schema({"query": "artifact_hash", "limit": 5})
    query_outputs = criminal.tools.search_query_outputs({"query": "acl_scope", "limit": 5})

    assert docs["packages"]
    assert schema["packages"]
    assert query_outputs["packages"]
    assert docs["packages"][0]["locator"].startswith("tests/fixtures/knowledge/document.md")
    assert schema["packages"][0]["locator"].startswith("tests/fixtures/knowledge/schema.sql")
    assert query_outputs["packages"][0]["locator"].startswith("tests/fixtures/knowledge/query-output.txt")
    assert docs["packages"][0]["estimated_bytes"] < sum(len(text.encode("utf-8")) for text in (_load("document.md"), _load("schema.sql"), _load("query-output.txt")))


def test_delegated_interface_requires_explicit_coordinator():
    factory = _factory()
    criminal = factory.create("criminal")

    with pytest.raises(NotImplementedError):
        criminal.tools.search_delegated_domain_evidence(
            {
                "query": "build_query",
                "delegated_domain_ids": ["encargos_sigef"],
                "limit": 5,
            }
        )


def test_retrieval_packages_are_more_compact_than_loading_full_corpora():
    factory = _factory()
    criminal = factory.create("criminal")

    docs = criminal.tools.search_docs({"query": "immutable originals", "limit": 5})
    package = docs["packages"][0]

    raw_bytes = sum(len(_load(name).encode("utf-8")) for name in ("document.md", "schema.sql", "query-output.txt", "logs.txt", "service.py"))
    raw_tokens = sum(_estimate for _estimate in (len(_load(name)) // 4 for name in ("document.md", "schema.sql", "query-output.txt", "logs.txt", "service.py")))

    assert package["estimated_bytes"] < raw_bytes
    assert package["estimated_tokens"] < raw_tokens


def test_contencioso_judicial_is_shared_factory_profile_with_confirmed_sicjut_ownership():
    factory = _factory()
    agent = factory.create("SICJUTPF".lower())
    profile = agent.profile

    assert isinstance(agent, ContenciosoJudicialDomainAgent)
    assert agent.gateway is factory.gateway
    assert agent.knowledge is factory.knowledge
    assert set(("SICJUT", "SICJUTPF", "SICJUTINDBAT")) == set(profile.owned_systems)
    assert set(("CJTCAADWS", "ISICJUTWS", "WSAFTAF", "WSCEXECF")) == set(profile.owned_components)
    assert set(CONTENCIOSO_JUDICIAL_ALIASES).issubset(profile.aliases)
    assert profile.delegated_domains == CONTENCIOSO_JUDICIAL_DELEGATED_DOMAINS


def test_contencioso_judicial_exact_and_symbol_search_is_acl_and_domain_scoped():
    factory = _factory()
    factory.knowledge.ingest(
        source_id="sicjut-code",
        source_uri="sicjut/service.py",
        content="class SICJUTCase:\n    def persist_process(self):\n        return 'judicial'\n",
        acl_scope="internal",
        domain="contencioso_judicial",
        system="SICJUT",
        component_id="CJTCAADWS",
        environment="test",
        source_type="code",
        parser_version="1",
    )
    agent = factory.create("contencioso_judicial")

    exact = agent.tools.search_code_exact({"query": "persist_process"})
    symbols = agent.tools.search_code_symbols({"class_name": "SICJUTCase"})

    assert exact["packages"][0]["domain_id"] == "contencioso_judicial"
    assert symbols["packages"][0]["evidence_refs"][0]["metadata"]["match_kind"] == "symbol"
    assert exact["filters"]["principal_acl_scopes"] == {"internal"}
    with pytest.raises(PermissionError, match="cannot widen"):
        agent.tools.search_code_exact({"query": "persist_process", "principal_acl_scopes": ["public"]})
    with pytest.raises(PermissionError, match="delegated interface"):
        agent.tools.search_code_exact({"query": "persist_process", "domain_ids": ["encargos_sigef"]})


class _CrossDomainCoordinator:
    def search(self, payload):
        assert payload["delegated_domain_ids"] == ["encargos_sigef"]
        return {"evidence_refs": [{"source_id": "sigef-case", "locator": "case:17"}]}


def test_contencioso_judicial_delegation_is_explicit_and_structured():
    factory = DomainAgentFactory(
        gateway=ModelGateway.from_env({}),
        knowledge=_platform(),
        diagnostic_broker=_CrossDomainCoordinator(),
        runtime_factory=lambda model_adapter, tools: _FakeRuntime(model_adapter, tools),
    )
    agent = factory.create("sicjut")

    result = agent.tools.search_delegated_domain_evidence(
        {"query": "return code", "delegated_domain_ids": ["encargos_sigef"]}
    )

    assert result["cross_domain_required"] is True
    assert result["delegation"]["source_domain_id"] == "contencioso_judicial"
    assert result["suspected_edge"]["relation_type"] == "delegates-to"
    assert result["suspected_edge"]["evidence_refs"] == result["evidence_refs"]
    with pytest.raises(PermissionError, match="Administrative and SIGEF"):
        agent.tools.search_delegated_domain_evidence(
            {"query": "x", "delegated_domain_ids": ["criminal"]}
        )
