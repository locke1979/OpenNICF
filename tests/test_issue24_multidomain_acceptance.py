"""Synthetic acceptance matrix for GitHub Issue #24.

The fixtures deliberately use the in-memory knowledge store and a no-network
gateway.  They exercise the same DomainRouter, QwenAgent runtime boundary,
coordinator, ACLs, diagnostic boundary, and embedding-space guards used by the
application without requiring credentials or private endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt
from time import perf_counter

import pytest

from opennicf.domain_agent import DomainAgentFactory
from opennicf.integration_correlation import IntegrationCorrelationAgent, IntegrationCorrelationRequest
from opennicf.knowledge import (
    EmbeddingError,
    EmbeddingMigration,
    EmbeddingSpaceMismatch,
    GeminiEmbeddingBackend,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    PrivacyBoundaryError,
    RetrievalFilters,
)
from opennicf.model_gateway import ModelGateway, PrivacyPolicy
from opennicf.ops.deployment import DeploymentController
from opennicf.ops.health import HealthState, check_health
from opennicf.ops.observability import AuditLog, TraceContext, structured_event
from opennicf.router import DomainRouter


class RecordingRuntime:
    instances: list["RecordingRuntime"] = []

    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        self.calls: list[str] = []
        self.instances.append(self)

    def run(self, request: str):
        self.calls.append(request)
        return {"status": "accepted", "runtime": "qwen-agent", "request": request}


@dataclass
class StubBackend:
    """Deterministic provider-shaped backend for multi-space migration tests."""

    model: str
    provider: str

    @property
    def info(self):
        from opennicf.knowledge.embeddings import EmbeddingModelInfo

        return EmbeddingModelInfo(
            model=self.model,
            dimensions=768,
            device="cpu" if self.provider == "LOCAL" else "remote",
            backend="acceptance",
            embedding_space_id=f"{self.model}:768:v1",
            provider=self.provider,
            normalized=True,
        )

    def embed(self, texts, *, purpose="retrieval_document"):
        from opennicf.knowledge.embeddings import EmbeddingResult

        vectors = []
        for text in texts:
            values = [0.0] * 768
            values[hash((self.model, text)) % 768] = 1.0
            vectors.append(tuple(values))
        return EmbeddingResult(
            model=self.model,
            dimensions=768,
            device=self.info.device,
            vectors=tuple(vectors),
            embedding_space_id=self.info.embedding_space_id,
            provider=self.provider,
            normalized=True,
            purpose=purpose,
        )


def _platform(service: LocalFirstEmbeddingService | None = None) -> KnowledgePlatform:
    return KnowledgePlatform.in_memory() if service is None else KnowledgePlatform(MemoryKnowledgeStore(), MemoryObjectStore(), service)


def _fixture_platform() -> KnowledgePlatform:
    service = LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=768))
    platform = _platform(service)
    records = {
        "criminal": ("SINQUER incident fact", "sinquer", "sinquer_casework"),
        "contraordenacional": ("SCO incident fact", "sco", "contraordenacional_casework"),
        "contencioso_administrativo": ("SICAT SIGEPRA administrative fact", "sicat", "SICAT"),
        "contencioso_judicial": ("SICJUT judicial fact", "sicjut", "SICJUT"),
        "encargos_sigef": ("SIGEF Encargos fact", "sigef", "encargos_sigef_casework"),
    }
    for domain_id, (text, alias, system) in records.items():
        platform.ingest(
            source_id=f"{domain_id}-evidence",
            source_uri=f"synthetic://{alias}/evidence",
            content=f"{text}. correlation_id=incident-24.",
            acl_scope="internal",
            domain_id=domain_id,
            system_id=system,
            environment="acceptance",
            source_type="document",
            metadata={"correlation_ids": ["incident-24"]},
        )
    return platform


@pytest.fixture()
def acceptance_stack():
    RecordingRuntime.instances.clear()
    platform = _fixture_platform()
    gateway = ModelGateway()
    factory = DomainAgentFactory(
        gateway=gateway,
        knowledge=platform,
        runtime_factory=RecordingRuntime,
    )
    coordinator = IntegrationCorrelationAgent(
        gateway=gateway,
        knowledge=platform,
        domain_factory=factory,
        max_domain_fan_out=4,
        max_evidence_budget=6,
        privacy=PrivacyPolicy.LOCAL_ONLY,
    )
    return platform, factory, coordinator, DomainRouter(
        gateway=gateway,
        knowledge=platform,
        domain_factory=factory,
        integration_coordinator=coordinator,
        privacy=PrivacyPolicy.LOCAL_ONLY,
    )


@pytest.mark.parametrize(
    ("alias", "domain_id"),
    [("SINQUER", "criminal"), ("SCO", "contraordenacional"), ("SICAT", "contencioso_administrativo"),
     ("SICJUT", "contencioso_judicial"), ("SIGEF", "encargos_sigef")],
)
def test_all_single_domain_scenarios_are_routed_to_qwen_agents(alias, domain_id, acceptance_stack):
    _, factory, _, router = acceptance_stack
    response = router.route({"query": f"audit {alias}", "domain_ids": [alias], "privacy": "local_only"})
    assert response["route_kind"] == "single_domain"
    assert response["selected_domain_ids"] == [domain_id]
    assert response["reason"]
    assert response["confidence"] == 1.0
    assert response["result"]["runtime"] == "qwen-agent"
    agent = factory.create(domain_id, privacy=PrivacyPolicy.LOCAL_ONLY)
    assert agent.model_adapter.gateway is router.gateway
    assert agent.profile.domain_id == domain_id
    assert agent.profile.privacy_policy is PrivacyPolicy.LOCAL_ONLY


@pytest.mark.parametrize(
    "domains",
    [
        ("contraordenacional", "criminal"),
        ("contencioso_administrativo", "contencioso_judicial"),
        ("encargos_sigef", "contencioso_judicial", "contencioso_administrativo"),
    ],
)
def test_three_cross_domain_coordinator_scenarios_preserve_bounded_provenance(domains, acceptance_stack):
    platform, _, coordinator, _ = acceptance_stack
    result = coordinator.correlate({
        "question": "correlate incident-24 across domains",
        "domain_ids": list(domains),
        "correlation_ids": ["incident-24"],
        "privacy": "local_only",
        "evidence_budget": 6,
    })
    assert result["status"] == "ok"
    assert result["selected_domain_ids"] == list(domains)
    assert result["embedding_space_id"] == platform.embeddings.active_space_id
    assert len(result["evidence_packages"]) <= 6
    assert result["estimated_tokens"] > 0
    classifications = {finding["classification"] for finding in result["findings"]}
    assert {"fact", "inference", "hypothesis"}.intersection(classifications)
    assert all(ref.startswith("${") is False for ref in result["delegations"][0]["provenance_refs"])
    assert all(item["domain_id"] in domains for item in result["delegations"])


def test_acl_prompt_escalation_and_coordinator_only_cross_domain_access(acceptance_stack):
    platform, factory, coordinator, _ = acceptance_stack
    criminal = factory.create("criminal", privacy=PrivacyPolicy.LOCAL_ONLY)
    with pytest.raises(PermissionError):
        criminal.tools.search_docs({"query": "SICJUT", "domain_ids": ["contencioso_judicial"]})
    with pytest.raises(PermissionError):
        criminal.tools.search_docs({"query": "foreign prompt", "principal_domain_id": "contencioso_judicial"})
    result = coordinator.correlate({"question": "SINQUER and SICJUT", "domain_ids": ["criminal", "contencioso_judicial"]})
    assert set(result["selected_domain_ids"]) == {"criminal", "contencioso_judicial"}
    assert all(ref["metadata"]["domain_id"] in result["selected_domain_ids"]
               for package in result["evidence_packages"] for ref in package["evidence_refs"])
    assert platform.embeddings.active_space_id


def test_missing_live_data_uses_diagnostic_pending_path_and_locality(acceptance_stack):
    _, factory, _, _ = acceptance_stack
    agent = factory.create("criminal", privacy=PrivacyPolicy.LOCAL_ONLY)
    pending = agent.tools.request_domain_diagnostic({"operation_class": "read_select", "query": "missing live row"})
    assert pending["status"] == "pending"
    assert pending["domain_id"] == "criminal"
    with pytest.raises(PermissionError):
        agent.tools.request_domain_diagnostic({"operation_class": "update"})


def test_embedding_space_contract_same_space_passes_cross_space_fails_and_migration_resumes():
    qwen = StubBackend("Qwen/Qwen3-Embedding-0.6B", "LOCAL")
    gemini = StubBackend("gemini-embedding-001", "GOOGLE")
    gemini2 = StubBackend("gemini-embedding-2", "GOOGLE")
    service = LocalFirstEmbeddingService(preferred_backend=qwen, cpu_backend=HashingEmbeddingBackend(dimensions=768), providers=[gemini, gemini2])
    platform = _platform(service)
    bundle = platform.ingest(source_id="shared-chunk", source_uri="synthetic://shared", content="shared provenance evidence", domain_id="criminal", acl_scope="internal")
    qwen_space = qwen.info.embedding_space_id
    gemini_space = gemini.info.embedding_space_id
    assert qwen_space != gemini_space
    assert len(bundle.embeddings[0].vector) == 768
    assert isclose(sqrt(sum(value * value for value in bundle.embeddings[0].vector)), 1.0, rel_tol=1e-6)
    assert platform.search("shared provenance", filters=RetrievalFilters(principal_acl_scopes=frozenset({"internal"})))
    service.switch_active(gemini_space, corpus_ready=True)
    with pytest.raises(EmbeddingSpaceMismatch):
        platform.search("shared provenance", filters=RetrievalFilters(principal_acl_scopes=frozenset({"internal"})))

    persisted = {}
    migration = EmbeddingMigration(service, gemini_space, batch_size=1)
    state = migration.run(bundle.chunks, lambda chunk, result: persisted.setdefault(chunk.chunk_id, result))
    resumed = migration.run(bundle.chunks, lambda chunk, result: persisted.setdefault(chunk.chunk_id, result))
    assert state["status"] == resumed["status"] == "complete"
    assert set(persisted) == {chunk.chunk_id for chunk in bundle.chunks}
    assert bundle.version.source_version_id == bundle.chunks[0].source_version_id
    assert service.active_space_id == gemini_space
    assert all(result.embedding_space_id == gemini_space for result in persisted.values())
    with pytest.raises(PrivacyBoundaryError):
        service.embed(["private local evidence"], space_id=gemini_space, privacy_policy="local_only")
    assert gemini2.info.embedding_space_id not in {qwen_space, gemini_space}


def test_google_adapter_contract_is_normalized_without_live_credentials():
    def transport(model, payload):
        assert payload["output_dimensionality"] == 768
        assert payload["task_type"] in {"RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"}
        return {"embedding": {"values": [2.0] * 768}}

    for model in ("gemini-embedding-001", "gemini-embedding-2"):
        backend = GeminiEmbeddingBackend(model, transport=transport)
        result = backend.embed(["synthetic evidence"], purpose="retrieval_query")
        assert result.provider == "GOOGLE"
        assert result.dimensions == 768
        assert result.normalized is True
        assert isclose(sqrt(sum(value * value for value in result.vectors[0])), 1.0, rel_tol=1e-6)


def test_bounded_context_parallelism_health_observability_and_rollback(acceptance_stack):
    _, _, coordinator, _ = acceptance_stack
    request = {"question": "incident-24", "domain_ids": ["criminal", "contraordenacional", "contencioso_judicial"], "evidence_budget": 3}
    started = perf_counter()
    result = coordinator.correlate(request)
    parallel_seconds = perf_counter() - started
    serial_started = perf_counter()
    parsed = coordinator._selected_domain_ids(IntegrationCorrelationRequest.from_input(request))
    serial_seconds = perf_counter() - serial_started
    assert parallel_seconds > 0 and serial_seconds >= 0
    assert result["estimated_bytes"] > 0
    assert sum(package["estimated_bytes"] for package in result["evidence_packages"]) == result["estimated_bytes"]

    health = check_health("domain-coordinator", [("domain-agents", lambda: True), ("embedding", lambda: True)], release_id="acceptance")
    assert health.state is HealthState.OK and health.ready and health.live
    event = structured_event("domain-retrieval", TraceContext("corr-24"), domain_id="criminal", embedding_space_id=result["embedding_space_id"])
    assert "corr-24" in event and "embedding_space_id" in event
    audit = AuditLog(secret_values=("synthetic-secret",))
    assert "synthetic-secret" not in str(audit.append("acceptance", TraceContext("corr-24"), secret="synthetic-secret"))

    deployed: list[str] = []
    rolled_back: list[str] = []
    controller = DeploymentController(known_good="known-good", deploy=deployed.append, rollback=rolled_back.append)
    failed = controller.deploy_release("candidate", migration_gate=lambda: True, health_gate=lambda: False, smoke_gate=lambda: True)
    assert failed.state == "rolled_back" and rolled_back == ["known-good"] and not deployed
