"""Deterministic final acceptance for GitHub Issue #11.

This is deliberately a composition test.  It consumes the #24 DomainRouter and
embedding-space contracts, then drives the same bounded path an application
request uses.  All evidence, channel payloads, and SQLcl output are synthetic.
The generated evidence manifest is written below pytest's temporary directory;
it contains hashes and identifiers only, never credentials or private endpoints.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from opennicf import (
    ChannelAllowList,
    ChannelAttachment,
    ChannelDelivery,
    ChannelEventLedger,
    ChannelResponse,
    DiagnosticBrokerStore,
    DiagnosticPolicyError,
    DiagnosticTemplate,
    DiagnosticTemplateRegistry,
    DiagnosticWorker,
    DomainAgentFactory,
    DomainRouter,
    FailureAuditEngine,
    HashingEmbeddingBackend,
    IngestionService,
    IntegrationCorrelationAgent,
    KnowledgePlatform,
    MemoryConversationTransport,
    MockSqlclFixtureRunner,
    ModelGateway,
    ModelRouter,
    PrivacyPolicy,
    QueryOutputProvenanceHook,
    SecureDiagnosticBroker,
    SqlclFixture,
    TelegramConversationAdapter,
    WebexConversationAdapter,
)
from opennicf.knowledge import EmbeddingSpaceMismatch, RetrievalFilters
from opennicf.ops.deployment import DeploymentController


class _ReportChannelHandler:
    def __init__(self, report: dict):
        self.report = report
        self.envelopes = []

    def handle(self, envelope):
        self.envelopes.append(envelope)
        artifact = ChannelAttachment.from_json("final-report.json", self.report)
        return ChannelResponse(
            intent="audit",
            status="completed",
            deliveries=(
                ChannelDelivery(
                    kind="message",
                    text="Synthetic audit completed.",
                    reply_target=envelope.event.reply_target,
                    thread_id=envelope.event.thread_id,
                ),
                ChannelDelivery(
                    kind="artifact",
                    text="Final report attached.",
                    reply_target=envelope.event.reply_target,
                    thread_id=envelope.event.thread_id,
                    attachments=(artifact,),
                ),
            ),
            metadata={"audit_id": self.report["audit_id"]},
        )


def _acceptance_stack():
    platform = KnowledgePlatform.in_memory()
    gateway = ModelGateway()
    factory = DomainAgentFactory(gateway=gateway, knowledge=platform)
    coordinator = IntegrationCorrelationAgent(
        gateway=gateway,
        knowledge=platform,
        domain_factory=factory,
        max_domain_fan_out=4,
        max_evidence_budget=8,
        privacy=PrivacyPolicy.LOCAL_ONLY,
    )
    router = DomainRouter(
        gateway=gateway,
        knowledge=platform,
        domain_factory=factory,
        integration_coordinator=coordinator,
        privacy=PrivacyPolicy.LOCAL_ONLY,
    )
    return platform, gateway, coordinator, router


def _broker(tmp_path: Path) -> SecureDiagnosticBroker:
    templates = DiagnosticTemplateRegistry(
        [DiagnosticTemplate(
            template_id="missing-ledger-fact",
            sql="select session_id, status from v$session where username = {{username}}",
            allowed_parameters=("username",),
        )]
    )
    return SecureDiagnosticBroker(
        DiagnosticBrokerStore(str(tmp_path / "broker.sqlite3")),
        signing_key=b"synthetic-issue-11-signing-key",
        templates=templates,
        lease_seconds=600,
        max_rows=20,
        max_runtime_seconds=5,
    )


def _ingest_corpus(platform: KnowledgePlatform) -> tuple[IngestionService, list[str]]:
    service = IngestionService(platform)
    corpus = [
        ("incident.log", "log", "2026-08-10T03:00:00Z ERROR payment finalization failed trace-id=run-11 File \"payment_service.py\", line 2\n"),
        ("splunk.csv", "splunk-csv", "timestamp,trace_id,status\n2026-08-10T03:00:00Z,run-11,failed\n"),
        ("payment_service.py", "code", "def finalize(order):\n    return ledger.commit(order)\n"),
        ("design.md", "document", "The ledger commit is verified by a read-only database check.\n"),
        ("schema.sql", "schema", "create table ledger_entry (entry_id integer, status varchar(16));\n"),
        ("query-output.txt", "query-output", "entry_id | status\n42 | pending\n"),
    ]
    for name, source_type, content in corpus:
        service.submit_manual(
            source_id=f"run-11-{name}",
            source_uri=f"synthetic://issue-11/{name}",
            content=content,
            source_type=source_type,
            source_kind=source_type,
            domain_id="criminal",
            system_id="SINQUER",
            component_id="payment-api",
            evidence_type=source_type,
            environment="acceptance",
            acl_scope="internal",
            metadata={"correlation_ids": ["run-11"], "test_run_id": "issue-11"},
        )
    # Duplicate delivery is intentionally submitted; the queue and platform
    # must retain one logical source/chunk lineage.
    service.submit_manual(
        source_id="run-11-incident.log",
        source_uri="synthetic://issue-11/incident.log",
        content=corpus[0][2],
        source_type="log",
        source_kind="log",
        domain_id="criminal",
        system_id="SINQUER",
        environment="acceptance",
        acl_scope="internal",
    )
    service.run()
    return service, [name for name, _, _ in corpus]


def test_issue11_synthetic_mixed_evidence_round_trip(tmp_path):
    platform, gateway, coordinator, router = _acceptance_stack()
    ingestion, corpus_names = _ingest_corpus(platform)

    routed = router.route({
        "question": "correlate run-11 payment failure across SINQUER and SICJUT",
        "domain_ids": ["criminal", "contencioso_judicial"],
        "correlation_ids": ["run-11"],
        "privacy": "local_only",
        "evidence_budget": 6,
    })
    assert routed["route_kind"] == "integration_correlation"
    assert routed["result"]["embedding_space_id"] == platform.embeddings.active_space_id
    assert routed["result"]["selected_domain_ids"] == ["criminal", "contencioso_judicial"]
    assert all(ref["provenance_ref"] for d in routed["result"]["delegations"] for ref in d["evidence_refs"])
    assert all(item["domain_id"] in {"criminal", "contencioso_judicial"} for item in routed["result"]["delegations"])

    # #24's strict vector-space invariant is part of the final acceptance.
    with pytest.raises(EmbeddingSpaceMismatch):
        platform.search(
            "run-11",
            filters=RetrievalFilters(
                principal_acl_scopes=frozenset({"internal"}),
                embedding_space_id="gemini-embedding-001:768:v1",
            ),
        )
    assert len({chunk.source_id for chunk in platform.store.chunks.values()}) == len(corpus_names)
    assert all(chunk.domain_id == "criminal" for chunk in platform.store.chunks.values())

    broker = _broker(tmp_path)

    class AuditBroker:
        name = broker.name

        def request(self, payload):
            return broker.request({
                **payload,
                "requester": "auditor-alice",
                "target_logical_system": "ledger",
                "policy_classification": "restricted-read",
                "authentication_material_ref": "ref:vault/sqlcl/reader",
                "database_profile_alias": "synthetic-ledger",
                "template_id": "missing-ledger-fact",
                "parameters": {"username": "APP"},
            })

    engine = FailureAuditEngine(platform, broker=AuditBroker())
    missing_report = engine.analyze({
        "query": "run-11 missing live database fact",
        "scope": "synthetic ledger incident",
        "systems": ["SINQUER"],
        "source_types": ["live-database"],
        "requested_outputs": ["findings", "verification"],
        "require_live_verification": True,
    })
    assert any(f.classification == "missing_evidence" for f in missing_report.findings)

    report = engine.analyze({
        "query": "run-11 payment_service.py payment finalization failed invoice_rows missing",
        "scope": "synthetic ledger incident",
        "systems": ["SINQUER"],
        "components": ["payment-api"],
        "symptoms": ["payment finalization failed", "invoice_rows missing"],
        "correlation_ids": ["run-11"],
        "source_types": ["log", "splunk-csv", "code", "document", "schema", "query-output"],
        "requested_outputs": ["findings", "json", "human report", "verification"],
        "require_live_verification": True,
    })
    classes = {finding.classification for finding in report.findings}
    assert {"fact", "inference"} <= classes
    assert "hypothesis" in {finding["classification"] for finding in routed["result"]["findings"]}
    assert missing_report.verification_requests[0].status == "pending"
    assert report.human_report and report.machine_report["audit_id"] == report.audit_id
    report_sources = {ref.source_id for f in report.findings for ref in f.evidence_refs}
    assert "run-11-incident.log" in report_sources
    assert {chunk.source_id for chunk in platform.store.chunks.values()} >= {f"run-11-{name}" for name in corpus_names}

    # The worker is pull-only and resumable: no worker means pending, then a
    # later poll executes and provenance-ingests the diagnostic output.
    runner = MockSqlclFixtureRunner({"missing-ledger-fact": SqlclFixture("session_id,status\n7,OPEN\n")})
    worker = DiagnosticWorker(broker, runner, tmp_path / "spool", worker_id="onprem-a", provenance_hook=QueryOutputProvenanceHook(platform))
    assert missing_report.verification_requests[0].status == "pending"
    result = worker.run_once()
    assert result and result.exit_code == 0 and result.result_hash
    assert any(chunk.source_type == "query-output" and chunk.source_id.startswith("diagnostic-") for chunk in platform.store.chunks.values())

    evidence = {
        "test_run_id": "issue-11-generated",
        "audit_id": report.audit_id,
        "embedding_space_id": platform.embeddings.active_space_id,
        "route": routed["route_kind"],
        "source_count": len(corpus_names),
        "finding_classes": sorted(classes),
        "broker_job_hash": missing_report.verification_requests[0].metadata["job_id"],
        "machine_report_sha256": sha256(report.to_json().encode()).hexdigest(),
        "channel_contracts": ["webex", "telegram"],
    }
    evidence_path = tmp_path / "issue-11-generated" / "acceptance.json"
    evidence_path.parent.mkdir()
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    assert json.loads(evidence_path.read_text(encoding="utf-8"))["test_run_id"].startswith("issue-11-")

    # Routing is policy-only and never conflates completion routing with
    # embedding routing: local-safe work is local; synthesis is remote-sized.
    model_router = ModelRouter()
    assert model_router.choose(task_class="classification", privacy=PrivacyPolicy.LOCAL_PREFERRED, local_healthy=True).provider == "lmstudio"
    assert model_router.choose(task_class="tool_planning", privacy=PrivacyPolicy.LOCAL_PREFERRED, local_healthy=True, complex_task=True).provider == "litellm_oci"
    assert model_router.choose(task_class="classification", privacy=PrivacyPolicy.LOCAL_PREFERRED, local_healthy=False).provider == "litellm_oci"
    with pytest.raises(RuntimeError):
        model_router.choose(task_class="classification", privacy=PrivacyPolicy.LOCAL_ONLY, local_healthy=False)
    assert gateway.audit_events == []


def test_issue11_channels_failures_and_deployment_safety(tmp_path):
    platform = KnowledgePlatform.in_memory()
    report = {"audit_id": "audit-11", "status": "complete", "evidence": "synthetic"}
    handler = _ReportChannelHandler(report)
    webex_transport = MemoryConversationTransport("webex")
    webex = WebexConversationAdapter(
        webex_transport,
        IngestionService(platform),
        handler,
        allow_list=ChannelAllowList(actor_ids=frozenset({"alice"}), conversation_ids=frozenset({"room-11"})),
        ledger=ChannelEventLedger(),
        sleep=lambda _: None,
    )
    event = {
        "id": "event-11",
        "roomId": "room-11",
        "personId": "alice",
        "text": "/audit run-11",
        "parentId": "thread-11",
        "attachments": [{"id": "attachment-11", "name": "report.txt", "contentType": "text/plain", "content": b"synthetic attachment"}],
    }
    first = webex.handle_webhook(event)
    duplicate = webex.handle_webhook(event)
    assert first.status == "completed" and duplicate.status == "duplicate"
    assert handler.envelopes[0].event.conversation.conversation_id == "room-11"
    assert any(a.filename == "final-report.json" for _, d in webex_transport.sent_deliveries for a in d.attachments)
    unauthorized = dict(event, id="event-unauthorized", personId="mallory")
    assert webex.handle_webhook(unauthorized).status == "rejected"

    telegram_transport = MemoryConversationTransport("telegram")
    telegram = TelegramConversationAdapter(
        telegram_transport,
        IngestionService(platform),
        handler,
        allow_list=ChannelAllowList(actor_ids=frozenset({"alice"}), conversation_ids=frozenset({"chat-11"})),
        ledger=ChannelEventLedger(),
        sleep=lambda _: None,
    )
    telegram_response = telegram.handle_webhook({"update_id": 11, "message": {"message_id": 11, "chat": {"id": "chat-11"}, "from": {"id": "alice"}, "text": "/audit run-11"}})
    assert telegram_response.status == "completed"
    assert len(handler.envelopes) == 2
    assert len(platform.store.sources) == 3  # Webex message+attachment and Telegram message; duplicate is suppressed

    broker = _broker(tmp_path)
    job = broker.create_job({
        "requester": "auditor-alice", "target_logical_system": "ledger", "policy_classification": "restricted-read",
        "authentication_material_ref": "ref:vault/sqlcl/reader", "database_profile_alias": "synthetic-ledger",
        "operation_class": "read_select", "sql_text": "select status from ledger_entry",
    })
    with pytest.raises(DiagnosticPolicyError):
        broker.verify_job(replace(job, sql_text="delete from ledger_entry"))
    deployed, rolled_back = [], []
    controller = DeploymentController(known_good="release-good", deploy=deployed.append, rollback=rolled_back.append)
    result = controller.deploy_release("release-bad", migration_gate=lambda: True, health_gate=lambda: False, smoke_gate=lambda: True)
    assert result.state == "rolled_back" and rolled_back == ["release-good"] and deployed == []
