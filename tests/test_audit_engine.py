from __future__ import annotations

from pathlib import Path

from opennicf import (
    EvidenceIndex,
    FailureAuditEngine,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    OpenNICFTools,
    RetrievalFilters,
)
from opennicf.audit import AuditRequest, SplunkCSVAnalyzer


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "audit"


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _platform() -> KnowledgePlatform:
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=64)),
    )
    platform.ingest(
        source_id="incident-log",
        source_uri="tests/fixtures/audit/incident.log",
        content=_load("incident.log"),
        acl_scope="internal",
        domain="audit",
        system="billing-api",
        environment="prod",
        source_type="log",
        parser_version="1",
    )
    platform.ingest(
        source_id="payment-code",
        source_uri="tests/fixtures/audit/payment_service.py",
        content=_load("payment_service.py"),
        acl_scope="internal",
        domain="audit",
        system="billing-api",
        environment="prod",
        source_type="code",
        parser_version="1",
    )
    platform.ingest(
        source_id="invoice-schema",
        source_uri="tests/fixtures/audit/schema.sql",
        content=_load("schema.sql"),
        acl_scope="internal",
        domain="audit",
        system="ledger-db",
        environment="prod",
        source_type="schema",
        parser_version="1",
    )
    platform.ingest(
        source_id="incident-doc",
        source_uri="tests/fixtures/audit/design.md",
        content=_load("design.md"),
        acl_scope="internal",
        domain="audit",
        system="billing-api",
        environment="prod",
        source_type="document",
        parser_version="1",
    )
    platform.ingest(
        source_id="query-output",
        source_uri="tests/fixtures/audit/query-output.txt",
        content=_load("query-output.txt"),
        acl_scope="internal",
        domain="audit",
        system="ledger-db",
        environment="prod",
        source_type="query-output",
        parser_version="1",
    )
    platform.ingest(
        source_id="splunk-export",
        source_uri="tests/fixtures/audit/splunk-export.csv",
        content=_load("splunk-export.csv"),
        acl_scope="internal",
        domain="audit",
        system="billing-api",
        environment="prod",
        source_type="splunk-csv",
        parser_version="1",
    )
    return platform


def test_failure_audit_engine_builds_mixed_evidence_report_with_broker_request():
    platform = _platform()
    engine = FailureAuditEngine(platform)

    report = engine.analyze(
        {
            "scope": "billing failure",
            "systems": ["billing-api", "ledger-db"],
            "components": ["payment_service.py", "invoice_runs"],
            "symptoms": ["payment finalization failed", "invoice_rows missing"],
            "requested_outputs": ["timeline", "findings", "json", "human report"],
            "correlation_ids": ["trace-91"],
            "source_types": ["log", "code", "schema", "document", "query-output"],
            "require_live_verification": True,
        }
    )

    classifications = {finding.classification for finding in report.findings}
    assert {"fact", "hypothesis", "missing_evidence", "recommendation"} <= classifications
    assert any(finding.classification == "fact" and any(ref.source_id == "incident-log" for ref in finding.evidence_refs) for finding in report.findings)
    assert any("source code evidence matches" in finding.statement.lower() for finding in report.findings)
    assert any(finding.classification == "missing_evidence" and finding.verification_status == "needs_live_verification" for finding in report.findings)
    assert report.verification_requests
    assert report.verification_requests[0].status == "pending"
    assert report.verification_requests[0].metadata["job_id"] == "broker-generated"
    assert any(point.correlation_ids for point in report.timeline)
    assert any("trace-91" in point.correlation_ids for point in report.timeline)

    machine = report.to_dict()
    finding_ids = {finding["finding_id"] for finding in machine["findings"]}
    assert finding_ids == {finding.finding_id for finding in report.findings}
    human = report.to_markdown()
    assert all(finding_id in human for finding_id in finding_ids)


def test_tools_surface_audit_analysis_and_preserve_brokered_verification():
    platform = _platform()
    tools = OpenNICFTools(EvidenceIndex(platform))

    result = tools.analyze_failure_audit(
        {
            "query": "trace-91",
            "scope": "splunk export check",
            "systems": ["billing-api"],
            "source_types": ["splunk-csv"],
            "requested_outputs": ["findings", "json"],
        }
    )

    assert any(finding["classification"] == "fact" for finding in result["findings"])
    assert any(tuple(finding["source_type_analyzers"]) == ("splunk-csv",) for finding in result["findings"])

    brokered = tools.request_controlled_verification({"operation_class": "read_select", "target": "database"})
    assert brokered["status"] == "pending"
    assert brokered["job_id"] == "broker-generated"


def test_splunk_csv_analyzer_is_source_type_specific():
    platform = _platform()
    hits = platform.search(
        "trace-91",
        filters=RetrievalFilters(principal_acl_scopes=frozenset({"internal"}), source_types=("splunk-csv",), limit=5),
        route="local",
    )
    observations = SplunkCSVAnalyzer().analyze(AuditRequest.from_input({"query": "trace-91", "source_types": ["splunk-csv"]}), hits)

    assert observations
    assert observations[0].classification == "fact"
    assert observations[0].source_type_analyzers == ("splunk-csv",)
