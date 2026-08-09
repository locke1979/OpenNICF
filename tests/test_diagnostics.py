from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from opennicf import (
    DiagnosticBrokerStore,
    DiagnosticJob,
    DiagnosticLeaseError,
    DiagnosticPolicyError,
    DiagnosticReplayError,
    DiagnosticTemplate,
    DiagnosticTemplateRegistry,
    DiagnosticWorker,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    MockSqlclFixtureRunner,
    QueryOutputProvenanceHook,
    SecureDiagnosticBroker,
    SqlclFixture,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "diagnostics"


def _templates() -> DiagnosticTemplateRegistry:
    return DiagnosticTemplateRegistry(
        [
            DiagnosticTemplate(
                template_id="select-sessions",
                sql="select session_id, username, status from v$session where username = {{username}}",
                allowed_parameters=("username",),
            ),
            DiagnosticTemplate(
                template_id="describe-users",
                sql="describe all_users",
                expected_output_format="text",
            ),
        ]
    )


def _broker(tmp_path: Path) -> SecureDiagnosticBroker:
    return SecureDiagnosticBroker(
        DiagnosticBrokerStore(str(tmp_path / "diagnostic-broker.sqlite3")),
        signing_key=b"test-signing-key",
        templates=_templates(),
        lease_seconds=600,
        max_rows=2,
        max_runtime_seconds=5.0,
        max_output_bytes=512,
    )


def _select_fixture() -> SqlclFixture:
    return SqlclFixture(
        stdout=(FIXTURE_DIR / "select_sessions.csv").read_text(encoding="utf-8"),
        runtime_seconds=0.2,
        output_format="csv",
    )


def test_broker_signs_template_jobs_and_rejects_tampering(tmp_path):
    broker = _broker(tmp_path)
    job = broker.create_job(
        {
            "audit_id": "audit-1",
            "finding_id": "finding-1",
            "database_profile_alias": "prod-ledger",
            "operation_class": "read_select",
            "template_id": "select-sessions",
            "parameters": {"username": "APP"},
            "max_rows": 10,
            "max_runtime_seconds": 3,
            "max_output_bytes": 256,
            "metadata": {"requested_by": "audit-engine"},
        }
    )

    assert job.signature
    assert job.payload_hash
    assert "select session_id" in job.sql_text.lower()
    broker.verify_job(job)

    tampered = replace(job, sql_text="select * from dual")
    with pytest.raises(DiagnosticPolicyError):
        broker.verify_job(tampered)


def test_broker_rejects_dml_and_arbitrary_powershell(tmp_path):
    broker = _broker(tmp_path)

    with pytest.raises(DiagnosticPolicyError):
        broker.create_job(
            {
                "database_profile_alias": "prod-ledger",
                "operation_class": "read_select",
                "sql_text": "drop table users",
                "max_rows": 10,
            }
        )

    with pytest.raises(DiagnosticPolicyError):
        broker.create_job(
            {
                "database_profile_alias": "prod-ledger",
                "operation_class": "read_select",
                "sql_text": "select * from dual",
                "wrapper_name": "Invoke-Expression",
            }
        )


def test_leases_are_single_use_and_expired_jobs_are_rejected(tmp_path):
    broker = _broker(tmp_path)
    job = broker.create_job(
        {
            "database_profile_alias": "prod-ledger",
            "operation_class": "describe",
            "template_id": "describe-users",
            "expires_in_seconds": 300,
        }
    )
    lease = broker.poll("worker-a")
    assert lease is not None
    runner = MockSqlclFixtureRunner({"describe-users": SqlclFixture(stdout=(FIXTURE_DIR / "describe_users.txt").read_text(encoding="utf-8"), output_format="text")})
    result = runner.execute(lease)
    broker.complete(lease, result)

    with pytest.raises(DiagnosticReplayError):
        broker.complete(lease, result)

    expired = broker.create_job(
        {
            "database_profile_alias": "prod-ledger",
            "operation_class": "read_select",
            "template_id": "select-sessions",
            "parameters": {"username": "APP"},
            "created_at": datetime.now(UTC) - timedelta(days=1),
            "expires_in_seconds": 30,
        }
    )
    assert expired.job_id
    assert broker.poll("worker-b") is None
    assert broker.store.get_job_row(expired.job_id)["status"] == "expired"


def test_worker_resume_from_offline_spool_without_duplicate_execution(tmp_path):
    broker = _broker(tmp_path)
    job = broker.create_job(
        {
            "audit_id": "audit-9",
            "finding_id": "finding-9",
            "database_profile_alias": "prod-ledger",
            "operation_class": "read_select",
            "template_id": "select-sessions",
            "parameters": {"username": "APP"},
            "max_rows": 2,
            "max_runtime_seconds": 3,
            "max_output_bytes": 512,
            "metadata": {"environment": "onprem"},
        }
    )
    lease = broker.poll("worker-a")
    assert lease is not None

    runner = MockSqlclFixtureRunner({"select-sessions": _select_fixture()})
    spool_dir = tmp_path / "spool"
    worker = DiagnosticWorker(broker, runner, spool_dir, worker_id="worker-a")
    worker._write_spool(lease)

    resumed = DiagnosticWorker(broker, runner, spool_dir, worker_id="worker-b")
    result = resumed.run_once()

    assert result is not None
    assert result.job_id == job.job_id
    assert runner.execution_count[job.job_id] == 1
    assert broker.store.get_job_row(job.job_id)["status"] == "completed"


def test_query_output_provenance_hook_records_hashes_and_audit_linkage(tmp_path):
    platform = KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=32)),
    )
    broker = _broker(tmp_path)
    broker.create_job(
        {
            "audit_id": "audit-9",
            "finding_id": "finding-9",
            "database_profile_alias": "prod-ledger",
            "operation_class": "read_select",
            "template_id": "select-sessions",
            "parameters": {"username": "APP"},
            "max_rows": 2,
            "max_runtime_seconds": 3,
            "max_output_bytes": 512,
            "metadata": {"environment": "onprem"},
        }
    )
    lease = broker.poll("worker-a")
    assert lease is not None

    runner = MockSqlclFixtureRunner({"select-sessions": _select_fixture()})
    worker = DiagnosticWorker(
        broker,
        runner,
        tmp_path / "spool",
        worker_id="worker-a",
        provenance_hook=QueryOutputProvenanceHook(platform),
    )
    worker._write_spool(lease)
    result = worker.run_once()

    assert result is not None
    assert result.stdout_hash
    source = platform.store.sources[f"diagnostic-{lease.job.job_id}"]
    assert source.source_type == "query-output"
    assert source.metadata["job_id"] == lease.job.job_id
    assert source.metadata["audit_id"] == lease.job.audit_id
    assert source.metadata["finding_id"] == lease.job.finding_id
    assert source.metadata["result_hash"] == result.result_hash
    assert source.metadata["stdout_hash"] == result.stdout_hash
    hits = platform.search("SESSION_ID", filters=None, route="local")
    assert hits
