import json
from pathlib import Path

from opennicf.ops.deployment import DeploymentController
from opennicf.ops.health import HealthState, check_health
from opennicf.ops.observability import AuditLog, Metrics, TraceContext, structured_event
from opennicf.ops.secrets import inventory, scan_text


def test_failing_gate_blocks_and_rolls_back():
    calls = []
    controller = DeploymentController(known_good="r1", deploy=calls.append, rollback=lambda release: calls.append(f"rollback:{release}"))
    result = controller.deploy_release("r2", migration_gate=lambda: True, health_gate=lambda: False, smoke_gate=lambda: True)
    assert result.state == "rolled_back"
    assert calls == ["rollback:r1"]
    assert [gate.name for gate in result.gates] == ["migration", "health"]


def test_deploy_exception_rolls_back_to_previous_release():
    calls = []

    def deploy(_release_id: str) -> None:
        raise RuntimeError("boom")

    controller = DeploymentController(
        known_good="r1",
        deploy=deploy,
        rollback=lambda release: calls.append(f"rollback:{release}"),
    )
    result = controller.deploy_release("r2", migration_gate=lambda: True, health_gate=lambda: True, smoke_gate=lambda: True)
    assert result.state == "rolled_back"
    assert result.rolled_back_to == "r1"
    assert calls == ["rollback:r1"]
    assert result.gates[-1].name == "deploy"


def test_synthetic_deployment_reports_release_id():
    deployed = []
    result = DeploymentController(known_good="r0", deploy=deployed.append, rollback=lambda _: None).synthetic_deploy("synthetic-42")
    assert result.state == "promoted"
    assert result.release_id == "synthetic-42"
    assert deployed == ["synthetic-42"]


def test_partial_health_is_degraded_and_identifies_dependency():
    report = check_health("gateway", [("postgres", lambda: True), ("lmstudio", lambda: False)], release_id="r2")
    assert report.state is HealthState.DEGRADED
    assert report.ready is False
    assert report.live is True
    assert report.as_dict()["dependencies"]["lmstudio"] == "failed"


def test_metrics_logs_and_audit_share_correlation_id_without_content():
    context = TraceContext(correlation_id="audit-123")
    metrics = Metrics(); metrics.inc("audit.count"); metrics.observe("audit.latency_ms", 12)
    audit = AuditLog(secret_values=["runtime-secret"])
    record = audit.append("audit.completed", context, finding_id="f-1", evidence="runtime-secret")
    assert record["correlation_id"] == "audit-123"
    assert record["fields"]["evidence"] == "[redacted]"
    assert '"correlation_id":"audit-123"' in structured_event("audit.completed", context, result="f-1")
    assert metrics.snapshot()["counters"]["audit.count"] == 1


def test_secret_fixture_detects_fake_secret():
    fixture = "GITHUB_TOKEN=" + "ghp_" + "EXAMPLE123456789012345678901234567890\n"
    assert scan_text(fixture)[0].kind == "github-token"


def test_secret_scan_flags_private_endpoint():
    endpoint = "https://" + "billing" + ".internal" + ".local/api"
    findings = scan_text(f"endpoint={endpoint}\n")
    assert any(finding.kind == "private-endpoint" for finding in findings)


def test_inventory_contains_names_only():
    entries = inventory()
    assert {entry["name"] for entry in entries} >= {"PROXMOX_API_CREDENTIALS", "TELEGRAM_TOKEN"}
    assert all("value" not in entry for entry in entries)


def test_config_inventory_document_matches_runtime_inventory():
    doc = json.loads(Path("docs/config-inventory.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == "opennicf.config-inventory/v1"
    assert doc["values"] == "omitted"
    assert {entry["name"] for entry in doc["secrets"]} == {entry["name"] for entry in inventory()}


def test_operations_docs_cover_backup_restore_and_traceability():
    text = Path("docs/operations.md").read_text(encoding="utf-8")
    assert "Backup, restore, and recovery" in text
    assert "correlation_id" in text
    assert "Secret/config inventory" in text
