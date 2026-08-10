"""Registered QwenAgent tools. Dangerous operations are service boundaries."""
from collections.abc import Mapping
from typing import Any

from .audit import FailureAuditBroker, FailureAuditEngine, LocalFailureAuditBroker
from .audit_workflow import PersistentAuditWorkflow
from .rag import EvidenceIndex


class OpenNICFTools:
    def __init__(
        self,
        evidence: EvidenceIndex,
        *,
        audit_engine: FailureAuditEngine | None = None,
        verification_broker: FailureAuditBroker | None = None,
        audit_workflow: PersistentAuditWorkflow | None = None,
    ):
        self.evidence = evidence
        self.verification_broker = verification_broker or LocalFailureAuditBroker()
        self.audit_engine = audit_engine or FailureAuditEngine(
            self.evidence.platform,
            broker=self.verification_broker,
        )
        self.audit_workflow = audit_workflow or PersistentAuditWorkflow(
            self.evidence.platform,
            engine=self.audit_engine,
        )

    def search_evidence(self, query: str):
        return [
            {
                "source_id": e.source_id,
                "locator": e.locator,
                "text": e.text,
                "namespace_id": e.namespace_id,
                "domain_id": e.domain_id,
                "system_id": e.system_id,
                "component_id": e.component_id,
                "environment": e.environment,
                "evidence_type": e.evidence_type,
                "acl_scope": e.acl_scope,
            }
            for e in self.evidence.search(query)
        ]

    def create_diagnostic_job(self, request: dict):
        if request.get("operation_class") not in {"read_select", "describe"}:
            raise PermissionError("diagnostic operation is not allow-listed")
        return dict(self.verification_broker.request(request))

    def request_controlled_verification(self, request: Mapping[str, Any]):
        if request.get("operation_class") not in {"read_select", "describe"}:
            raise PermissionError("diagnostic operation is not allow-listed")
        return dict(self.verification_broker.request(request))

    def analyze_failure_audit(self, request: str | Mapping[str, Any]):
        return self.audit_engine.analyze(request).to_dict()

    def start_failure_audit(
        self,
        request: str | Mapping[str, Any],
        *,
        principal_acl_scopes: tuple[str, ...] = ("internal",),
        domain_ids: tuple[str, ...] = (),
    ):
        """Create and run a durable audit under explicit caller scopes."""
        return self.audit_workflow.start(
            request,
            principal_acl_scopes=principal_acl_scopes,
            domain_ids=domain_ids,
        ).to_dict()

    def resume_failure_audit(self, audit_id: str, diagnostic_result: Mapping[str, Any]):
        return self.audit_workflow.resume(audit_id, diagnostic_result).to_dict()

    def get_audit_evidence_package(
        self,
        audit_id: str,
        *,
        principal_acl_scopes: tuple[str, ...] = ("internal",),
        domain_ids: tuple[str, ...] = (),
    ):
        return self.audit_workflow.evidence_package(
            audit_id,
            principal_acl_scopes=principal_acl_scopes,
            domain_ids=domain_ids,
        )

    def as_qwen_tools(self):
        return {
            "search_evidence": self.search_evidence,
            "create_diagnostic_job": self.create_diagnostic_job,
            "request_controlled_verification": self.request_controlled_verification,
            "analyze_failure_audit": self.analyze_failure_audit,
            "start_failure_audit": self.start_failure_audit,
            "resume_failure_audit": self.resume_failure_audit,
            "get_audit_evidence_package": self.get_audit_evidence_package,
        }
