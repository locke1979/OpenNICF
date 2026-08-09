"""Registered QwenAgent tools. Dangerous operations are service boundaries."""
from typing import Any, Mapping

from .audit import FailureAuditEngine, FailureAuditBroker, LocalFailureAuditBroker
from .rag import EvidenceIndex

class OpenNICFTools:
    def __init__(
        self,
        evidence: EvidenceIndex,
        *,
        audit_engine: FailureAuditEngine | None = None,
        verification_broker: FailureAuditBroker | None = None,
    ):
        self.evidence = evidence
        self.verification_broker = verification_broker or LocalFailureAuditBroker()
        self.audit_engine = audit_engine or FailureAuditEngine(
            self.evidence.platform,
            broker=self.verification_broker,
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

    def as_qwen_tools(self):
        return {
            "search_evidence": self.search_evidence,
            "create_diagnostic_job": self.create_diagnostic_job,
            "request_controlled_verification": self.request_controlled_verification,
            "analyze_failure_audit": self.analyze_failure_audit,
        }
