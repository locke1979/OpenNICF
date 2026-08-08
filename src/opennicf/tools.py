"""Registered QwenAgent tools. Dangerous operations are service boundaries."""
from .rag import EvidenceIndex

class OpenNICFTools:
    def __init__(self, evidence: EvidenceIndex):
        self.evidence = evidence

    def search_evidence(self, query: str):
        return [{"source_id": e.source_id, "locator": e.locator, "text": e.text} for e in self.evidence.search(query)]

    def create_diagnostic_job(self, request: dict):
        if request.get("operation_class") not in {"read_select", "describe"}:
            raise PermissionError("diagnostic operation is not allow-listed")
        return {"status": "pending", "job_id": "broker-generated"}

    def as_qwen_tools(self):
        return {"search_evidence": self.search_evidence, "create_diagnostic_job": self.create_diagnostic_job}

