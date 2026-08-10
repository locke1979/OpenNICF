"""Durable, resumable orchestration for evidence-grounded failure audits.

This module is a checkpointed service around :class:`FailureAuditEngine`.
QwenAgent remains the application orchestration authority; this service owns
only audit state transitions and persistence.  Evidence is always treated as
untrusted data and never as scope, policy, or tool authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from .audit import (
    AuditEvidenceCollection,
    AuditReport,
    AuditRequest,
    FailureAuditBroker,
    FailureAuditEngine,
)
from .knowledge import (
    AuditArtifactRecord,
    AuditEvidenceRefRecord,
    AuditWorkflowRecord,
    KnowledgePlatform,
    VerificationRequestRecord,
)

AUDIT_STATUSES = (
    "created",
    "collecting_evidence",
    "analyzing",
    "waiting_for_diagnostic",
    "resumed",
    "completed",
    "failed",
)
_TERMINAL_STATUSES = {"completed", "failed"}
_TRANSITIONS = {
    "created": {"collecting_evidence", "failed"},
    "collecting_evidence": {"analyzing", "failed"},
    "analyzing": {"waiting_for_diagnostic", "completed", "failed"},
    "waiting_for_diagnostic": {"resumed", "failed"},
    "resumed": {"analyzing", "failed"},
    "completed": set(),
    "failed": set(),
}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    for part in (prefix, *parts):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:24]}"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _safe_request_payload(request: AuditRequest) -> dict[str, Any]:
    """Persist only the typed request contract, never arbitrary authority keys."""

    return {
        "raw_request": request.raw_request,
        "scope": request.scope,
        "domain_ids": list(request.domain_ids),
        "systems": list(request.systems),
        "components": list(request.components),
        "start_time": request.start_time.isoformat() if request.start_time else None,
        "end_time": request.end_time.isoformat() if request.end_time else None,
        "symptoms": list(request.symptoms),
        "requested_outputs": list(request.requested_outputs),
        "correlation_ids": list(request.correlation_ids),
        "source_types": list(request.source_types),
        "require_live_verification": request.require_live_verification,
    }


def _ref_from_hit(hit: Any, *, query: str) -> AuditEvidenceRefRecord:
    return AuditEvidenceRefRecord(
        reference_id=_stable_id("eref", hit.source_id, hit.source_version_id, hit.locator, query),
        source_id=hit.source_id,
        source_version_id=hit.source_version_id,
        source_type=hit.source_type,
        locator=hit.locator,
        role="audit-evidence-package",
        provenance_ref=f"{hit.source_id}:{hit.locator}",
        excerpt_hash=hit.excerpt_hash,
        statement=hit.text.splitlines()[0].strip() if hit.text.strip() else hit.locator,
        metadata={
            "domain_id": hit.domain_id,
            "system_id": hit.system_id,
            "component_id": hit.component_id,
            "namespace_id": hit.namespace_id,
            "acl_scope": hit.acl_scope,
            "score": hit.score,
            "embedding_space_id": hit.embedding_space_id,
            "untrusted_content": True,
        },
    )


def _compact_package(
    collection: AuditEvidenceCollection,
    *,
    max_chunks: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Create the bounded package shape consumed by integration correlation."""

    hits = sorted(collection.hits, key=lambda hit: (-hit.score, hit.chunk_ordinal, hit.chunk_id))
    refs: list[AuditEvidenceRefRecord] = []
    used_tokens = 0
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk_id in seen or len(refs) >= max_chunks:
            continue
        statement = hit.text.splitlines()[0].strip() if hit.text.strip() else hit.locator
        tokens = max(1, (len(statement) + 3) // 4)
        if refs and used_tokens + tokens > max_tokens:
            continue
        refs.append(_ref_from_hit(hit, query=collection.query))
        seen.add(hit.chunk_id)
        used_tokens += tokens
        if used_tokens >= max_tokens:
            break
    ref_payload = [asdict(ref) for ref in refs]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for ref in ref_payload:
        grouped.setdefault((ref["source_id"], ref["source_version_id"]), []).append(ref)
    packages = []
    for (source_id, source_version_id), bucket in sorted(grouped.items()):
        primary = bucket[0]
        packages.append(
            {
                "package_id": _stable_id("pkg", collection.query, source_id, source_version_id),
                "query": collection.query,
                "retrieval_mode": "audit-resume-safe",
                "principal_domain_id": collection.filters.principal_domain_id,
                "source_id": source_id,
                "source_version_id": source_version_id,
                "domain_id": primary["metadata"].get("domain_id", ""),
                "system_id": primary["metadata"].get("system_id", ""),
                "component_id": primary["metadata"].get("component_id", ""),
                "evidence_type": primary["source_type"],
                "source_type": primary["source_type"],
                "locator": primary["locator"],
                "estimated_tokens": sum(max(1, (len(item["statement"]) + 3) // 4) for item in bucket),
                "estimated_bytes": sum(len(item["statement"].encode("utf-8")) for item in bucket),
                "evidence_refs": bucket,
                "metadata": {
                    "domain_ids": list(collection.filters.domain_ids),
                    "acl_scopes": sorted(collection.filters.principal_acl_scopes),
                    "untrusted_content": True,
                },
            }
        )
    return {
        "query": collection.query,
        "retrieval_mode": "audit-resume-safe",
        "domain_ids": list(collection.filters.domain_ids),
        "acl_scopes": sorted(collection.filters.principal_acl_scopes),
        "package_count": len(packages),
        "estimated_tokens": used_tokens,
        "estimated_bytes": sum(package["estimated_bytes"] for package in packages),
        "packages": packages,
    }


@dataclass(frozen=True)
class AuditWorkflowResult:
    record: AuditWorkflowRecord
    report: AuditReport | None = None
    evidence_package: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        return self.record.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit": asdict(self.record),
            "status": self.record.status,
            "report": self.report.to_dict() if self.report else None,
            "evidence_package": self.evidence_package,
        }


class AuditWorkflowError(RuntimeError):
    """Raised for invalid lifecycle operations or inaccessible audit state."""


class PersistentAuditWorkflow:
    """Checkpointed audit workflow using the existing knowledge abstractions."""

    def __init__(
        self,
        platform: KnowledgePlatform,
        *,
        engine: FailureAuditEngine | None = None,
        broker: FailureAuditBroker | None = None,
        max_package_chunks: int = 8,
        max_package_tokens: int = 512,
    ) -> None:
        self.platform = platform
        self.engine = engine or FailureAuditEngine(platform, broker=broker)
        self.max_package_chunks = max(1, int(max_package_chunks))
        self.max_package_tokens = max(1, int(max_package_tokens))

    @property
    def store(self) -> Any:
        return self.platform.store

    def create(
        self,
        request: str | Mapping[str, Any],
        *,
        principal_acl_scopes: Sequence[str] = ("internal",),
        domain_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
    ) -> AuditWorkflowRecord:
        parsed = AuditRequest.from_input(request)
        safe_payload = _safe_request_payload(parsed)
        effective_domains = tuple(dict.fromkeys(str(item) for item in (domain_ids or domains or parsed.domain_ids) if str(item)))
        acl_scopes = tuple(dict.fromkeys(str(item) for item in principal_acl_scopes if str(item)))
        if not acl_scopes:
            raise PermissionError("an audit requires at least one explicit ACL scope")
        request_hash = sha256(_canonical({"request": safe_payload, "domains": effective_domains, "acl": acl_scopes}).encode()).hexdigest()
        audit_id = _stable_id("audit", request_hash)
        existing = self.store.get_audit_workflow(audit_id) if hasattr(self.store, "get_audit_workflow") else None
        if existing is not None:
            return existing
        now = _utcnow()
        record = AuditWorkflowRecord(
            audit_id=audit_id,
            request_hash=request_hash,
            request_payload=safe_payload,
            scope=str(parsed.scope or parsed.raw_request[:240]),
            domain_ids=effective_domains,
            system_ids=parsed.systems,
            component_ids=parsed.components,
            acl_scopes=acl_scopes,
            status="created",
            created_at=now,
            updated_at=now,
            metadata={"status_history": [{"status": "created", "at": now.isoformat()}], "untrusted_evidence": True},
        )
        self._save(record)
        return record

    def start(self, request: str | Mapping[str, Any], **kwargs: Any) -> AuditWorkflowResult:
        record = self.create(request, **kwargs)
        return self.run(record.audit_id)

    def get(self, audit_id: str) -> AuditWorkflowRecord:
        record = self.store.get_audit_workflow(audit_id)
        if record is None:
            raise KeyError(f"unknown audit: {audit_id}")
        return record

    def run(self, audit_id: str) -> AuditWorkflowResult:
        record = self.get(audit_id)
        if record.status in _TERMINAL_STATUSES or record.status == "waiting_for_diagnostic":
            return AuditWorkflowResult(record)
        if record.status == "created":
            record = self._transition(record, "collecting_evidence")
        parsed = AuditRequest.from_input(record.request_payload)
        try:
            collection = self.engine.collect_evidence(
                parsed,
                principal_acl_scopes=frozenset(record.acl_scopes),
                domain_ids=record.domain_ids,
            )
            package = _compact_package(
                collection,
                max_chunks=self.max_package_chunks,
                max_tokens=self.max_package_tokens,
            )
            record = replace(
                record,
                evidence_refs=tuple(
                    ref for package_item in package["packages"] for ref in (
                        AuditEvidenceRefRecord(**ref) for ref in package_item["evidence_refs"]
                    )
                ),
                metadata={**record.metadata, "evidence_package": package},
                updated_at=_utcnow(),
            )
            self._save(record)
            record = self._transition(record, "analyzing")
            report = self.engine.analyze(
                parsed,
                principal_acl_scopes=frozenset(record.acl_scopes),
                domain_ids=record.domain_ids,
                audit_id=record.audit_id,
                collection=collection,
            )
            return self._finish(record, report, package)
        except Exception as exc:
            failed = replace(record, status="failed", last_error=str(exc), updated_at=_utcnow())
            self._save(failed)
            raise

    def resume(
        self,
        audit_id: str,
        diagnostic_result: Mapping[str, Any] | None = None,
        *,
        diagnostic_results: Sequence[Mapping[str, Any]] = (),
    ) -> AuditWorkflowResult:
        record = self.get(audit_id)
        if record.status != "waiting_for_diagnostic":
            if record.status in _TERMINAL_STATUSES:
                return AuditWorkflowResult(record)
            raise AuditWorkflowError(f"audit {audit_id} is not waiting for a diagnostic")
        results = [dict(item) for item in diagnostic_results]
        if diagnostic_result is not None:
            results.insert(0, dict(diagnostic_result))
        if not results:
            raise ValueError("resume requires at least one diagnostic result")
        safe_results = [self._safe_diagnostic_result(item) for item in results]
        record = self._transition(record, "resumed")
        record = replace(
            record,
            resume_count=record.resume_count + 1,
            metadata={**record.metadata, "diagnostic_results": safe_results},
            updated_at=_utcnow(),
        )
        self._save(record)
        parsed = AuditRequest.from_input(record.request_payload)
        try:
            collection = self.engine.collect_evidence(
                parsed,
                principal_acl_scopes=frozenset(record.acl_scopes),
                domain_ids=record.domain_ids,
            )
            record = self._transition(record, "analyzing")
            existing_requests = tuple(
                VerificationRequestRecord(
                    request_id=request_id,
                    audit_id=record.audit_id,
                    finding_id=None,
                    broker_name="broker",
                    request_payload={"operation_class": "read_select", "audit_id": record.audit_id},
                    status=str(safe_results[0].get("status") or "completed"),
                    created_at=record.created_at,
                    metadata={"resumed": True},
                )
                for request_id in record.diagnostic_request_ids
            )
            report = self.engine.analyze(
                parsed,
                principal_acl_scopes=frozenset(record.acl_scopes),
                domain_ids=record.domain_ids,
                audit_id=record.audit_id,
                collection=collection,
                diagnostic_results=safe_results,
                existing_verification_requests=existing_requests,
            )
            package = _compact_package(collection, max_chunks=self.max_package_chunks, max_tokens=self.max_package_tokens)
            return self._finish(record, report, package)
        except Exception as exc:
            failed = replace(record, status="failed", last_error=str(exc), updated_at=_utcnow())
            self._save(failed)
            raise

    def evidence_package(
        self,
        audit_id: str,
        *,
        principal_acl_scopes: Sequence[str],
        domain_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
    ) -> dict[str, Any]:
        record = self.get(audit_id)
        if not set(principal_acl_scopes).issuperset(record.acl_scopes):
            raise PermissionError("audit package is outside the requester's ACL scope")
        if not set(domain_ids or domains).issuperset(record.domain_ids):
            raise PermissionError("audit package is outside the requester's domain scope")
        return dict(record.metadata.get("evidence_package", {"packages": []}))

    def read_artifact(
        self,
        audit_id: str,
        artifact_id: str,
        *,
        principal_acl_scopes: Sequence[str],
        domain_ids: Sequence[str] = (),
        domains: Sequence[str] = (),
    ) -> bytes:
        record = self.get(audit_id)
        if not set(principal_acl_scopes).issuperset(record.acl_scopes):
            raise PermissionError("artifact is outside the requester's ACL scope")
        if not set(domain_ids or domains).issuperset(record.domain_ids):
            raise PermissionError("artifact is outside the requester's domain scope")
        artifacts = {item.artifact_id: item for item in self.store.list_audit_artifacts(audit_id)}
        artifact = artifacts.get(artifact_id)
        if artifact is None or not set(principal_acl_scopes).issuperset(artifact.acl_scopes):
            raise PermissionError("artifact is unavailable")
        return self.platform.object_store.get_bytes(artifact.object_key)

    def _safe_diagnostic_result(self, result: Mapping[str, Any]) -> dict[str, Any]:
        request_id = str(result.get("request_id") or "")
        status = str(result.get("status") or "completed")
        if status not in {"completed", "succeeded", "failed", "rejected"}:
            raise ValueError("diagnostic result has an unsupported status")
        digest = sha256(_canonical(dict(result)).encode()).hexdigest()
        return {"request_id": request_id, "status": status, "result_hash": digest}

    def _finish(self, record: AuditWorkflowRecord, report: AuditReport, package: dict[str, Any]) -> AuditWorkflowResult:
        pending = any(item.status in {"pending", "queued", "running"} for item in report.verification_requests)
        diagnostic_failed = any(
            item.status in {"failed", "rejected"} for item in report.verification_requests
        ) or any(
            result.get("status") in {"failed", "rejected"}
            for result in record.metadata.get("diagnostic_results", ())
        )
        status = "waiting_for_diagnostic" if pending else "failed" if diagnostic_failed else "completed"
        checkpoint = replace(
            record,
            finding_ids=tuple(finding.finding_id for finding in report.findings),
            missing_evidence=tuple(finding.statement for finding in report.findings if finding.classification == "missing_evidence"),
            diagnostic_request_ids=tuple(item.request_id for item in report.verification_requests),
            timeline_event_ids=tuple(_stable_id("event", point.provenance_ref, point.statement) for point in report.timeline),
            updated_at=_utcnow(),
            metadata={**record.metadata, "report_audit_id": report.audit_id},
        )
        self._save(checkpoint)
        record = self._transition(checkpoint, status)
        artifact_ids, manifest_id, manifest_hash = self._write_artifacts(record, report, package, status=status)
        record = replace(record, report_artifact_ids=artifact_ids, manifest_artifact_id=manifest_id, manifest_hash=manifest_hash)
        self._save(record)
        return AuditWorkflowResult(record, report=report, evidence_package=package)

    def _write_artifacts(self, record: AuditWorkflowRecord, report: AuditReport, package: dict[str, Any], *, status: str) -> tuple[tuple[str, ...], str, str]:
        artifacts: list[AuditArtifactRecord] = []
        payloads = {
            "human_report": (self._human_report(report, record, package), "text/markdown; charset=utf-8"),
            "machine_report": (report.to_json(), "application/json"),
            "evidence_package": (_canonical(package), "application/json"),
        }
        for artifact_type, (content, mime_type) in payloads.items():
            content_bytes = content.encode("utf-8")
            reference = self.platform.object_store.put_bytes(content_bytes, mime_type=mime_type, metadata={"audit_id": record.audit_id, "artifact_type": artifact_type})
            artifact = AuditArtifactRecord(
                artifact_id=_stable_id("aart", record.audit_id, artifact_type, reference.content_hash),
                audit_id=record.audit_id,
                artifact_type=artifact_type,
                backend=reference.backend,
                object_key=reference.object_key,
                content_hash=reference.content_hash,
                mime_type=mime_type,
                size_bytes=reference.size_bytes,
                acl_scopes=record.acl_scopes,
                domain_ids=record.domain_ids,
                metadata={"status": status, "provenance_backed": True},
            )
            self.store.record_audit_artifact(artifact)
            artifacts.append(artifact)
        manifest_payload = {
            "manifest_version": 1,
            "audit_id": record.audit_id,
            "status": status,
            "request_hash": record.request_hash,
            "scope": record.scope,
            "domain_ids": list(record.domain_ids),
            "acl_scopes": list(record.acl_scopes),
            "evidence_refs": [asdict(ref) for ref in record.evidence_refs],
            "finding_ids": list(record.finding_ids),
            "missing_evidence": list(record.missing_evidence),
            "artifacts": [asdict(item) for item in artifacts],
            "provenance": {"source": "OpenNICF knowledge platform", "untrusted_evidence": True},
        }
        manifest_content = _canonical(manifest_payload).encode("utf-8")
        manifest_reference = self.platform.object_store.put_bytes(manifest_content, mime_type="application/json", metadata={"audit_id": record.audit_id, "artifact_type": "manifest"})
        manifest_hash = manifest_reference.content_hash
        manifest = AuditArtifactRecord(
            artifact_id=_stable_id("aart", record.audit_id, "manifest", manifest_hash),
            audit_id=record.audit_id,
            artifact_type="manifest",
            backend=manifest_reference.backend,
            object_key=manifest_reference.object_key,
            content_hash=manifest_hash,
            mime_type="application/json",
            size_bytes=manifest_reference.size_bytes,
            acl_scopes=record.acl_scopes,
            domain_ids=record.domain_ids,
            metadata={"status": status, "provenance_backed": True},
        )
        self.store.record_audit_artifact(manifest)
        return tuple(item.artifact_id for item in artifacts), manifest.artifact_id, manifest_hash

    def _human_report(self, report: AuditReport, record: AuditWorkflowRecord, package: Mapping[str, Any]) -> str:
        findings_by_class = {classification: [finding for finding in report.findings if finding.classification == classification] for classification in ("fact", "inference", "hypothesis")}
        technical_lines = [f"- {chain}" for chain in report.causal_chains] or ["- No causal chain established."]
        lines = [
            report.to_markdown(),
            "",
            "## Evidence",
            f"- Compact evidence packages: {package.get('package_count', 0)}",
            f"- Provenance references: {len(record.evidence_refs)}",
            "",
            "## Technical Analysis",
            *technical_lines,
            "",
            "## Facts, Inferences, and Hypotheses",
        ]
        for classification, findings in findings_by_class.items():
            lines.append(f"### {classification.title()}")
            lines.extend(f"- `{finding.finding_id}` {finding.statement}" for finding in findings) or lines.append("- None recorded.")
        lines.extend(["", "## Missing Evidence"])
        lines.extend(f"- {item}" for item in record.missing_evidence) or lines.append("- None recorded.")
        lines.extend(["", "## Verification and Remediation"])
        lines.extend(f"- {item}" for item in report.recommended_verification_steps) or lines.append("- No additional controlled verification requested.")
        lines.extend(["", "## Appendix: Manifest", f"- Audit ID: `{record.audit_id}`", f"- Request hash: `{record.request_hash}`", f"- ACL scopes: {', '.join(record.acl_scopes)}"])
        lines.extend(f"- `{ref.reference_id}` → `{ref.provenance_ref}` (excerpt `{ref.excerpt_hash}`)" for ref in record.evidence_refs)
        return "\n".join(lines).strip() + "\n"

    def _transition(self, record: AuditWorkflowRecord, status: str) -> AuditWorkflowRecord:
        if status not in AUDIT_STATUSES or status not in _TRANSITIONS[record.status]:
            raise AuditWorkflowError(f"invalid audit transition {record.status} -> {status}")
        now = _utcnow()
        history = [*record.metadata.get("status_history", []), {"status": status, "at": now.isoformat()}]
        updated = replace(record, status=status, updated_at=now, metadata={**record.metadata, "status_history": history})
        self._save(updated)
        return updated

    def _save(self, record: AuditWorkflowRecord) -> None:
        if not hasattr(self.store, "record_audit_workflow"):
            raise AuditWorkflowError("knowledge store does not support durable audit workflows")
        self.store.record_audit_workflow(record)


AuditWorkflow = PersistentAuditWorkflow
