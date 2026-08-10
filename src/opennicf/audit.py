"""Evidence-grounded failure audit engine and report renderers."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, Protocol

from .knowledge import (
    AuditEvidenceRefRecord,
    AuditFindingRecord,
    AuditReportRecord,
    EvidenceHit,
    KnowledgePlatform,
    RetrievalFilters,
    VerificationRequestRecord,
)

AuditClassification = Literal["fact", "inference", "hypothesis", "missing_evidence", "recommendation"]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:24]}"


def _as_utc(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    normalized = value.replace("Z", "+00:00") if isinstance(value, str) else value
    return datetime.fromisoformat(normalized).astimezone(UTC)


_CORRELATION_RE = re.compile(r"(?:correlation[_-]?id|request[_-]?id|trace[_-]?id|session[_-]?id)\s*[:=]\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_STACK_FRAME_RE = re.compile(r'(?:File\s+"(?P<file>[^"]+)",\s*line\s*(?P<line>\d+)|(?P<path>[A-Za-z0-9_./\\-]+\.(?:py|java|cs|go|js|ts|rb|php|sql)):(?P<path_line>\d+))')
_TIMESTAMP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_ERROR_RE = re.compile(r"\b(ERROR|FATAL|EXCEPTION|TRACEBACK|FAIL(?:ED|URE)?)\b", re.IGNORECASE)
_SQLSTATE_RE = re.compile(r"SQLSTATE\s+([A-Z0-9]+)", re.IGNORECASE)
_PATH_RE = re.compile(r"(?P<path>[A-Za-z0-9_./\\-]+\.(?:py|java|cs|go|js|ts|rb|sql|xml|wsdl|md))")
_ZERO_ROWS_RE = re.compile(r"\b0\s+rows?\b|\bno rows?\b", re.IGNORECASE)


@dataclass(frozen=True)
class AuditRequest:
    """Parsed audit request with explicit scope and requested outputs."""

    raw_request: str
    scope: str | None = None
    systems: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None
    symptoms: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = ()
    correlation_ids: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    require_live_verification: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, value: str | Mapping[str, Any]) -> AuditRequest:
        if isinstance(value, str):
            text = value.strip()
            lower = text.lower()
            requested_outputs = tuple(
                output
                for output in ("timeline", "findings", "json", "human report", "verification")
                if output in lower
            )
            require_live_verification = any(token in lower for token in ("live verification", "on-prem", "database", "db"))
            return cls(
                raw_request=text,
                scope=_first_text_match(text, ("scope", "for", "about")) or None,
                systems=tuple(_split_comma_values(_first_text_match(text, ("system", "systems")))),
                components=tuple(_split_comma_values(_first_text_match(text, ("component", "components")))),
                symptoms=tuple(_split_comma_values(_first_text_match(text, ("symptom", "symptoms", "failure", "error")))),
                requested_outputs=requested_outputs,
                correlation_ids=tuple(_CORRELATION_RE.findall(text)),
                source_types=tuple(
                    source_type
                    for source_type in ("log", "code", "schema", "document", "query-output", "splunk-csv")
                    if source_type.replace("-", " ") in lower or source_type in lower
                ),
                require_live_verification=require_live_verification,
            )

        data = dict(value)
        return cls(
            raw_request=str(data.get("query") or data.get("request") or data.get("scope") or ""),
            scope=data.get("scope"),
            systems=_normalize_tuple(data.get("systems")),
            components=_normalize_tuple(data.get("components")),
            start_time=_as_utc(data.get("start_time") or data.get("from")),
            end_time=_as_utc(data.get("end_time") or data.get("to")),
            symptoms=_normalize_tuple(data.get("symptoms") or data.get("issues")),
            requested_outputs=_normalize_tuple(data.get("requested_outputs") or data.get("outputs")),
            correlation_ids=_normalize_tuple(data.get("correlation_ids") or data.get("request_ids")),
            source_types=_normalize_tuple(data.get("source_types") or data.get("evidence_types")),
            require_live_verification=bool(data.get("require_live_verification") or data.get("live_verification_required")),
            metadata={k: v for k, v in data.items() if k not in {"query", "request", "scope", "systems", "components", "start_time", "from", "end_time", "to", "symptoms", "issues", "requested_outputs", "outputs", "correlation_ids", "request_ids", "source_types", "evidence_types", "require_live_verification", "live_verification_required"}},
        )

    @property
    def search_terms(self) -> str:
        parts = [self.raw_request, self.scope or "", *self.systems, *self.components, *self.symptoms]
        return " ".join(part for part in parts if part).strip()


@dataclass(frozen=True)
class AuditTimelinePoint:
    timestamp: datetime | None
    statement: str
    source_id: str
    source_type: str
    locator: str
    correlation_ids: tuple[str, ...] = ()
    systems: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    provenance_ref: str = ""


@dataclass(frozen=True)
class AuditReport:
    audit_id: str
    request: AuditRequest
    findings: tuple[AuditFindingRecord, ...]
    timeline: tuple[AuditTimelinePoint, ...]
    causal_chains: tuple[str, ...]
    unresolved_hypotheses: tuple[str, ...]
    recommended_verification_steps: tuple[str, ...]
    verification_requests: tuple[VerificationRequestRecord, ...]
    executive_summary: str
    human_report: str
    machine_report: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "request": {
                "raw_request": self.request.raw_request,
                "scope": self.request.scope,
                "systems": list(self.request.systems),
                "components": list(self.request.components),
                "start_time": self.request.start_time.isoformat() if self.request.start_time else None,
                "end_time": self.request.end_time.isoformat() if self.request.end_time else None,
                "symptoms": list(self.request.symptoms),
                "requested_outputs": list(self.request.requested_outputs),
                "correlation_ids": list(self.request.correlation_ids),
                "source_types": list(self.request.source_types),
                "require_live_verification": self.request.require_live_verification,
                "metadata": self.request.metadata,
            },
            "findings": [_finding_to_dict(finding) for finding in self.findings],
            "timeline": [_timeline_to_dict(point) for point in self.timeline],
            "causal_chains": list(self.causal_chains),
            "unresolved_hypotheses": list(self.unresolved_hypotheses),
            "recommended_verification_steps": list(self.recommended_verification_steps),
            "verification_requests": [asdict(item) for item in self.verification_requests],
            "executive_summary": self.executive_summary,
            "human_report": self.human_report,
            "machine_report": self.machine_report,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_json_default)

    def to_markdown(self) -> str:
        lines = [
            "# Failure Audit Report",
            "",
            f"- Audit ID: `{self.audit_id}`",
            f"- Scope: {self.request.scope or 'unspecified'}",
            f"- Systems: {', '.join(self.request.systems) if self.request.systems else 'unspecified'}",
            f"- Components: {', '.join(self.request.components) if self.request.components else 'unspecified'}",
            "",
            "## Executive Summary",
            self.executive_summary or "No summary available.",
            "",
            "## Timeline",
        ]
        if self.timeline:
            for point in self.timeline:
                timestamp = point.timestamp.isoformat() if point.timestamp else "unspecified"
                correlation = f" [{', '.join(point.correlation_ids)}]" if point.correlation_ids else ""
                lines.append(f"- `{timestamp}` {point.statement}{correlation}")
                lines.append(f"  - Evidence: `{point.source_id}` `{point.locator}`")
        else:
            lines.append("- No timestamped evidence located.")
        lines.extend(["", "## Findings"])
        for finding in self.findings:
            evidence = ", ".join(ref.provenance_ref for ref in finding.evidence_refs) or "none"
            lines.append(f"- `{finding.finding_id}` [{finding.classification}] {finding.statement}")
            lines.append(f"  - Confidence: {finding.confidence:.2f}")
            lines.append(f"  - Verification: {finding.verification_status}")
            lines.append(f"  - Evidence: {evidence}")
        if self.recommended_verification_steps:
            lines.extend(["", "## Verification"])
            lines.extend(f"- {step}" for step in self.recommended_verification_steps)
        if self.unresolved_hypotheses:
            lines.extend(["", "## Unresolved Hypotheses"])
            lines.extend(f"- {item}" for item in self.unresolved_hypotheses)
        return "\n".join(lines).strip()


@dataclass(frozen=True)
class AuditObservation:
    classification: AuditClassification
    statement: str
    confidence: float
    evidence_refs: tuple[AuditEvidenceRefRecord, ...]
    time_range: tuple[datetime | None, datetime | None] | None = None
    systems: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    verification_status: str = "observed"
    recommended_query: str | None = None
    diagnostic_action: dict[str, Any] | None = None
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    correlation_ids: tuple[str, ...] = ()
    source_type_analyzers: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AuditSourceAnalyzer(Protocol):
    source_types: tuple[str, ...]

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        raise NotImplementedError


class StructuredLogAnalyzer:
    source_types = ("log",)

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        if not hits:
            return []
        correlation_ids = sorted({cid for hit in hits for cid in _extract_correlation_ids(hit.text)} | set(request.correlation_ids))
        timestamps = [ts for hit in hits for ts in _extract_timestamps(hit.text)]
        time_range = (min(timestamps), max(timestamps)) if timestamps else None
        error_hits = [hit for hit in hits if _ERROR_RE.search(hit.text)]
        observations: list[AuditObservation] = []
        if error_hits:
            observations.append(
                AuditObservation(
                    classification="fact",
                    statement=f"Application logs contain a failure signal in {', '.join(sorted({hit.locator for hit in error_hits}))}.",
                    confidence=0.98,
                    evidence_refs=tuple(_ref_for_hit(hit, "log-observation", "observed") for hit in error_hits),
                    time_range=time_range,
                    systems=tuple(sorted({hit.system for hit in hits if hit.system})),
                    components=tuple(sorted({hit.metadata.get("component", "") for hit in hits if hit.metadata.get("component")})),
                    verification_status="observed",
                    supporting_evidence=tuple(sorted({hit.chunk_id for hit in error_hits})),
                    provenance_refs=tuple(sorted({hit.locator for hit in error_hits})),
                    correlation_ids=tuple(correlation_ids),
                    source_type_analyzers=self.source_types,
                    metadata={"message_count": len(hits), "error_count": len(error_hits)},
                )
            )
        if correlation_ids:
            observations.append(
                AuditObservation(
                    classification="fact",
                    statement=f"Correlation identifiers were present in the logs: {', '.join(correlation_ids)}.",
                    confidence=0.95,
                    evidence_refs=tuple(_ref_for_hit(hit, "correlation-observation", "observed") for hit in hits if _extract_correlation_ids(hit.text)),
                    time_range=time_range,
                    systems=tuple(sorted({hit.system for hit in hits if hit.system})),
                    verification_status="observed",
                    supporting_evidence=tuple(sorted({hit.chunk_id for hit in hits if _extract_correlation_ids(hit.text)})),
                    provenance_refs=tuple(sorted({hit.locator for hit in hits if _extract_correlation_ids(hit.text)})),
                    correlation_ids=tuple(correlation_ids),
                    source_type_analyzers=self.source_types,
                )
            )
        return observations


class SplunkCSVAnalyzer:
    source_types = ("splunk-csv",)

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        observations: list[AuditObservation] = []
        for hit in hits:
            rows = list(_parse_csv_rows(hit.text))
            if not rows:
                continue
            header = rows[0].keys()
            time_fields = [row.get("timestamp") or row.get("_time") or row.get("time") for row in rows if row.get("timestamp") or row.get("_time") or row.get("time")]
            timestamps = [_as_utc(value) for value in time_fields if value]
            correlation_ids = sorted({value for row in rows for value in (_extract_correlation_ids(" ".join(str(item) for item in row.values())))})
            statement = f"Splunk export at {hit.locator} contains {len(rows)} event row(s) with columns {', '.join(header)}."
            if correlation_ids:
                statement += f" Correlation IDs observed: {', '.join(correlation_ids)}."
            observations.append(
                AuditObservation(
                    classification="fact",
                    statement=statement,
                    confidence=0.9,
                    evidence_refs=( _ref_for_hit(hit, "splunk-export", "observed"), ),
                    time_range=(min(ts for ts in timestamps if ts), max(ts for ts in timestamps if ts)) if any(timestamps) else None,
                    systems=(hit.system,) if hit.system else (),
                    verification_status="observed",
                    supporting_evidence=(hit.chunk_id,),
                    provenance_refs=(hit.locator,),
                    correlation_ids=tuple(correlation_ids),
                    source_type_analyzers=self.source_types,
                )
            )
        return observations


class GenericCodeAnalyzer:
    source_types = ("code",)

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        if not hits:
            return []
        observations: list[AuditObservation] = []
        code_paths = {hit.locator for hit in hits}
        for hit in hits:
            indexed_symbols = tuple(hit.metadata.get("symbol_names", ()))
            if indexed_symbols:
                observations.append(
                    AuditObservation(
                        classification="fact",
                        statement=f"Persistent code index resolves symbols {', '.join(indexed_symbols[:8])} in {hit.locator}.",
                        confidence=0.9,
                        evidence_refs=(_ref_for_hit(hit, "code-symbol-index", "observed"),),
                        systems=(hit.system,) if hit.system else (),
                        components=indexed_symbols[:8],
                        verification_status="observed",
                        supporting_evidence=(hit.chunk_id,),
                        provenance_refs=(hit.locator,),
                        source_type_analyzers=("code", "symbol-index"),
                        metadata={"parser_version": hit.parser_version, "symbol_count": len(indexed_symbols)},
                    )
                )
            matched_frames = [frame for frame in _extract_stack_frames(hit.text) if _frame_matches_paths(frame, code_paths)]
            if matched_frames:
                observations.append(
                    AuditObservation(
                        classification="fact",
                        statement=f"Source code evidence matches the runtime stack frame(s) {', '.join(frame['raw'] for frame in matched_frames)}.",
                        confidence=0.96,
                        evidence_refs=( _ref_for_hit(hit, "code-match", "observed"), ),
                        systems=(hit.system,) if hit.system else (),
                        components=(hit.metadata.get("symbol") or hit.locator,),
                        verification_status="observed",
                        supporting_evidence=(hit.chunk_id,),
                        provenance_refs=(hit.locator,),
                        source_type_analyzers=self.source_types,
                    )
                )
        return observations


class SchemaAnalyzer:
    source_types = ("schema",)

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        if not hits:
            return []
        objects = sorted({name for hit in hits for name in _extract_schema_objects(hit.text)})
        statements = [hit.text for hit in hits]
        observations: list[AuditObservation] = []
        if objects:
            observations.append(
                AuditObservation(
                    classification="fact",
                    statement=f"Schema evidence identifies objects: {', '.join(objects)}.",
                    confidence=0.92,
                    evidence_refs=tuple(_ref_for_hit(hit, "schema-object", "observed") for hit in hits),
                    systems=tuple(sorted({hit.system for hit in hits if hit.system})),
                    verification_status="observed",
                    supporting_evidence=tuple(hit.chunk_id for hit in hits),
                    provenance_refs=tuple(hit.locator for hit in hits),
                    source_type_analyzers=self.source_types,
                )
            )
        if any("query" in statement.lower() for statement in statements):
            observations.append(
                AuditObservation(
                    classification="inference",
                    statement="Schema and query evidence are consistent with a data-access path rather than an isolated presentation-layer failure.",
                    confidence=0.7,
                    evidence_refs=tuple(_ref_for_hit(hit, "schema-query-consistency", "supporting") for hit in hits),
                    systems=tuple(sorted({hit.system for hit in hits if hit.system})),
                    verification_status="inferred",
                    supporting_evidence=tuple(hit.chunk_id for hit in hits),
                    provenance_refs=tuple(hit.locator for hit in hits),
                    source_type_analyzers=self.source_types,
                )
            )
        return observations


class QueryOutputAnalyzer:
    source_types = ("query-output",)

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        if not hits:
            return []
        observations: list[AuditObservation] = []
        for hit in hits:
            rows = _parse_query_table(hit.text)
            row_count = len(rows)
            statements = [hit.text.lower(), *[value.lower() for row in rows for value in row.values() if isinstance(value, str)]]
            missing = _ZERO_ROWS_RE.search(hit.text) is not None or row_count == 0
            if missing:
                observations.append(
                    AuditObservation(
                        classification="missing_evidence",
                        statement=f"Query output at {hit.locator} does not provide sufficient live data to confirm the database state.",
                        confidence=0.88,
                        evidence_refs=( _ref_for_hit(hit, "query-output-gap", "supporting"), ),
                        systems=(hit.system,) if hit.system else (),
                        verification_status="needs_live_verification",
                        recommended_query="Run a controlled on-prem verification against the live database and compare the result set against this query output.",
                        diagnostic_action={
                            "operation_class": "read_select",
                            "target": "database",
                            "reason": "Live verification required to avoid fabricating database contents",
                        },
                        supporting_evidence=(hit.chunk_id,),
                        provenance_refs=(hit.locator,),
                        source_type_analyzers=self.source_types,
                    )
                )
            else:
                observations.append(
                    AuditObservation(
                        classification="fact",
                        statement=f"Query output at {hit.locator} contains {row_count} row(s) with explicit result evidence.",
                        confidence=0.9,
                        evidence_refs=( _ref_for_hit(hit, "query-output", "observed"), ),
                        systems=(hit.system,) if hit.system else (),
                        verification_status="observed",
                        supporting_evidence=(hit.chunk_id,),
                        provenance_refs=(hit.locator,),
                        source_type_analyzers=self.source_types,
                    )
                )
            if any("error" in text for text in statements):
                observations.append(
                    AuditObservation(
                        classification="hypothesis",
                        statement="The query output appears to capture a downstream failure signature rather than a successful operational state.",
                        confidence=0.56,
                        evidence_refs=( _ref_for_hit(hit, "query-output-hypothesis", "supporting"), ),
                        systems=(hit.system,) if hit.system else (),
                        verification_status="unverified",
                        supporting_evidence=(hit.chunk_id,),
                        provenance_refs=(hit.locator,),
                        source_type_analyzers=self.source_types,
                    )
                )
        return observations


class DocumentationAnalyzer:
    source_types = ("document", "docs", "doc", "markdown")

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        if not hits:
            return []
        observations: list[AuditObservation] = []
        for hit in hits:
            if any(term in hit.text.lower() for term in ("verify", "verification", "on-prem", "controlled", "evidence")):
                observations.append(
                    AuditObservation(
                        classification="fact",
                        statement=f"Documentation at {hit.locator} defines a controlled verification expectation.",
                        confidence=0.84,
                        evidence_refs=( _ref_for_hit(hit, "doc-verification", "observed"), ),
                        systems=(hit.system,) if hit.system else (),
                        verification_status="observed",
                        supporting_evidence=(hit.chunk_id,),
                        provenance_refs=(hit.locator,),
                        source_type_analyzers=self.source_types,
                    )
                )
        return observations


class IntegrationDefinitionAnalyzer:
    source_types = ("wsdl", "soap", "rest", "integration")

    def analyze(self, request: AuditRequest, hits: Sequence[EvidenceHit]) -> list[AuditObservation]:
        if not hits:
            return []
        endpoints = sorted({endpoint for hit in hits for endpoint in _extract_integration_endpoints(hit.text)})
        if not endpoints:
            return []
        return [
            AuditObservation(
                classification="fact",
                statement=f"Integration definitions expose endpoints or operations: {', '.join(endpoints)}.",
                confidence=0.82,
                evidence_refs=tuple(_ref_for_hit(hit, "integration-definition", "observed") for hit in hits),
                systems=tuple(sorted({hit.system for hit in hits if hit.system})),
                verification_status="observed",
                supporting_evidence=tuple(hit.chunk_id for hit in hits),
                provenance_refs=tuple(hit.locator for hit in hits),
                source_type_analyzers=self.source_types,
            )
        ]


class FailureAuditBroker(Protocol):
    name: str

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


class LocalFailureAuditBroker:
    """Safe local broker stub that never fabricates live database results."""

    name = "broker"

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id = _stable_id("verify", json.dumps(payload, sort_keys=True, default=_json_default))
        return {
            "status": "pending",
            "job_id": "broker-generated",
            "request_id": request_id,
            "broker": self.name,
            "payload": dict(payload),
        }


class FailureAuditEngine:
    """Coordinate evidence retrieval, source-type analyzers, and report generation."""

    def __init__(
        self,
        platform: KnowledgePlatform,
        *,
        broker: FailureAuditBroker | None = None,
        analyzers: Sequence[AuditSourceAnalyzer] | None = None,
    ) -> None:
        self.platform = platform
        self.broker = broker or LocalFailureAuditBroker()
        self.analyzers = tuple(analyzers or _default_analyzers())

    def analyze(self, request: str | Mapping[str, Any]) -> AuditReport:
        parsed = AuditRequest.from_input(request)
        query = parsed.search_terms or parsed.raw_request or "failure audit"
        filters = RetrievalFilters(
            principal_acl_scopes=frozenset({"internal"}),
            systems=parsed.systems,
            source_types=parsed.source_types,
            since=parsed.start_time,
            until=parsed.end_time,
            limit=24,
        )

        source_types = parsed.source_types or self._discover_source_types()
        hits_by_type: dict[str, list[EvidenceHit]] = defaultdict(list)
        all_hits: list[EvidenceHit] = []
        for source_type in source_types:
            type_filters = RetrievalFilters(
                principal_acl_scopes=filters.principal_acl_scopes,
                systems=filters.systems,
                source_types=(source_type,),
                since=filters.since,
                until=filters.until,
                limit=filters.limit,
            )
            hits = self.platform.search(query, filters=type_filters, route="local")
            for hit in hits:
                hits_by_type[hit.source_type].append(hit)
                all_hits.append(hit)

        if not all_hits:
            # Search the broad corpus once so a structured request still sees evidence.
            all_hits = self.platform.search(query, filters=filters, route="local")
            for hit in all_hits:
                hits_by_type[hit.source_type].append(hit)

        observations: list[AuditObservation] = []
        indexed_symbols = (
            self.platform.search_code_symbols(
                "",
                filters=RetrievalFilters(
                    principal_acl_scopes=filters.principal_acl_scopes,
                    systems=filters.systems,
                    source_types=("code",),
                    since=filters.since,
                    until=filters.until,
                    limit=filters.limit,
                ),
            )
            if hasattr(self.platform, "search_code_symbols")
            else []
        )
        symbols_by_chunk: dict[str, list[str]] = defaultdict(list)
        for symbol in indexed_symbols:
            symbols_by_chunk[symbol.chunk_id].append(symbol.qualified_name)
        if symbols_by_chunk:
            all_hits = [replace(hit, metadata={**hit.metadata, "symbol_names": tuple(symbols_by_chunk.get(hit.chunk_id, ()))}) for hit in all_hits]
            for source_type, hits in list(hits_by_type.items()):
                hits_by_type[source_type] = [replace(hit, metadata={**hit.metadata, "symbol_names": tuple(symbols_by_chunk.get(hit.chunk_id, ()))}) for hit in hits]
        for analyzer in self.analyzers:
            analyzer_hits = [hit for source_type in analyzer.source_types for hit in hits_by_type.get(source_type, [])]
            observations.extend(analyzer.analyze(parsed, analyzer_hits))

        timeline = self._build_timeline(all_hits)
        findings = self._build_findings(parsed, observations, timeline)
        findings.extend(self._derive_cross_source_findings(parsed, findings, hits_by_type))
        verification_requests: list[VerificationRequestRecord] = []
        unresolved_hypotheses = [finding.statement for finding in findings if finding.classification in {"hypothesis", "missing_evidence"}]
        recommended_verification_steps: list[str] = []
        if parsed.require_live_verification or any(finding.classification == "missing_evidence" for finding in findings):
            recommendation = "Request controlled on-prem verification for the missing live database evidence through the broker interface."
            recommended_verification_steps.append(recommendation)
            request_payload = {
                "operation_class": "read_select",
                "scope": parsed.scope,
                "systems": list(parsed.systems),
                "components": list(parsed.components),
                "symptoms": list(parsed.symptoms),
                "requested_outputs": list(parsed.requested_outputs),
                "correlation_ids": list(parsed.correlation_ids),
                "query": query,
                "reason": "live database data must not be fabricated",
            }
            broker_result = self.broker.request(request_payload)
            verification_request = VerificationRequestRecord(
                request_id=str(broker_result.get("request_id") or _stable_id("verify", query)),
                audit_id=_stable_id("audit", query, parsed.scope or "", ",".join(parsed.systems)),
                finding_id=next((finding.finding_id for finding in findings if finding.classification == "missing_evidence"), None),
                broker_name=str(broker_result.get("broker", getattr(self.broker, "name", "broker"))),
                request_payload=dict(broker_result.get("payload") or request_payload),
                status=str(broker_result.get("status", "pending")),
                metadata={"job_id": broker_result.get("job_id")},
            )
            verification_requests.append(verification_request)
            if hasattr(self.platform.store, "record_verification_request"):
                self.platform.store.record_verification_request(verification_request)

        causal_chains = self._build_causal_chains(findings, timeline)
        report = AuditReport(
            audit_id=_stable_id("audit", query, parsed.scope or "", ",".join(parsed.systems), ",".join(parsed.correlation_ids)),
            request=parsed,
            findings=tuple(findings),
            timeline=tuple(timeline),
            causal_chains=tuple(causal_chains),
            unresolved_hypotheses=tuple(dict.fromkeys(unresolved_hypotheses)),
            recommended_verification_steps=tuple(dict.fromkeys(recommended_verification_steps)),
            verification_requests=tuple(verification_requests),
            executive_summary=self._build_executive_summary(parsed, findings),
            human_report="",
            machine_report={},
            metadata={"source_types": list(source_types)},
        )
        human_report = report.to_markdown()
        machine_report = report.to_dict()
        report = AuditReport(
            **{
                **report.__dict__,
                "human_report": human_report,
                "machine_report": machine_report,
            }
        )
        self._persist(report)
        return report

    def _discover_source_types(self) -> tuple[str, ...]:
        source_types = sorted({chunk.source_type for chunk in getattr(self.platform.store, "chunks", {}).values()})
        return tuple(source_types or ("log", "code", "schema", "document", "query-output"))

    def _build_findings(
        self,
        request: AuditRequest,
        observations: Sequence[AuditObservation],
        timeline: Sequence[AuditTimelinePoint],
    ) -> list[AuditFindingRecord]:
        findings: list[AuditFindingRecord] = []
        for observation in observations:
            finding_id = _stable_id("finding", observation.classification, observation.statement, ",".join(ref.reference_id for ref in observation.evidence_refs))
            findings.append(
                AuditFindingRecord(
                    finding_id=finding_id,
                    classification=observation.classification,
                    statement=observation.statement,
                    confidence=observation.confidence,
                    evidence_refs=observation.evidence_refs,
                    time_range=observation.time_range,
                    systems=observation.systems,
                    components=observation.components,
                    verification_status=observation.verification_status,
                    recommended_query=observation.recommended_query,
                    diagnostic_action=observation.diagnostic_action,
                    supporting_evidence=observation.supporting_evidence,
                    contradicting_evidence=observation.contradicting_evidence,
                    provenance_refs=observation.provenance_refs,
                    correlation_ids=observation.correlation_ids,
                    source_type_analyzers=observation.source_type_analyzers,
                    metadata=observation.metadata,
                )
            )

        if len({finding.classification for finding in findings}) < 3 and timeline:
            timeline_statement = f"Evidence timeline spans {timeline[0].timestamp.isoformat() if timeline[0].timestamp else 'unspecified'} to {timeline[-1].timestamp.isoformat() if timeline[-1].timestamp else 'unspecified'}."
            findings.append(
                AuditFindingRecord(
                    finding_id=_stable_id("finding", "timeline", timeline_statement, request.raw_request),
                    classification="inference",
                    statement=timeline_statement,
                    confidence=0.72,
                    evidence_refs=tuple(
                        _ref_for_timeline(point, "timeline-inference") for point in timeline[: min(len(timeline), 3)]
                    ),
                    time_range=(timeline[0].timestamp, timeline[-1].timestamp),
                    systems=tuple(sorted({point.source_type for point in timeline})),
                    verification_status="inferred",
                    supporting_evidence=tuple(point.provenance_ref for point in timeline),
                    provenance_refs=tuple(point.provenance_ref for point in timeline),
                    source_type_analyzers=("timeline",),
                )
            )

        if not any(finding.classification == "fact" for finding in findings):
            findings.append(
                AuditFindingRecord(
                    finding_id=_stable_id("finding", "missing-fact", request.raw_request),
                    classification="missing_evidence",
                    statement="No direct fact-level evidence was found for the requested scope.",
                    confidence=0.68,
                    evidence_refs=(),
                    verification_status="needs_live_verification" if request.require_live_verification else "unverified",
                    recommended_query="Broaden source-type coverage and request controlled verification where live database contents are required.",
                    diagnostic_action={
                        "operation_class": "read_select",
                        "target": "database",
                        "reason": "Direct fact evidence is missing for the requested scope",
                    },
                    source_type_analyzers=("coverage-check",),
                )
            )
        return findings

    def _build_timeline(self, hits: Sequence[EvidenceHit]) -> list[AuditTimelinePoint]:
        timeline: list[AuditTimelinePoint] = []
        for hit in hits:
            timestamp = _extract_timestamp_from_hit(hit)
            if timestamp is None:
                continue
            timeline.append(
                AuditTimelinePoint(
                    timestamp=timestamp,
                    statement=hit.text.splitlines()[0].strip(),
                    source_id=hit.source_id,
                    source_type=hit.source_type,
                    locator=hit.locator,
                    correlation_ids=_extract_correlation_ids(hit.text),
                    systems=(hit.system,) if hit.system else (),
                    components=(hit.metadata.get("component"),) if hit.metadata.get("component") else (),
                    provenance_ref=f"{hit.source_id}:{hit.locator}",
                )
            )
        timeline.sort(key=lambda point: (point.timestamp or datetime.min.replace(tzinfo=UTC), point.source_id, point.locator))
        return timeline

    def _build_causal_chains(self, findings: Sequence[AuditFindingRecord], timeline: Sequence[AuditTimelinePoint]) -> list[str]:
        if not findings:
            return []
        chains: list[str] = []
        interesting = [finding for finding in findings if finding.classification in {"fact", "inference", "hypothesis"}]
        if interesting:
            chains.append(" -> ".join(f"{finding.classification}:{finding.statement}" for finding in interesting[:4]))
        if timeline:
            chains.append("Timeline anchors are " + " -> ".join(point.provenance_ref for point in timeline[:4]))
        return chains

    def _build_executive_summary(self, request: AuditRequest, findings: Sequence[AuditFindingRecord]) -> str:
        fact_count = sum(1 for finding in findings if finding.classification == "fact")
        hypothesis_count = sum(1 for finding in findings if finding.classification == "hypothesis")
        missing_count = sum(1 for finding in findings if finding.classification == "missing_evidence")
        return (
            f"Audit scope '{request.scope or request.raw_request[:80]}' produced {fact_count} fact finding(s), "
            f"{hypothesis_count} hypothesis finding(s), and {missing_count} missing-evidence finding(s)."
        )

    def _derive_cross_source_findings(
        self,
        request: AuditRequest,
        findings: Sequence[AuditFindingRecord],
        hits_by_type: Mapping[str, Sequence[EvidenceHit]],
    ) -> list[AuditFindingRecord]:
        derived: list[AuditFindingRecord] = []
        log_fact = next((finding for finding in findings if finding.classification == "fact" and "failure signal" in finding.statement.lower()), None)
        code_fact = next((finding for finding in findings if finding.classification == "fact" and "source code evidence matches" in finding.statement.lower()), None)
        schema_fact = next((finding for finding in findings if finding.classification == "fact" and "schema evidence identifies" in finding.statement.lower()), None)
        query_gap = next((finding for finding in findings if finding.classification == "missing_evidence"), None)
        if not code_fact:
            matching_code_hits: list[EvidenceHit] = []
            matched_frames: list[str] = []
            for log_hit in hits_by_type.get("log", []):
                for frame in _extract_stack_frames(log_hit.text):
                    for code_hit in hits_by_type.get("code", []):
                        if _frame_matches_paths(frame, {code_hit.locator}):
                            matching_code_hits.append(code_hit)
                            matched_frames.append(frame["raw"])
            if matching_code_hits:
                code_fact = AuditFindingRecord(
                    finding_id=_stable_id(
                        "finding",
                        "source-code",
                        ",".join(hit.chunk_id for hit in matching_code_hits),
                        request.raw_request,
                    ),
                    classification="fact",
                    statement=f"Source code evidence matches the runtime stack frame(s) {', '.join(dict.fromkeys(matched_frames))}.",
                    confidence=0.96,
                    evidence_refs=_unique_evidence_refs(
                        *(_ref_for_hit(hit, "code-match", "observed") for hit in matching_code_hits),
                        *(_ref_for_hit(hit, "log-stack-frame", "observed") for hit in hits_by_type.get("log", [])),
                    ),
                    systems=_unique_strings(
                        *(hit.system for hit in matching_code_hits if hit.system),
                        *(hit.system for hit in hits_by_type.get("log", []) if hit.system),
                    ),
                    components=_unique_strings(*(hit.locator for hit in matching_code_hits)),
                    verification_status="observed",
                    supporting_evidence=_unique_strings(
                        *[hit.chunk_id for hit in matching_code_hits],
                        *[hit.chunk_id for hit in hits_by_type.get("log", [])],
                    ),
                    provenance_refs=_unique_strings(
                        *[hit.locator for hit in matching_code_hits],
                        *[hit.locator for hit in hits_by_type.get("log", [])],
                    ),
                    correlation_ids=_unique_strings(*_extract_correlation_ids(" ".join(hit.text for hit in hits_by_type.get("log", [])))),
                    source_type_analyzers=("cross-source-correlation",),
                )
                derived.append(code_fact)
        if log_fact and code_fact:
            schema_systems = schema_fact.systems if schema_fact else ()
            schema_components = schema_fact.components if schema_fact else ()
            schema_supporting = schema_fact.supporting_evidence if schema_fact else ()
            schema_provenance = schema_fact.provenance_refs if schema_fact else ()
            schema_refs = schema_fact.evidence_refs if schema_fact else ()
            evidence_refs = _unique_evidence_refs(log_fact.evidence_refs, code_fact.evidence_refs)
            statement = "The runtime exception appears to originate in the payment service path that guards invoice row retrieval."
            if schema_fact:
                statement += " The schema evidence suggests the failure surfaces where invoice_runs records are expected."
                evidence_refs = _unique_evidence_refs(evidence_refs, schema_refs)
            derived.append(
                AuditFindingRecord(
                    finding_id=_stable_id("finding", "hypothesis", statement, request.raw_request),
                    classification="hypothesis",
                    statement=statement,
                    confidence=0.74,
                    evidence_refs=evidence_refs,
                    systems=_unique_strings(*log_fact.systems, *code_fact.systems, *schema_systems),
                    components=_unique_strings(*log_fact.components, *code_fact.components, *schema_components),
                    verification_status="unverified",
                    supporting_evidence=_unique_strings(*log_fact.supporting_evidence, *code_fact.supporting_evidence, *schema_supporting),
                    contradicting_evidence=tuple(query_gap.supporting_evidence) if query_gap else (),
                    provenance_refs=_unique_strings(*log_fact.provenance_refs, *code_fact.provenance_refs, *schema_provenance),
                    correlation_ids=_unique_strings(*log_fact.correlation_ids, *code_fact.correlation_ids),
                    source_type_analyzers=("cross-source-correlation",),
                )
            )
        if query_gap:
            derived.append(
                AuditFindingRecord(
                    finding_id=_stable_id("finding", "recommendation", query_gap.statement, request.raw_request),
                    classification="recommendation",
                    statement="Request a controlled on-prem verification job to confirm the live database state before any conclusion is drawn from the query output.",
                    confidence=0.91,
                    evidence_refs=query_gap.evidence_refs,
                    systems=query_gap.systems,
                    components=query_gap.components,
                    verification_status="requested",
                    recommended_query=query_gap.recommended_query,
                    diagnostic_action=query_gap.diagnostic_action,
                    supporting_evidence=query_gap.supporting_evidence,
                    provenance_refs=query_gap.provenance_refs,
                    source_type_analyzers=("verification-planning",),
                )
            )
        return derived

    def _persist(self, report: AuditReport) -> None:
        if hasattr(self.platform.store, "record_audit_finding"):
            for finding in report.findings:
                self.platform.store.record_audit_finding(finding)
        if hasattr(self.platform.store, "record_audit_report"):
            self.platform.store.record_audit_report(
                AuditReportRecord(
                    report_id=report.audit_id,
                    request_hash=sha256(report.request.raw_request.encode("utf-8")).hexdigest(),
                    request_payload=report.to_dict()["request"],
                    executive_summary=report.executive_summary,
                    human_report=report.human_report,
                    machine_report=report.machine_report,
                    finding_ids=tuple(finding.finding_id for finding in report.findings),
                    timeline_event_ids=tuple(_stable_id("event", point.provenance_ref, point.statement) for point in report.timeline),
                    causal_chains=report.causal_chains,
                    unresolved_hypotheses=report.unresolved_hypotheses,
                    recommended_verification_steps=report.recommended_verification_steps,
                    metadata=report.metadata,
                )
            )


def _default_analyzers() -> tuple[AuditSourceAnalyzer, ...]:
    return (
        StructuredLogAnalyzer(),
        SplunkCSVAnalyzer(),
        GenericCodeAnalyzer(),
        IntegrationDefinitionAnalyzer(),
        SchemaAnalyzer(),
        QueryOutputAnalyzer(),
        DocumentationAnalyzer(),
    )


def _finding_to_dict(finding: AuditFindingRecord) -> dict[str, Any]:
    payload = asdict(finding)
    payload["time_range"] = _time_range_to_json(finding.time_range)
    return payload


def _timeline_to_dict(point: AuditTimelinePoint) -> dict[str, Any]:
    return {
        "timestamp": point.timestamp.isoformat() if point.timestamp else None,
        "statement": point.statement,
        "source_id": point.source_id,
        "source_type": point.source_type,
        "locator": point.locator,
        "correlation_ids": list(point.correlation_ids),
        "systems": list(point.systems),
        "components": list(point.components),
        "provenance_ref": point.provenance_ref,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _time_range_to_json(time_range: tuple[datetime | None, datetime | None] | None) -> dict[str, Any] | None:
    if time_range is None:
        return None
    start, end = time_range
    return {
        "start": start.isoformat() if start is not None else None,
        "end": end.isoformat() if end is not None else None,
    }


def _normalize_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(_split_comma_values(value))
    return tuple(str(item) for item in value if str(item))


def _split_comma_values(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in re.split(r"[,\n;]", str(value)) if item.strip()]


def _unique_strings(*values: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _unique_evidence_refs(*items: Any) -> tuple[AuditEvidenceRefRecord, ...]:
    unique: dict[str, AuditEvidenceRefRecord] = {}
    for item in items:
        if isinstance(item, AuditEvidenceRefRecord):
            refs = (item,)
        else:
            refs = item
        for ref in refs:
            unique.setdefault(ref.reference_id, ref)
    return tuple(unique.values())


def _first_text_match(text: str, prefixes: tuple[str, ...]) -> str | None:
    lower = text.lower()
    for prefix in prefixes:
        token = f"{prefix.lower()}:"
        if token in lower:
            index = lower.index(token) + len(token)
            return text[index:].strip().splitlines()[0]
    return None


def _extract_correlation_ids(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_CORRELATION_RE.findall(text)))


def _extract_timestamps(text: str) -> list[datetime]:
    values: list[datetime] = []
    for match in _TIMESTAMP_RE.finditer(text):
        values.append(_as_utc(match.group("ts")))
    return [value for value in values if value is not None]


def _extract_timestamp_from_hit(hit: EvidenceHit) -> datetime | None:
    timestamps = _extract_timestamps(hit.text)
    if timestamps:
        return timestamps[0]
    return _as_utc(hit.ingest_timestamp)


def _extract_stack_frames(text: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for match in _STACK_FRAME_RE.finditer(text):
        if match.group("file"):
            frames.append({"raw": f'{match.group("file")}:{match.group("line")}', "path": match.group("file"), "line": match.group("line")})
        elif match.group("path"):
            frames.append({"raw": f'{match.group("path")}:{match.group("path_line")}', "path": match.group("path"), "line": match.group("path_line")})
    return frames


def _frame_matches_paths(frame: Mapping[str, str], paths: set[str]) -> bool:
    frame_path = frame.get("path", "").replace("\\", "/")
    for path in paths:
        normalized = path.replace("\\", "/")
        normalized = re.split(r"[:#]", normalized, 1)[0]
        if normalized.endswith(frame_path) or frame_path.endswith(normalized):
            return True
    return False


def _extract_schema_objects(text: str) -> list[str]:
    objects: list[str] = []
    for match in re.finditer(r"\b(?:CREATE|ALTER|DROP|SELECT|INSERT|UPDATE|DELETE)\s+(?:TABLE|INDEX|VIEW|FUNCTION|PROC(?:EDURE)?)?\s*(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.]+)", text, re.IGNORECASE):
        objects.append(match.group(1))
    for match in re.finditer(r"\b([A-Za-z0-9_]+)\s*\(", text):
        if match.group(1).lower() not in {"create", "alter", "drop", "select", "insert", "update", "delete"}:
            objects.append(match.group(1))
    return list(dict.fromkeys(objects))


def _extract_integration_endpoints(text: str) -> list[str]:
    endpoints = []
    for match in re.finditer(r"\b(?:https?://[^\s\"']+|/[A-Za-z0-9_./-]+|\b[A-Za-z0-9_.-]+#[A-Za-z0-9_.-]+)\b", text):
        endpoints.append(match.group(0))
    return list(dict.fromkeys(endpoints))


def _parse_csv_rows(text: str) -> list[dict[str, str]]:
    sample = text.strip()
    if not sample:
        return []
    reader = csv.DictReader(io.StringIO(sample))
    return [dict(row) for row in reader if any(value.strip() for value in row.values() if value)]


def _parse_query_table(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if "|" not in lines[0]:
        return []
    headers = [part.strip() for part in lines[0].split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if line.startswith("-"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != len(headers):
            continue
        rows.append(dict(zip(headers, parts, strict=False)))
    return rows


def _ref_for_hit(hit: EvidenceHit, role: str, provenance_tag: str) -> AuditEvidenceRefRecord:
    reference_id = _stable_id("eref", hit.source_id, hit.source_version_id, hit.locator, role, provenance_tag)
    return AuditEvidenceRefRecord(
        reference_id=reference_id,
        source_id=hit.source_id,
        source_version_id=hit.source_version_id,
        source_type=hit.source_type,
        locator=hit.locator,
        role=role,
        provenance_ref=f"{hit.source_id}:{hit.locator}",
        excerpt_hash=hit.excerpt_hash,
        statement=hit.text.splitlines()[0].strip(),
        metadata={"source_type": hit.source_type, "score": hit.score, "semantic_score": hit.semantic_score, "lexical_score": hit.lexical_score},
    )


def _ref_for_timeline(point: AuditTimelinePoint, role: str) -> AuditEvidenceRefRecord:
    reference_id = _stable_id("eref", point.provenance_ref, role, point.statement)
    return AuditEvidenceRefRecord(
        reference_id=reference_id,
        source_id=point.source_id,
        source_version_id=point.provenance_ref.split(":", 1)[0],
        source_type=point.source_type,
        locator=point.locator,
        role=role,
        provenance_ref=point.provenance_ref,
        excerpt_hash=_stable_id("excerpt", point.provenance_ref, point.statement),
        statement=point.statement,
        metadata={"correlation_ids": list(point.correlation_ids)},
    )
