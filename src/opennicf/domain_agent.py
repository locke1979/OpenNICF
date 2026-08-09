"""Domain-agent contracts built on the shared QwenAgent runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import PurePosixPath
import ast
import re
from typing import Any, Callable, Mapping

from .knowledge import AuditEvidenceRefRecord, EvidenceHit, KnowledgePlatform, RetrievalFilters, SearchCandidate
from .model_gateway import ModelGateway, PrivacyPolicy
from .qwen_adapter import OpenNICFChatModel
from .qwen_runtime import QwenAgentRuntime


DEFAULT_DOMAIN_IDS = (
    "criminal",
    "contraordenacional",
    "contencioso_administrativo",
    "contencioso_judicial",
    "encargos_sigef",
)

# Sanitized ownership from the NICF application map.  These identifiers are
# deliberately names only: runtime connection details remain in secret stores.
CONTENCIOSO_ADMINISTRATIVO_SYSTEMS = (
    "SICAT",
    "SICATPF",
    "SIGEPRA",
    "WSCAT",
    "ISIGEPRAWS",
    "JTRECLBAT",
)
CONTENCIOSO_ADMINISTRATIVO_DELEGATION_TARGETS = (
    "contencioso_judicial",
    "encargos_sigef",
)
CONTENCIOSO_JUDICIAL_DOMAIN_ID = "contencioso_judicial"
CONTENCIOSO_JUDICIAL_ALIASES = (
    "sicjut",
    "sicjutpf",
    "sicjutindbat",
    "cjtcaadws",
    "isicjutws",
    "wsaftaf",
    "wscexecf",
)
CONTENCIOSO_JUDICIAL_DELEGATED_DOMAINS = (
    "contencioso_administrativo",
    "encargos_sigef",
)
_CONTENCIOSO_JUDICIAL_DELEGATION_ALIASES = {
    "administrative": "contencioso_administrativo",
    "sigepra": "contencioso_administrativo",
    "sigef": "encargos_sigef",
    "encargos": "encargos_sigef",
}

# These are sanitized registry labels only.  They identify the application
# ownership boundary; connection details and deployment endpoints stay in the
# runtime configuration and are never part of a profile.
CRIMINAL_DOMAIN_ALIASES = (
    "criminal",
    "criminal-law",
    "penal",
    "sinquer",
    "sinquer-criminal",
    "criminal-sinquer",
)

GENERIC_RETRIEVAL_TOOL_NAMES = (
    "search_code_exact",
    "search_code_symbols",
    "search_code_semantic",
    "search_logs",
    "search_docs",
    "search_schema",
    "search_query_outputs",
)

LEGACY_RETRIEVAL_TOOL_NAMES = ("search_domain_evidence",)

_QUERY_ID_KEYS = ("correlation_ids", "request_ids", "session_ids")
_CODE_SOURCE_TYPES = ("code",)
_LOG_SOURCE_TYPES = ("log",)
_DOC_SOURCE_TYPES = ("document",)
_SCHEMA_SOURCE_TYPES = ("schema",)
_QUERY_OUTPUT_SOURCE_TYPES = ("query-output",)
_DEFAULT_HYBRID_SOURCE_TYPES = ("document", "log", "code", "schema", "query-output")
_TIMESTAMP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_IDENTIFIER_RE = re.compile(r"(?i)\b(?:correlation|request|session)[-_ ]?id\b\s*[:=]\s*([A-Za-z0-9._:-]+)")
_PY_CLASS_RE = re.compile(r"(?m)^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")
_PY_FUNCTION_RE = re.compile(r"(?m)^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
_SQL_OBJECT_RE = re.compile(r"(?i)\b(?:create\s+(?:table|view|index)|alter\s+table|insert\s+into|update|from|join)\s+([A-Za-z_][A-Za-z0-9_\.]*)")
_MARKDOWN_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+)$")


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(b"\0")
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:24]}"


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _normalize_terms(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(term for term in re.split(r"[\s,;]+", value.strip()) if term)


def _normalize_multi_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in re.split(r"[\s,;]+", value) if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize_request(request: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(request, str):
        return {"query": request}
    return dict(request)


def _source_path(source_uri: str) -> str:
    path = PurePosixPath(source_uri)
    return path.as_posix()


def _source_name_terms(source_uri: str) -> tuple[str, ...]:
    path = PurePosixPath(source_uri)
    terms = {path.name, path.stem}
    terms.update(part for part in path.parts if part not in {"", "."})
    return tuple(term for term in terms if term)


def _extract_symbol_names(text: str, source_uri: str) -> tuple[str, ...]:
    lowered_uri = source_uri.lower()
    symbols: set[str] = set(_source_name_terms(source_uri))
    if lowered_uri.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add(node.name)
                    if isinstance(node, ast.ClassDef):
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                symbols.add(child.name)
        symbols.update(_PY_CLASS_RE.findall(text))
        symbols.update(_PY_FUNCTION_RE.findall(text))
    elif lowered_uri.endswith(".sql"):
        symbols.update(match.group(1) for match in _SQL_OBJECT_RE.finditer(text))
    else:
        symbols.update(match.group(2) for match in _MARKDOWN_HEADING_RE.finditer(text))
    return tuple(sorted(symbols))


def _extract_log_ids(text: str, metadata: Mapping[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    metadata = metadata or {}
    correlation_ids = set(_normalize_multi_value(metadata.get("correlation_ids")))
    request_ids = set(_normalize_multi_value(metadata.get("request_ids")))
    session_ids = set(_normalize_multi_value(metadata.get("session_ids")))
    for match in _IDENTIFIER_RE.finditer(text):
        value = match.group(1)
        lowered = match.group(0).lower()
        if "correlation" in lowered:
            correlation_ids.add(value)
        elif "session" in lowered:
            session_ids.add(value)
        else:
            request_ids.add(value)
    return {
        "correlation_ids": tuple(sorted(correlation_ids)),
        "request_ids": tuple(sorted(request_ids)),
        "session_ids": tuple(sorted(session_ids)),
    }


def _extract_log_timestamps(text: str, metadata: Mapping[str, Any] | None = None) -> tuple[datetime, ...]:
    timestamps: list[datetime] = []
    for match in _TIMESTAMP_RE.finditer(text):
        value = match.group("ts")
        normalized = value.replace("Z", "+00:00")
        try:
            timestamps.append(datetime.fromisoformat(normalized).astimezone(timezone.utc))
        except ValueError:
            continue
    metadata = metadata or {}
    for value in _normalize_multi_value(metadata.get("timestamps")):
        normalized = value.replace("Z", "+00:00")
        try:
            timestamps.append(datetime.fromisoformat(normalized).astimezone(timezone.utc))
        except ValueError:
            continue
    return tuple(dict.fromkeys(timestamps))


def _term_score(query_terms: tuple[str, ...], haystack: str) -> float:
    if not query_terms:
        return 0.0
    lowered = haystack.lower()
    if all(term.lower() in lowered for term in query_terms):
        return 1.0
    matched = sum(1 for term in query_terms if term.lower() in lowered)
    return matched / len(query_terms)


@dataclass(frozen=True)
class DomainProfile:
    """Declarative configuration for one application domain."""

    domain_id: str
    name: str
    description: str
    owned_systems: tuple[str, ...] = ()
    # Optional hard system ACL.  Legacy profiles retain their domain-only
    # behavior; profiles with this field cannot widen retrieval by request.
    retrieval_system_ids: tuple[str, ...] = ()
    system_aliases: tuple[str, ...] = ()
    owned_components: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    delegated_domains: tuple[str, ...] = ()
    integration_boundary_aliases: tuple[str, ...] = ()
    permitted_evidence_classes: tuple[str, ...] = ()
    default_retrieval_filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    permitted_tools: tuple[str, ...] = (
        "search_code_exact",
        "search_code_symbols",
        "search_code_semantic",
        "search_logs",
        "search_docs",
        "search_schema",
        "search_query_outputs",
        "request_domain_diagnostic",
    )
    privacy_policy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED
    task_class: str = "simple_rag"
    system_prompt: str | None = None

    def normalized(self) -> "DomainProfile":
        filters = self.default_retrieval_filters.normalized()
        if self.domain_id:
            filters = replace(filters, domains=(self.domain_id,))
        return replace(self, default_retrieval_filters=filters)


def _profile(
    domain_id: str,
    *,
    name: str,
    description: str,
    owned_systems: tuple[str, ...] = (),
    retrieval_system_ids: tuple[str, ...] = (),
    system_aliases: tuple[str, ...] = (),
    owned_components: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    delegated_domains: tuple[str, ...] = (),
    integration_boundary_aliases: tuple[str, ...] = (),
    permitted_evidence_classes: tuple[str, ...] = (),
    permitted_tools: tuple[str, ...] = (
        "search_code_exact",
        "search_code_symbols",
        "search_code_semantic",
        "search_logs",
        "search_docs",
        "search_schema",
        "search_query_outputs",
        "request_domain_diagnostic",
    ),
    privacy_policy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED,
    task_class: str = "simple_rag",
    system_prompt: str | None = None,
    default_retrieval_filters: RetrievalFilters | None = None,
) -> DomainProfile:
    return DomainProfile(
        domain_id=domain_id,
        name=name,
        description=description,
        owned_systems=owned_systems,
        retrieval_system_ids=retrieval_system_ids,
        system_aliases=system_aliases,
        owned_components=owned_components,
        aliases=aliases,
        delegated_domains=delegated_domains,
        integration_boundary_aliases=integration_boundary_aliases,
        permitted_evidence_classes=permitted_evidence_classes,
        default_retrieval_filters=default_retrieval_filters or RetrievalFilters(),
        permitted_tools=permitted_tools,
        privacy_policy=privacy_policy,
        task_class=task_class,
        system_prompt=system_prompt,
    ).normalized()


DEFAULT_DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "criminal": _profile(
        "criminal",
        name="Criminal",
        description="Criminal-law matters and supporting evidence.",
        owned_systems=("criminal_casework", "criminal_evidence", "sinquer", "sinquer_criminal"),
        integration_boundary_aliases=CRIMINAL_DOMAIN_ALIASES[1:],
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the criminal domain agent. Stay inside the criminal evidence namespace.",
    ),
    "contraordenacional": _profile(
        "contraordenacional",
        name="Contraordenação",
        description="Administrative offense matters and supporting evidence.",
        owned_systems=("contraordenacional_casework", "contraordenacional_evidence"),
        integration_boundary_aliases=("administrative-offense", "administrative-penalty"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the contraordenacional domain agent. Stay inside the contraordenacional evidence namespace.",
    ),
    "contencioso_administrativo": _profile(
        "contencioso_administrativo",
        name="Contencioso Administrativo",
        description="Administrative litigation matters and supporting evidence.",
        owned_systems=CONTENCIOSO_ADMINISTRATIVO_SYSTEMS,
        retrieval_system_ids=CONTENCIOSO_ADMINISTRATIVO_SYSTEMS,
        system_aliases=(
            "sicat",
            "sicatpf",
            "sigepra",
            "wscat",
            "isigepraws",
            "jtreclbat",
        ),
        integration_boundary_aliases=("administrative-litigation", "contencioso-admin"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        permitted_tools=(
            "search_code_exact",
            "search_code_symbols",
            "search_code_semantic",
            "search_logs",
            "search_docs",
            "search_schema",
            "search_query_outputs",
            "request_domain_diagnostic",
            "request_cross_domain_collaboration",
        ),
        system_prompt="You are the contencioso administrativo domain agent. Stay inside the administrative-litigation namespace.",
    ),
    "contencioso_judicial": _profile(
        "contencioso_judicial",
        name="Contencioso Judicial",
        description="Judicial litigation matters and supporting evidence.",
        owned_systems=("SICJUT", "SICJUTPF", "SICJUTINDBAT"),
        owned_components=("CJTCAADWS", "ISICJUTWS", "WSAFTAF", "WSCEXECF"),
        aliases=CONTENCIOSO_JUDICIAL_ALIASES,
        delegated_domains=CONTENCIOSO_JUDICIAL_DELEGATED_DOMAINS,
        integration_boundary_aliases=("judicial-litigation", "SIGEPRA", "SIGEF", "Encargos"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        default_retrieval_filters=RetrievalFilters(
            principal_acl_scopes=frozenset({"internal"}),
            principal_domain_id=CONTENCIOSO_JUDICIAL_DOMAIN_ID,
            domains=(CONTENCIOSO_JUDICIAL_DOMAIN_ID,),
        ),
        system_prompt=(
            "You are the Contencioso Judicial domain agent. Stay inside the "
            "contencioso_judicial namespace. Use exact or symbol retrieval first; "
            "request Administrative or SIGEF evidence through coordinator delegation only."
        ),
    ),
    "encargos_sigef": _profile(
        "encargos_sigef",
        name="Encargos SIGEF",
        description="SIGEF assignment operations and supporting evidence.",
        owned_systems=("encargos_sigef_casework", "encargos_sigef_evidence"),
        integration_boundary_aliases=("sigef", "assignment-ops"),
        permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
        system_prompt="You are the encargos SIGEF domain agent. Stay inside the SIGEF evidence namespace.",
    ),
}


class DomainTools:
    """Domain-scoped tool surface that enforces the profile's domain boundary."""

    def __init__(self, profile: DomainProfile, platform: KnowledgePlatform, *, diagnostic_broker: Any | None = None):
        self.profile = profile
        self.platform = platform
        self.diagnostic_broker = diagnostic_broker

    def _normalize(self, request: str | Mapping[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
        payload = _normalize_request(request or {}) if request is not None else {}
        payload.update(overrides)
        payload.setdefault("query", "")
        payload.setdefault("limit", self.profile.default_retrieval_filters.limit)
        payload.setdefault("neighbor_window", self.profile.default_retrieval_filters.neighbor_window)
        return payload

    def _enforce_domain_boundary(self, payload: Mapping[str, Any], *, allow_delegation: bool = False) -> tuple[str, ...]:
        requested_principal = str(payload.get("principal_domain_id") or "").strip()
        if requested_principal and requested_principal != self.profile.domain_id:
            raise PermissionError("principal_domain_id cannot override the registered domain boundary")

        requested_domains = _normalize_multi_value(payload.get("domain_ids") or payload.get("domains"))
        delegated_domains = _normalize_multi_value(payload.get("delegated_domain_ids"))
        if requested_domains and any(domain != self.profile.domain_id for domain in requested_domains):
            if not allow_delegation:
                raise PermissionError("cross-domain retrieval requires the delegated interface")
        if delegated_domains and not allow_delegation:
            raise PermissionError("delegated domain retrieval requires the delegated interface")
        return delegated_domains

    def _filters(
        self,
        payload: Mapping[str, Any],
        *,
        source_types: tuple[str, ...],
        allow_delegation: bool = False,
    ) -> RetrievalFilters:
        delegated_domains = self._enforce_domain_boundary(payload, allow_delegation=allow_delegation)
        defaults = self.profile.default_retrieval_filters
        query_source_types = _normalize_multi_value(payload.get("source_types"))
        query_evidence_types = _normalize_multi_value(payload.get("evidence_types"))
        source_ids = _normalize_multi_value(payload.get("source_ids"))
        namespace_ids = _normalize_multi_value(payload.get("namespace_ids"))
        requested_system_ids = _normalize_multi_value(payload.get("system_ids"))
        if self.profile.retrieval_system_ids:
            owned = set(self.profile.retrieval_system_ids)
            aliases = {alias.lower(): system for alias, system in zip(self.profile.system_aliases, self.profile.retrieval_system_ids)}
            normalized_requested = tuple(aliases.get(system.lower(), system) for system in requested_system_ids)
            if normalized_requested and any(system not in owned for system in normalized_requested):
                raise PermissionError("requested system is outside the registered domain ACL")
            system_ids = normalized_requested or self.profile.retrieval_system_ids
        else:
            system_ids = requested_system_ids
        component_ids = _normalize_multi_value(payload.get("component_ids"))
        systems = _normalize_multi_value(payload.get("systems"))
        environments = _normalize_multi_value(payload.get("environments"))
        requested_acl_scopes = frozenset(_normalize_multi_value(payload.get("principal_acl_scopes")))
        if requested_acl_scopes and not requested_acl_scopes.issubset(defaults.principal_acl_scopes):
            raise PermissionError("principal_acl_scopes cannot widen the registered domain ACL")
        filters = replace(
            defaults,
            principal_acl_scopes=defaults.principal_acl_scopes,
            principal_domain_id=self.profile.domain_id,
            domain_ids=(self.profile.domain_id,),
            delegated_domain_ids=delegated_domains if allow_delegation else (),
            system_ids=system_ids or defaults.system_ids,
            component_ids=component_ids or defaults.component_ids,
            evidence_types=query_evidence_types or source_types or defaults.evidence_types,
            namespace_ids=namespace_ids or defaults.namespace_ids,
            domains=(self.profile.domain_id,),
            systems=systems or defaults.systems,
            environments=environments or defaults.environments,
            source_types=query_source_types or source_types or defaults.source_types,
            source_ids=source_ids or defaults.source_ids,
            since=payload.get("since") or defaults.since,
            until=payload.get("until") or defaults.until,
            limit=max(1, int(payload.get("limit") or defaults.limit)),
            neighbor_window=max(0, int(payload.get("neighbor_window") or defaults.neighbor_window)),
        )
        return filters.normalized()

    def _candidate_matches_terms(
        self,
        candidate: SearchCandidate,
        *,
        query: str,
        path: str | None = None,
        module: str | None = None,
        class_name: str | None = None,
        method_name: str | None = None,
        mode: str,
    ) -> bool:
        text = candidate.chunk.text
        metadata = dict(candidate.chunk.metadata)
        source_uri = candidate.source.source_uri
        haystack = " ".join(
            [
                text,
                candidate.chunk.locator,
                source_uri,
                metadata.get("module_name", ""),
                metadata.get("class_name", ""),
                metadata.get("method_name", ""),
                " ".join(_extract_symbol_names(text, source_uri)),
            ]
        ).lower()
        query_terms = _normalize_terms(query)
        if query and mode in {"exact", "symbol"} and not all(term.lower() in haystack for term in query_terms):
            return False
        if path:
            normalized_path = _source_path(path).lower()
            if normalized_path not in source_uri.lower() and normalized_path not in candidate.chunk.locator.lower():
                return False
        if module:
            module_terms = {module.lower(), PurePosixPath(source_uri).stem.lower()}
            if metadata.get("module_name"):
                module_terms.add(str(metadata["module_name"]).lower())
            if module.lower() not in module_terms:
                return False
        if class_name:
            symbols = {symbol.lower() for symbol in _extract_symbol_names(text, source_uri)}
            if class_name.lower() not in symbols and class_name.lower() not in haystack:
                return False
        if method_name:
            symbols = {symbol.lower() for symbol in _extract_symbol_names(text, source_uri)}
            if method_name.lower() not in symbols and method_name.lower() not in haystack:
                return False
        return True

    def _hit_from_candidate(
        self,
        candidate: SearchCandidate,
        *,
        query: str,
        retrieval_mode: str,
        score: float,
        lexical_score: float,
        semantic_score: float,
        match_kind: str,
        neighboring_chunk_ids: tuple[str, ...] = (),
    ) -> EvidenceHit:
        source_path = _source_path(candidate.source.source_uri)
        symbols = _extract_symbol_names(candidate.chunk.text, candidate.source.source_uri)
        log_ids = _extract_log_ids(candidate.chunk.text, candidate.chunk.metadata)
        log_timestamps = _extract_log_timestamps(candidate.chunk.text, candidate.chunk.metadata)
        metadata = {
            **dict(candidate.chunk.metadata),
            "match_kind": match_kind,
            "retrieval_mode": retrieval_mode,
            "query": query,
            "source_path": source_path,
            "symbol_names": list(symbols),
            "estimated_tokens": _estimate_tokens(candidate.chunk.text),
            "estimated_bytes": len(candidate.chunk.text.encode("utf-8")),
            **log_ids,
        }
        if log_timestamps:
            metadata["timestamps"] = [timestamp.isoformat() for timestamp in log_timestamps]
        return EvidenceHit(
            source_id=candidate.chunk.source_id,
            source_version_id=candidate.chunk.source_version_id,
            artifact_hash=candidate.chunk.artifact_hash,
            locator=candidate.chunk.locator,
            text=candidate.chunk.text,
            excerpt_hash=candidate.chunk.chunk_hash,
            ingest_timestamp=candidate.version.ingest_timestamp,
            parser_version=candidate.chunk.parser_version,
            namespace_id=candidate.chunk.namespace_id,
            domain_id=candidate.chunk.domain_id,
            system_id=candidate.chunk.system_id,
            component_id=candidate.chunk.component_id,
            environment=candidate.chunk.environment,
            evidence_type=candidate.chunk.evidence_type,
            acl_scope=candidate.chunk.acl_scope,
            source_type=candidate.chunk.source_type,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            score=score,
            model=self.platform.embeddings.info().model,
            dimensions=self.platform.embeddings.info().dimensions,
            chunk_id=candidate.chunk.chunk_id,
            chunk_ordinal=candidate.chunk.ordinal,
            domain=candidate.chunk.domain,
            system=candidate.chunk.system,
            neighboring_chunk_ids=neighboring_chunk_ids,
            metadata=metadata,
        )

    def _package_hits(self, query: str, retrieval_mode: str, hits: list[EvidenceHit]) -> dict[str, Any]:
        packages: list[dict[str, Any]] = []
        grouped: dict[str, list[EvidenceHit]] = {}
        for hit in hits:
            grouped.setdefault(hit.source_version_id, []).append(hit)
        for source_version_id, bucket in grouped.items():
            bucket.sort(key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))
            primary = bucket[0]
            evidence_refs = []
            for hit in bucket:
                evidence_refs.append(
                    asdict(
                        AuditEvidenceRefRecord(
                            reference_id=_stable_id("eref", hit.source_id, hit.source_version_id, hit.locator, retrieval_mode, query),
                            source_id=hit.source_id,
                            source_version_id=hit.source_version_id,
                            source_type=hit.source_type,
                            locator=hit.locator,
                            role="retrieval",
                            provenance_ref=f"{hit.source_id}:{hit.locator}",
                            excerpt_hash=hit.excerpt_hash,
                            statement=hit.text.splitlines()[0].strip() if hit.text.strip() else hit.locator,
                            metadata={
                                "retrieval_mode": retrieval_mode,
                                "score": hit.score,
                                "semantic_score": hit.semantic_score,
                                "lexical_score": hit.lexical_score,
                                "domain_id": hit.domain_id,
                                "namespace_id": hit.namespace_id,
                                "component_id": hit.component_id,
                                "system_id": hit.system_id,
                                "evidence_type": hit.evidence_type,
                                "correlation_ids": list(hit.metadata.get("correlation_ids", [])),
                                "request_ids": list(hit.metadata.get("request_ids", [])),
                                "session_ids": list(hit.metadata.get("session_ids", [])),
                                "timestamps": list(hit.metadata.get("timestamps", [])),
                                "estimated_tokens": hit.metadata.get("estimated_tokens", _estimate_tokens(hit.text)),
                                "estimated_bytes": hit.metadata.get("estimated_bytes", len(hit.text.encode("utf-8"))),
                                "match_kind": hit.metadata.get("match_kind", retrieval_mode),
                            },
                        )
                    )
                )
            estimated_tokens = sum(int(hit.metadata.get("estimated_tokens", _estimate_tokens(hit.text))) for hit in bucket)
            estimated_bytes = sum(int(hit.metadata.get("estimated_bytes", len(hit.text.encode("utf-8")))) for hit in bucket)
            packages.append(
                {
                    "package_id": _stable_id("pkg", query, retrieval_mode, source_version_id),
                    "query": query,
                    "retrieval_mode": retrieval_mode,
                    "principal_domain_id": self.profile.domain_id,
                    "source_id": primary.source_id,
                    "source_version_id": source_version_id,
                    "namespace_id": primary.namespace_id,
                    "domain_id": primary.domain_id,
                    "system_id": primary.system_id,
                    "component_id": primary.component_id,
                    "evidence_type": primary.evidence_type,
                    "source_type": primary.source_type,
                    "locator": primary.locator,
                    "score": primary.score,
                    "lexical_score": primary.lexical_score,
                    "semantic_score": primary.semantic_score,
                    "estimated_tokens": estimated_tokens,
                    "estimated_bytes": estimated_bytes,
                    "neighboring_chunk_ids": list(dict.fromkeys(chunk_id for hit in bucket for chunk_id in hit.neighboring_chunk_ids)),
                    "evidence_refs": evidence_refs,
                    "metadata": {
                        "domain_id": primary.domain_id,
                        "namespace_id": primary.namespace_id,
                        "system_id": primary.system_id,
                        "component_id": primary.component_id,
                        "evidence_type": primary.evidence_type,
                        "source_type": primary.source_type,
                        "integration_edges": list(primary.metadata.get("integration_edges", [])),
                    },
                }
            )
        packages.sort(key=lambda item: (-item["score"], item["source_version_id"], item["package_id"]))
        return {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "principal_domain_id": self.profile.domain_id,
            "package_count": len(packages),
            "estimated_tokens": sum(package["estimated_tokens"] for package in packages),
            "estimated_bytes": sum(package["estimated_bytes"] for package in packages),
            "packages": packages,
        }

    def _exact_search(self, request: str | Mapping[str, Any], *, source_types: tuple[str, ...], retrieval_mode: str) -> dict[str, Any]:
        payload = self._normalize(request)
        query = str(payload.get("query", "")).strip()
        path = payload.get("path") or payload.get("source_uri")
        module = payload.get("module") or payload.get("module_name")
        class_name = payload.get("class") or payload.get("class_name")
        method_name = payload.get("method") or payload.get("method_name")
        filters = self._filters(payload, source_types=source_types)
        candidates = self.platform.store.search_candidates(filters)
        query_terms = _normalize_terms(query)
        hits: list[EvidenceHit] = []
        for candidate in candidates:
            if not self._candidate_matches_terms(
                candidate,
                query=query,
                path=path,
                module=module,
                class_name=class_name,
                method_name=method_name,
                mode="exact",
            ):
                continue
            text = candidate.chunk.text
            source_uri = candidate.source.source_uri
            haystack = " ".join([text, candidate.chunk.locator, source_uri, " ".join(_extract_symbol_names(text, source_uri))]).lower()
            lexical_score = _term_score(query_terms, haystack)
            if query and not lexical_score:
                continue
            score = 1.0 if query and query.lower() in haystack else max(0.5, lexical_score)
            hits.append(
                self._hit_from_candidate(
                    candidate,
                    query=query,
                    retrieval_mode=retrieval_mode,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=0.0,
                    match_kind="exact",
                )
            )
        hits.sort(key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))
        limited = hits[: filters.limit]
        result = self._package_hits(query, retrieval_mode, limited)
        result["filters"] = asdict(filters)
        return result

    def _symbol_search(self, request: str | Mapping[str, Any], *, source_types: tuple[str, ...], retrieval_mode: str) -> dict[str, Any]:
        payload = self._normalize(request)
        query = str(payload.get("query", "")).strip()
        path = payload.get("path") or payload.get("source_uri")
        module = payload.get("module") or payload.get("module_name")
        class_name = payload.get("class") or payload.get("class_name")
        method_name = payload.get("method") or payload.get("method_name")
        filters = self._filters(payload, source_types=source_types)
        candidates = self.platform.store.search_candidates(filters)
        query_terms = _normalize_terms(query)
        hits: list[EvidenceHit] = []
        for candidate in candidates:
            if not self._candidate_matches_terms(
                candidate,
                query=query,
                path=path,
                module=module,
                class_name=class_name,
                method_name=method_name,
                mode="symbol",
            ):
                continue
            symbols = {symbol.lower() for symbol in _extract_symbol_names(candidate.chunk.text, candidate.source.source_uri)}
            haystack = " ".join([candidate.chunk.locator, candidate.source.source_uri, " ".join(symbols)]).lower()
            lexical_score = _term_score(query_terms, haystack)
            symbol_score = 1.0 if query and query.lower() in symbols else max(lexical_score, 0.75 if not query or query_terms else 0.0)
            if not symbol_score:
                continue
            hits.append(
                self._hit_from_candidate(
                    candidate,
                    query=query,
                    retrieval_mode=retrieval_mode,
                    score=symbol_score,
                    lexical_score=lexical_score,
                    semantic_score=0.0,
                    match_kind="symbol",
                )
            )
        hits.sort(key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))
        limited = hits[: filters.limit]
        result = self._package_hits(query, retrieval_mode, limited)
        result["filters"] = asdict(filters)
        return result

    def _semantic_search(self, request: str | Mapping[str, Any], *, source_types: tuple[str, ...], retrieval_mode: str) -> dict[str, Any]:
        payload = self._normalize(request)
        query = str(payload.get("query", "")).strip()
        filters = self._filters(payload, source_types=source_types)
        hits = self.platform.search(query or " ", filters=filters, route="local")
        packages = self._package_hits(query, retrieval_mode, hits[: filters.limit])
        packages["filters"] = asdict(filters)
        return packages

    def _log_search(self, request: str | Mapping[str, Any], *, retrieval_mode: str) -> dict[str, Any]:
        payload = self._normalize(request)
        query = str(payload.get("query", "")).strip()
        filter_payload = dict(payload)
        filter_payload.pop("since", None)
        filter_payload.pop("until", None)
        filters = self._filters(filter_payload, source_types=_LOG_SOURCE_TYPES)
        candidates = self.platform.store.search_candidates(filters)
        query_terms = _normalize_terms(query)
        requested_correlation_ids = {value.lower() for value in _normalize_multi_value(payload.get("correlation_ids"))}
        requested_request_ids = {value.lower() for value in _normalize_multi_value(payload.get("request_ids"))}
        requested_session_ids = {value.lower() for value in _normalize_multi_value(payload.get("session_ids"))}
        since = payload.get("since")
        until = payload.get("until")
        if isinstance(since, str):
            since = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if isinstance(until, str):
            until = datetime.fromisoformat(until.replace("Z", "+00:00"))
        hits: list[EvidenceHit] = []
        for candidate in candidates:
            if candidate.chunk.evidence_type != "log" and candidate.chunk.source_type != "log":
                continue
            log_ids = _extract_log_ids(candidate.chunk.text, candidate.chunk.metadata)
            candidate_timestamps = _extract_log_timestamps(candidate.chunk.text, candidate.chunk.metadata)
            if requested_correlation_ids and not requested_correlation_ids.intersection({item.lower() for item in log_ids["correlation_ids"]}):
                continue
            if requested_request_ids and not requested_request_ids.intersection({item.lower() for item in log_ids["request_ids"]}):
                continue
            if requested_session_ids and not requested_session_ids.intersection({item.lower() for item in log_ids["session_ids"]}):
                continue
            if since and candidate_timestamps and all(timestamp < since for timestamp in candidate_timestamps):
                continue
            if until and candidate_timestamps and all(timestamp > until for timestamp in candidate_timestamps):
                continue
            text = candidate.chunk.text
            haystack = " ".join([text, candidate.chunk.locator, candidate.source.source_uri]).lower()
            lexical_score = _term_score(query_terms, haystack)
            structured_boost = 0.0
            if requested_correlation_ids:
                structured_boost += 0.5
            if requested_request_ids:
                structured_boost += 0.5
            if requested_session_ids:
                structured_boost += 0.5
            if candidate_timestamps and (since or until):
                structured_boost += 0.25
            if query and not lexical_score and structured_boost == 0.0:
                continue
            score = max(lexical_score, structured_boost)
            if not query and score == 0.0:
                score = 0.25
            hits.append(
                self._hit_from_candidate(
                    candidate,
                    query=query,
                    retrieval_mode=retrieval_mode,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=0.0,
                    match_kind="log",
                )
            )
        hits.sort(key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))
        limited = hits[: filters.limit]
        result = self._package_hits(query, retrieval_mode, limited)
        result["filters"] = asdict(filters)
        return result

    def search_domain_evidence(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._semantic_search(payload, source_types=_DEFAULT_HYBRID_SOURCE_TYPES, retrieval_mode="domain_evidence")

    def search_code_exact(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._exact_search(payload, source_types=_CODE_SOURCE_TYPES, retrieval_mode="code_exact")

    def search_code_symbols(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._symbol_search(payload, source_types=_CODE_SOURCE_TYPES, retrieval_mode="code_symbols")

    def search_code_semantic(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._semantic_search(payload, source_types=_CODE_SOURCE_TYPES, retrieval_mode="code_semantic")

    def search_logs(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._log_search(payload, retrieval_mode="logs")

    def search_docs(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._semantic_search(payload, source_types=_DOC_SOURCE_TYPES, retrieval_mode="docs")

    def search_schema(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._semantic_search(payload, source_types=_SCHEMA_SOURCE_TYPES, retrieval_mode="schema")

    def search_query_outputs(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        return self._semantic_search(payload, source_types=_QUERY_OUTPUT_SOURCE_TYPES, retrieval_mode="query_outputs")

    def search_delegated_domain_evidence(self, request: str | Mapping[str, Any], *, limit: int | None = None) -> dict[str, Any]:
        payload = self._normalize(request)
        if limit is not None:
            payload["limit"] = limit
        if self.diagnostic_broker is None:
            raise NotImplementedError("cross-domain coordinator is not implemented")
        requested_delegated_domains = _normalize_multi_value(payload.get("delegated_domain_ids"))
        delegated_domains = tuple(
            _CONTENCIOSO_JUDICIAL_DELEGATION_ALIASES.get(domain.lower(), domain)
            for domain in requested_delegated_domains
        )
        if not delegated_domains:
            raise PermissionError("delegated_domain_ids are required for cross-domain retrieval")
        if not set(delegated_domains).issubset(self.profile.delegated_domains):
            raise PermissionError("delegation is limited to the registered Administrative and SIGEF boundaries")
        payload["delegated_domain_ids"] = list(delegated_domains)
        payload["allow_delegation"] = True
        delegated = dict(self.diagnostic_broker.search(payload))
        delegated.setdefault("evidence_refs", [])
        delegated["cross_domain_required"] = True
        delegated["delegation"] = {
            "source_domain_id": self.profile.domain_id,
            "target_domain_ids": list(delegated_domains),
            "coordinator": "diagnostic_broker",
        }
        delegated["suspected_edge"] = {
            "source_domain_id": self.profile.domain_id,
            "target_domain_ids": list(delegated_domains),
            "relation_type": "delegates-to",
            "evidence_refs": delegated["evidence_refs"],
        }
        return delegated

    def request_domain_diagnostic(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["domain_id"] = self.profile.domain_id
        if payload.get("operation_class") not in {"read_select", "describe"}:
            raise PermissionError("diagnostic operation is not allow-listed")
        if self.diagnostic_broker is not None:
            return dict(self.diagnostic_broker.request(payload))
        return {
            "status": "pending",
            "job_id": "broker-generated",
            "domain_id": self.profile.domain_id,
            "request": payload,
        }

    def request_delegated_domain_diagnostic(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        delegated_domains = _normalize_multi_value(payload.get("delegated_domain_ids"))
        if not delegated_domains:
            raise PermissionError("delegated_domain_ids are required for cross-domain diagnostics")
        if self.diagnostic_broker is None:
            raise NotImplementedError("cross-domain coordinator is not implemented")
        payload["domain_id"] = self.profile.domain_id
        return dict(self.diagnostic_broker.request(payload))

    def request_cross_domain_collaboration(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return a coordinator request without widening this agent's ACL."""
        payload = dict(request)
        target = str(payload.get("target_domain_id") or "").strip()
        if target not in CONTENCIOSO_ADMINISTRATIVO_DELEGATION_TARGETS:
            raise PermissionError("delegation target is not approved for this domain")
        evidence_refs = payload.get("evidence_refs", ())
        if not isinstance(evidence_refs, (list, tuple)) or not evidence_refs:
            raise ValueError("evidence_refs are required for cross-domain collaboration")
        return {
            "status": "cross_domain_required",
            "source_domain_id": self.profile.domain_id,
            "target_domain_id": target,
            "integration_edge": payload.get("integration_edge"),
            "evidence_refs": list(evidence_refs),
            "query": str(payload.get("query") or "").strip(),
        }

    def as_qwen_tools(self) -> dict[str, Callable[..., Any]]:
        return {
            "search_code_exact": self.search_code_exact,
            "search_code_symbols": self.search_code_symbols,
            "search_code_semantic": self.search_code_semantic,
            "search_logs": self.search_logs,
            "search_docs": self.search_docs,
            "search_schema": self.search_schema,
            "search_query_outputs": self.search_query_outputs,
            "search_domain_evidence": self.search_domain_evidence,
            "request_domain_diagnostic": self.request_domain_diagnostic,
            "search_delegated_domain_evidence": self.search_delegated_domain_evidence,
            "request_delegated_domain_diagnostic": self.request_delegated_domain_diagnostic,
            "request_cross_domain_collaboration": self.request_cross_domain_collaboration,
        }

    def permitted_qwen_tools(self) -> dict[str, Callable[..., Any]]:
        registry = self.as_qwen_tools()
        selected: dict[str, Callable[..., Any]] = {}
        missing = [tool_name for tool_name in self.profile.permitted_tools if tool_name not in registry]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"unknown permitted tool(s) for {self.profile.domain_id}: {missing_list}")
        for tool_name in self.profile.permitted_tools:
            selected[tool_name] = registry[tool_name]
        return selected


class DomainAgent:
    """QwenAgent-backed runtime agent scoped to one declarative domain profile."""

    def __init__(
        self,
        profile: DomainProfile,
        *,
        gateway: ModelGateway | None = None,
        knowledge: KnowledgePlatform | None = None,
        diagnostic_broker: Any | None = None,
        runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
    ) -> None:
        self.profile = profile.normalized()
        self.gateway = gateway or ModelGateway.from_env()
        self.knowledge = knowledge or KnowledgePlatform.in_memory()
        self.model_adapter = OpenNICFChatModel(
            gateway=self.gateway,
            task_class=self.profile.task_class,
            privacy=self.profile.privacy_policy,
        )
        self.tools = DomainTools(self.profile, self.knowledge, diagnostic_broker=diagnostic_broker)
        self._runtime_factory = runtime_factory or QwenAgentRuntime
        self._runtime: Any | None = None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_factory(self.model_adapter, self.tools.permitted_qwen_tools())
        return self._runtime

    @property
    def agent(self) -> Any:
        return self.runtime

    def run(self, request: str) -> Any:
        return self.runtime.run(request)


class ContenciosoAdministrativoDomainAgent(DomainAgent):
    """Shared-factory agent scoped to Administrative Litigation evidence."""


class CriminalDomainAgent(DomainAgent):
    """Criminal/SINQUER specialization over the shared DomainAgent runtime.

    The specialization contains domain policy and response shaping only.  It
    deliberately reuses ``DomainTools`` and the QwenAgent runtime supplied by
    ``DomainAgentFactory``; it does not introduce another orchestration loop.
    """

    domain_id = "criminal"
    aliases = CRIMINAL_DOMAIN_ALIASES

    def __init__(self, profile: DomainProfile, **kwargs: Any) -> None:
        if profile.domain_id != self.domain_id:
            raise ValueError("CriminalDomainAgent requires the criminal profile")
        super().__init__(profile, **kwargs)

    @staticmethod
    def _cross_domain_edge(package: Mapping[str, Any]) -> dict[str, Any] | None:
        for edge in package.get("metadata", {}).get("integration_edges", ()):
            if isinstance(edge, Mapping):
                target = str(edge.get("target_domain_id") or edge.get("target_domain") or "").strip()
                if target and target != "criminal":
                    return dict(edge)
        return None

    def retrieve(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        """Retrieve criminal evidence or return an explicit domain handoff."""
        payload = _normalize_request(request)
        query = str(payload.get("query") or "").strip()
        # Known identifiers take the deterministic code path before semantic
        # retrieval, keeping broad matching from obscuring ownership edges.
        if payload.get("retrieval_mode") in {"exact", "symbol"} or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", query):
            result = self.tools.search_code_exact(payload)
        else:
            result = self.tools.search_domain_evidence(payload)
        for package in result.get("packages", ()):
            edge = self._cross_domain_edge(package)
            if edge is not None:
                return {
                    "status": "cross_domain_required",
                    "cross_domain_required": True,
                    "principal_domain_id": self.domain_id,
                    "requested_domain_id": edge.get("target_domain_id") or edge.get("target_domain"),
                    "integration_edge": edge,
                    "evidence": package.get("evidence_refs", []),
                }
        result["cross_domain_required"] = False
        return result

    def run_failure_audit(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        """Return a bounded, provenance-preserving audit package.

        This is intentionally a result adapter for QwenAgent tools, not an
        agent loop.  Classification remains explicit so the model cannot turn
        an unverified observation into a fact without evidence.
        """
        result = self.retrieve(request)
        if result.get("cross_domain_required"):
            return result
        evidence = [ref for package in result.get("packages", ()) for ref in package.get("evidence_refs", ())]
        findings = [
            {
                "classification": "fact",
                "statement": f"Criminal-domain evidence matched: {ref.get('statement', ref.get('locator', 'evidence'))}",
                "confidence": 1.0,
                "provenance_ref": ref.get("provenance_ref"),
                "evidence_ref": ref,
            }
            for ref in evidence
        ]
        return {
            "status": "ok",
            "principal_domain_id": self.domain_id,
            "cross_domain_required": False,
            "findings": findings,
            "evidence": evidence,
        }

    def request_live_verification(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Use only the allow-listed diagnostic broker interface."""
        return self.tools.request_domain_diagnostic(request)
class ContenciosoJudicialDomainAgent(DomainAgent):
    """Judicial domain facade created by the shared factory."""

    pass


class DomainAgentFactory:
    """Factory that builds isolated domain agents over shared runtime services."""

    def __init__(
        self,
        *,
        gateway: ModelGateway | None = None,
        knowledge: KnowledgePlatform | None = None,
        diagnostic_broker: Any | None = None,
        profiles: Mapping[str, DomainProfile] | None = None,
        runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
    ) -> None:
        self.gateway = gateway or ModelGateway.from_env()
        self.knowledge = knowledge or KnowledgePlatform.in_memory()
        self.diagnostic_broker = diagnostic_broker
        self.profiles = dict(profiles or DEFAULT_DOMAIN_PROFILES)
        self._runtime_factory = runtime_factory or QwenAgentRuntime

    def available_domains(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def profile_for(self, domain_id: str) -> DomainProfile:
        normalized = str(domain_id).strip().lower().replace("_", "-")
        for candidate_id, profile in self.profiles.items():
            aliases = tuple(
                str(alias).lower().replace("_", "-")
                for alias in (
                    *getattr(profile, "aliases", ()),
                    *getattr(profile, "system_aliases", ()),
                    *profile.integration_boundary_aliases,
                    *profile.owned_systems,
                )
            )
            if normalized == candidate_id.replace("_", "-") or normalized in aliases:
                return profile
        try:
            return self.profiles[lookup]
        except KeyError as exc:
            raise KeyError(f"unknown domain_id: {domain_id}") from exc

    def resolve_domain_id(self, domain_id: str) -> str:
        return self.profile_for(domain_id).domain_id

    def create(self, domain_id: str) -> DomainAgent:
        resolved_domain_id = self.domain_for_alias(domain_id)
        profile = self.profile_for(resolved_domain_id)
        agent_class = (
            ContenciosoAdministrativoDomainAgent
            if resolved_domain_id == "contencioso_administrativo"
            else CriminalDomainAgent
            if resolved_domain_id == "criminal"
            else ContenciosoJudicialDomainAgent
            if resolved_domain_id == CONTENCIOSO_JUDICIAL_DOMAIN_ID
            else DomainAgent
        )
        if resolved_domain_id == "encargos_sigef":
            # Keep SIGEF on the common factory and shared runtime while using
            # its specialized profile and boundary tools.
            from .encargos_sigef import ENCARGOS_SIGEF_PROFILE, EncargosSigefDomainAgent

            agent_class = EncargosSigefDomainAgent
            profile = ENCARGOS_SIGEF_PROFILE
        return agent_class(
            profile,
            gateway=self.gateway,
            knowledge=self.knowledge,
            diagnostic_broker=self.diagnostic_broker,
            runtime_factory=self._runtime_factory,
        )

    def domain_for_alias(self, value: str) -> str:
        candidate = str(value).strip()
        if candidate in self.profiles:
            return candidate
        lowered = candidate.lower()
        for domain_id, profile in self.profiles.items():
            aliases = (*profile.system_aliases, *profile.integration_boundary_aliases, *profile.owned_systems)
            if lowered in {str(alias).lower() for alias in aliases}:
                return domain_id
        from .encargos_sigef import SIGEF_SYSTEM_ALIASES

        if lowered in {alias.lower() for aliases in SIGEF_SYSTEM_ALIASES.values() for alias in aliases}:
            return "encargos_sigef"
        raise KeyError(f"unknown domain or system alias: {value}")


def create_domain_agent_factory(
    *,
    gateway: ModelGateway | None = None,
    knowledge: KnowledgePlatform | None = None,
    diagnostic_broker: Any | None = None,
    profiles: Mapping[str, DomainProfile] | None = None,
    runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
) -> DomainAgentFactory:
    return DomainAgentFactory(
        gateway=gateway,
        knowledge=knowledge,
        diagnostic_broker=diagnostic_broker,
        profiles=profiles,
        runtime_factory=runtime_factory,
    )


def create_contencioso_administrativo_agent(
    *,
    gateway: ModelGateway | None = None,
    knowledge: KnowledgePlatform | None = None,
    diagnostic_broker: Any | None = None,
    runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
) -> ContenciosoAdministrativoDomainAgent:
    """Create the Administrative agent through the shared domain factory."""
    agent = create_domain_agent_factory(
        gateway=gateway,
        knowledge=knowledge,
        diagnostic_broker=diagnostic_broker,
        runtime_factory=runtime_factory,
    ).create("contencioso_administrativo")
    assert isinstance(agent, ContenciosoAdministrativoDomainAgent)
    return agent
