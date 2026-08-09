"""Namespace classification, registry records, and sanitization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
import re


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


_SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "client_secret",
    "private_key",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "authorization",
    "endpoint",
    "uri",
    "url",
}

_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b("
    r"api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token|client[-_ ]?secret|"
    r"private[-_ ]?key|password|passwd|secret|token|credential"
    r")\b(\s*[:=]\s*)([^\s,\"']+)"
)

_BEARER_PATTERN = re.compile(r"(?i)\bAuthorization(\s*:\s*Bearer\s+)([^\s,\"']+)")

_PRIVATE_HOST_PATTERNS = (
    "localhost",
    "127.",
    "0.0.0.0",
    "10.",
    "192.168.",
    "172.16.",
    "172.17.",
    "172.18.",
    "172.19.",
    "172.20.",
    "172.21.",
    "172.22.",
    "172.23.",
    "172.24.",
    "172.25.",
    "172.26.",
    "172.27.",
    "172.28.",
    "172.29.",
    "172.30.",
    "172.31.",
)

_PRIVATE_ENDPOINT_PATTERN = re.compile(
    r"(?i)\bhttps?://(?:[^/\s:@]+(?::[^/\s:@]+)?@)?"
    r"(?:(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})"
    r"|(?:[A-Za-z0-9-]+\.)*(?:internal|local|corp|lan))"
    r"(?::\d+)?(?:/[^\s\"']*)?"
)

DEFAULT_DOMAIN_NAMESPACE_HINTS: dict[str, dict[str, str]] = {
    "criminal": {
        "system_id": "criminal_casework",
        "component_id": "criminal_evidence",
        "environment": "prod",
        "evidence_type": "document",
        "acl_scope": "internal",
    },
    "contraordenacional": {
        "system_id": "contraordenacional_casework",
        "component_id": "contraordenacional_evidence",
        "environment": "prod",
        "evidence_type": "document",
        "acl_scope": "internal",
    },
    "contencioso_administrativo": {
        "system_id": "contencioso_administrativo_casework",
        "component_id": "contencioso_administrativo_evidence",
        "environment": "prod",
        "evidence_type": "document",
        "acl_scope": "internal",
    },
    "contencioso_judicial": {
        "system_id": "contencioso_judicial_casework",
        "component_id": "contencioso_judicial_evidence",
        "environment": "prod",
        "evidence_type": "document",
        "acl_scope": "internal",
    },
    "encargos_sigef": {
        "system_id": "encargos_sigef_casework",
        "component_id": "encargos_sigef_evidence",
        "environment": "prod",
        "evidence_type": "document",
        "acl_scope": "internal",
    },
}


def _clean(value: Any, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(b"\0")
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_private_endpoint(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        split = urlsplit(value)
        host = (split.hostname or "").lower()
        if not host:
            return True
        if host.startswith(_PRIVATE_HOST_PATTERNS):
            return True
        if host.endswith((".internal", ".local", ".corp", ".lan")):
            return True
        if split.username or split.password:
            return True
        return False
    return any(pattern in lowered for pattern in _PRIVATE_HOST_PATTERNS) or lowered.endswith((".internal", ".local", ".corp", ".lan"))


def _redact_text(text: str, secrets: Iterable[str]) -> str:
    redacted = text
    for secret in sorted({secret for secret in secrets if secret}, key=len, reverse=True):
        redacted = redacted.replace(secret, "[redacted]")
    redacted = _CREDENTIAL_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", redacted)
    redacted = _BEARER_PATTERN.sub(lambda match: f"Authorization{match.group(1)}[redacted]", redacted)
    redacted = _PRIVATE_ENDPOINT_PATTERN.sub("[redacted]", redacted)
    return redacted


def _sanitize_value(value: Any, *, path: tuple[str, ...], redacted_values: list[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _sanitize_value(item, path=(*path, str(key)), redacted_values=redacted_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, path=path, redacted_values=redacted_values) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item, path=path, redacted_values=redacted_values) for item in value)
    if isinstance(value, set):
        return sorted(_sanitize_value(item, path=path, redacted_values=redacted_values) for item in value)
    if isinstance(value, str):
        key_name = path[-1].lower() if path else ""
        if any(token in key_name for token in _SENSITIVE_KEY_NAMES):
            redacted_values.append(value)
            return "[redacted]"
        if any(token in key_name for token in {"endpoint", "uri", "url"}) and _is_private_endpoint(value):
            redacted_values.append(value)
            return "[redacted]"
        return value
    return value


def sanitize_metadata(metadata: Mapping[str, Any] | None) -> tuple[dict[str, Any], tuple[str, ...]]:
    redacted_values: list[str] = []
    sanitized = _sanitize_value(dict(metadata or {}), path=(), redacted_values=redacted_values)
    ordered = tuple(dict.fromkeys(redacted_values))
    if ordered:
        sanitized = dict(sanitized)
        sanitized["redacted_fields"] = list(ordered)
    return sanitized, ordered


def sanitize_text(text: str, secrets: Iterable[str] = ()) -> str:
    return _redact_text(text, secrets)


@dataclass(frozen=True)
class KnowledgeNamespaceRecord:
    namespace_id: str
    domain_id: str
    system_id: str
    component_id: str
    environment: str
    evidence_type: str
    acl_scope: str
    asset_name: str
    component_name: str
    sanitized_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class IntegrationEdgeRecord:
    edge_id: str
    source_namespace_id: str
    target_namespace_id: str
    relation_type: str
    domain_id: str
    system_id: str
    component_id: str
    environment: str
    evidence_type: str
    acl_scope: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)


def build_namespace_record(
    *,
    domain_id: str,
    system_id: str | None = None,
    component_id: str | None = None,
    environment: str | None = None,
    evidence_type: str | None = None,
    acl_scope: str | None = None,
    asset_name: str | None = None,
    component_name: str | None = None,
    summary: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_uri: str | None = None,
    source_kind: str | None = None,
    source_type: str | None = None,
) -> KnowledgeNamespaceRecord:
    fallback = DEFAULT_DOMAIN_NAMESPACE_HINTS.get(domain_id, {})
    sanitized_metadata, redacted_values = sanitize_metadata(metadata)
    system_id = _clean(system_id or fallback.get("system_id"), fallback="unknown-system")
    component_id = _clean(component_id or fallback.get("component_id"), fallback="unknown-component")
    environment = _clean(environment or fallback.get("environment"), fallback="unknown")
    evidence_type = _clean(evidence_type or source_type or fallback.get("evidence_type") or source_kind, fallback="document")
    acl_scope = _clean(acl_scope or fallback.get("acl_scope"), fallback="internal")
    asset_name_value = sanitize_text(asset_name, redacted_values) if asset_name else sanitized_metadata.get("asset_name")
    if isinstance(asset_name_value, str):
        asset_name_value = sanitize_text(asset_name_value, redacted_values)
    component_name_value = sanitize_text(component_name, redacted_values) if component_name else sanitized_metadata.get("component_name")
    if isinstance(component_name_value, str):
        component_name_value = sanitize_text(component_name_value, redacted_values)
    asset_name = _clean(asset_name_value or domain_id, fallback=domain_id)
    component_name = _clean(component_name_value or component_id, fallback=component_id)
    if summary:
        summary_value = sanitize_text(summary, redacted_values)
    else:
        summary_value = sanitized_metadata.get("namespace_summary") or sanitized_metadata.get("summary")
        if isinstance(summary_value, str):
            summary_value = sanitize_text(summary_value, redacted_values)
    summary = _clean(
        summary_value or f"{domain_id}/{system_id}/{component_id} [{evidence_type}] in {environment}",
        fallback=domain_id,
    )
    if source_uri:
        summary = f"{summary} :: {sanitize_text(source_uri, redacted_values)}"
    if redacted_values and "redacted_fields" not in sanitized_metadata:
        sanitized_metadata = dict(sanitized_metadata)
        sanitized_metadata["redacted_fields"] = list(redacted_values)
    namespace_id = f"ns_{_stable_id(domain_id, system_id, component_id, environment, evidence_type, acl_scope)[:32]}"
    return KnowledgeNamespaceRecord(
        namespace_id=namespace_id,
        domain_id=domain_id,
        system_id=system_id,
        component_id=component_id,
        environment=environment,
        evidence_type=evidence_type,
        acl_scope=acl_scope,
        asset_name=asset_name,
        component_name=component_name,
        sanitized_summary=summary,
        metadata=dict(sanitized_metadata),
    )


def build_integration_edge_record(
    *,
    source_namespace: KnowledgeNamespaceRecord,
    target_namespace: KnowledgeNamespaceRecord,
    relation_type: str,
    metadata: Mapping[str, Any] | None = None,
) -> IntegrationEdgeRecord:
    sanitized_metadata, _ = sanitize_metadata(metadata)
    edge_id = f"edge_{_stable_id(source_namespace.namespace_id, target_namespace.namespace_id, relation_type)[:32]}"
    return IntegrationEdgeRecord(
        edge_id=edge_id,
        source_namespace_id=source_namespace.namespace_id,
        target_namespace_id=target_namespace.namespace_id,
        relation_type=_clean(relation_type, fallback="integrates-with"),
        domain_id=source_namespace.domain_id,
        system_id=source_namespace.system_id,
        component_id=source_namespace.component_id,
        environment=source_namespace.environment,
        evidence_type=source_namespace.evidence_type,
        acl_scope=source_namespace.acl_scope,
        metadata=dict(sanitized_metadata),
    )


def build_integration_records(
    source_namespace: KnowledgeNamespaceRecord,
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[KnowledgeNamespaceRecord, ...], tuple[IntegrationEdgeRecord, ...]]:
    payload = dict(metadata or {})
    specs = payload.get("integration_edges", [])
    if not isinstance(specs, list):
        return (), ()

    target_namespaces: dict[str, KnowledgeNamespaceRecord] = {}
    edges: list[IntegrationEdgeRecord] = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        target_spec = spec.get("target") if isinstance(spec.get("target"), Mapping) else spec
        target_namespace = build_namespace_record(
            domain_id=_clean(target_spec.get("domain_id", source_namespace.domain_id), fallback=source_namespace.domain_id),
            system_id=target_spec.get("system_id", source_namespace.system_id),
            component_id=target_spec.get("component_id", source_namespace.component_id),
            environment=target_spec.get("environment", source_namespace.environment),
            evidence_type=target_spec.get("evidence_type", source_namespace.evidence_type),
            acl_scope=target_spec.get("acl_scope", source_namespace.acl_scope),
            asset_name=target_spec.get("asset_name"),
            component_name=target_spec.get("component_name"),
            summary=target_spec.get("summary") or target_spec.get("namespace_summary"),
            metadata=target_spec.get("metadata") if isinstance(target_spec.get("metadata"), Mapping) else {},
            source_uri=target_spec.get("source_uri"),
            source_kind=target_spec.get("source_kind"),
            source_type=target_spec.get("source_type"),
        )
        target_namespaces.setdefault(target_namespace.namespace_id, target_namespace)
        edge = build_integration_edge_record(
            source_namespace=source_namespace,
            target_namespace=target_namespace,
            relation_type=str(spec.get("relation_type") or spec.get("edge_type") or "integrates-with"),
            metadata=spec.get("metadata") if isinstance(spec.get("metadata"), Mapping) else {},
        )
        edges.append(edge)
    return tuple(target_namespaces.values()), tuple(edges)
