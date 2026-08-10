"""Shared integration-correlation coordinator built on the QwenAgent runtime.

The coordinator reuses the common gateway, knowledge platform, and domain-agent
factory.  It fans out to existing domain agents for scoped retrieval, preserves
provenance-bearing evidence refs, and fan-ins into an integration graph without
introducing a second orchestration framework.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from .domain_agent import DEFAULT_DOMAIN_IDS, DomainAgentFactory
from .knowledge import KnowledgePlatform
from .model_gateway import ModelGateway, PrivacyPolicy
from .qwen_adapter import OpenNICFChatModel
from .qwen_runtime import QwenAgentRuntime

_COORDINATOR_DOMAIN_ALIASES: dict[str, str] = {
    "administrative": "contencioso_administrativo",
    "criminal": "criminal",
    "contraordenacional": "contraordenacional",
    "judicial": "contencioso_judicial",
    "sco": "contraordenacional",
    "sigef": "encargos_sigef",
    "sigepra": "contencioso_administrativo",
    "sicjut": "contencioso_judicial",
    "sinquer": "criminal",
    "encargos": "encargos_sigef",
}

_DOMAIN_DISPLAY_NAMES: dict[str, str] = {
    "criminal": "Criminal",
    "contraordenacional": "Contraordenacional",
    "contencioso_administrativo": "Contencioso Administrativo",
    "contencioso_judicial": "Contencioso Judicial",
    "encargos_sigef": "Encargos SIGEF",
}

_SUCCESS_CODE = 0
_PARTIAL_CODE = 1
_ERROR_CODE_MAP = {
    PermissionError: 403,
    KeyError: 404,
    NotImplementedError: 501,
}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(b"\0")
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:24]}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _normalize_multi_value(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.replace(";", ",").split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _normalize_privacy_policy(value: Any) -> PrivacyPolicy:
    if isinstance(value, PrivacyPolicy):
        return value
    if isinstance(value, bool):
        return PrivacyPolicy.LOCAL_ONLY if value else PrivacyPolicy.LOCAL_PREFERRED
    if value is None:
        return PrivacyPolicy.LOCAL_PREFERRED
    text = str(value).strip()
    if not text:
        return PrivacyPolicy.LOCAL_PREFERRED
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "y", "local", "local_only"}:
        return PrivacyPolicy.LOCAL_ONLY
    if lowered in {"0", "false", "no", "n", "local_preferred", "preferred", "remote_allowed"}:
        return PrivacyPolicy.LOCAL_PREFERRED if lowered != "remote_allowed" else PrivacyPolicy.REMOTE_ALLOWED
    return PrivacyPolicy(text)


def _normalize_request(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return {"query": value}
    return dict(value)


def _normalize_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)  # noqa: FURB162


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _text_contains_term(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def _infer_domains_from_text(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    ordered: list[str] = []
    for needle, domain_id in (
        ("sinquer", "criminal"),
        ("criminal", "criminal"),
        ("sco", "contraordenacional"),
        ("contraordenacional", "contraordenacional"),
        ("sigef", "encargos_sigef"),
        ("sigepra", "contencioso_administrativo"),
        ("administrative", "contencioso_administrativo"),
        ("judicial", "contencioso_judicial"),
        ("sicjut", "contencioso_judicial"),
        ("encargos", "encargos_sigef"),
    ):
        if needle in lowered and domain_id not in ordered:
            ordered.append(domain_id)
    return tuple(ordered)


def _error_code_for(exc: Exception) -> int:
    for exc_type, code in _ERROR_CODE_MAP.items():
        if isinstance(exc, exc_type):
            return code
    return 500


@dataclass(frozen=True)
class IntegrationCorrelationRequest:
    raw_request: str
    question: str
    domain_ids: tuple[str, ...] = ()
    systems: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    correlation_ids: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    session_ids: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    start_time: datetime | None = None
    end_time: datetime | None = None
    evidence_budget: int = 8
    token_budget: int = 12_000
    retrieval_mode: str = "domain_evidence"
    privacy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_input(cls, value: str | Mapping[str, Any]) -> IntegrationCorrelationRequest:
        if isinstance(value, str):
            question = value.strip()
            return cls(
                raw_request=question,
                question=question,
                domain_ids=_infer_domains_from_text(question),
            )

        data = dict(value)
        question = str(data.get("question") or data.get("query") or data.get("request") or "").strip()
        raw_request = str(data.get("raw_request") or question or data.get("scope") or "").strip()
        domain_ids = _normalize_multi_value(data.get("domain_ids") or data.get("domains") or data.get("selected_domain_ids"))
        if not domain_ids:
            domain_ids = _infer_domains_from_text(question or raw_request)
        evidence_budget = int(data.get("evidence_budget") or data.get("limit") or data.get("max_evidence_packages") or 8)
        token_budget = int(data.get("token_budget") or data.get("estimated_input_tokens") or 12_000)
        privacy = _normalize_privacy_policy(data.get("privacy") or data.get("privacy_policy") or data.get("local_only"))
        return cls(
            raw_request=raw_request or question,
            question=question or raw_request,
            domain_ids=domain_ids,
            systems=_normalize_multi_value(data.get("systems") or data.get("system_ids")),
            components=_normalize_multi_value(data.get("components") or data.get("component_ids")),
            correlation_ids=_normalize_multi_value(data.get("correlation_ids")),
            request_ids=_normalize_multi_value(data.get("request_ids")),
            session_ids=_normalize_multi_value(data.get("session_ids")),
            source_types=_normalize_multi_value(data.get("source_types") or data.get("evidence_types")),
            start_time=_normalize_datetime(data.get("start_time") or data.get("from") or data.get("since")),
            end_time=_normalize_datetime(data.get("end_time") or data.get("to") or data.get("until")),
            evidence_budget=max(1, evidence_budget),
            token_budget=max(1, token_budget),
            retrieval_mode=str(data.get("retrieval_mode") or data.get("mode") or "domain_evidence"),
            privacy=privacy,
            metadata={
                key: value
                for key, value in data.items()
                if key
                not in {
                    "question",
                    "query",
                    "request",
                    "raw_request",
                    "domain_ids",
                    "domains",
                    "selected_domain_ids",
                    "systems",
                    "system_ids",
                    "components",
                    "component_ids",
                    "correlation_ids",
                    "request_ids",
                    "session_ids",
                    "source_types",
                    "evidence_types",
                    "start_time",
                    "from",
                    "since",
                    "end_time",
                    "to",
                    "until",
                    "evidence_budget",
                    "limit",
                    "max_evidence_packages",
                    "token_budget",
                    "estimated_input_tokens",
                    "retrieval_mode",
                    "mode",
                    "privacy",
                    "privacy_policy",
                    "local_only",
                }
            },
        )

    @property
    def time_window(self) -> dict[str, str | None]:
        return {
            "start": self.start_time.isoformat() if self.start_time else None,
            "end": self.end_time.isoformat() if self.end_time else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_request": self.raw_request,
            "question": self.question,
            "domain_ids": list(self.domain_ids),
            "systems": list(self.systems),
            "components": list(self.components),
            "correlation_ids": list(self.correlation_ids),
            "request_ids": list(self.request_ids),
            "session_ids": list(self.session_ids),
            "source_types": list(self.source_types),
            "time_window": self.time_window,
            "evidence_budget": self.evidence_budget,
            "token_budget": self.token_budget,
            "retrieval_mode": self.retrieval_mode,
            "privacy": self.privacy.value,
            "metadata": self.metadata,
        }


class IntegrationCorrelationTools:
    """Tool surface used by the shared QwenAgent runtime."""

    def __init__(self, agent: IntegrationCorrelationAgent) -> None:
        self.agent = agent

    def request_integration_correlation(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        return self.agent.correlate(request)

    def as_qwen_tools(self) -> dict[str, Callable[..., Any]]:
        return {"request_integration_correlation": self.request_integration_correlation}


class IntegrationCorrelationAgent:
    """QwenAgent-backed coordinator for cross-domain integration correlation."""

    def __init__(
        self,
        *,
        gateway: ModelGateway | None = None,
        knowledge: KnowledgePlatform | None = None,
        domain_factory: DomainAgentFactory | None = None,
        diagnostic_broker: Any | None = None,
        allowed_domain_ids: Sequence[str] | None = None,
        max_domain_fan_out: int = 4,
        max_evidence_budget: int = 16,
        synthesis_task_class: str = "tool_planning",
        privacy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED,
        runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
    ) -> None:
        self.gateway = gateway or ModelGateway.from_env()
        self.knowledge = knowledge or KnowledgePlatform.in_memory()
        self.privacy = _normalize_privacy_policy(privacy)
        self.domain_factory = domain_factory or DomainAgentFactory(
            gateway=self.gateway,
            knowledge=self.knowledge,
            diagnostic_broker=diagnostic_broker,
        )
        self.allowed_domain_ids = tuple(allowed_domain_ids or DEFAULT_DOMAIN_IDS)
        self.max_domain_fan_out = max(1, int(max_domain_fan_out))
        self.max_evidence_budget = max(1, int(max_evidence_budget))
        self.model_adapter = OpenNICFChatModel(
            gateway=self.gateway,
            task_class=synthesis_task_class,
            privacy=self.privacy,
            complex_input_chars=6_000,
        )
        self.tools = IntegrationCorrelationTools(self)
        self._runtime_factory = runtime_factory or QwenAgentRuntime
        self._runtime: Any | None = None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_factory(self.model_adapter, self.tools.as_qwen_tools())
        return self._runtime

    def run(self, request: str) -> Any:
        return self.runtime.run(request)

    def _resolve_domain_id(self, value: str) -> str:
        candidate = str(value).strip()
        if not candidate:
            raise KeyError("unknown domain or system alias: <empty>")
        if candidate in self.allowed_domain_ids:
            return candidate
        lowered = candidate.lower()
        mapped = _COORDINATOR_DOMAIN_ALIASES.get(lowered)
        if mapped:
            candidate = mapped
        return self.domain_factory.resolve_domain_id(candidate)

    def _selected_domain_ids(self, request: IntegrationCorrelationRequest) -> tuple[str, ...]:
        candidates: list[str] = []
        for value in (*request.domain_ids, *request.systems):
            resolved = self._resolve_domain_id(value)
            if resolved not in candidates:
                candidates.append(resolved)
        for value in request.components:
            try:
                resolved = self._resolve_domain_id(value)
            except KeyError:
                continue
            if resolved not in candidates:
                candidates.append(resolved)

        if not candidates:
            candidates = list(_infer_domains_from_text(request.question or request.raw_request))
        if not candidates:
            raise ValueError("integration correlation requires at least one selected domain or system")

        disallowed = [domain_id for domain_id in candidates if domain_id not in self.allowed_domain_ids]
        if disallowed:
            raise PermissionError(f"coordinator ACL blocks domain(s): {', '.join(disallowed)}")

        return tuple(candidates[: self.max_domain_fan_out])

    def _partition_systems(self, request: IntegrationCorrelationRequest, domain_ids: Sequence[str]) -> dict[str, tuple[str, ...]]:
        if not request.systems:
            return {domain_id: () for domain_id in domain_ids}
        resolved: dict[str, list[str]] = {domain_id: [] for domain_id in domain_ids}
        for system in request.systems:
            try:
                domain_id = self._resolve_domain_id(system)
            except KeyError:
                continue
            if domain_id in resolved and system not in resolved[domain_id]:
                resolved[domain_id].append(system)
        return {domain_id: tuple(values) for domain_id, values in resolved.items()}

    def _choose_retrieval_method(self, request: IntegrationCorrelationRequest) -> str:
        lowered = request.question.lower()
        explicit_mode = request.retrieval_mode.strip().lower()
        if explicit_mode in {"exact", "symbol", "logs", "docs", "schema", "query_outputs"}:
            return explicit_mode
        if request.source_types == ("log",) or any(term in lowered for term in ("correlation id", "request id", "session id")):
            return "logs"
        if request.source_types == ("code",):
            return "exact" if " " not in request.question.strip() else "symbol"
        if request.source_types == ("schema",):
            return "schema"
        if request.source_types == ("query-output",):
            return "query_outputs"
        return "domain_evidence"

    def _package_request_payload(
        self,
        request: IntegrationCorrelationRequest,
        *,
        correlation_id: str,
        domain_id: str,
        domain_system_ids: Sequence[str],
        evidence_budget: int,
        token_budget: int,
    ) -> dict[str, Any]:
        return {
            "query": request.question or request.raw_request,
            "principal_domain_id": domain_id,
            "domain_ids": [domain_id],
            "system_ids": list(domain_system_ids),
            "component_ids": list(request.components),
            "correlation_ids": list(request.correlation_ids),
            "request_ids": list(request.request_ids),
            "session_ids": list(request.session_ids),
            "source_types": list(request.source_types),
            "since": request.start_time,
            "until": request.end_time,
            "limit": evidence_budget,
            "neighbor_window": 1,
            "correlation_id": correlation_id,
            "evidence_budget": evidence_budget,
            "token_budget": token_budget,
            "time_window": request.time_window,
        }

    def _domain_agent_summary(self, domain_id: str) -> dict[str, Any]:
        profile = self.domain_factory.profile_for(domain_id)
        return {
            "domain_id": profile.domain_id,
            "name": profile.name,
            "description": profile.description,
            "owned_systems": list(profile.owned_systems),
            "owned_components": list(profile.owned_components),
            "aliases": list(profile.aliases),
            "system_aliases": list(profile.system_aliases),
        }

    def _delegate_domain(
        self,
        request: IntegrationCorrelationRequest,
        *,
        correlation_id: str,
        task_id: str,
        domain_id: str,
        domain_system_ids: Sequence[str],
        evidence_budget: int,
        token_budget: int,
    ) -> dict[str, Any]:
        started_at = _utcnow()
        try:
            agent = self.domain_factory.create(domain_id, privacy=request.privacy)
        except TypeError:
            agent = self.domain_factory.create(domain_id)
        retrieval_method = self._choose_retrieval_method(request)
        domain_payload = self._package_request_payload(
            request,
            correlation_id=correlation_id,
            domain_id=domain_id,
            domain_system_ids=domain_system_ids,
            evidence_budget=evidence_budget,
            token_budget=token_budget,
        )

        try:
            tool = {
                "exact": agent.tools.search_code_exact,
                "symbol": agent.tools.search_code_symbols,
                "logs": agent.tools.search_logs,
                "docs": agent.tools.search_docs,
                "schema": agent.tools.search_schema,
                "query_outputs": agent.tools.search_query_outputs,
                "domain_evidence": agent.tools.search_domain_evidence,
            }[retrieval_method]
            retrieval = dict(tool(domain_payload, limit=evidence_budget))
            packages = list(retrieval.get("packages", ()))[:evidence_budget]
            evidence_refs = [ref for package in packages for ref in package.get("evidence_refs", ())]
            provenance_refs = tuple(
                dict.fromkeys(
                    ref.get("provenance_ref")
                    for ref in evidence_refs
                    if isinstance(ref, Mapping) and ref.get("provenance_ref")
                )
            )
            correlation_refs = tuple(
                dict.fromkeys(
                    value
                    for ref in evidence_refs
                    for value in _normalize_multi_value(ref.get("metadata", {}).get("correlation_ids"))
                )
            )
            timestamps = tuple(
                dict.fromkeys(
                    value
                    for ref in evidence_refs
                    for value in _normalize_multi_value(ref.get("metadata", {}).get("timestamps"))
                )
            )
            facts = [
                {
                    "classification": "fact",
                    "statement": f"{_DOMAIN_DISPLAY_NAMES.get(domain_id, domain_id)} retrieval produced {len(packages)} evidence package(s).",
                    "confidence": 1.0 if packages else 0.5,
                    "evidence_refs": evidence_refs[:1],
                    "provenance_refs": list(provenance_refs),
                    "correlation_ids": list(correlation_refs),
                    "timestamps": list(timestamps),
                    "contradicting_evidence": [],
                }
            ]
            if len(packages) > 1:
                facts.append(
                    {
                        "classification": "inference",
                        "statement": f"{_DOMAIN_DISPLAY_NAMES.get(domain_id, domain_id)} evidence remained compact within the requested budget.",
                        "confidence": 0.84,
                        "evidence_refs": evidence_refs[:2],
                        "provenance_refs": list(provenance_refs),
                        "correlation_ids": list(correlation_refs),
                        "timestamps": list(timestamps),
                        "contradicting_evidence": [],
                    }
                )
            if retrieval.get("package_count", 0) == 0:
                facts.append(
                    {
                        "classification": "hypothesis",
                        "statement": f"No direct evidence was located for {_DOMAIN_DISPLAY_NAMES.get(domain_id, domain_id)} within the bounded request window.",
                        "confidence": 0.55,
                        "evidence_refs": [],
                        "provenance_refs": [],
                        "correlation_ids": list(request.correlation_ids),
                        "timestamps": [],
                        "contradicting_evidence": list(evidence_refs),
                    }
                )
            integration_edges = tuple(
                edge
                for package in packages
                for edge in package.get("metadata", {}).get("integration_edges", ())
                if isinstance(edge, Mapping)
            )
            return {
                "task_id": task_id,
                "correlation_id": correlation_id,
                "domain_id": domain_id,
                "domain": self._domain_agent_summary(domain_id),
                "status": "ok",
                "return_code": _SUCCESS_CODE,
                "error_code": None,
                "error_message": None,
                "started_at": started_at.isoformat(),
                "finished_at": _utcnow().isoformat(),
                "request": {
                    "question": request.question or request.raw_request,
                    "time_window": request.time_window,
                    "identifiers": {
                        "correlation_ids": list(request.correlation_ids),
                        "request_ids": list(request.request_ids),
                        "session_ids": list(request.session_ids),
                    },
                    "evidence_budget": evidence_budget,
                    "token_budget": token_budget,
                    "retrieval_mode": retrieval_method,
                },
                "retrieval": {
                    "mode": retrieval.get("retrieval_mode", retrieval_method),
                    "filters": retrieval.get("filters", {}),
                    "package_count": retrieval.get("package_count", len(packages)),
                    "estimated_tokens": max(1, int(retrieval.get("estimated_tokens", 0) or 0)),
                    "estimated_bytes": max(1, int(retrieval.get("estimated_bytes", 0) or 0)),
                    "embedding_space_id": next(
                        (
                            package.get("embedding_space_id")
                            or package.get("metadata", {}).get("embedding_space_id")
                            for package in packages
                            if isinstance(package, Mapping)
                        ),
                        self.knowledge.embeddings.active_space_id,
                    ),
                },
                "component_ownership": {
                    "owned_systems": list(agent.profile.owned_systems),
                    "owned_components": list(agent.profile.owned_components),
                    "aliases": list(agent.profile.aliases),
                    "system_aliases": list(agent.profile.system_aliases),
                },
                "evidence_packages": packages,
                "evidence_refs": evidence_refs,
                "provenance_refs": list(provenance_refs),
                "correlation_ids": list(correlation_refs),
                "timestamps": list(timestamps),
                "facts": facts,
                "integration_edges": [dict(edge) for edge in integration_edges],
                "contradicting_evidence": [],
                "summary_hint": f"{_DOMAIN_DISPLAY_NAMES.get(domain_id, domain_id)} yielded {len(packages)} package(s).",
            }
        except Exception as exc:  # noqa: BLE001 - package every delegate failure explicitly
            failure_code = _error_code_for(exc)
            return {
                "task_id": task_id,
                "correlation_id": correlation_id,
                "domain_id": domain_id,
                "domain": self._domain_agent_summary(domain_id),
                "status": "failed",
                "return_code": _PARTIAL_CODE,
                "error_code": failure_code,
                "error_message": str(exc),
                "started_at": started_at.isoformat(),
                "finished_at": _utcnow().isoformat(),
                "request": {
                    "question": request.question or request.raw_request,
                    "time_window": request.time_window,
                    "identifiers": {
                        "correlation_ids": list(request.correlation_ids),
                        "request_ids": list(request.request_ids),
                        "session_ids": list(request.session_ids),
                    },
                    "evidence_budget": evidence_budget,
                    "token_budget": token_budget,
                    "retrieval_mode": retrieval_method,
                },
                "retrieval": {
                    "mode": retrieval_method,
                    "filters": {},
                    "package_count": 0,
                    "estimated_tokens": 0,
                    "estimated_bytes": 0,
                },
                "component_ownership": {
                    "owned_systems": list(self.domain_factory.profile_for(domain_id).owned_systems),
                    "owned_components": list(self.domain_factory.profile_for(domain_id).owned_components),
                    "aliases": list(self.domain_factory.profile_for(domain_id).aliases),
                    "system_aliases": list(self.domain_factory.profile_for(domain_id).system_aliases),
                },
                "evidence_packages": [],
                "evidence_refs": [],
                "provenance_refs": [],
                "correlation_ids": list(request.correlation_ids),
                "timestamps": [],
                "facts": [
                    {
                        "classification": "hypothesis",
                        "statement": f"{_DOMAIN_DISPLAY_NAMES.get(domain_id, domain_id)} delegation failed: {exc}",
                        "confidence": 0.5,
                        "evidence_refs": [],
                        "provenance_refs": [],
                        "correlation_ids": list(request.correlation_ids),
                        "timestamps": [],
                        "contradicting_evidence": [],
                    }
                ],
                "integration_edges": [],
                "contradicting_evidence": [],
            }

    def _build_integration_graph(
        self,
        request: IntegrationCorrelationRequest,
        delegations: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        nodes = [self._domain_agent_summary(delegation["domain_id"]) for delegation in delegations]
        edges: list[dict[str, Any]] = []
        for delegation in delegations:
            for edge in delegation.get("integration_edges", ()):
                if not isinstance(edge, Mapping):
                    continue
                source_domain = str(edge.get("domain_id") or delegation["domain_id"])
                target_domain = str(edge.get("target_domain_id") or edge.get("target_domain") or "")
                if not target_domain:
                    continue
                edges.append(
                    {
                        "edge_id": edge.get("edge_id") or _stable_id("edge", request.question, source_domain, target_domain, str(edge.get("relation_type") or "")),
                        "source_domain_id": source_domain,
                        "target_domain_id": target_domain,
                        "relation_type": str(edge.get("relation_type") or "integration"),
                        "component_id": str(edge.get("component_id") or edge.get("component") or ""),
                        "system_id": str(edge.get("system_id") or ""),
                        "evidence_refs": list(edge.get("evidence_refs", [])) if isinstance(edge.get("evidence_refs"), list) else [],
                        "correlation_ids": list(
                            dict.fromkeys(
                                [*request.correlation_ids, *delegation.get("correlation_ids", [])]
                            )
                        ),
                        "timestamps": list(delegation.get("timestamps", [])),
                        "provenance_refs": list(delegation.get("provenance_refs", [])),
                    }
                )
        unique_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for edge in edges:
            key = (edge["source_domain_id"], edge["target_domain_id"], edge["relation_type"])
            unique_edges.setdefault(key, edge)

        if not unique_edges and len(delegations) > 1:
            relation_type = "integration-correlation"
            lowered_question = (request.question or request.raw_request).lower()
            if "handoff" in lowered_question:
                relation_type = "handoff"
            elif "ownership" in lowered_question or "case" in lowered_question:
                relation_type = "case-reference"
            elif "delegate" in lowered_question:
                relation_type = "delegates-to"
            for left_index, left in enumerate(delegations):
                for right in delegations[left_index + 1 :]:
                    left_refs = list(left.get("evidence_refs", []))
                    right_refs = list(right.get("evidence_refs", []))
                    forward = {
                        "edge_id": _stable_id("edge", request.question, left["domain_id"], right["domain_id"], relation_type),
                        "source_domain_id": left["domain_id"],
                        "target_domain_id": right["domain_id"],
                        "relation_type": relation_type,
                        "component_id": "",
                        "system_id": "",
                        "evidence_refs": left_refs[:1] + right_refs[:1],
                        "correlation_ids": list(dict.fromkeys([*request.correlation_ids, *left.get("correlation_ids", []), *right.get("correlation_ids", [])])),
                        "timestamps": list(dict.fromkeys([*left.get("timestamps", []), *right.get("timestamps", [])])),
                        "provenance_refs": list(dict.fromkeys([*left.get("provenance_refs", []), *right.get("provenance_refs", [])])),
                    }
                    reverse = {
                        **forward,
                        "edge_id": _stable_id("edge", request.question, right["domain_id"], left["domain_id"], relation_type),
                        "source_domain_id": right["domain_id"],
                        "target_domain_id": left["domain_id"],
                    }
                    unique_edges.setdefault((forward["source_domain_id"], forward["target_domain_id"], relation_type), forward)
                    unique_edges.setdefault((reverse["source_domain_id"], reverse["target_domain_id"], relation_type), reverse)
        return {
            "nodes": nodes,
            "edges": list(unique_edges.values()),
        }

    def _build_findings(
        self,
        request: IntegrationCorrelationRequest,
        delegations: Sequence[dict[str, Any]],
        graph: dict[str, Any],
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        seen_provenance: list[str] = []
        for delegation in delegations:
            if delegation["status"] == "ok":
                findings.extend(delegation.get("facts", []))
                seen_provenance.extend(delegation.get("provenance_refs", []))
        if graph["edges"]:
            edge_statements = []
            for edge in graph["edges"]:
                source = _DOMAIN_DISPLAY_NAMES.get(edge["source_domain_id"], edge["source_domain_id"])
                target = _DOMAIN_DISPLAY_NAMES.get(edge["target_domain_id"], edge["target_domain_id"])
                edge_statements.append(f"{source} links to {target} via {edge['relation_type']}")
            findings.append(
                {
                    "classification": "inference",
                    "statement": "; ".join(edge_statements),
                    "confidence": 0.88,
                    "evidence_refs": [ref for delegation in delegations for ref in delegation.get("evidence_refs", [])][:4],
                    "provenance_refs": list(dict.fromkeys(seen_provenance)),
                    "correlation_ids": list(request.correlation_ids),
                    "timestamps": [timestamp for delegation in delegations for timestamp in delegation.get("timestamps", [])],
                    "supporting_evidence": list(dict.fromkeys(seen_provenance)),
                    "contradicting_evidence": [],
                }
            )
        failures = [delegation for delegation in delegations if delegation["status"] != "ok"]
        if failures:
            findings.append(
                {
                    "classification": "hypothesis",
                    "statement": f"{len(failures)} delegated retrieval task(s) failed within the bounded correlation run.",
                    "confidence": 0.61,
                    "evidence_refs": [],
                    "provenance_refs": [],
                    "correlation_ids": list(request.correlation_ids),
                    "timestamps": [ts for delegation in failures for ts in delegation.get("timestamps", [])],
                    "supporting_evidence": [],
                    "contradicting_evidence": [
                        {
                            "domain_id": delegation["domain_id"],
                            "error_code": delegation["error_code"],
                            "error_message": delegation["error_message"],
                            "return_code": delegation["return_code"],
                        }
                        for delegation in failures
                    ],
                }
            )
            findings.append(
                {
                    "classification": "missing_evidence",
                    "statement": "One or more requested domains did not return retrieval evidence.",
                    "confidence": 0.58,
                    "evidence_refs": [],
                    "provenance_refs": [],
                    "correlation_ids": list(request.correlation_ids),
                    "timestamps": [],
                    "supporting_evidence": [],
                    "contradicting_evidence": [
                        {
                            "domain_id": delegation["domain_id"],
                            "error_code": delegation["error_code"],
                            "error_message": delegation["error_message"],
                            "return_code": delegation["return_code"],
                        }
                        for delegation in failures
                    ],
                }
            )
        if not findings:
            findings.append(
                {
                    "classification": "missing_evidence",
                    "statement": "No integration evidence was retrieved in the requested domain set.",
                    "confidence": 0.5,
                    "evidence_refs": [],
                    "provenance_refs": [],
                    "correlation_ids": list(request.correlation_ids),
                    "timestamps": [],
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                }
            )
        return findings

    def _summarize(self, request: IntegrationCorrelationRequest, delegations: Sequence[dict[str, Any]], graph: dict[str, Any]) -> str:
        if not (self.gateway.local_provider or self.gateway.remote_provider):
            domain_names = ", ".join(_DOMAIN_DISPLAY_NAMES.get(delegation["domain_id"], delegation["domain_id"]) for delegation in delegations)
            return (
                f"Correlated {len(delegations)} domain(s) [{domain_names}] with {len(graph['edges'])} integration edge(s) "
                f"and {sum(len(delegation.get('evidence_refs', [])) for delegation in delegations)} provenance-bearing evidence ref(s)."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OpenNICF integration-correlation coordinator. "
                    "Summarize only the retrieved evidence, preserve contradictions, "
                    "and avoid inventing cross-domain ownership or endpoints."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": request.question or request.raw_request,
                        "time_window": request.time_window,
                        "domains": [delegation["domain_id"] for delegation in delegations],
                        "graph": graph,
                        "findings": [
                            finding["classification"] + ": " + finding["statement"]
                            for finding in self._build_findings(request, delegations, graph)[:4]
                        ],
                    },
                    sort_keys=True,
                    default=_json_default,
                ),
            },
        ]
        estimated_input_tokens = sum(_estimate_tokens(str(message.get("content", ""))) for message in messages)
        try:
            result = self.gateway.chat(
                messages,
                task_class=self.model_adapter.task_class,
                privacy=request.privacy,
                complex_task=len(delegations) > 2 or estimated_input_tokens > 2_000,
                estimated_input_tokens=estimated_input_tokens,
                source_fan_in=len(delegations),
                tool_complexity=1,
                extra_payload={"response_format": {"type": "text"}},
            )
        except Exception:  # noqa: BLE001 - synthesis failure is returned as an explicit result
            domain_names = ", ".join(_DOMAIN_DISPLAY_NAMES.get(delegation["domain_id"], delegation["domain_id"]) for delegation in delegations)
            return (
                f"Correlated {len(delegations)} domain(s) [{domain_names}] with {len(graph['edges'])} integration edge(s) "
                f"and {sum(len(delegation.get('evidence_refs', [])) for delegation in delegations)} provenance-bearing evidence ref(s)."
            )
        if isinstance(result, list):
            if result and hasattr(result[0], "content"):
                return str(result[0].content)
            if result and isinstance(result[0], Mapping):
                message = result[0].get("message") or result[0]
                if isinstance(message, Mapping):
                    return str(message.get("content") or "")
            return ""
        return str(getattr(result, "content", ""))

    def correlate(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        parsed = IntegrationCorrelationRequest.from_input(request)
        selected_domain_ids = self._selected_domain_ids(parsed)
        partitioned_systems = self._partition_systems(parsed, selected_domain_ids)
        correlation_id = _stable_id(
            "icorr",
            parsed.question or parsed.raw_request,
            ",".join(selected_domain_ids),
            ",".join(parsed.correlation_ids),
            ",".join(parsed.request_ids),
            ",".join(parsed.session_ids),
            parsed.time_window["start"] or "",
            parsed.time_window["end"] or "",
        )
        task_budget = max(1, min(self.max_evidence_budget, parsed.evidence_budget))
        per_task_budget = max(1, task_budget // max(1, len(selected_domain_ids)))
        remainder = task_budget % max(1, len(selected_domain_ids))
        tasks: list[dict[str, Any]] = []
        for index, domain_id in enumerate(selected_domain_ids):
            evidence_budget = per_task_budget + (1 if index < remainder else 0)
            evidence_budget = min(evidence_budget, self.max_evidence_budget)
            task_id = _stable_id("itask", correlation_id, domain_id, str(index))
            tasks.append(
                {
                    "task_id": task_id,
                    "domain_id": domain_id,
                    "domain_system_ids": partitioned_systems.get(domain_id, ()),
                    "evidence_budget": evidence_budget,
                    "token_budget": max(1, parsed.token_budget // max(1, len(selected_domain_ids))),
                }
            )

        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(len(tasks), self.max_domain_fan_out)) as executor:
            future_map = {
                executor.submit(
                    self._delegate_domain,
                    parsed,
                    correlation_id=correlation_id,
                    task_id=task["task_id"],
                    domain_id=task["domain_id"],
                    domain_system_ids=task["domain_system_ids"],
                    evidence_budget=task["evidence_budget"],
                    token_budget=task["token_budget"],
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                results[task["task_id"]] = future.result()

        delegations = [results[task["task_id"]] for task in tasks]
        graph = self._build_integration_graph(parsed, delegations)
        findings = self._build_findings(parsed, delegations, graph)
        total_estimated_tokens = sum(int(delegation.get("retrieval", {}).get("estimated_tokens", 0)) for delegation in delegations)
        total_estimated_bytes = sum(int(delegation.get("retrieval", {}).get("estimated_bytes", 0)) for delegation in delegations)
        summary = self._summarize(parsed, delegations, graph)
        status = "ok"
        if any(delegation["status"] != "ok" for delegation in delegations):
            status = "partial" if any(delegation["status"] == "ok" for delegation in delegations) else "failed"
        return {
            "status": status,
            "return_code": _SUCCESS_CODE if status == "ok" else _PARTIAL_CODE,
            "error_code": None if status == "ok" else 207,
            "correlation_id": correlation_id,
            "question": parsed.question or parsed.raw_request,
            "time_window": parsed.time_window,
            "identifiers": {
                "correlation_ids": list(parsed.correlation_ids),
                "request_ids": list(parsed.request_ids),
                "session_ids": list(parsed.session_ids),
            },
            "evidence_budget": parsed.evidence_budget,
            "token_budget": parsed.token_budget,
            "selected_domain_ids": list(selected_domain_ids),
            "selected_domain_count": len(selected_domain_ids),
            "bounded_fan_out": len(selected_domain_ids) >= self.max_domain_fan_out,
            "delegations": delegations,
            "evidence_packages": [package for delegation in delegations for package in delegation.get("evidence_packages", [])],
            "integration_graph": graph,
            "findings": findings,
            "contradicting_evidence": [
                item for finding in findings for item in finding.get("contradicting_evidence", []) if item
            ],
            "summary": summary,
            "summary_mode": "model_router" if self.gateway.local_provider or self.gateway.remote_provider else "local",
            "estimated_tokens": total_estimated_tokens,
            "estimated_bytes": total_estimated_bytes,
            "created_at": _utcnow().isoformat(),
            "privacy": parsed.privacy.value,
            "embedding_space_id": self.knowledge.embeddings.active_space_id,
            "metadata": parsed.metadata,
        }


def create_integration_correlation_agent(
    *,
    gateway: ModelGateway | None = None,
    knowledge: KnowledgePlatform | None = None,
    domain_factory: DomainAgentFactory | None = None,
    diagnostic_broker: Any | None = None,
    allowed_domain_ids: Sequence[str] | None = None,
    max_domain_fan_out: int = 4,
    max_evidence_budget: int = 16,
    synthesis_task_class: str = "tool_planning",
    privacy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED,
    runtime_factory: Callable[[Any, dict[str, Callable[..., Any]]], Any] | None = None,
) -> IntegrationCorrelationAgent:
    return IntegrationCorrelationAgent(
        gateway=gateway,
        knowledge=knowledge,
        domain_factory=domain_factory,
        diagnostic_broker=diagnostic_broker,
        allowed_domain_ids=allowed_domain_ids,
        max_domain_fan_out=max_domain_fan_out,
        max_evidence_budget=max_evidence_budget,
        synthesis_task_class=synthesis_task_class,
        privacy=_normalize_privacy_policy(privacy),
        runtime_factory=runtime_factory,
    )


__all__ = [
    "IntegrationCorrelationAgent",
    "IntegrationCorrelationRequest",
    "IntegrationCorrelationTools",
    "create_integration_correlation_agent",
]
