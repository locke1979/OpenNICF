"""Top-level router for the shared QwenAgent-backed OpenNICF stack."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

from .domain_agent import DEFAULT_DOMAIN_IDS, DomainAgentFactory
from .integration_correlation import (
    IntegrationCorrelationAgent,
    IntegrationCorrelationRequest,
)
from .knowledge import KnowledgePlatform
from .model_gateway import ModelGateway, PrivacyPolicy


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


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return asdict(value)
    return str(value)


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
    if lowered in {"0", "false", "no", "n", "local_preferred", "preferred"}:
        return PrivacyPolicy.LOCAL_PREFERRED
    if lowered == "remote_allowed":
        return PrivacyPolicy.REMOTE_ALLOWED
    if lowered == "remote_required":
        return PrivacyPolicy.REMOTE_REQUIRED
    return PrivacyPolicy(text)


def _normalize_domain_ids(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    return tuple(str(value).strip() for value in values if str(value).strip())


def _parse_domain_ids(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, Mapping):
        for key in ("domain_ids", "domains", "selected_domain_ids"):
            domain_ids = _normalize_domain_ids(payload.get(key))
            if domain_ids:
                return domain_ids
        return ()
    if isinstance(payload, str):
        candidate = payload.strip()
        if not candidate:
            return ()
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            return _normalize_domain_ids(candidate)
        if isinstance(decoded, Mapping):
            for key in ("domain_ids", "domains", "selected_domain_ids"):
                domain_ids = _normalize_domain_ids(decoded.get(key))
                if domain_ids:
                    return domain_ids
        if isinstance(decoded, list):
            return _normalize_domain_ids(decoded)
        if isinstance(decoded, str):
            return _normalize_domain_ids(decoded)
    return ()


def _coerce_model_result(result: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(result, list):
        if result and hasattr(result[0], "content"):
            content = str(result[0].content)
            provider = str(getattr(result[0], "provider", ""))
            return content, {"provider": provider}
        if result and isinstance(result[0], Mapping):
            message = result[0].get("message") or result[0]
            if isinstance(message, Mapping):
                return str(message.get("content") or ""), dict(result[0])
        return "", {"content": result}
    if hasattr(result, "content"):
        metadata = {"provider": getattr(result, "provider", None), "model": getattr(result, "model", None)}
        if getattr(result, "route", None) is not None:
            metadata["route"] = asdict(result.route)
        return str(result.content), {key: value for key, value in metadata.items() if value is not None}
    if isinstance(result, Mapping):
        content = result.get("content")
        if content is not None:
            return str(content), dict(result)
    return str(result), {"content": str(result)}


@dataclass(frozen=True)
class DomainRoutePlan:
    routing_id: str
    question: str
    route_kind: str
    route_source: str
    privacy: str
    local_only: bool
    selected_domain_ids: tuple[str, ...]
    deterministic_domain_ids: tuple[str, ...] = ()
    delegate_domain_id: str | None = None
    classification: dict[str, Any] = field(default_factory=dict)
    correlation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_id": self.routing_id,
            "question": self.question,
            "route_kind": self.route_kind,
            "route_source": self.route_source,
            "privacy": self.privacy,
            "local_only": self.local_only,
            "selected_domain_ids": list(self.selected_domain_ids),
            "deterministic_domain_ids": list(self.deterministic_domain_ids),
            "delegate_domain_id": self.delegate_domain_id,
            "classification": self.classification,
            "correlation": self.correlation,
            "metadata": self.metadata,
            "reason": self.reason,
            "confidence": self.confidence,
        }


class DomainRouter:
    """Top-level decision layer for deterministic alias routing and delegation."""

    def __init__(
        self,
        *,
        gateway: ModelGateway | None = None,
        knowledge: KnowledgePlatform | None = None,
        domain_factory: DomainAgentFactory | None = None,
        integration_coordinator: IntegrationCorrelationAgent | None = None,
        coordinator: IntegrationCorrelationAgent | None = None,
        diagnostic_broker: Any | None = None,
        allowed_domain_ids: Sequence[str] | None = None,
        max_domain_fan_out: int = 4,
        max_evidence_budget: int = 16,
        routing_task_class: str = "classification",
        synthesis_task_class: str = "tool_planning",
        privacy: PrivacyPolicy | str = PrivacyPolicy.LOCAL_PREFERRED,
        runtime_factory: Any | None = None,
    ) -> None:
        self.privacy = _normalize_privacy_policy(privacy)
        candidate_coordinator = integration_coordinator or coordinator
        if gateway is None and candidate_coordinator is not None:
            gateway = candidate_coordinator.gateway
        if knowledge is None and candidate_coordinator is not None:
            knowledge = candidate_coordinator.knowledge
        if domain_factory is None and candidate_coordinator is not None:
            domain_factory = candidate_coordinator.domain_factory

        self.gateway = gateway or ModelGateway.from_env()
        self.knowledge = knowledge or KnowledgePlatform.in_memory()
        self.domain_factory = domain_factory or DomainAgentFactory(
            gateway=self.gateway,
            knowledge=self.knowledge,
            diagnostic_broker=diagnostic_broker,
        )
        self.coordinator = candidate_coordinator or IntegrationCorrelationAgent(
            gateway=self.gateway,
            knowledge=self.knowledge,
            domain_factory=self.domain_factory,
            diagnostic_broker=diagnostic_broker,
            allowed_domain_ids=allowed_domain_ids or DEFAULT_DOMAIN_IDS,
            max_domain_fan_out=max_domain_fan_out,
            max_evidence_budget=max_evidence_budget,
            synthesis_task_class=synthesis_task_class,
            privacy=self.privacy,
            runtime_factory=runtime_factory,
        )
        self.routing_task_class = routing_task_class
        self._runtime_factory = runtime_factory
        self.allowed_domain_ids = tuple(allowed_domain_ids or self.coordinator.allowed_domain_ids)

    def _classify(self, request: IntegrationCorrelationRequest) -> dict[str, Any]:
        classification_prompt = {
            "question": request.question or request.raw_request,
            "allowed_domain_ids": list(self.allowed_domain_ids),
            "privacy": request.privacy.value,
            "local_only": request.privacy is PrivacyPolicy.LOCAL_ONLY,
            "time_window": request.time_window,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OpenNICF domain router. "
                    "Return JSON with a domain_ids array selecting one or more allowed domains. "
                    "Do not broaden scope beyond the allowed domains."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(classification_prompt, sort_keys=True, default=_json_default),
            },
        ]
        estimated_input_tokens = _estimate_tokens(json.dumps(classification_prompt, sort_keys=True, default=_json_default))
        result = self.gateway.chat(
            messages,
            task_class=self.routing_task_class,
            privacy=request.privacy,
            complex_task=False,
            estimated_input_tokens=estimated_input_tokens,
            source_fan_in=0,
            tool_complexity=0,
            extra_payload={"response_format": {"type": "json_object"}},
        )
        content, metadata = _coerce_model_result(result)
        payload: dict[str, Any]
        try:
            decoded = json.loads(content) if content else {}
            payload = decoded if isinstance(decoded, dict) else {"domain_ids": decoded}
        except json.JSONDecodeError:
            payload = {"content": content}
        payload.setdefault("content", content)
        payload.setdefault("domain_ids", list(_parse_domain_ids(payload)))
        payload["provider"] = metadata.get("provider")
        if "route" in metadata:
            payload["route"] = metadata["route"]
        if "model" in metadata:
            payload["model"] = metadata["model"]
        payload["estimated_input_tokens"] = estimated_input_tokens
        return payload

    def plan(self, request: str | Mapping[str, Any]) -> DomainRoutePlan:
        parsed = IntegrationCorrelationRequest.from_input(request)
        deterministic_domain_ids: tuple[str, ...] = ()
        route_source = "classification"
        classification: dict[str, Any] = {}

        try:
            deterministic_domain_ids = self.coordinator._selected_domain_ids(parsed)
        except ValueError:
            deterministic_domain_ids = ()
        if deterministic_domain_ids:
            selected_domain_ids = deterministic_domain_ids
            route_source = "deterministic_alias"
        else:
            classification = self._classify(parsed)
            selected_domain_ids = _parse_domain_ids(classification)
            if not selected_domain_ids:
                raise ValueError("domain routing classification did not return any domain ids")
            route_source = "classification"

        route_kind = "single_domain" if len(selected_domain_ids) == 1 else "integration_correlation"
        delegate_domain_id = selected_domain_ids[0] if route_kind == "single_domain" else "integration_correlation"
        correlation: dict[str, Any] = {}
        if route_kind == "integration_correlation":
            correlation = {
                "selected_domain_ids": list(selected_domain_ids),
                "selected_domain_count": len(selected_domain_ids),
            }
        routing_id = _stable_id(
            "route",
            parsed.question or parsed.raw_request,
            route_kind,
            route_source,
            self.privacy.value,
            ",".join(selected_domain_ids),
        )
        return DomainRoutePlan(
            routing_id=routing_id,
            question=parsed.question or parsed.raw_request,
            route_kind=route_kind,
            route_source=route_source,
            privacy=parsed.privacy.value,
            local_only=parsed.privacy is PrivacyPolicy.LOCAL_ONLY,
            selected_domain_ids=selected_domain_ids,
            deterministic_domain_ids=deterministic_domain_ids,
            delegate_domain_id=delegate_domain_id,
            classification=classification,
            correlation=correlation,
            metadata=dict(parsed.metadata),
            reason=(
                "deterministic domain/system alias match"
                if route_source == "deterministic_alias"
                else "QwenAgent classification constrained to the registered domain allow-list"
            ),
            confidence=(
                1.0
                if route_source == "deterministic_alias"
                else float(classification.get("confidence", classification.get("score", 0.0)) or 0.0)
            ),
        )

    def route(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        plan = self.plan(request)
        parsed = IntegrationCorrelationRequest.from_input(request)
        payload = parsed.to_dict()
        payload["domain_ids"] = list(plan.selected_domain_ids)
        payload["selected_domain_ids"] = list(plan.selected_domain_ids)
        payload["privacy"] = parsed.privacy.value

        if plan.route_kind == "single_domain":
            agent = self.domain_factory.create(plan.delegate_domain_id or plan.selected_domain_ids[0], privacy=parsed.privacy)
            result = agent.run(parsed.question or parsed.raw_request)
        else:
            result = self.coordinator.correlate(payload)
            plan = DomainRoutePlan(
                routing_id=plan.routing_id,
                question=plan.question,
                route_kind=plan.route_kind,
                route_source=plan.route_source,
                privacy=plan.privacy,
                local_only=plan.local_only,
                selected_domain_ids=plan.selected_domain_ids,
                deterministic_domain_ids=plan.deterministic_domain_ids,
                delegate_domain_id=plan.delegate_domain_id,
                classification=plan.classification,
                correlation={
                    **plan.correlation,
                    "correlation_id": result.get("correlation_id"),
                    "status": result.get("status"),
                },
                metadata=plan.metadata,
                reason=plan.reason,
                confidence=plan.confidence,
            )

        response = plan.to_dict()
        response["plan"] = plan.to_dict()
        response["result"] = result
        return response

    def dispatch(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        return self.route(request)

    def run(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        return self.route(request)


def create_domain_router(
    *,
    gateway: ModelGateway | None = None,
    knowledge: KnowledgePlatform | None = None,
    domain_factory: DomainAgentFactory | None = None,
    integration_coordinator: IntegrationCorrelationAgent | None = None,
    coordinator: IntegrationCorrelationAgent | None = None,
    diagnostic_broker: Any | None = None,
    allowed_domain_ids: Sequence[str] | None = None,
    max_domain_fan_out: int = 4,
    max_evidence_budget: int = 16,
    routing_task_class: str = "classification",
    synthesis_task_class: str = "tool_planning",
    privacy: PrivacyPolicy | str = PrivacyPolicy.LOCAL_PREFERRED,
    runtime_factory: Any | None = None,
) -> DomainRouter:
    return DomainRouter(
        gateway=gateway,
        knowledge=knowledge,
        domain_factory=domain_factory,
        integration_coordinator=integration_coordinator,
        coordinator=coordinator,
        diagnostic_broker=diagnostic_broker,
        allowed_domain_ids=allowed_domain_ids,
        max_domain_fan_out=max_domain_fan_out,
        max_evidence_budget=max_evidence_budget,
        routing_task_class=routing_task_class,
        synthesis_task_class=synthesis_task_class,
        privacy=privacy,
        runtime_factory=runtime_factory,
    )


__all__ = ["DomainRoutePlan", "DomainRouter", "create_domain_router"]
