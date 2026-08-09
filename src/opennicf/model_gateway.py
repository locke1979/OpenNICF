"""Provider-neutral model routing and gateway helpers.

The gateway keeps provider credentials and endpoints in runtime configuration,
prefers LM Studio for local-safe tasks, and escalates to LiteLLM OCI when the
task is too large or complex for the local budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class PrivacyPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    LOCAL_PREFERRED = "local_preferred"
    REMOTE_ALLOWED = "remote_allowed"
    REMOTE_REQUIRED = "remote_required"


LOCAL_FIRST_TASK_CLASSES = {
    "classification",
    "routing",
    "metadata_extraction",
    "summarisation",
    "summarization",
    "simple_rag",
    "chunk_tagging",
    "tool_planning",
}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.25
    circuit_breaker_failures: int = 3
    circuit_breaker_reset_seconds: float = 30.0


@dataclass(frozen=True)
class RouterConfig:
    local_context_limit: int = 8192
    local_max_tool_complexity: int = 1
    local_max_attachment_fan_in: int = 4
    local_max_source_fan_in: int = 4


@dataclass(frozen=True)
class Route:
    provider: str
    reason: str
    model: str | None = None
    base_url: str | None = None
    streaming: bool = False


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str
    arguments: Any
    type: str = "function"


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ModelResult:
    provider: str
    route: Route
    model: str | None
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    latency_seconds: float = 0.0
    response_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamChunk:
    provider: str
    route: Route
    delta: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    healthy: bool
    latency_seconds: float
    models: tuple[str, ...] = ()
    detail: str | None = None
    status_code: int | None = None


@dataclass(frozen=True)
class AuditEvent:
    timestamp: float
    task_class: str
    privacy: PrivacyPolicy
    provider: str
    reason: str
    request_hash: str
    streaming: bool
    estimated_input_tokens: int | None = None
    complex_task: bool = False
    attachment_count: int = 0
    source_fan_in: int = 0
    tool_complexity: int = 0
    route: Route | None = None


@dataclass
class _BreakerState:
    failures: int = 0
    opened_until: float = 0.0


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        reset_after_seconds: float = 30.0,
        clock=time.monotonic,
    ):
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_after_seconds = float(reset_after_seconds)
        self._clock = clock
        self._state = _BreakerState()

    def is_open(self) -> bool:
        now = self._clock()
        if self._state.opened_until and now >= self._state.opened_until:
            self._state = _BreakerState()
        return self._state.opened_until > now

    def record_success(self) -> None:
        self._state = _BreakerState()

    def record_failure(self) -> None:
        now = self._clock()
        failures = self._state.failures + 1
        opened_until = self._state.opened_until
        if failures >= self.failure_threshold:
            opened_until = now + self.reset_after_seconds
        self._state = _BreakerState(failures=failures, opened_until=opened_until)


class GatewayError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retriable: bool = False,
        details: Mapping[str, Any] | None = None,
        route: Route | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retriable = retriable
        self.details = dict(details or {})
        self.route = route

    def __str__(self) -> str:
        base = super().__str__()
        suffix = [f"provider={self.provider}"]
        if self.status_code is not None:
            suffix.append(f"status={self.status_code}")
        if self.retriable:
            suffix.append("retriable=True")
        return f"{base} ({', '.join(suffix)})"


def _env_int(env: Mapping[str, str], key: str, default: int | None = None) -> int | None:
    value = env.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(env: Mapping[str, str], key: str, default: float | None = None) -> float | None:
    value = env.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _join_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _build_request(
    base_url: str,
    path: str,
    *,
    method: str,
    payload: Mapping[str, Any] | None = None,
    api_key: str | None = None,
    stream: bool = False,
) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if stream:
        headers["Accept"] = "text/event-stream"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return urllib.request.Request(
        _join_url(base_url, path),
        data=data,
        headers=headers,
        method=method,
    )


def _redact_text(text: str, secrets: Iterable[str]) -> str:
    redacted = text
    for secret in {secret for secret in secrets if secret}:
        redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _redact_structure(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, list):
        return [_redact_structure(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_structure(item, secrets) for item in value)
    if isinstance(value, dict):
        return {key: _redact_structure(item, secrets) for key, item in value.items()}
    return value


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _extract_usage(data: Mapping[str, Any]) -> Usage | None:
    usage = data.get("usage")
    if not isinstance(usage, Mapping):
        return None
    return Usage(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _extract_tool_calls(raw_calls: Any) -> tuple[ToolCall, ...]:
    if not isinstance(raw_calls, list):
        return ()
    calls: list[ToolCall] = []
    for item in raw_calls:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            arguments = function.get("arguments")
        else:
            name = item.get("name")
            arguments = item.get("arguments")
        if not isinstance(name, str):
            continue
        calls.append(
            ToolCall(
                id=item.get("id") if isinstance(item.get("id"), str) else None,
                name=name,
                arguments=arguments,
                type=item.get("type") if isinstance(item.get("type"), str) else "function",
            )
        )
    return tuple(calls)


def _extract_content(choice: Mapping[str, Any]) -> str:
    message = choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    return ""


def _extract_tool_calls_from_choice(choice: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    message = choice.get("message")
    if isinstance(message, Mapping):
        tool_calls = _extract_tool_calls(message.get("tool_calls"))
        if tool_calls:
            return tool_calls
    delta = choice.get("delta")
    if isinstance(delta, Mapping):
        return _extract_tool_calls(delta.get("tool_calls"))
    return ()


class ModelRouter:
    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()

    def choose(
        self,
        *,
        task_class: str,
        privacy: PrivacyPolicy,
        local_healthy: bool = True,
        complex_task: bool = False,
        estimated_input_tokens: int | None = None,
        attachment_count: int = 0,
        source_fan_in: int = 0,
        tool_complexity: int = 0,
    ) -> Route:
        local_safe = (
            local_healthy
            and not complex_task
            and tool_complexity <= self.config.local_max_tool_complexity
            and attachment_count <= self.config.local_max_attachment_fan_in
            and source_fan_in <= self.config.local_max_source_fan_in
            and (
                estimated_input_tokens is None
                or estimated_input_tokens <= self.config.local_context_limit
            )
        )

        if privacy is PrivacyPolicy.REMOTE_REQUIRED:
            return Route(
                provider="litellm_oci",
                reason=f"policy requires remote provider for {task_class}",
            )

        if privacy is PrivacyPolicy.LOCAL_ONLY:
            if not local_safe:
                raise RuntimeError("local_only task cannot be routed to OCI")
            return Route(
                provider="lmstudio",
                reason=f"local_only task fits local budget for {task_class}",
            )

        if local_safe:
            if task_class in LOCAL_FIRST_TASK_CLASSES:
                reason = f"local-first task class: {task_class}"
            else:
                reason = f"local provider can satisfy task class: {task_class}"
            return Route(provider="lmstudio", reason=reason)

        if privacy is PrivacyPolicy.LOCAL_PREFERRED:
            return Route(
                provider="litellm_oci",
                reason=f"local provider unavailable or over budget for {task_class}",
            )

        return Route(
            provider="litellm_oci",
            reason=f"remote escalation required for {task_class}",
        )


class ModelGateway:
    def __init__(
        self,
        *,
        router: ModelRouter | None = None,
        local_provider: ProviderConfig | None = None,
        remote_provider: ProviderConfig | None = None,
        clock=time.monotonic,
        sleep=time.sleep,
        opener=urllib.request.urlopen,
    ):
        self.router = router or ModelRouter()
        self.local_provider = local_provider
        self.remote_provider = remote_provider
        self._clock = clock
        self._sleep = sleep
        self._open = opener
        self.audit_events: list[AuditEvent] = []
        self._health: dict[str, ProviderHealth] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        for provider in filter(None, (local_provider, remote_provider)):
            self._breakers[provider.name] = CircuitBreaker(
                failure_threshold=provider.circuit_breaker_failures,
                reset_after_seconds=provider.circuit_breaker_reset_seconds,
                clock=clock,
            )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> "ModelGateway":
        env = os.environ if env is None else env
        router_config = RouterConfig(
            local_context_limit=_env_int(env, "MODEL_ROUTER_LOCAL_CONTEXT_LIMIT", 8192) or 8192,
            local_max_tool_complexity=_env_int(env, "MODEL_ROUTER_LOCAL_MAX_TOOL_COMPLEXITY", 1) or 1,
            local_max_attachment_fan_in=_env_int(env, "MODEL_ROUTER_LOCAL_MAX_ATTACHMENT_FAN_IN", 4) or 4,
            local_max_source_fan_in=_env_int(env, "MODEL_ROUTER_LOCAL_MAX_SOURCE_FAN_IN", 4) or 4,
        )
        local_provider = cls._provider_from_env("LMSTUDIO", "lmstudio", env)
        remote_provider = cls._provider_from_env("LITELLM", "litellm_oci", env)
        return cls(
            router=router or ModelRouter(router_config),
            local_provider=local_provider,
            remote_provider=remote_provider,
        )

    @staticmethod
    def _provider_from_env(
        prefix: str,
        provider_name: str,
        env: Mapping[str, str],
    ) -> ProviderConfig | None:
        base_url = env.get(f"{prefix}_BASE_URL")
        model = env.get(f"{prefix}_MODEL")
        if not base_url or not model:
            return None
        return ProviderConfig(
            name=provider_name,
            base_url=_normalize_base_url(base_url),
            model=model,
            api_key=env.get(f"{prefix}_API_KEY"),
            timeout_seconds=_env_float(env, f"{prefix}_TIMEOUT_SECONDS", 30.0) or 30.0,
            max_retries=_env_int(env, f"{prefix}_MAX_RETRIES", 1) or 1,
            retry_backoff_seconds=_env_float(env, f"{prefix}_RETRY_BACKOFF_SECONDS", 0.25) or 0.25,
            circuit_breaker_failures=_env_int(env, f"{prefix}_CIRCUIT_BREAKER_FAILURES", 3) or 3,
            circuit_breaker_reset_seconds=_env_float(env, f"{prefix}_CIRCUIT_BREAKER_RESET_SECONDS", 30.0) or 30.0,
        )

    def _provider(self, name: str) -> ProviderConfig | None:
        if self.local_provider and self.local_provider.name == name:
            return self.local_provider
        if self.remote_provider and self.remote_provider.name == name:
            return self.remote_provider
        return None

    def _breaker(self, name: str) -> CircuitBreaker | None:
        return self._breakers.get(name)

    def _provider_healthy(self, provider: ProviderConfig | None) -> bool:
        if provider is None:
            return False
        breaker = self._breaker(provider.name)
        return breaker is not None and not breaker.is_open()

    def _route_for(
        self,
        *,
        task_class: str,
        privacy: PrivacyPolicy,
        complex_task: bool = False,
        estimated_input_tokens: int | None = None,
        attachment_count: int = 0,
        source_fan_in: int = 0,
        tool_complexity: int = 0,
    ) -> Route:
        local_healthy = self._provider_healthy(self.local_provider)
        return self.router.choose(
            task_class=task_class,
            privacy=privacy,
            local_healthy=local_healthy,
            complex_task=complex_task,
            estimated_input_tokens=estimated_input_tokens,
            attachment_count=attachment_count,
            source_fan_in=source_fan_in,
            tool_complexity=tool_complexity,
        )

    def health_check(self, provider_name: str) -> ProviderHealth:
        provider = self._provider(provider_name)
        started = self._clock()
        if provider is None:
            health = ProviderHealth(
                provider=provider_name,
                healthy=False,
                latency_seconds=0.0,
                detail="provider is not configured",
            )
            self._health[provider_name] = health
            return health

        request = _build_request(
            provider.base_url,
            "/v1/models",
            method="GET",
            api_key=provider.api_key,
        )
        try:
            with self._open(request, timeout=provider.timeout_seconds) as response:
                raw = response.read()
                latency = self._clock() - started
                payload = json.loads(raw.decode("utf-8") or "{}") if raw else {}
                models = tuple(
                    item.get("id")
                    for item in payload.get("data", [])
                    if isinstance(item, Mapping) and isinstance(item.get("id"), str)
                )
                health = ProviderHealth(
                    provider=provider.name,
                    healthy=True,
                    latency_seconds=latency,
                    models=models,
                    status_code=getattr(response, "status", 200),
                )
                self._health[provider.name] = health
                breaker = self._breaker(provider.name)
                if breaker:
                    breaker.record_success()
                return health
        except Exception as exc:  # pragma: no cover - the specific branch is tested
            latency = self._clock() - started
            status_code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            detail = str(exc)
            health = ProviderHealth(
                provider=provider.name,
                healthy=False,
                latency_seconds=latency,
                detail=detail,
                status_code=status_code,
            )
            self._health[provider.name] = health
            breaker = self._breaker(provider.name)
            if breaker:
                breaker.record_failure()
            return health

    def _request_once(
        self,
        provider: ProviderConfig,
        *,
        payload: Mapping[str, Any],
        path: str = "/v1/chat/completions",
    ) -> ModelResult:
        request = _build_request(
            provider.base_url,
            path,
            method="POST",
            payload=payload,
            api_key=provider.api_key,
        )
        started = self._clock()
        try:
            with self._open(request, timeout=provider.timeout_seconds) as response:
                raw = response.read()
                latency = self._clock() - started
                data = json.loads(raw.decode("utf-8") or "{}") if raw else {}
                breaker = self._breaker(provider.name)
                if breaker:
                    breaker.record_success()
                health = ProviderHealth(
                    provider=provider.name,
                    healthy=True,
                    latency_seconds=latency,
                    models=(),
                    status_code=getattr(response, "status", 200),
                )
                self._health[provider.name] = health
                return self._normalize_result(
                    provider=provider,
                    payload=data,
                    latency=latency,
                )
        except Exception as exc:
            error = self._normalize_error(provider=provider, exc=exc, started=started)
            breaker = self._breaker(provider.name)
            if breaker:
                breaker.record_failure()
            raise error from exc

    def _stream_once(
        self,
        provider: ProviderConfig,
        *,
        payload: Mapping[str, Any],
        route: Route,
        path: str = "/v1/chat/completions",
    ):
        request = _build_request(
            provider.base_url,
            path,
            method="POST",
            payload=payload,
            api_key=provider.api_key,
            stream=True,
        )
        started = self._clock()
        try:
            with self._open(request, timeout=provider.timeout_seconds) as response:
                yield from self._parse_stream(
                    response,
                    provider=provider,
                    started=started,
                    route=route,
                )
        except Exception as exc:
            error = self._normalize_error(provider=provider, exc=exc, started=started)
            breaker = self._breaker(provider.name)
            if breaker:
                breaker.record_failure()
            raise error from exc

    def _normalize_error(self, *, provider: ProviderConfig, exc: Exception, started: float) -> GatewayError:
        status_code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        detail: dict[str, Any] = {}
        retriable = False
        message = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            body = exc.read()
            if body:
                try:
                    detail = json.loads(body.decode("utf-8"))
                    if isinstance(detail, Mapping):
                        error_payload = detail.get("error")
                        if isinstance(error_payload, Mapping):
                            message = str(error_payload.get("message", message))
                            detail = dict(error_payload)
                except Exception:
                    detail = {"body": body.decode("utf-8", errors="replace")}
            retriable = exc.code >= 500 or exc.code in {408, 409, 425, 429}
        elif isinstance(exc, urllib.error.URLError):
            retriable = True
            detail = {"reason": getattr(exc, "reason", message)}
        latency = self._clock() - started
        secrets = [provider.api_key] if provider.api_key else []
        message = _redact_text(message, secrets)
        detail = _redact_structure(detail, secrets)
        detail.setdefault("latency_seconds", latency)
        return GatewayError(
            message,
            provider=provider.name,
            status_code=status_code,
            retriable=retriable,
            details=detail,
        )

    def _normalize_result(
        self,
        *,
        provider: ProviderConfig,
        payload: Mapping[str, Any],
        latency: float,
    ) -> ModelResult:
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, Mapping):
            choice = {}
        content = _extract_content(choice)
        tool_calls = _extract_tool_calls_from_choice(choice)
        usage = _extract_usage(payload)
        response_id = payload.get("id") if isinstance(payload.get("id"), str) else None
        route = Route(
            provider=provider.name,
            reason=f"completed via {provider.name}",
            model=provider.model,
            base_url=provider.base_url,
        )
        return ModelResult(
            provider=provider.name,
            route=route,
            model=payload.get("model") if isinstance(payload.get("model"), str) else provider.model,
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            latency_seconds=latency,
            response_id=response_id,
            raw=dict(payload),
        )

    def _parse_stream(
        self,
        response,
        *,
        provider: ProviderConfig,
        started: float,
        route: Route | None,
    ):
        current_route = route or Route(
            provider=provider.name,
            reason=f"streaming via {provider.name}",
            model=provider.model,
            base_url=provider.base_url,
            streaming=True,
        )
        for raw_line in iter(response.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    breaker = self._breaker(provider.name)
                    if breaker:
                        breaker.record_success()
                    self._health[provider.name] = ProviderHealth(
                        provider=provider.name,
                        healthy=True,
                        latency_seconds=self._clock() - started,
                        models=(),
                        status_code=getattr(response, "status", 200),
                    )
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = {}
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, Mapping):
                        choice = first_choice
                yield StreamChunk(
                    provider=provider.name,
                    route=current_route,
                    delta=_extract_content(choice),
                    tool_calls=_extract_tool_calls_from_choice(choice),
                    usage=_extract_usage(event),
                    finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
                    raw=dict(event),
                )

    def _fallback_allowed(self, privacy: PrivacyPolicy, provider_name: str) -> bool:
        return privacy is not PrivacyPolicy.LOCAL_ONLY and provider_name == "lmstudio"

    def _call_with_retries(
        self,
        provider: ProviderConfig,
        *,
        payload: Mapping[str, Any],
    ) -> ModelResult:
        last_error: GatewayError | None = None
        attempts = provider.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._request_once(provider, payload=payload)
            except GatewayError as exc:
                last_error = exc
                if attempt < provider.max_retries and exc.retriable:
                    backoff = provider.retry_backoff_seconds * (2**attempt)
                    self._sleep(backoff)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("unreachable")

    def _stream_with_retries(
        self,
        provider: ProviderConfig,
        *,
        payload: Mapping[str, Any],
        route: Route,
    ):
        last_error: GatewayError | None = None
        attempts = provider.max_retries + 1
        for attempt in range(attempts):
            yielded_any = False
            try:
                for chunk in self._stream_once(provider, payload=payload, route=route):
                    yielded_any = True
                    yield chunk
                return
            except GatewayError as exc:
                last_error = exc
                # Once a provider has emitted output, retrying can replay a
                # request that already executed side effects such as tools.
                if not yielded_any and attempt < provider.max_retries and exc.retriable:
                    backoff = provider.retry_backoff_seconds * (2**attempt)
                    self._sleep(backoff)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("unreachable")

    def _select_provider(self, route: Route) -> ProviderConfig:
        provider = self._provider(route.provider)
        if provider is None:
            raise GatewayError(
                f"provider {route.provider} is not configured",
                provider=route.provider,
                retriable=False,
                details={"route": asdict(route)},
                route=route,
            )
        return provider

    def _audit(
        self,
        *,
        task_class: str,
        privacy: PrivacyPolicy,
        route: Route,
        messages: list[Mapping[str, Any]],
        estimated_input_tokens: int | None,
        complex_task: bool,
        attachment_count: int,
        source_fan_in: int,
        tool_complexity: int,
        streaming: bool,
    ) -> None:
        secrets = []
        for provider in filter(None, (self.local_provider, self.remote_provider)):
            if provider.api_key:
                secrets.append(provider.api_key)
        safe_messages = _redact_structure(messages, secrets)
        self.audit_events.append(
            AuditEvent(
                timestamp=self._clock(),
                task_class=task_class,
                privacy=privacy,
                provider=route.provider,
                reason=route.reason,
                request_hash=_hash_payload(safe_messages),
                streaming=streaming,
                estimated_input_tokens=estimated_input_tokens,
                complex_task=complex_task,
                attachment_count=attachment_count,
                source_fan_in=source_fan_in,
                tool_complexity=tool_complexity,
                route=route,
            )
        )

    def chat(
        self,
        messages: list[Mapping[str, Any]],
        *,
        task_class: str,
        privacy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED,
        complex_task: bool = False,
        estimated_input_tokens: int | None = None,
        attachment_count: int = 0,
        source_fan_in: int = 0,
        tool_complexity: int = 0,
        stream: bool = False,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> ModelResult | list[StreamChunk]:
        if stream:
            return list(
                self.stream_chat(
                    messages,
                    task_class=task_class,
                    privacy=privacy,
                    complex_task=complex_task,
                    estimated_input_tokens=estimated_input_tokens,
                    attachment_count=attachment_count,
                    source_fan_in=source_fan_in,
                    tool_complexity=tool_complexity,
                    extra_payload=extra_payload,
                )
            )
        route = self._route_for(
            task_class=task_class,
            privacy=privacy,
            complex_task=complex_task,
            estimated_input_tokens=estimated_input_tokens,
            attachment_count=attachment_count,
            source_fan_in=source_fan_in,
            tool_complexity=tool_complexity,
        )
        self._audit(
            task_class=task_class,
            privacy=privacy,
            route=route,
            messages=messages,
            estimated_input_tokens=estimated_input_tokens,
            complex_task=complex_task,
            attachment_count=attachment_count,
            source_fan_in=source_fan_in,
            tool_complexity=tool_complexity,
            streaming=stream,
        )
        provider = self._select_provider(route)
        safe_messages = _redact_structure(
            messages,
            [secret for secret in (provider.api_key,) if secret],
        )
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": safe_messages,
            "stream": stream,
        }
        if extra_payload:
            payload.update(extra_payload)

        try:
            return self._call_with_retries(provider, payload=payload)
        except GatewayError as exc:
            if self._fallback_allowed(privacy, provider.name):
                fallback = self._select_provider(
                    Route(
                        provider="litellm_oci",
                        reason=f"fallback after {provider.name} failure",
                    )
                )
                fallback_payload = dict(payload)
                fallback_payload["model"] = fallback.model
                return self._call_with_retries(fallback, payload=fallback_payload)
            raise exc

    def stream_chat(
        self,
        messages: list[Mapping[str, Any]],
        *,
        task_class: str,
        privacy: PrivacyPolicy = PrivacyPolicy.LOCAL_PREFERRED,
        complex_task: bool = False,
        estimated_input_tokens: int | None = None,
        attachment_count: int = 0,
        source_fan_in: int = 0,
        tool_complexity: int = 0,
        extra_payload: Mapping[str, Any] | None = None,
    ):
        route = self._route_for(
            task_class=task_class,
            privacy=privacy,
            complex_task=complex_task,
            estimated_input_tokens=estimated_input_tokens,
            attachment_count=attachment_count,
            source_fan_in=source_fan_in,
            tool_complexity=tool_complexity,
        )
        self._audit(
            task_class=task_class,
            privacy=privacy,
            route=route,
            messages=messages,
            estimated_input_tokens=estimated_input_tokens,
            complex_task=complex_task,
            attachment_count=attachment_count,
            source_fan_in=source_fan_in,
            tool_complexity=tool_complexity,
            streaming=True,
        )
        provider = self._select_provider(route)
        safe_messages = _redact_structure(
            messages,
            [secret for secret in (provider.api_key,) if secret],
        )
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": safe_messages,
            "stream": True,
        }
        if extra_payload:
            payload.update(extra_payload)
        yielded_any = False
        try:
            for chunk in self._stream_with_retries(provider, payload=payload, route=route):
                yielded_any = True
                yield chunk
        except GatewayError as exc:
            if yielded_any or not self._fallback_allowed(privacy, provider.name):
                raise exc
            fallback = self._select_provider(
                Route(
                    provider="litellm_oci",
                    reason=f"fallback after {provider.name} failure",
                )
            )
            fallback_payload = dict(payload)
            fallback_payload["model"] = fallback.model
            fallback_route = Route(
                provider=fallback.name,
                reason=f"fallback after {provider.name} failure",
                model=fallback.model,
                base_url=fallback.base_url,
                streaming=True,
            )
            for chunk in self._stream_with_retries(
                fallback,
                payload=fallback_payload,
                route=fallback_route,
            ):
                yield chunk
