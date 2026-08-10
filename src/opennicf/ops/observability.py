"""Small dependency-free observability primitives with safe serialization."""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class TraceContext:
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None

    def fields(self) -> dict[str, str]:
        value = {"correlation_id": self.correlation_id}
        if self.parent_id:
            value["parent_id"] = self.parent_id
        return value


class Metrics:
    """In-process counters and observations; export is intentionally deterministic."""
    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._observations: dict[str, list[float]] = {}

    def inc(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        self._observations.setdefault(name, []).append(float(value))

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": dict(sorted(self._counters.items())),
            "observations": {k: list(v) for k, v in sorted(self._observations.items())},
        }


class AuditLog:
    """Append-only audit records with hash chaining and default content redaction."""
    def __init__(self, *, secret_values: Iterable[str] = ()) -> None:
        self._records: list[dict[str, Any]] = []
        self._secrets = tuple(s for s in secret_values if s)

    def append(self, action: str, context: TraceContext, **fields: Any) -> dict[str, Any]:
        safe = _redact(fields, self._secrets)
        previous = self._records[-1]["record_hash"] if self._records else "GENESIS"
        record = {"action": action, **context.fields(), "fields": safe, "previous_hash": previous}
        record["record_hash"] = sha256(_canonical(record).encode()).hexdigest()
        self._records.append(record)
        return dict(record)

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)


def structured_event(event: str, context: TraceContext, **fields: Any) -> str:
    """Return one JSON log line. Callers should pass identifiers, not prompt/evidence bodies."""
    return _canonical({"event": event, **context.fields(), **fields})


def _redact(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {str(k): _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v, secrets) for v in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
