"""Liveness/readiness contracts that distinguish dependency failures."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum


class HealthState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class HealthReport:
    service: str
    state: HealthState
    dependencies: dict[str, HealthState]
    release_id: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == HealthState.OK

    @property
    def live(self) -> bool:
        return True

    def as_dict(self) -> dict:
        return {"service": self.service, "state": self.state.value, "ready": self.ready,
                "live": self.live, "release_id": self.release_id,
                "dependencies": {k: v.value for k, v in sorted(self.dependencies.items())}}


def check_health(service: str, checks: Iterable[tuple[str, Callable[[], bool]]], *, release_id: str | None = None) -> HealthReport:
    dependencies: dict[str, HealthState] = {}
    for name, check in checks:
        try:
            dependencies[name] = HealthState.OK if check() else HealthState.FAILED
        except Exception:  # noqa: BLE001 - dependency probes should never bubble
            dependencies[name] = HealthState.FAILED
    if all(v == HealthState.OK for v in dependencies.values()):
        state = HealthState.OK
    elif any(v == HealthState.OK for v in dependencies.values()):
        state = HealthState.DEGRADED
    else:
        state = HealthState.FAILED
    return HealthReport(service, state, dependencies, release_id)
