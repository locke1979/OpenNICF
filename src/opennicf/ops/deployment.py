"""Controlled deployment gates with deterministic promotion and rollback."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DeploymentResult:
    release_id: str
    state: str
    gates: tuple[GateResult, ...]
    rolled_back_to: str | None = None


class DeploymentController:
    """The runner owns this state; GitHub CI only builds and verifies artifacts."""
    def __init__(self, *, known_good: str, deploy: Callable[[str], None], rollback: Callable[[str], None]) -> None:
        self.known_good = known_good
        self._deploy = deploy
        self._rollback = rollback

    def deploy_release(self, release_id: str, *, migration_gate: Callable[[], bool], health_gate: Callable[[], bool], smoke_gate: Callable[[], bool]) -> DeploymentResult:
        gates: list[GateResult] = []
        for name, gate in (("migration", migration_gate), ("health", health_gate), ("smoke", smoke_gate)):
            try:
                passed = bool(gate())
                detail = "passed" if passed else "failed"
            except Exception as exc:  # noqa: BLE001 - gate failures must trigger rollback
                passed, detail = False, f"failed: {type(exc).__name__}"
            gates.append(GateResult(name, passed, detail))
            if not passed:
                self._rollback(self.known_good)
                return DeploymentResult(release_id, "rolled_back", tuple(gates), self.known_good)
        try:
            self._deploy(release_id)
        except Exception as exc:  # noqa: BLE001 - deployment failures must trigger rollback
            gates.append(GateResult("deploy", False, f"failed: {type(exc).__name__}"))
            self._rollback(self.known_good)
            return DeploymentResult(release_id, "rolled_back", tuple(gates), self.known_good)
        self.known_good = release_id
        return DeploymentResult(release_id, "promoted", tuple(gates))

    def synthetic_deploy(self, release_id: str) -> DeploymentResult:
        return self.deploy_release(release_id, migration_gate=lambda: True, health_gate=lambda: True, smoke_gate=lambda: True)
