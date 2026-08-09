"""SIGEF/Encargos domain behavior on top of the shared domain-agent runtime.

This module contains domain vocabulary and boundary policy only.  It does not
create a model loop, provider client, or a second knowledge platform.
"""

from __future__ import annotations

from typing import Any, Mapping

from .domain_agent import DomainAgent, DomainProfile, DomainTools


SIGEF_DOMAIN_ID = "encargos_sigef"

# These are source-system aliases, not additional domains.  They are kept
# explicit so a prompt cannot turn a shared integration into owned evidence.
SIGEF_SYSTEM_ALIASES: dict[str, tuple[str, ...]] = {
    "sigef": ("sigef", "SIGEF"),
    "sigefbat": ("sigefbat", "SIGEFBAT"),
    "ef": ("ef", "EF", "ef_batch", "EF_BATCH"),
    "encargos": ("encargos", "ENCARGOS", "encargos_batch", "ENCARGOS_BATCH"),
}
SIGEF_OWNED_SYSTEMS = tuple(alias for aliases in SIGEF_SYSTEM_ALIASES.values() for alias in aliases)

# GFF, CCTDCWS and WSCTIGPS are deliberately boundary components.  SICJUT
# and SIGEPRA are collaboration targets, never retrieval namespaces owned by
# this agent.
INTEGRATION_BOUNDARIES = ("GFF", "CCTDCWS", "WSCTIGPS", "SICJUT", "SIGEPRA")
COLLABORATION_TARGETS = {
    "sicjut": ("contencioso_judicial", "SICJUT"),
    "sigepra": ("contencioso_administrativo", "SIGEPRA"),
    "gff": (None, "GFF"),
}

ENCARGOS_SIGEF_PROFILE = DomainProfile(
    domain_id=SIGEF_DOMAIN_ID,
    name="Encargos SIGEF",
    description=(
        "SIGEF and SIGEFBAT execution, EF/encargos batches, and their "
        "evidence-grounded financial integration outcomes."
    ),
    owned_systems=SIGEF_OWNED_SYSTEMS,
    integration_boundary_aliases=INTEGRATION_BOUNDARIES,
    permitted_evidence_classes=("document", "log", "code", "schema", "query-output"),
    permitted_tools=(
        "search_code_exact",
        "search_code_symbols",
        "search_code_semantic",
        "search_logs",
        "search_docs",
        "search_schema",
        "search_query_outputs",
        "correlate_encargos_batch",
        "request_integration_collaboration",
        "request_domain_diagnostic",
    ),
    system_prompt=(
        "You are the Encargos SIGEF domain agent. Restrict evidence to "
        "domain_id=encargos_sigef and SIGEF/SIGEFBAT/EF/encargos batch "
        "systems. Treat GFF, CCTDCWS, WSCTIGPS, SICJUT and SIGEPRA as "
        "integration boundaries; request explicit brokered collaboration "
        "instead of retrieving foreign-domain evidence."
    ),
).normalized()


def _system_aliases(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        matches = SIGEF_SYSTEM_ALIASES.get(key)
        if matches is None:
            raise PermissionError(f"system is outside the SIGEF/Encargos boundary: {text}")
        result.extend(matches)
    return tuple(dict.fromkeys(result))


class EncargosSigefTools(DomainTools):
    """SIGEF tools with batch correlation and explicit boundary delegation."""

    def correlate_encargos_batch(self, request: str | Mapping[str, Any]) -> dict[str, Any]:
        payload = self._normalize(request)
        batch_id = str(payload.get("batch_id") or "").strip()
        if not batch_id:
            raise ValueError("batch_id is required for EF/encargos correlation")
        payload["query"] = " ".join(
            value
            for value in (
                batch_id,
                str(payload.get("predecessor_id") or "").strip(),
                str(payload.get("successor_id") or "").strip(),
            )
            if value
        )
        payload["system_ids"] = _system_aliases(payload.get("systems") or payload.get("system_ids"))
        # DomainTools performs the domain and ACL checks before searching.
        evidence = self._semantic_search(
            payload,
            source_types=("document", "log", "code", "schema", "query-output"),
            retrieval_mode="encargos_batch_correlation",
        )
        return {
            "domain_id": SIGEF_DOMAIN_ID,
            "batch_id": batch_id,
            "predecessor_id": payload.get("predecessor_id"),
            "successor_id": payload.get("successor_id"),
            "integration_boundaries": list(INTEGRATION_BOUNDARIES),
            "evidence": evidence,
        }

    def request_integration_collaboration(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        target = str(payload.get("target") or payload.get("system") or "").strip().lower()
        if target not in COLLABORATION_TARGETS:
            raise PermissionError("collaboration target is not an approved SIGEF integration boundary")
        target_domain, target_system = COLLABORATION_TARGETS[target]
        delegated = payload.get("delegated_domain_ids")
        if target_domain is not None:
            if isinstance(delegated, str):
                delegated = (delegated,)
            if tuple(delegated or ()) != (target_domain,):
                raise PermissionError("foreign-domain collaboration requires explicit single-domain delegation")
        if self.diagnostic_broker is None:
            raise NotImplementedError("integration collaboration requires the shared broker")
        payload.update(
            {
                "domain_id": SIGEF_DOMAIN_ID,
                "target_system": target_system,
                "operation_class": payload.get("operation_class", "read_select"),
            }
        )
        if payload["operation_class"] not in {"read_select", "describe"}:
            raise PermissionError("collaboration operation is not read-only")
        return dict(self.diagnostic_broker.request(payload))

    def as_qwen_tools(self) -> dict[str, Any]:
        tools = super().as_qwen_tools()
        tools.update(
            {
                "correlate_encargos_batch": self.correlate_encargos_batch,
                "request_integration_collaboration": self.request_integration_collaboration,
            }
        )
        return tools


class EncargosSigefDomainAgent(DomainAgent):
    """QwenAgent-backed SIGEF agent produced by :class:`DomainAgentFactory`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tools = EncargosSigefTools(
            self.profile,
            self.knowledge,
            diagnostic_broker=kwargs.get("diagnostic_broker"),
        )


__all__ = [
    "COLLABORATION_TARGETS",
    "ENCARGOS_SIGEF_PROFILE",
    "EncargosSigefDomainAgent",
    "INTEGRATION_BOUNDARIES",
    "SIGEF_DOMAIN_ID",
    "SIGEF_OWNED_SYSTEMS",
    "SIGEF_SYSTEM_ALIASES",
]
