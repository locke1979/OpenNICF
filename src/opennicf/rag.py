"""Compatibility wrapper over the provenance-aware knowledge platform."""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge import KnowledgePlatform, RetrievalFilters


@dataclass(frozen=True)
class Evidence:
    source_id: str
    text: str
    locator: str
    namespace_id: str = ""
    domain_id: str = ""
    system_id: str = ""
    component_id: str = ""
    environment: str = ""
    evidence_type: str = ""
    acl_scope: str = ""


class EvidenceIndex:
    """Thin adapter kept for the existing tool surface."""

    def __init__(self, platform: KnowledgePlatform | None = None):
        self.platform = platform or KnowledgePlatform.in_memory()

    def add(self, evidence: Evidence) -> None:
        self.platform.ingest(
            source_id=evidence.source_id,
            source_uri=evidence.locator,
            explicit_locator=evidence.locator,
            content=evidence.text,
            acl_scope="internal",
            source_kind="ad_hoc",
            source_type="evidence",
            channel="manual",
            domain="general",
            system="unknown",
            environment="unknown",
            parser_name="identity",
            parser_version="1",
        )

    def search(self, query: str) -> list[Evidence]:
        hits = self.platform.search(
            query,
            filters=RetrievalFilters(principal_acl_scopes=frozenset({"internal"})),
        )
        return [
            Evidence(
                hit.source_id,
                hit.text,
                hit.locator,
                namespace_id=hit.namespace_id,
                domain_id=hit.domain_id,
                system_id=hit.system_id,
                component_id=hit.component_id,
                environment=hit.environment,
                evidence_type=hit.evidence_type,
                acl_scope=hit.acl_scope,
            )
            for hit in hits
        ]
