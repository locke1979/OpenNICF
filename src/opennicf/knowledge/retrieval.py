"""Provenance-aware hybrid retrieval with pre-model ACL filtering."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import sqrt

from .embeddings import EmbeddingSpaceMismatch, LocalFirstEmbeddingService
from .models import EvidenceHit, RetrievalEventRecord, RetrievalFilters, SearchCandidate


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_vector = tuple(left)
    right_vector = tuple(right)
    if len(left_vector) != len(right_vector) or not left_vector:
        return 0.0
    dot = sum(a * b for a, b in zip(left_vector, right_vector))
    left_norm = sqrt(sum(value * value for value in left_vector)) or 1.0
    right_norm = sqrt(sum(value * value for value in right_vector)) or 1.0
    return dot / (left_norm * right_norm)


_LEXICAL_TOKEN_RE = re.compile(r"[\w]+(?:[./:-][\w]+)*", re.UNICODE)
_LEXICAL_COMPONENT_RE = re.compile(r"[._/:+-]+")


def _lexical_terms(text: str) -> set[str]:
    """Return whole terms and identifier components for evidence text.

    Evidence frequently stores fields as ``source_version_id`` while users
    ask for the human-readable components. Keeping the complete identifier
    preserves exact-match behavior and adding components makes schema, log,
    and query-output retrieval usable without a separate analyzer.
    """
    terms: set[str] = set()
    for token in _LEXICAL_TOKEN_RE.findall(text):
        normalized = token.casefold()
        terms.add(normalized)
        terms.update(component for component in _LEXICAL_COMPONENT_RE.split(normalized) if component)
    return terms


def _lexical_score(query: str, text: str) -> float:
    query_terms = _lexical_terms(query)
    text_terms = _lexical_terms(text)
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


@dataclass
class RetrievalDecision:
    query_hash: str
    candidates: list[EvidenceHit] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, store, embeddings: LocalFirstEmbeddingService, *, semantic_weight: float = 0.75, lexical_weight: float = 0.25):
        self.store = store
        self.embeddings = embeddings
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def _score_candidate(self, query_vector: tuple[float, ...], candidate: SearchCandidate, query: str) -> EvidenceHit | None:
        embedding = candidate.embedding
        if embedding is None:
            semantic_score = 0.0
            model = self.embeddings.info().model
            dimensions = self.embeddings.info().dimensions
        else:
            query_space = self.embeddings.space()
            candidate_space_id = embedding.embedding_space_id or f"{embedding.model}:{embedding.dimensions}:v1"
            if candidate_space_id != query_space.embedding_space_id:
                raise EmbeddingSpaceMismatch(
                    f"query space {query_space.embedding_space_id} cannot search {candidate_space_id}"
                )
            semantic_score = cosine_similarity(query_vector, embedding.vector)
            model = embedding.model
            dimensions = embedding.dimensions
        lexical_score = _lexical_score(query, candidate.chunk.text)
        score = (semantic_score * self.semantic_weight) + (lexical_score * self.lexical_weight)
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
            model=model,
            dimensions=dimensions,
            chunk_id=candidate.chunk.chunk_id,
            chunk_ordinal=candidate.chunk.ordinal,
            embedding_space_id=(embedding.embedding_space_id if embedding else self.embeddings.active_space_id),
            domain=candidate.chunk.domain,
            system=candidate.chunk.system,
            metadata=dict(candidate.chunk.metadata),
        )

    def search(self, query: str, *, filters: RetrievalFilters | None = None, route: str | None = None) -> list[EvidenceHit]:
        filters = (filters or RetrievalFilters()).normalized()
        active_space_id = filters.embedding_space_id or self.embeddings.active_space_id
        filters = replace(filters, embedding_space_id=active_space_id)
        query_result = self.embeddings.embed([query], purpose="retrieval_query", prefer_gpu=True, space_id=active_space_id)
        query_vector = query_result.vectors[0] if query_result.vectors else ()
        candidates = self.store.search_candidates(filters)
        stored_embeddings = getattr(self.store, "embeddings", None)
        if stored_embeddings is not None and filters.embedding_space_id:
            mismatched_chunks = [
                candidate.chunk.chunk_id
                for candidate in candidates
                if candidate.embedding is None
                and any(chunk_id == candidate.chunk.chunk_id for chunk_id, _ in stored_embeddings)
            ]
            if mismatched_chunks:
                available = sorted({space_id for chunk_id, space_id in stored_embeddings if chunk_id in mismatched_chunks})
                raise EmbeddingSpaceMismatch(
                    f"query space {filters.embedding_space_id} has no compatible index for chunks "
                    f"{mismatched_chunks[:3]}; available spaces={available}"
                )
        allowed_domains = filters.effective_domain_ids()
        allowed_systems = filters.effective_system_ids()
        allowed_components = filters.effective_component_ids()
        allowed_evidence_types = filters.effective_evidence_types()
        allowed_namespaces = filters.effective_namespace_ids()
        scored: list[EvidenceHit] = []
        for candidate in candidates:
            if filters.principal_acl_scopes and candidate.chunk.acl_scope not in filters.principal_acl_scopes:
                continue
            if allowed_namespaces and candidate.chunk.namespace_id not in allowed_namespaces:
                continue
            if allowed_domains and candidate.chunk.domain_id not in allowed_domains:
                continue
            if allowed_systems and candidate.chunk.system_id not in allowed_systems:
                continue
            if allowed_components and candidate.chunk.component_id not in allowed_components:
                continue
            if allowed_evidence_types and candidate.chunk.evidence_type not in allowed_evidence_types:
                continue
            hit = self._score_candidate(query_vector, candidate, query)
            if hit is not None:
                scored.append(hit)
        scored.sort(key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))

        if filters.neighbor_window > 0 and scored:
            by_source: dict[str, list[SearchCandidate]] = {}
            for candidate in candidates:
                by_source.setdefault(candidate.chunk.source_version_id, []).append(candidate)
            for bucket in by_source.values():
                bucket.sort(key=lambda item: item.chunk.ordinal)
            expanded: list[EvidenceHit] = []
            seen = {hit.chunk_id for hit in scored}
            for hit in scored:
                expanded.append(hit)
                if filters.neighbor_window <= 0:
                    continue
                source_bucket = by_source.get(hit.source_version_id, [])
                if not source_bucket:
                    continue
                position = next((index for index, candidate in enumerate(source_bucket) if candidate.chunk.chunk_id == hit.chunk_id), None)
                if position is None:
                    continue
                for offset in range(1, filters.neighbor_window + 1):
                    for neighbor_index in (position - offset, position + offset):
                        if not 0 <= neighbor_index < len(source_bucket):
                            continue
                        neighbor = source_bucket[neighbor_index]
                        if neighbor.chunk.chunk_id in seen:
                            continue
                        seen.add(neighbor.chunk.chunk_id)
                        neighbor_hit = self._score_candidate(query_vector, neighbor, query)
                        if neighbor_hit is None:
                            continue
                        expanded.append(
                            EvidenceHit(
                                **{
                                    **neighbor_hit.__dict__,
                                    "score": max(0.0, neighbor_hit.score * 0.9),
                                    "neighboring_chunk_ids": (hit.chunk_id,),
                                }
                            )
                        )
            scored = expanded

        unique: dict[str, EvidenceHit] = {}
        for hit in scored:
            existing = unique.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                unique[hit.chunk_id] = hit
        final_hits = sorted(unique.values(), key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))[: filters.limit]
        event = RetrievalEventRecord(
            event_id=sha256(f"{query}:{len(final_hits)}".encode()).hexdigest(),
            query_hash=sha256(query.encode("utf-8")).hexdigest(),
            filters={
                "principal_acl_scopes": sorted(filters.principal_acl_scopes),
                "principal_domain_id": filters.principal_domain_id,
                "domain_ids": list(filters.domain_ids),
                "delegated_domain_ids": list(filters.delegated_domain_ids),
                "system_ids": list(filters.system_ids),
                "component_ids": list(filters.component_ids),
                "evidence_types": list(filters.evidence_types),
                "namespace_ids": list(filters.namespace_ids),
                "domains": list(filters.domains),
                "systems": list(filters.systems),
                "environments": list(filters.environments),
                "source_types": list(filters.source_types),
                "source_ids": list(filters.source_ids),
                "limit": filters.limit,
                "embedding_space_id": filters.embedding_space_id,
                "neighbor_window": filters.neighbor_window,
            },
            selected_chunks=tuple(hit.chunk_id for hit in final_hits),
            scores={hit.chunk_id: hit.score for hit in final_hits},
            reranker="lexical+semantic",
            model_route=route,
        )
        self.store.record_retrieval_event(event)
        return final_hits
