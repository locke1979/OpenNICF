"""Provenance-aware hybrid retrieval with pre-model ACL filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import sqrt
from typing import Any, Iterable

from .embeddings import EmbeddingResult, LocalFirstEmbeddingService
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


def _lexical_score(query: str, text: str) -> float:
    query_terms = {term.lower() for term in query.split() if term.strip()}
    text_terms = {term.lower() for term in text.split() if term.strip()}
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
            acl_scope=candidate.chunk.acl_scope,
            domain=candidate.chunk.domain,
            system=candidate.chunk.system,
            environment=candidate.chunk.environment,
            source_type=candidate.chunk.source_type,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            score=score,
            model=model,
            dimensions=dimensions,
            chunk_id=candidate.chunk.chunk_id,
            chunk_ordinal=candidate.chunk.ordinal,
            metadata=dict(candidate.chunk.metadata),
        )

    def search(self, query: str, *, filters: RetrievalFilters | None = None, route: str | None = None) -> list[EvidenceHit]:
        filters = (filters or RetrievalFilters()).normalized()
        query_result = self.embeddings.embed([query], prefer_gpu=True)
        query_vector = query_result.vectors[0] if query_result.vectors else ()
        candidates = self.store.search_candidates(filters)
        scored: list[EvidenceHit] = []
        for candidate in candidates:
            if filters.principal_acl_scopes and candidate.chunk.acl_scope not in filters.principal_acl_scopes:
                continue
            hit = self._score_candidate(query_vector, candidate, query)
            if hit is not None:
                scored.append(hit)
        scored.sort(key=lambda item: (-item.score, item.chunk_ordinal, item.chunk_id))

        if filters.neighbor_window > 0 and scored:
            candidate_map = {candidate.chunk.chunk_id: candidate for candidate in candidates}
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
            event_id=sha256(f"{query}:{len(final_hits)}".encode("utf-8")).hexdigest(),
            query_hash=sha256(query.encode("utf-8")).hexdigest(),
            filters={
                "principal_acl_scopes": sorted(filters.principal_acl_scopes),
                "domains": list(filters.domains),
                "systems": list(filters.systems),
                "environments": list(filters.environments),
                "source_types": list(filters.source_types),
                "source_ids": list(filters.source_ids),
                "limit": filters.limit,
                "neighbor_window": filters.neighbor_window,
            },
            selected_chunks=tuple(hit.chunk_id for hit in final_hits),
            scores={hit.chunk_id: hit.score for hit in final_hits},
            reranker="lexical+semantic",
            model_route=route,
        )
        self.store.record_retrieval_event(event)
        return final_hits

