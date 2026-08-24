"""Deterministic retrieval benchmark for the versioned representative corpus.

The benchmark deliberately measures the existing knowledge path in memory.  It
uses the same chunk records, embedding service, filters, and persistent symbol
index as production retrieval, while keeping the workload small enough for CI.
The result manifest contains only hashes and provenance references; fixture
text is never copied into a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from .embeddings import EmbeddingSpaceMismatch
from .models import EvidenceHit, RetrievalFilters, SearchCandidate
from .retrieval import _lexical_score, cosine_similarity
from .store import KnowledgePlatform

_TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)
_SECRET_RE = re.compile(r"(?:BEGIN [A-Z ]+PRIVATE KEY|(?:api[_-]?key|password|secret|token)\s*[:=])", re.IGNORECASE)
MODES = ("lexical", "vector", "hybrid", "symbol_aware")
DEFAULT_K = 5
MANIFEST_VERSION = 1


class BenchmarkManifestError(ValueError):
    """Raised when a benchmark manifest is missing, unsafe, or tampered with."""


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    path: str
    source_id: str
    source_type: str
    domain_id: str
    system_id: str
    acl_scope: str
    language: str
    content_sha256: str
    parser_version: str = "benchmark-v1"
    mime_type: str = "text/plain"


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str
    relevant_source_ids: tuple[str, ...]
    modes: tuple[str, ...] = MODES
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    relevant_symbols: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _fixture_from_payload(raw: Mapping[str, Any]) -> FixtureSpec:
    required = ("fixture_id", "path", "source_id", "source_type", "domain_id", "acl_scope", "language", "content_sha256")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise BenchmarkManifestError(f"fixture is missing required fields: {missing}")
    return FixtureSpec(
        fixture_id=str(raw["fixture_id"]),
        path=str(raw["path"]),
        source_id=str(raw["source_id"]),
        source_type=str(raw["source_type"]),
        domain_id=str(raw["domain_id"]),
        system_id=str(raw.get("system_id") or "benchmark"),
        acl_scope=str(raw["acl_scope"]),
        language=str(raw["language"]),
        content_sha256=str(raw["content_sha256"]),
        parser_version=str(raw.get("parser_version") or "benchmark-v1"),
        mime_type=str(raw.get("mime_type") or "text/plain"),
    )


def load_manifest(path: str | Path) -> tuple[dict[str, Any], tuple[FixtureSpec, ...], tuple[BenchmarkQuery, ...]]:
    """Load and verify a versioned fixture manifest, including content hashes."""
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkManifestError(f"cannot read benchmark manifest: {manifest_path}") from exc
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise BenchmarkManifestError("unsupported benchmark manifest version")
    fixtures = tuple(_fixture_from_payload(item) for item in payload.get("fixtures", ()))
    if not fixtures:
        raise BenchmarkManifestError("benchmark manifest has no fixtures")
    fixture_ids: set[str] = set()
    source_ids: set[str] = set()
    for fixture in fixtures:
        if fixture.fixture_id in fixture_ids or fixture.source_id in source_ids:
            raise BenchmarkManifestError("fixture and source IDs must be unique")
        fixture_ids.add(fixture.fixture_id)
        source_ids.add(fixture.source_id)
        fixture_path = (manifest_path.parent / fixture.path).resolve()
        if manifest_path.parent not in fixture_path.parents:
            raise BenchmarkManifestError(f"fixture escapes manifest directory: {fixture.path}")
        try:
            text = fixture_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BenchmarkManifestError(f"fixture does not exist: {fixture.path}") from exc
        if _SECRET_RE.search(text):
            raise BenchmarkManifestError(f"fixture is not sanitized: {fixture.path}")
        if _sha256(fixture_path) != fixture.content_sha256:
            raise BenchmarkManifestError(f"fixture hash mismatch: {fixture.path}")

    defaults = payload.get("default_filters") or {}
    queries: list[BenchmarkQuery] = []
    for raw in payload.get("queries", ()):
        if not raw.get("query_id") or not raw.get("text") or not raw.get("relevant_source_ids"):
            raise BenchmarkManifestError("query requires query_id, text, and relevant_source_ids")
        modes = tuple(raw.get("modes") or MODES)
        if any(mode not in MODES for mode in modes):
            raise BenchmarkManifestError(f"query has an unsupported retrieval mode: {modes}")
        if any(source_id not in source_ids for source_id in raw["relevant_source_ids"]):
            raise BenchmarkManifestError(f"query references an unknown source: {raw['query_id']}")
        filters = RetrievalFilters(
            principal_acl_scopes=frozenset(raw.get("principal_acl_scopes", defaults.get("principal_acl_scopes", ()))),
            domain_ids=tuple(raw.get("domain_ids", defaults.get("domain_ids", ()))),
            system_ids=tuple(raw.get("system_ids", defaults.get("system_ids", ()))),
            source_types=tuple(raw.get("source_types", defaults.get("source_types", ()))),
            embedding_space_id=raw.get("embedding_space_id"),
            limit=int(raw.get("limit", DEFAULT_K)),
            neighbor_window=int(raw.get("neighbor_window", 0)),
        )
        queries.append(BenchmarkQuery(
            query_id=str(raw["query_id"]),
            text=str(raw["text"]),
            relevant_source_ids=tuple(str(item) for item in raw["relevant_source_ids"]),
            modes=modes,
            filters=filters,
            relevant_symbols=tuple(str(item) for item in raw.get("relevant_symbols", ())),
        ))
    if not queries:
        raise BenchmarkManifestError("benchmark manifest has no queries")
    return payload, fixtures, tuple(queries)


def _hit(candidate: SearchCandidate, *, lexical: float, semantic: float, score: float, model: str, dimensions: int, space_id: str) -> EvidenceHit:
    chunk = candidate.chunk
    return EvidenceHit(
        source_id=chunk.source_id,
        source_version_id=chunk.source_version_id,
        artifact_hash=chunk.artifact_hash,
        locator=chunk.locator,
        text=chunk.text,
        excerpt_hash=chunk.chunk_hash,
        ingest_timestamp=candidate.version.ingest_timestamp,
        parser_version=chunk.parser_version,
        namespace_id=chunk.namespace_id,
        domain_id=chunk.domain_id,
        system_id=chunk.system_id,
        component_id=chunk.component_id,
        environment=chunk.environment,
        evidence_type=chunk.evidence_type,
        acl_scope=chunk.acl_scope,
        source_type=chunk.source_type,
        semantic_score=semantic,
        lexical_score=lexical,
        score=score,
        model=model,
        dimensions=dimensions,
        chunk_id=chunk.chunk_id,
        chunk_ordinal=chunk.ordinal,
        embedding_space_id=space_id,
        domain=chunk.domain,
        system=chunk.system,
        metadata=dict(chunk.metadata),
    )


def _rank_key(hit: EvidenceHit) -> tuple[float, int, str]:
    return (-hit.score, hit.chunk_ordinal, hit.chunk_id)


def compact_hits(hits: Iterable[EvidenceHit], *, max_chunks: int = 4, max_tokens: int = 256) -> tuple[EvidenceHit, ...]:
    """Bound context after ranking, deduplicating chunks and enforcing a token budget."""
    selected: list[EvidenceHit] = []
    used_tokens = 0
    seen: set[str] = set()
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        if len(selected) >= max_chunks:
            break
        tokens = _token_count(hit.text)
        if selected and used_tokens + tokens > max_tokens:
            continue
        selected.append(hit)
        seen.add(hit.chunk_id)
        used_tokens += min(tokens, max_tokens)
        if used_tokens >= max_tokens:
            break
    return tuple(selected)


def _metrics(ranked: tuple[EvidenceHit, ...], relevant: set[str], *, k: int, latency_ms: float, compacted: tuple[EvidenceHit, ...]) -> dict[str, Any]:
    first_rank = next((index for index, hit in enumerate(ranked, 1) if hit.source_id in relevant), None)
    return {
        "recall_at_k": 1.0 if any(hit.source_id in relevant for hit in ranked[:k]) else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "latency_ms": round(latency_ms, 3),
        "returned_chunks": len(ranked),
        "context_chunks_after_compaction": len(compacted),
        "context_tokens_after_compaction": sum(_token_count(hit.text) for hit in compacted),
    }


class RetrievalBenchmark:
    """Run four deterministic retrieval strategies against a KnowledgePlatform."""

    def __init__(
        self,
        platform: KnowledgePlatform,
        manifest_path: str | Path,
        *,
        clock: Callable[[], int] | None = None,
        k: int = DEFAULT_K,
        max_context_chunks: int = 4,
        max_context_tokens: int = 256,
    ):
        self.platform = platform
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest, self.fixtures, self.queries = load_manifest(self.manifest_path)
        self.clock = clock or time.perf_counter_ns
        self.k = max(1, k)
        self.max_context_chunks = max(1, max_context_chunks)
        self.max_context_tokens = max(1, max_context_tokens)
        self.fixture_by_source = {fixture.source_id: fixture for fixture in self.fixtures}

    def ingest_fixtures(self) -> None:
        """Ingest the manifest corpus using the same public ingestion path as production."""
        for fixture in self.fixtures:
            path = self.manifest_path.parent / fixture.path
            self.platform.ingest(
                source_id=fixture.source_id,
                source_uri=fixture.path,
                content=path.read_text(encoding="utf-8"),
                mime_type=fixture.mime_type,
                source_kind=fixture.source_type,
                source_type=fixture.source_type,
                evidence_type=fixture.source_type,
                domain=fixture.domain_id,
                domain_id=fixture.domain_id,
                system=fixture.system_id,
                system_id=fixture.system_id,
                environment="benchmark",
                acl_scope=fixture.acl_scope,
                parser_name="benchmark",
                parser_version=fixture.parser_version,
                metadata={"benchmark_fixture_id": fixture.fixture_id, "language": fixture.language},
            )

    def _candidates(self, query: BenchmarkQuery) -> list[SearchCandidate]:
        filters = query.filters.normalized()
        return self.platform.store.search_candidates(filters)

    def _vector(self, query: BenchmarkQuery, *, space_id: str) -> tuple[float, ...]:
        result = self.platform.embeddings.embed(
            [query.text], purpose="retrieval_query", prefer_gpu=False, space_id=space_id, privacy_policy="local_only"
        )
        return result.vectors[0] if result.vectors else ()

    def _rank(self, mode: str, query: BenchmarkQuery) -> tuple[EvidenceHit, ...]:
        candidates = self._candidates(query)
        space_id = self.platform.embeddings.active_space_id
        # Report the active benchmark space, not the optional CPU backend used
        # only when fallback is explicitly allowed.
        info = self.platform.embeddings.info(prefer_gpu=True)
        query_vector: tuple[float, ...] = ()
        if mode in {"vector", "hybrid"}:
            query_vector = self._vector(query, space_id=space_id)
        ranked: list[EvidenceHit] = []
        if mode == "symbol_aware":
            by_chunk = {candidate.chunk.chunk_id: candidate for candidate in candidates}
            symbols = [
                symbol for symbol in self.platform.search_code_symbols(query.text, filters=query.filters)
                if symbol.chunk_id in by_chunk
            ]
            query_terms = {term.lower() for term in _TOKEN_RE.findall(query.text)}
            def symbol_match(symbol: Any) -> tuple[float, str, int]:
                names = {symbol.name.lower(), symbol.qualified_name.lower(), symbol.kind.lower()}
                matched = sum(any(term in name for name in names) for term in query_terms)
                exact = 1.0 if query.text.lower() in names else 0.0
                return (exact + matched / max(1, len(query_terms)), symbol.name.lower(), symbol.line_start)
            symbols.sort(key=symbol_match, reverse=True)
            seen: set[str] = set()
            for position, symbol in enumerate(symbols):
                candidate = by_chunk.get(symbol.chunk_id)
                if candidate is None or candidate.chunk.chunk_id in seen:
                    continue
                seen.add(candidate.chunk.chunk_id)
                lexical = _lexical_score(query.text, candidate.chunk.text)
                ranked.append(_hit(candidate, lexical=lexical, semantic=0.0, score=1.0 / (position + 1), model=info.model, dimensions=info.dimensions, space_id=space_id))
        else:
            for candidate in candidates:
                lexical = _lexical_score(query.text, candidate.chunk.text)
                semantic = 0.0
                if mode in {"vector", "hybrid"}:
                    if candidate.embedding is None:
                        continue
                    candidate_space = candidate.embedding.embedding_space_id or f"{candidate.embedding.model}:{candidate.embedding.dimensions}:v1"
                    if candidate_space != space_id:
                        raise EmbeddingSpaceMismatch(f"query space {space_id} cannot search {candidate_space}")
                    semantic = cosine_similarity(query_vector, candidate.embedding.vector)
                score = lexical if mode == "lexical" else semantic if mode == "vector" else semantic * self.platform.retriever.semantic_weight + lexical * self.platform.retriever.lexical_weight
                ranked.append(_hit(candidate, lexical=lexical, semantic=semantic, score=score, model=info.model, dimensions=info.dimensions, space_id=space_id))
        return tuple(sorted(ranked, key=_rank_key)[: query.filters.limit])

    def run(self, *, ingest: bool = True) -> dict[str, Any]:
        if ingest:
            self.ingest_fixtures()
        started = self.clock()
        results: list[dict[str, Any]] = []
        for mode in MODES:
            applicable = [query for query in self.queries if mode in query.modes]
            for query in applicable:
                begin = self.clock()
                ranked = self._rank(mode, query)
                elapsed_ms = (self.clock() - begin) / 1_000_000
                compacted = compact_hits(ranked, max_chunks=self.max_context_chunks, max_tokens=self.max_context_tokens)
                query_metrics = _metrics(ranked, set(query.relevant_source_ids), k=self.k, latency_ms=elapsed_ms, compacted=compacted)
                results.append({
                    "mode": mode,
                    "query_id": query.query_id,
                    "query_hash": hashlib.sha256(query.text.encode()).hexdigest(),
                    "relevant_source_ids": list(query.relevant_source_ids),
                    "metrics": query_metrics,
                    "ranked": [
                        {
                            "rank": rank,
                            "chunk_id": hit.chunk_id,
                            "source_id": hit.source_id,
                            "source_version_id": hit.source_version_id,
                            "artifact_hash": hit.artifact_hash,
                            "locator": hit.locator,
                            "excerpt_hash": hit.excerpt_hash,
                            "score": round(hit.score, 8),
                            "embedding_space_id": hit.embedding_space_id,
                        }
                        for rank, hit in enumerate(ranked, 1)
                    ],
                    "compacted_chunk_ids": [hit.chunk_id for hit in compacted],
                })
        summaries: dict[str, Any] = {}
        for mode in MODES:
            mode_rows = [row for row in results if row["mode"] == mode]
            metric_rows = [row["metrics"] for row in mode_rows]
            summaries[mode] = {
                "query_count": len(mode_rows),
                "recall_at_k": round(mean(row["recall_at_k"] for row in metric_rows), 6) if metric_rows else 0.0,
                "mrr": round(mean(row["mrr"] for row in metric_rows), 6) if metric_rows else 0.0,
                "median_latency_ms": round(median(row["latency_ms"] for row in metric_rows), 3) if metric_rows else 0.0,
                "mean_returned_chunks": round(mean(row["returned_chunks"] for row in metric_rows), 3) if metric_rows else 0.0,
                "mean_context_tokens_after_compaction": round(mean(row["context_tokens_after_compaction"] for row in metric_rows), 3) if metric_rows else 0.0,
            }
        info = self.platform.embeddings.info(prefer_gpu=True)
        return {
            "benchmark_version": MANIFEST_VERSION,
            "fixture_manifest": self.manifest_path.name,
            "fixture_manifest_sha256": _manifest_hash(self.manifest),
            "fixture_set": self.manifest.get("fixture_set", "unknown"),
            "execution": {
                "knowledge_path": "KnowledgePlatform.in_memory",
                "fallback": bool(info.fallback),
                "fallback_reason": "deterministic_cpu_backend" if info.fallback else None,
                "embedding_space_id": self.platform.embeddings.active_space_id,
                "embedding_provider": info.provider,
                "embedding_model": info.model,
                "elapsed_ms": round((self.clock() - started) / 1_000_000, 3),
            },
            "metrics_definition": {
                "k": self.k,
                "recall_at_k": "fraction of queries with a relevant source in the first k chunks",
                "mrr": "mean reciprocal rank of the first relevant source",
                "latency_ms": "wall-clock retrieval duration per query; ranking is deterministic",
                "context_tokens_after_compaction": "Unicode word-token count after deduplication and bounded compaction",
            },
            "summaries": summaries,
            "results": results,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    platform = KnowledgePlatform.in_memory()
    report = RetrievalBenchmark(platform, args.manifest).run()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
