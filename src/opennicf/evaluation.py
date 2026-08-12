"""Model-free evaluation pipeline contracts for issue #88 Gate A.

The module is intentionally independent of the production knowledge store and
model gateway.  It provides deterministic orchestration around ranked IDs,
provider-neutral reranker contracts, and bounded post-rerank context assembly.
No method here downloads, invokes, or persists a model artifact.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from .evaluation_contract import (
    EmbeddingVariant,
    InMemoryRepresentationStore,
    ManifestValidationError,
    RepresentationRecord,
    compare_embeddings,
    rrf_rank,
    stable_representation_id,
    validate_manifest,
)

_TOKEN_RE = re.compile(r"[\w./:-]+", re.UNICODE)
_PAGE_RE = re.compile(r"(?:^|[:/#=_ -])page[:/#=_ -]?([0-9]+)(?:$|[:/#=_ -])", re.IGNORECASE)
FAILURE_POLICIES = {"FAIL_QUERY", "FALLBACK_TO_FUSED_ORDER", "SKIP_UNSUPPORTED_CANDIDATE"}
MODALITIES = {"text", "page_image", "region_image", "mixed"}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE_RESULT_RE = re.compile(
    r"(?:api[_-]?key|password|secret|authorization)\s*[:=]|"
    r"(?:/var/lib/opennicf|/etc/opennicf|production[_-]?data|private[_-]?evidence|raw[_-]?evidence)",
    re.IGNORECASE,
)


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True)
class RerankerInfo:
    """Provider-neutral identity and declared capabilities of a reranker."""

    id: str
    model: str = "evaluation-fake"
    model_revision: str = "model-free"
    quantization: str = "none"
    capabilities: frozenset[str] = frozenset({"text"})
    route: str = "evaluation"
    failure_policy: str = "FAIL_QUERY"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("reranker id is required")
        capabilities = frozenset(self.capabilities)
        if not capabilities or not capabilities <= MODALITIES:
            raise ValueError("reranker capabilities must contain known modalities")
        if self.failure_policy not in FAILURE_POLICIES:
            raise ValueError(f"unsupported reranker failure policy: {self.failure_policy}")
        object.__setattr__(self, "capabilities", capabilities)


@dataclass(frozen=True)
class RerankQuery:
    text: str
    query_id: str = ""


@dataclass(frozen=True)
class RerankCandidate:
    """A reranker input whose provenance cannot be reconstructed by a model."""

    candidate_id: str
    representation_id: str
    text: str
    representation_type: str
    source_id: str
    source_version_id: str
    artifact_hash: str
    locator: str
    chunk_id: str | None = None
    parent_chunk_id: str | None = None
    acl_scope: str = ""
    domain_id: str = ""
    system_id: str = ""
    component_id: str = "unknown-component"
    environment: str = ""
    evidence_type: str = ""
    lifecycle_status: str = "ready"
    pixel_width: int | None = None
    pixel_height: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.representation_id:
            raise ValueError("candidate and representation IDs are required")
        if self.representation_type not in MODALITIES:
            raise ValueError(f"unsupported representation_type: {self.representation_type}")
        if self.lifecycle_status not in {"pending", "ready", "failed", "quarantined", "retired"}:
            raise ValueError(f"unsupported lifecycle_status: {self.lifecycle_status}")
        if self.representation_type != "text" and (
            self.pixel_width is not None and self.pixel_width <= 0
            or self.pixel_height is not None and self.pixel_height <= 0
        ):
            raise ValueError("image dimensions must be positive when provided")

    @property
    def image(self) -> bool:
        return self.representation_type != "text"

    @property
    def pixels(self) -> int:
        return (self.pixel_width or 0) * (self.pixel_height or 0)


@dataclass(frozen=True)
class RerankLimits:
    """Explicit resource limits passed to every provider implementation."""

    max_candidate_pairs: int = 50
    max_tokens: int = 4096
    max_images: int = 4
    max_pixels: int = 4_000_000
    batch_size: int = 8
    timeout_ms: int = 2_000

    def __post_init__(self) -> None:
        for name in ("max_candidate_pairs", "max_tokens", "max_images", "max_pixels", "batch_size", "timeout_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RerankedCandidate:
    candidate: RerankCandidate
    score: float
    original_rank: int

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id


@dataclass(frozen=True)
class RerankResult:
    reranked: tuple[RerankedCandidate, ...]
    requested_reranker: str
    actual_reranker: str | None
    elapsed_ms: float
    eligible_count: int
    scored_count: int
    skipped_count: int = 0
    failure_classification: str | None = None
    error: str | None = None
    reranker_applied: bool = True
    fallback_invoked: bool = False
    skipped_candidate_ids: tuple[str, ...] = ()


class RerankerError(RuntimeError):
    """A bounded, classifiable reranker failure."""

    classification = "reranker_error"

    def __init__(self, message: str, *, candidate_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.candidate_ids = tuple(candidate_ids)


class UnsupportedModalityError(RerankerError):
    classification = "unsupported_modality"


class RerankerTimeout(RerankerError):
    classification = "timeout"


class RerankerOutOfMemory(RerankerError):
    classification = "oom"


class MalformedRerankerOutput(RerankerError):
    classification = "malformed_output"


class Reranker(Protocol):
    @property
    def info(self) -> RerankerInfo: ...

    def score(
        self,
        query: RerankQuery,
        candidates: Sequence[RerankCandidate],
        *,
        limits: RerankLimits,
    ) -> RerankResult: ...


class FakeReranker:
    """Deterministic fake for contract and coordinator tests.

    ``scores`` lets a test describe an expected ranking without depending on a
    model.  With no mapping, a small lexical overlap score is used.  Failures
    are injected by classification name and are never hidden.
    """

    def __init__(
        self,
        info: RerankerInfo | None = None,
        *,
        scores: Mapping[str, float] | None = None,
        failure: str | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        self._info = info or RerankerInfo(id="FAKE_TEXT_RERANKER_V1")
        self.scores = dict(scores or {})
        self.failure = failure
        self.elapsed_ms = elapsed_ms
        self.calls: list[tuple[str, tuple[str, ...], RerankLimits]] = []

    @property
    def info(self) -> RerankerInfo:
        return self._info

    def _supports(self, candidate: RerankCandidate) -> bool:
        return candidate.representation_type in self.info.capabilities

    def _score(self, query: RerankQuery, candidate: RerankCandidate) -> float:
        if candidate.candidate_id in self.scores:
            return float(self.scores[candidate.candidate_id])
        query_terms = set(_TOKEN_RE.findall(query.text.casefold()))
        candidate_terms = set(_TOKEN_RE.findall(candidate.text.casefold()))
        return len(query_terms & candidate_terms) / max(1, len(query_terms))

    def score(self, query: RerankQuery, candidates: Sequence[RerankCandidate], *, limits: RerankLimits) -> RerankResult:
        candidate_tuple = tuple(candidates)
        self.calls.append((query.query_id or query.text, tuple(item.candidate_id for item in candidate_tuple), limits))
        if self.failure:
            errors: dict[str, type[RerankerError]] = {
                "timeout": RerankerTimeout,
                "oom": RerankerOutOfMemory,
                "malformed_output": MalformedRerankerOutput,
                "unsupported_modality": UnsupportedModalityError,
            }
            error_type = errors.get(self.failure, RerankerError)
            raise error_type(f"fake reranker failure: {self.failure}")
        unsupported = tuple(item.candidate_id for item in candidate_tuple if not self._supports(item))
        if unsupported:
            raise UnsupportedModalityError("reranker does not support candidate modality", candidate_ids=unsupported)
        if len(candidate_tuple) > limits.max_candidate_pairs:
            raise RerankerError("candidate pair limit exceeded")
        total_tokens = sum(_token_count(item.text) for item in candidate_tuple) + _token_count(query.text)
        if total_tokens > limits.max_tokens:
            raise RerankerError("token limit exceeded")
        image_count = sum(item.image for item in candidate_tuple)
        if image_count > limits.max_images:
            raise RerankerError("image limit exceeded")
        if sum(item.pixels for item in candidate_tuple) > limits.max_pixels:
            raise RerankerError("pixel limit exceeded")
        if self.elapsed_ms > limits.timeout_ms:
            raise RerankerTimeout("fake reranker exceeded timeout")
        ranked = [
            RerankedCandidate(item, self._score(query, item), rank)
            for rank, item in enumerate(candidate_tuple, 1)
        ]
        ranked.sort(key=lambda item: (-item.score, item.original_rank, item.candidate_id))
        return RerankResult(
            reranked=tuple(ranked),
            requested_reranker=self.info.id,
            actual_reranker=self.info.id,
            elapsed_ms=self.elapsed_ms,
            eligible_count=len(candidate_tuple),
            scored_count=len(ranked),
        )


@dataclass(frozen=True)
class EvaluationFilters:
    """Pre-retrieval ACL/domain/system boundary for evaluation candidates."""

    acl_scopes: frozenset[str] = frozenset()
    domain_ids: frozenset[str] = frozenset()
    system_ids: frozenset[str] = frozenset()

    def allows(self, candidate: RerankCandidate) -> bool:
        return (
            candidate.lifecycle_status == "ready"
            and (not self.acl_scopes or candidate.acl_scope in self.acl_scopes)
            and (not self.domain_ids or candidate.domain_id in self.domain_ids)
            and (not self.system_ids or candidate.system_id in self.system_ids)
        )


@dataclass(frozen=True)
class ContextLimits:
    max_candidates: int = 4
    max_tokens: int = 256
    max_images: int = 4
    max_pixels: int = 4_000_000
    max_per_source: int = 2
    max_per_page: int = 1

    def __post_init__(self) -> None:
        for name in ("max_candidates", "max_tokens", "max_images", "max_pixels", "max_per_source", "max_per_page"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SuppressedCandidate:
    candidate: RerankCandidate
    reason: str


@dataclass(frozen=True)
class ContextSelectionResult:
    selected: tuple[RerankedCandidate, ...]
    suppressed: tuple[SuppressedCandidate, ...]
    token_count: int
    image_count: int
    pixel_count: int


def _page_key(candidate: RerankCandidate) -> tuple[str, str] | None:
    match = _PAGE_RE.search(candidate.locator)
    if match:
        return candidate.source_id, match.group(1)
    if candidate.representation_type in {"page_image", "region_image"} and candidate.locator:
        return candidate.source_id, candidate.locator
    return None


def select_context(
    candidates: Sequence[RerankedCandidate],
    *,
    limits: ContextLimits,
) -> ContextSelectionResult:
    """Deduplicate and bound context after reranking, preserving suppression reasons."""
    selected: list[RerankedCandidate] = []
    suppressed: list[SuppressedCandidate] = []
    seen_representations: set[str] = set()
    source_counts: dict[str, int] = {}
    page_counts: dict[tuple[str, str], int] = {}
    tokens = images = pixels = 0
    for ranked in candidates:
        candidate = ranked.candidate
        if candidate.representation_id in seen_representations:
            suppressed.append(SuppressedCandidate(candidate, "duplicate_representation"))
            continue
        if source_counts.get(candidate.source_id, 0) >= limits.max_per_source:
            suppressed.append(SuppressedCandidate(candidate, "source_limit"))
            continue
        page = _page_key(candidate)
        if page is not None and page_counts.get(page, 0) >= limits.max_per_page:
            suppressed.append(SuppressedCandidate(candidate, "page_limit"))
            continue
        candidate_tokens = _token_count(candidate.text)
        candidate_images = int(candidate.image)
        candidate_pixels = candidate.pixels
        if len(selected) >= limits.max_candidates:
            suppressed.append(SuppressedCandidate(candidate, "candidate_limit"))
            continue
        if tokens + candidate_tokens > limits.max_tokens:
            suppressed.append(SuppressedCandidate(candidate, "token_limit"))
            continue
        if images + candidate_images > limits.max_images:
            suppressed.append(SuppressedCandidate(candidate, "image_limit"))
            continue
        if pixels + candidate_pixels > limits.max_pixels:
            suppressed.append(SuppressedCandidate(candidate, "pixel_limit"))
            continue
        selected.append(ranked)
        seen_representations.add(candidate.representation_id)
        source_counts[candidate.source_id] = source_counts.get(candidate.source_id, 0) + 1
        if page is not None:
            page_counts[page] = page_counts.get(page, 0) + 1
        tokens += candidate_tokens
        images += candidate_images
        pixels += candidate_pixels
    return ContextSelectionResult(tuple(selected), tuple(suppressed), tokens, images, pixels)


@dataclass(frozen=True)
class EvaluationResult:
    query: RerankQuery
    filtered_candidates: tuple[RerankCandidate, ...]
    lane_orders: Mapping[str, tuple[str, ...]]
    fused_candidates: tuple[RerankCandidate, ...]
    candidate_recall: float | None
    relevant_candidate_count: int
    relevant_candidates_found: int
    rerank: RerankResult | None
    reranked_candidates: tuple[RerankedCandidate, ...]
    context: ContextSelectionResult
    phase_order: tuple[str, ...]


def _candidate_provenance(candidate: RerankCandidate) -> dict[str, Any]:
    """Return the immutable, non-content fields allowed in a result manifest."""
    return {
        "candidate_id": candidate.candidate_id,
        "representation_id": candidate.representation_id,
        "representation_type": candidate.representation_type,
        "source_id": candidate.source_id,
        "source_version_id": candidate.source_version_id,
        "artifact_hash": candidate.artifact_hash,
        "locator": candidate.locator,
        "chunk_id": candidate.chunk_id,
        "parent_chunk_id": candidate.parent_chunk_id,
        "domain_id": candidate.domain_id,
        "system_id": candidate.system_id,
        "component_id": candidate.component_id,
        "environment": candidate.environment,
        "evidence_type": candidate.evidence_type,
    }


def build_result_manifest(
    result: EvaluationResult,
    *,
    experiment_manifest_digest: str,
    arm_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Serialize an evaluation result without query text, evidence text, ACLs, or metadata."""
    rerank = result.rerank
    payload: dict[str, Any] = {
        "result_manifest_version": 1,
        "production": False,
        "production_route_changed": False,
        "experiment_manifest_digest": experiment_manifest_digest,
        "arm_id": arm_id,
        "run_id": run_id,
        "query_id": result.query.query_id,
        "phase_order": list(result.phase_order),
        "lane_orders": {name: list(ids) for name, ids in sorted(result.lane_orders.items())},
        "candidate_recall": result.candidate_recall,
        "relevant_candidate_count": result.relevant_candidate_count,
        "relevant_candidates_found": result.relevant_candidates_found,
        "fused_candidates": [_candidate_provenance(item) for item in result.fused_candidates],
        "reranked_candidates": [
            {**_candidate_provenance(item.candidate), "score": item.score, "original_rank": item.original_rank}
            for item in result.reranked_candidates
        ],
        "context": {
            "selected_ids": [item.candidate_id for item in result.context.selected],
            "suppressions": [
                {"candidate_id": item.candidate.candidate_id, "reason": item.reason}
                for item in result.context.suppressed
            ],
            "token_count": result.context.token_count,
            "image_count": result.context.image_count,
            "pixel_count": result.context.pixel_count,
        },
        "rerank": None if rerank is None else {
            "requested_reranker": rerank.requested_reranker,
            "actual_reranker": rerank.actual_reranker,
            "elapsed_ms": rerank.elapsed_ms,
            "eligible_count": rerank.eligible_count,
            "scored_count": rerank.scored_count,
            "skipped_count": rerank.skipped_count,
            "failure_classification": rerank.failure_classification,
            "reranker_applied": rerank.reranker_applied,
            "fallback_invoked": rerank.fallback_invoked,
            "skipped_candidate_ids": list(rerank.skipped_candidate_ids),
        },
    }
    validate_result_manifest(payload)
    return payload


def validate_result_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed if a generated result contract is incomplete or unsafe."""
    errors: list[str] = []
    if manifest.get("result_manifest_version") != 1:
        errors.append("unsupported result manifest version")
    if manifest.get("production") is not False or manifest.get("production_route_changed") is not False:
        errors.append("result manifest must remain evaluation-only")
    if not _HEX64_RE.fullmatch(str(manifest.get("experiment_manifest_digest", ""))):
        errors.append("experiment manifest digest must be SHA-256")
    if not manifest.get("arm_id") or not manifest.get("run_id") or not manifest.get("query_id"):
        errors.append("arm, run, and query IDs are required")
    expected_order = list(EvaluationCoordinator.PHASE_ORDER)
    if manifest.get("phase_order") != expected_order:
        errors.append("result phase order is invalid")
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if _UNSAFE_RESULT_RE.search(raw):
        errors.append("unsafe result content is forbidden")
    forbidden_keys = {"text", "metadata", "acl_scope", "error"}
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys.intersection(value):
                errors.append("raw content, metadata, ACLs, and errors are forbidden")
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
    walk(manifest)
    for section in ("fused_candidates", "reranked_candidates"):
        for candidate in manifest.get(section, []):
            if not _HEX64_RE.fullmatch(str(candidate.get("artifact_hash", ""))):
                errors.append(f"{section} artifact hash must be SHA-256")
    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))


CandidateSource: TypeAlias = Mapping[str, RerankCandidate] | Sequence[RerankCandidate]


class EvaluationCoordinator:
    """Execute the model-free evaluation ordering as an auditable pipeline."""

    PHASE_ORDER = (
        "filter",
        "independent_ranked_lists",
        "rrf",
        "candidate_recall",
        "rerank",
        "late_deduplication",
        "context_selection",
    )

    def __init__(
        self,
        *,
        k_rrf: int = 60,
        rerank_limits: RerankLimits | None = None,
        context_limits: ContextLimits | None = None,
        candidate_depth: int | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if k_rrf <= 0:
            raise ValueError("k_rrf must be positive")
        if candidate_depth is not None and candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        self.k_rrf = k_rrf
        self.rerank_limits = rerank_limits or RerankLimits()
        self.context_limits = context_limits or ContextLimits()
        self.candidate_depth = candidate_depth
        self.clock = clock or time.perf_counter_ns

    @staticmethod
    def _candidate_map(candidates: CandidateSource) -> dict[str, RerankCandidate]:
        if isinstance(candidates, Mapping):
            result = dict(candidates)
        else:
            result = {candidate.candidate_id: candidate for candidate in candidates}
        if len(result) != (len(candidates) if not isinstance(candidates, Mapping) else len(candidates)):
            raise ValueError("candidate IDs must be unique")
        return result

    @staticmethod
    def _lane_map(ranked_lanes: Mapping[str, Sequence[str]] | Sequence[Sequence[str]]) -> dict[str, tuple[str, ...]]:
        if isinstance(ranked_lanes, Mapping):
            return {str(name): tuple(ids) for name, ids in ranked_lanes.items()}
        return {f"lane-{index}": tuple(ids) for index, ids in enumerate(ranked_lanes)}

    @staticmethod
    def _failure_result(reranker: Reranker, elapsed_ms: float, error: RerankerError) -> RerankResult:
        return RerankResult(
            reranked=(),
            requested_reranker=reranker.info.id,
            actual_reranker=reranker.info.id,
            elapsed_ms=elapsed_ms,
            eligible_count=0,
            scored_count=0,
            failure_classification=error.classification,
            error=str(error),
            reranker_applied=False,
            skipped_candidate_ids=error.candidate_ids,
        )

    def run(
        self,
        query: RerankQuery | str,
        ranked_lanes: Mapping[str, Sequence[str]] | Sequence[Sequence[str]],
        candidates: CandidateSource,
        *,
        filters: EvaluationFilters | None = None,
        reranker: Reranker | None = None,
        relevant_candidate_ids: Sequence[str] = (),
        candidate_depth: int | None = None,
    ) -> EvaluationResult:
        query_obj = query if isinstance(query, RerankQuery) else RerankQuery(query)
        all_candidates = self._candidate_map(candidates)
        policy = filters or EvaluationFilters()
        filtered = tuple(candidate for candidate in all_candidates.values() if policy.allows(candidate))
        filtered_map = {candidate.candidate_id: candidate for candidate in filtered}
        lanes = self._lane_map(ranked_lanes)
        filtered_lanes = {
            name: tuple(candidate_id for candidate_id in ids if candidate_id in filtered_map)
            for name, ids in lanes.items()
        }
        fused_ids = rrf_rank(list(filtered_lanes.values()), k_rrf=self.k_rrf)
        depth = candidate_depth if candidate_depth is not None else self.candidate_depth
        if depth is not None:
            fused_ids = fused_ids[:depth]
        fused = tuple(filtered_map[candidate_id] for candidate_id in fused_ids)
        relevant = set(relevant_candidate_ids)
        found = len(relevant.intersection(fused_ids))
        recall = found / len(relevant) if relevant else None

        rerank_result: RerankResult | None = None
        reranked: tuple[RerankedCandidate, ...]
        if reranker is None:
            reranked = tuple(RerankedCandidate(candidate, 0.0, rank) for rank, candidate in enumerate(fused, 1))
        else:
            supported = tuple(candidate for candidate in fused if candidate.representation_type in reranker.info.capabilities)
            skipped = tuple(candidate for candidate in fused if candidate.representation_type not in reranker.info.capabilities)
            started = self.clock()
            try:
                if skipped and reranker.info.failure_policy != "SKIP_UNSUPPORTED_CANDIDATE":
                    raise UnsupportedModalityError(
                        "reranker does not support candidate modality",
                        candidate_ids=tuple(candidate.candidate_id for candidate in skipped),
                    )
                rerank_result = reranker.score(query_obj, supported, limits=self.rerank_limits)
                self._validate_rerank_result(rerank_result, supported)
                if skipped:
                    rerank_result = RerankResult(
                        **{
                            **rerank_result.__dict__,
                            "skipped_count": len(skipped),
                            "skipped_candidate_ids": tuple(candidate.candidate_id for candidate in skipped),
                        }
                    )
                    reranked = rerank_result.reranked + tuple(
                        RerankedCandidate(candidate, 0.0, fused.index(candidate) + 1) for candidate in skipped
                    )
                else:
                    reranked = rerank_result.reranked
            except RerankerError as error:
                elapsed_ms = (self.clock() - started) / 1_000_000
                rerank_result = self._failure_result(reranker, elapsed_ms, error)
                if error.candidate_ids:
                    rerank_result = RerankResult(
                        **{
                            **rerank_result.__dict__,
                            "skipped_count": len(error.candidate_ids),
                            "skipped_candidate_ids": error.candidate_ids,
                        }
                    )
                if reranker.info.failure_policy == "FALLBACK_TO_FUSED_ORDER":
                    rerank_result = RerankResult(
                        **{**rerank_result.__dict__, "fallback_invoked": True}
                    )
                    reranked = tuple(RerankedCandidate(candidate, 0.0, rank) for rank, candidate in enumerate(fused, 1))
                else:
                    reranked = ()

        context = select_context(reranked, limits=self.context_limits) if reranked else ContextSelectionResult((), (), 0, 0, 0)
        return EvaluationResult(
            query=query_obj,
            filtered_candidates=filtered,
            lane_orders=filtered_lanes,
            fused_candidates=fused,
            candidate_recall=recall,
            relevant_candidate_count=len(relevant),
            relevant_candidates_found=found,
            rerank=rerank_result,
            reranked_candidates=reranked,
            context=context,
            phase_order=self.PHASE_ORDER,
        )

    @staticmethod
    def _validate_rerank_result(result: RerankResult, candidates: Sequence[RerankCandidate]) -> None:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        seen: set[str] = set()
        for item in result.reranked:
            if item.candidate_id in seen or item.candidate_id not in by_id:
                raise MalformedRerankerOutput("reranker returned an unknown or duplicate candidate")
            if item.candidate != by_id[item.candidate_id]:
                raise MalformedRerankerOutput("reranker changed candidate provenance")
            seen.add(item.candidate_id)


__all__ = [
    "ContextLimits",
    "ContextSelectionResult",
    "EmbeddingVariant",
    "EvaluationCoordinator",
    "EvaluationFilters",
    "EvaluationResult",
    "FakeReranker",
    "InMemoryRepresentationStore",
    "ManifestValidationError",
    "MalformedRerankerOutput",
    "RepresentationRecord",
    "RerankCandidate",
    "RerankLimits",
    "RerankQuery",
    "RerankResult",
    "RerankedCandidate",
    "Reranker",
    "RerankerError",
    "RerankerInfo",
    "RerankerOutOfMemory",
    "RerankerTimeout",
    "SuppressedCandidate",
    "UnsupportedModalityError",
    "compare_embeddings",
    "build_result_manifest",
    "rrf_rank",
    "select_context",
    "stable_representation_id",
    "validate_manifest",
    "validate_result_manifest",
]
