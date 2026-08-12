"""Model-free contracts for the isolated Gate C multimodal evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any, Callable, Iterable, Mapping


def _digest(*parts: str | bytes) -> str:
    value = b"\0".join(p if isinstance(p, bytes) else p.encode() for p in parts)
    return sha256(value).hexdigest()


class Lifecycle(StrEnum):
    READY = "READY"
    QUARANTINED = "QUARANTINED"


class ReviewStatus(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    ACCEPT = "ACCEPT"
    REWRITE = "REWRITE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class RenderLimits:
    max_pixels: int = 1_500_000
    max_pages: int = 32
    max_images: int = 8
    max_batch: int = 4

    def validate(self, *, width: int, height: int, page: int, images: int, batch: int) -> None:
        if min(width, height, page, images, batch) < 1:
            raise ValueError("render dimensions/counts must be positive")
        if width * height > self.max_pixels:
            raise ValueError("pixel limit exceeded")
        if page > self.max_pages or images > self.max_images or batch > self.max_batch:
            raise ValueError("page/image/batch limit exceeded")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_sha256: str
    media_type: str
    domain: str
    system: str
    acl: tuple[str, ...]
    production: bool = False


@dataclass(frozen=True)
class Representation:
    representation_id: str
    source_id: str
    modality: str
    page: int
    renderer_version: str
    rendition_sha256: str
    width: int
    height: int
    domain: str
    system: str
    acl: tuple[str, ...]
    lifecycle: Lifecycle
    failure_code: str | None = None
    retryable: bool = False
    embedded_content_is_untrusted: bool = True


class SanitizedCorpusBuilder:
    """Deterministic representation builder; it never interprets document instructions."""

    def __init__(self, *, renderer_version: str, limits: RenderLimits = RenderLimits()) -> None:
        self.renderer_version = renderer_version
        self.limits = limits
        self._records: dict[str, Representation] = {}

    def render(
        self,
        source: SourceRecord,
        *,
        modality: str,
        page: int,
        width: int,
        height: int,
        renderer: Callable[[], bytes],
        images: int = 1,
        batch: int = 1,
    ) -> Representation:
        if source.production:
            raise ValueError("production material is forbidden in Gate C fixtures")
        self.limits.validate(width=width, height=height, page=page, images=images, batch=batch)
        identity_seed = _digest(source.source_sha256, modality, str(page), self.renderer_version)
        try:
            rendered = renderer()
            if not rendered or rendered.startswith(b"MALFORMED"):
                raise ValueError("malformed rendition")
            rendition_sha = sha256(rendered).hexdigest()
            representation_id = f"rep-{_digest(identity_seed, rendition_sha)[:24]}"
            record = Representation(
                representation_id, source.source_id, modality, page,
                self.renderer_version, rendition_sha, width, height,
                source.domain, source.system, source.acl, Lifecycle.READY,
            )
        except Exception as exc:  # renderer boundary intentionally isolates failures
            representation_id = f"rep-{_digest(identity_seed, 'failed')[:24]}"
            record = Representation(
                representation_id, source.source_id, modality, page,
                self.renderer_version, "", width, height, source.domain,
                source.system, source.acl, Lifecycle.QUARANTINED,
                failure_code=type(exc).__name__, retryable=True,
            )
        self._records[representation_id] = record
        return record

    def manifest(self) -> list[dict[str, object]]:
        return [asdict(self._records[key]) for key in sorted(self._records)]


@dataclass(frozen=True)
class MultimodalQuery:
    query_id: str
    text: str
    language: str
    modality: str
    domain: str
    category: str
    relevant_representation_ids: tuple[str, ...]
    highly_relevant_representation_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]
    text_only_evidence_sufficient: bool
    review_status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    review_notes: str = ""
    leakage_indicators: tuple[str, ...] = ()
    hard_negative_category: str = "visually_similar_wrong_evidence"


def validate_review_contract(queries: Iterable[MultimodalQuery], representations: Iterable[Representation]) -> None:
    known = {record.representation_id for record in representations}
    for query in queries:
        refs = (*query.relevant_representation_ids, *query.highly_relevant_representation_ids, *query.hard_negative_ids)
        if not refs or any(ref not in known for ref in refs):
            raise ValueError(f"unresolved representation reference in {query.query_id}")
        if query.review_status != ReviewStatus.PENDING_REVIEW and not query.review_notes.strip():
            raise ValueError("human decisions require review notes")


def apply_multimodal_reviews(
    manifest: Mapping[str, Any],
    decisions: Iterable[Mapping[str, Any]],
    *,
    reviewer: str,
    reviewed_at: str,
) -> dict[str, Any]:
    """Apply genuine review decisions while preserving query and evidence identity."""
    if not reviewer.strip() or not reviewed_at.strip():
        raise ValueError("reviewer and reviewed_at are required")
    known = {item["representation_id"] for item in manifest.get("representations", [])}
    originals = {item["query_id"]: item for item in manifest.get("queries", [])}
    supplied: dict[str, Mapping[str, Any]] = {}
    for item in decisions:
        query_id = str(item.get("query_id"))
        if query_id in supplied or query_id not in originals:
            raise ValueError(f"duplicate or unknown query ID: {query_id}")
        supplied[query_id] = item
    output = []
    for query_id in sorted(originals):
        original = dict(originals[query_id])
        review = supplied.get(query_id)
        if review is None:
            output.append(original)
            continue
        decision = str(review.get("reviewer_decision"))
        if decision not in {"ACCEPT", "REWRITE", "REJECT"}:
            raise ValueError(f"invalid reviewer decision for {query_id}")
        notes = str(review.get("reviewer_notes") or "").strip()
        if not notes:
            raise ValueError(f"reviewer notes are required for {query_id}")
        for field in ("relevant_representation_ids", "highly_relevant_representation_ids"):
            if field in review and tuple(review[field]) != tuple(original.get(field, ())):
                raise ValueError(f"{field} changed for {query_id}")
        negatives = tuple(review.get("hard_negative_ids", original.get("hard_negative_ids", ())))
        if any(item not in known for item in negatives):
            raise ValueError(f"invalid hard-negative ID for {query_id}")
        rewritten = review.get("rewritten_query_text")
        if decision == "REWRITE" and not str(rewritten or "").strip():
            raise ValueError(f"rewritten query text is required for {query_id}")
        original.update({
            "original_text": original.get("text"),
            "text": str(rewritten).strip() if decision == "REWRITE" else original.get("text"),
            "hard_negative_ids": list(negatives),
            "review_status": decision,
            "review_notes": notes,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        })
        output.append(original)
    payload = {**manifest, "queries": output, "review_source": reviewer, "reviewed_at": reviewed_at}
    import json
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    payload["reviewed_label_digest"] = sha256(canonical).hexdigest()
    return payload


@dataclass(frozen=True)
class TelemetrySample:
    phase: str
    duration_ms: float
    items: int
    peak_rss_bytes: int
    peak_vram_bytes: int | None
    queue_latency_ms: float
    input_bytes: int
    stored_bytes: int
    failures: int = 0
    oom: int = 0
    nonfinite_vectors: int = 0
    concurrency: int = 1
    restart_cycle: int = 0

    def __post_init__(self) -> None:
        if self.concurrency not in (1, 2, 4):
            raise ValueError("concurrency must be one of 1, 2, 4")
        if min(self.duration_ms, self.queue_latency_ms, self.items, self.input_bytes, self.stored_bytes) < 0:
            raise ValueError("telemetry counters cannot be negative")

    @property
    def throughput_per_second(self) -> float | None:
        return None if self.duration_ms == 0 else self.items * 1000 / self.duration_ms

    @property
    def storage_amplification(self) -> float | None:
        return None if self.input_bytes == 0 else self.stored_bytes / self.input_bytes


HARD_NEGATIVE_CATEGORIES = (
    "same_application_wrong_screen", "same_document_wrong_page",
    "same_diagram_labels_wrong_flow", "same_table_wrong_operation",
    "current_obsolete", "production_non_production", "same_error_wrong_service",
    "visually_similar_wrong_evidence",
)
