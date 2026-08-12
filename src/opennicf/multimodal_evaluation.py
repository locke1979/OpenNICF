"""Model-free contracts for the isolated Gate C multimodal evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Callable, Iterable, Mapping


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


class FeasibilityStage(StrEnum):
    METADATA = "METADATA"
    CPU_FORWARD = "CPU_FORWARD"
    CUDA_LOAD = "CUDA_LOAD"
    CUDA_FORWARD = "CUDA_FORWARD"
    RESTART = "RESTART"
    THROUGHPUT = "THROUGHPUT"


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    bytes: int
    sha256: str | None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.path or self.bytes < 0:
            raise ValueError("invalid artifact file")
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("artifact SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class VLArtifactContract:
    role: str
    repository: str
    revision: str
    license: str | None
    architecture: str
    files: tuple[ArtifactFile, ...]
    native_dimension: int | None = None
    supported_dimensions: tuple[int, ...] = ()
    output_normalized: bool | None = None
    processor_revision: str | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    quantization_method: str | None = None
    weight_bits: int | None = None
    base_repository: str | None = None
    base_revision: str | None = None
    calibration_contract: str | None = None
    custom_code: bool = False

    def validate_metadata(self) -> None:
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("artifact revision must be a pinned commit")
        if not self.files or not any(f.path.endswith((".safetensors", ".gguf")) for f in self.files):
            raise ValueError("weight file is required")
        if any(f.required and f.bytes == 0 for f in self.files):
            raise ValueError("required file size is unknown")
        if self.processor_revision is not None and self.processor_revision != self.revision:
            raise ValueError("processor and weights must use the same pinned revision")
        if self.max_pixels is not None and (
            self.min_pixels is None or self.min_pixels <= 0 or self.max_pixels < self.min_pixels
        ):
            raise ValueError("invalid pixel contract")
        if self.weight_bits is not None:
            if self.weight_bits != 4 or not self.quantization_method:
                raise ValueError("W4 requires an explicit weight quantization method")
            if not self.base_repository or not self.base_revision or not self.calibration_contract:
                raise ValueError("W4 requires pinned base identity and calibration contract")


@dataclass(frozen=True)
class RuntimeQualification:
    name: str
    version: str | None
    installed: bool
    exact_architecture_supported: bool
    cuda_version: str | None
    compute_capability: str | None
    cuda_architecture_evidence: str | None
    multimodal_forward_proven: bool = False
    embedding_head_proven: bool = False
    reranker_head_proven: bool = False

    def executable_for(self, artifact: VLArtifactContract, *, cuda: bool) -> bool:
        artifact.validate_metadata()
        if not artifact.license or not self.installed or not self.version:
            return False
        if not self.exact_architecture_supported:
            return False
        if cuda and (not self.cuda_version or not self.compute_capability or not self.cuda_architecture_evidence):
            return False
        if artifact.role == "embedding" and not self.embedding_head_proven:
            return False
        if artifact.role == "reranker" and not self.reranker_head_proven:
            return False
        return self.multimodal_forward_proven


@dataclass(frozen=True)
class StageResult:
    stage: FeasibilityStage
    passed: bool
    measured: bool
    evidence: str
    stop_reason: str | None = None


def validate_feasibility_ladder(results: Iterable[StageResult]) -> None:
    ordered = list(results)
    expected = list(FeasibilityStage)
    if [r.stage for r in ordered] != expected[:len(ordered)]:
        raise ValueError("feasibility stages must be contiguous and ordered")
    for index, result in enumerate(ordered):
        if not result.evidence.strip():
            raise ValueError("each stage requires evidence")
        if not result.passed:
            if not result.stop_reason:
                raise ValueError("a failed stage requires a stop reason")
            if index != len(ordered) - 1:
                raise ValueError("no stages may run after a failed stage")


def validate_artifact_inventory(
    artifacts: Iterable[VLArtifactContract],
    remote_hashes: Mapping[tuple[str, str, str], str],
) -> None:
    """Fail closed when pinned metadata and authoritative remote hashes diverge."""
    seen: set[tuple[str, str, str]] = set()
    for artifact in artifacts:
        artifact.validate_metadata()
        for file in artifact.files:
            key = (artifact.repository, artifact.revision, file.path)
            if key in seen:
                raise ValueError("duplicate artifact file identity")
            seen.add(key)
            expected = remote_hashes.get(key)
            if file.sha256 is not None and expected != file.sha256:
                raise ValueError(f"artifact digest mismatch: {file.path}")
