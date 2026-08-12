"""Model-free contracts for the issue #86 evaluation.

This module is deliberately evaluation-only. It does not select a provider,
write production persistence, or execute a model. It makes rendition identity,
space identity, and manifest safety rules executable before model acquisition.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


REPRESENTATION_TYPES = {"text", "page_image", "region_image", "mixed"}
REPRESENTATION_LIFECYCLE = {"pending", "ready", "failed", "quarantined", "retired"}
MANIFEST_STATES = {"DRAFT", "EXECUTABLE"}
FAILURE_POLICIES = {"FAIL_QUERY", "FALLBACK_TO_FUSED_ORDER", "SKIP_UNSUPPORTED_CANDIDATE"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER = re.compile(r"^(?:exact |recorded |path/to/|TODO|placeholder)", re.I)


class ManifestValidationError(ValueError):
    """Raised when an issue #86 manifest cannot be safely executed."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_representation_id(
    *,
    source_version_id: str,
    representation_type: str,
    locator: str,
    parent_chunk_id: str | None,
    rendition_hash: str,
    render_name: str,
    render_version: str,
) -> str:
    """Return a rendition identity independent of embedding space/model."""
    if representation_type not in REPRESENTATION_TYPES:
        raise ValueError(f"unsupported representation_type: {representation_type}")
    canonical = {
        "source_version_id": source_version_id,
        "representation_type": representation_type,
        "locator": locator,
        "parent_chunk_id": parent_chunk_id,
        "rendition_hash": rendition_hash,
        "render_name": render_name,
        "render_version": render_version,
    }
    return "rep_" + hashlib.sha256(_canonical(canonical).encode()).hexdigest()


@dataclass(frozen=True)
class RepresentationRecord:
    representation_id: str
    source_id: str
    source_version_id: str
    artifact_hash: str
    representation_type: str
    locator: str
    parent_chunk_id: str | None
    object_key: str | None
    mime_type: str
    rendition_hash: str
    render_name: str
    render_version: str
    lifecycle_status: str = "ready"
    failure_classification: str | None = None
    attempt_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    pixel_width: int | None = None
    pixel_height: int | None = None
    acl_scope: str = ""
    domain_id: str = ""
    system_id: str = ""
    component_id: str = "unknown-component"
    environment: str = ""
    evidence_type: str = ""


@dataclass(frozen=True)
class EmbeddingVariant:
    representation_id: str
    embedding_space_id: str
    model: str
    model_revision: str
    quantization: str
    native_dimensions: int
    dimensions: int
    normalized: bool
    vector: tuple[float, ...] = ()


def compare_embeddings(left: EmbeddingVariant, right: EmbeddingVariant) -> float:
    """Compare only vectors from the same semantic space."""
    if left.embedding_space_id != right.embedding_space_id:
        raise ValueError("embedding-space mismatch: vectors cannot be compared")
    if len(left.vector) != len(right.vector):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left.vector, right.vector))


class InMemoryRepresentationStore:
    """Idempotent isolated contract store used by model-free tests."""

    def __init__(self) -> None:
        self.representations: dict[str, RepresentationRecord] = {}
        self.embeddings: dict[tuple[str, str], EmbeddingVariant] = {}

    def put_representation(self, record: RepresentationRecord) -> RepresentationRecord:
        if record.representation_type not in REPRESENTATION_TYPES:
            raise ValueError(f"unsupported representation_type: {record.representation_type}")
        if record.lifecycle_status not in REPRESENTATION_LIFECYCLE:
            raise ValueError(f"unsupported lifecycle_status: {record.lifecycle_status}")
        if record.attempt_count < 0:
            raise ValueError("attempt_count must be non-negative")
        if record.representation_type != "text" and not record.object_key:
            raise ValueError("non-text representations require an object_key")
        for name, value in (("pixel_width", record.pixel_width), ("pixel_height", record.pixel_height)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        expected = stable_representation_id(
            source_version_id=record.source_version_id,
            representation_type=record.representation_type,
            locator=record.locator,
            parent_chunk_id=record.parent_chunk_id,
            rendition_hash=record.rendition_hash,
            render_name=record.render_name,
            render_version=record.render_version,
        )
        if record.representation_id != expected:
            raise ValueError("representation_id does not match canonical rendition identity")
        existing = self.representations.get(record.representation_id)
        if existing is not None and existing != record:
            raise ValueError("representation identity collision with different rendition metadata")
        self.representations[record.representation_id] = record
        return record

    def put_embedding(self, variant: EmbeddingVariant) -> EmbeddingVariant:
        if variant.representation_id not in self.representations:
            raise KeyError("embedding requires an existing representation")
        if variant.dimensions <= 0 or variant.native_dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        key = (variant.representation_id, variant.embedding_space_id)
        existing = self.embeddings.get(key)
        if existing is not None and existing != variant:
            raise ValueError("embedding-space identity collision")
        self.embeddings[key] = variant
        return variant

    def get_representation(self, representation_id: str) -> RepresentationRecord | None:
        return self.representations.get(representation_id)

    def get_embedding(self, representation_id: str, embedding_space_id: str) -> EmbeddingVariant | None:
        return self.embeddings.get((representation_id, embedding_space_id))

    def list_representations(self, *, lifecycle_status: str | None = None) -> tuple[RepresentationRecord, ...]:
        records = self.representations.values()
        if lifecycle_status is not None:
            if lifecycle_status not in REPRESENTATION_LIFECYCLE:
                raise ValueError(f"unsupported lifecycle_status: {lifecycle_status}")
            records = (record for record in records if record.lifecycle_status == lifecycle_status)
        return tuple(sorted(records, key=lambda record: record.representation_id))

    def eligible_representations(self, embedding_space_id: str) -> tuple[RepresentationRecord, ...]:
        """Return ready renditions with an embedding in one explicit space."""
        return tuple(
            record
            for record in self.list_representations(lifecycle_status="ready")
            if (record.representation_id, embedding_space_id) in self.embeddings
        )


def rrf_rank(
    ranked_lanes: list[list[str]],
    *,
    k_rrf: int = 60,
) -> list[str]:
    """Fuse ranked candidate IDs without comparing raw lane scores."""
    if k_rrf <= 0:
        raise ValueError("k_rrf must be positive")
    scores: dict[str, float] = {}
    first_rank: dict[str, int] = {}
    for lane in ranked_lanes:
        seen_in_lane: set[str] = set()
        for rank, candidate_id in enumerate(lane, 1):
            # A repeated representation is one candidate in a lane.  Keep its
            # first (best) rank and do not let a later occurrence contribute a
            # second reciprocal-rank term.
            if candidate_id in seen_in_lane:
                continue
            seen_in_lane.add(candidate_id)
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k_rrf + rank)
            first_rank[candidate_id] = min(first_rank.get(candidate_id, rank), rank)
    return sorted(scores, key=lambda candidate_id: (-scores[candidate_id], first_rank[candidate_id], candidate_id))


def _fail(errors: list[str], condition: bool, message: str) -> None:
    if condition:
        errors.append(message)


def validate_manifest(manifest: dict[str, Any], *, execution: bool = False) -> None:
    """Validate a sanitized issue #86 manifest; fail closed on every violation."""
    errors: list[str] = []
    state = manifest.get("state")
    _fail(errors, state not in MANIFEST_STATES, "state must be DRAFT or EXECUTABLE")
    _fail(errors, manifest.get("production") is not False, "production must be false")
    _fail(errors, manifest.get("production_migration_authorized") is not False, "production migration must be unauthorized")
    _fail(errors, execution and state != "EXECUTABLE", "DRAFT manifests cannot execute")
    if state == "EXECUTABLE":
        required = ("corpus_manifest_digest", "reviewed_label_digest", "runtime_digest")
        _fail(errors, any(not manifest.get(key) or PLACEHOLDER.search(str(manifest[key])) for key in required), "EXECUTABLE manifest requires resolved corpus, label, and runtime digests")

    spaces = manifest.get("spaces", [])
    if isinstance(spaces, dict):
        spaces = list(spaces.values())
    space_ids: set[str] = set()
    for space in spaces:
        sid = space.get("id")
        _fail(errors, not sid or sid in space_ids, "space IDs must be present and unique")
        if sid:
            space_ids.add(sid)
        for key in ("model", "model_revision", "artifact_sha256", "runtime", "quantization", "native_dimensions", "dimensions", "normalized", "preprocessing", "modality"):
            value = space.get(key)
            _fail(errors, value in (None, "") or (state == "EXECUTABLE" and PLACEHOLDER.search(str(value)) is not None), f"space {sid or '<unknown>'} missing resolved {key}")
        _fail(errors, not isinstance(space.get("dimensions"), int) or space.get("dimensions", 0) <= 0, f"space {sid or '<unknown>'} has invalid dimensions")
        _fail(errors, not isinstance(space.get("native_dimensions"), int) or space.get("native_dimensions", 0) <= 0, f"space {sid or '<unknown>'} has invalid native dimensions")
        _fail(errors, not isinstance(space.get("normalized"), bool), f"space {sid or '<unknown>'} must declare normalization")
        artifact = str(space.get("artifact_sha256", ""))
        _fail(errors, state == "EXECUTABLE" and not HEX64.fullmatch(artifact), f"space {sid or '<unknown>'} has invalid artifact SHA-256")
    modalities = {str(x.get("modality")) for x in spaces}
    _fail(errors, len(space_ids) != len(spaces), "duplicate space IDs")
    _fail(errors, len([x for x in spaces if x.get("modality") == "text"]) > 1 and len({x.get("id") for x in spaces if x.get("modality") == "text"}) != len([x for x in spaces if x.get("modality") == "text"]), "duplicate text spaces")
    for representation_type in manifest.get("representation_types", ["text"]):
        _fail(errors, representation_type not in REPRESENTATION_TYPES, f"unknown representation type: {representation_type}")

    rerankers = manifest.get("rerankers", [])
    if isinstance(rerankers, dict):
        rerankers = list(rerankers.values())
    reranker_ids: set[str] = set()
    for reranker in rerankers:
        rid = reranker.get("id")
        _fail(errors, not rid or rid in reranker_ids, "reranker IDs must be present and unique")
        if rid:
            reranker_ids.add(rid)
        for key in ("model", "model_revision", "artifact_sha256", "runtime", "quantization", "capabilities", "failure_policy"):
            value = reranker.get(key)
            _fail(errors, value in (None, "") or (state == "EXECUTABLE" and PLACEHOLDER.search(str(value)) is not None), f"reranker {rid or '<unknown>'} missing resolved {key}")
        _fail(errors, reranker.get("failure_policy") not in FAILURE_POLICIES, f"reranker {rid or '<unknown>'} has invalid failure policy")
        _fail(errors, state == "EXECUTABLE" and not HEX64.fullmatch(str(reranker.get("artifact_sha256", ""))), f"reranker {rid or '<unknown>'} has invalid artifact SHA-256")

    fusion = manifest.get("fusion", {})
    _fail(errors, not isinstance(fusion, dict) or fusion.get("method") != "rrf", "fusion must use RRF")
    if isinstance(fusion, dict):
        _fail(errors, not isinstance(fusion.get("k_rrf"), int) or fusion.get("k_rrf", 0) <= 0, "k_rrf must be a positive integer")
        _fail(errors, fusion.get("raw_score_fusion") is True, "raw-score fusion is forbidden")

    depths = manifest.get("candidate_depths", [])
    _fail(errors, not depths or any(not isinstance(x, int) or x < 1 or x > 50 for x in depths), "candidate depths must be integers in 1..50")
    arms = manifest.get("arms", [])
    arm_ids: set[str] = set()
    for arm in arms:
        aid = arm.get("id")
        _fail(errors, not aid or aid in arm_ids, "arm IDs must be present and unique")
        if aid:
            arm_ids.add(aid)
        _fail(errors, not set(arm.get("spaces", [])) <= space_ids, f"arm {aid or '<unknown>'} references an undefined space")
        reranker = arm.get("reranker")
        _fail(errors, reranker not in (None, "") and reranker not in reranker_ids, f"arm {aid or '<unknown>'} references an undefined reranker")
        _fail(errors, not set(arm.get("candidate_depths", depths)) <= set(depths), f"arm {aid or '<unknown>'} has an unconfigured candidate depth")
        if reranker:
            capabilities = next((x.get("capabilities", []) for x in rerankers if x.get("id") == reranker), [])
            modalities_used = {next((x.get("modality") for x in spaces if x.get("id") == sid), None) for sid in arm.get("spaces", [])}
            visual_modalities = {"multimodal", "page_image", "region_image", "mixed"}
            _fail(errors, bool(modalities_used & visual_modalities) and not (visual_modalities & set(capabilities)), f"arm {aid or '<unknown>'} sends visual candidates to an unsupported reranker")

    raw = _canonical(manifest)
    _fail(errors, bool(re.search(r"(?:api[_-]?key|password|secret|authorization)\s*[:=]", raw, re.I)), "credentials/secrets are forbidden")
    _fail(errors, bool(re.search(r"(?:/var/lib/opennicf|/etc/opennicf|production[_-]?data|private[_-]?evidence|raw[_-]?evidence)", raw, re.I)), "production paths or raw private evidence are forbidden")
    if errors:
        raise ManifestValidationError("; ".join(errors))
