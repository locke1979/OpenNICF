"""Model-free safety tests for the issue #86 Phase 0 contract."""

import json
from pathlib import Path

import pytest

from opennicf.evaluation_contract import (
    EmbeddingVariant,
    InMemoryRepresentationStore,
    ManifestValidationError,
    RepresentationRecord,
    compare_embeddings,
    rrf_rank,
    stable_representation_id,
    validate_manifest,
)

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"


def test_phase0_manifest_is_evaluation_only_and_space_isolated():
    payload = json.loads((DOCS / "issue86-phase0-manifest.example.json").read_text())
    assert payload["issue"] == 86
    assert payload["baseline_issue"] == 85
    assert payload["production"] is False
    assert payload["production_migration_authorized"] is False
    assert payload["text_space_id"] != payload["vl_space_id"]
    assert payload["fusion"]["method"] == "rrf"
    assert payload["fusion"]["k_rrf"] == 60
    assert payload["candidate_recall_before_rerank_required"] is True
    validate_manifest(payload)
    with pytest.raises(ManifestValidationError):
        validate_manifest(payload, execution=True)


def _representation(**overrides):
    values = {
        "source_version_id": "sv-1",
        "representation_type": "page_image",
        "locator": "page:1",
        "parent_chunk_id": "chunk-1",
        "rendition_hash": "a" * 64,
        "render_name": "pdfium",
        "render_version": "1",
    }
    values.update(overrides)
    rep_id = stable_representation_id(**values)
    return RepresentationRecord(
        representation_id=rep_id,
        source_id="src-1",
        source_version_id=values["source_version_id"],
        artifact_hash="b" * 64,
        representation_type=values["representation_type"],
        locator=values["locator"],
        parent_chunk_id=values["parent_chunk_id"],
        object_key="renditions/page-1.png",
        mime_type="image/png",
        rendition_hash=values["rendition_hash"],
        render_name=values["render_name"],
        render_version=values["render_version"],
    )


def test_rendition_identity_is_independent_of_embedding_space_and_quantization():
    store = InMemoryRepresentationStore()
    record = _representation()
    store.put_representation(record)
    store.put_embedding(EmbeddingVariant(record.representation_id, "VL_QWEN3_2B_BF16_768_V1", "qwen-vl", "r1", "BF16", 2048, 768, True, (1.0,)))
    store.put_embedding(EmbeddingVariant(record.representation_id, "VL_QWEN3_2B_W4_768_V1", "qwen-vl", "r1", "W4", 2048, 768, True, (1.0,)))
    assert len(store.representations) == 1
    assert len(store.embeddings) == 2


def test_changed_rendition_or_renderer_version_changes_representation_id():
    original = _representation()
    assert original.representation_id != _representation(rendition_hash="c" * 64).representation_id
    assert original.representation_id != _representation(render_version="2").representation_id


def test_unchanged_rendition_write_is_idempotent_and_embedding_spaces_cannot_compare():
    store = InMemoryRepresentationStore()
    record = _representation()
    assert store.put_representation(record) == store.put_representation(record)
    left = EmbeddingVariant(record.representation_id, "TEXT_QWEN3_4B_Q4_K_M_768_V1", "text", "r1", "Q4_K_M", 2560, 2, True, (1.0, 0.0))
    right = EmbeddingVariant(record.representation_id, "VL_QWEN3_2B_W4_768_V1", "vl", "r1", "W4", 2048, 2, True, (1.0, 0.0))
    with pytest.raises(ValueError, match="space"):
        compare_embeddings(left, right)


def test_rrf_is_deterministic_and_uses_rank_not_raw_scores():
    assert rrf_rank([["b", "a"], ["a", "c"]], k_rrf=1) == ["a", "b", "c"]
    assert rrf_rank([["b", "a"], ["a", "c"]], k_rrf=60) == ["a", "b", "c"]


def _executable_manifest():
    payload = json.loads((DOCS / "issue86-phase0-manifest.example.json").read_text())
    payload["state"] = "EXECUTABLE"
    payload["corpus_manifest_digest"] = "c" * 64
    payload["reviewed_label_digest"] = "d" * 64
    payload["runtime_digest"] = "e" * 64
    for space in payload["spaces"]:
        space.update({"model_revision": "r1", "artifact_sha256": "a" * 64, "runtime": "runtime:r1", "preprocessing": "prep:r1"})
    for reranker in payload["rerankers"]:
        reranker.update({"model_revision": "r1", "artifact_sha256": "b" * 64, "runtime": "runtime:r1", "quantization": "W4"})
    return payload


def test_executable_manifest_validates_and_fail_closed_cases_are_rejected():
    validate_manifest(_executable_manifest(), execution=True)
    invalid = _executable_manifest()
    invalid["production"] = True
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["spaces"][1]["id"] = invalid["spaces"][0]["id"]
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["arms"][0]["spaces"] = ["missing-space"]
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["fusion"]["raw_score_fusion"] = True
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["rerankers"][0]["artifact_sha256"] = "secret=bad"
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["representation_types"] = ["video_stream"]
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid.pop("reviewed_label_digest")
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["arms"][0]["candidate_depths"] = [100]
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["arms"][1]["reranker"] = "missing-reranker"
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["arms"].append(dict(invalid["arms"][0]))
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["arms"][1]["spaces"] = ["VL_QWEN3_2B_W4_768_V1"]
    invalid["arms"][1]["reranker"] = "TEXT_RERANKER_V1"
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)
    invalid = _executable_manifest()
    invalid["runtime_path"] = "/var/lib/opennicf/production"
    with pytest.raises(ManifestValidationError):
        validate_manifest(invalid, execution=True)


def test_phase0_docs_preserve_runtime_authority_and_fail_closed_rules():
    architecture = (DOCS / "architecture-delta.md").read_text()
    api = (DOCS / "schema-api-compatibility.md").read_text()
    plan = (DOCS / "phase0-test-plan.md").read_text()
    for text in (architecture, api, plan):
        assert "QwenAgent" in text
        assert "ACL" in text
        assert "provenance" in text.lower()
    assert "RRF" in architecture
    assert "fail-closed" in api
    assert "image-only candidates" in plan


def test_phase0_artifacts_do_not_contain_credentials_or_production_paths():
    for path in DOCS.glob("*phase0*"):
        text = path.read_text()
        assert "OPENAI_API_KEY" not in text
        assert "Authorization: Bearer" not in text
        assert "/var/lib/opennicf" not in text
