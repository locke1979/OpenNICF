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
from opennicf.evaluation import (
    ContextLimits,
    EvaluationCoordinator,
    EvaluationFilters,
    FakeReranker,
    RerankCandidate,
    RerankQuery,
    RerankLimits,
    select_context,
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


def _candidate(candidate_id, *, representation_type="text", source_id="src", locator="chunk:1", acl="allow", text=None):
    return RerankCandidate(
        candidate_id=candidate_id,
        representation_id=f"rep-{candidate_id}",
        text=text or candidate_id,
        representation_type=representation_type,
        source_id=source_id,
        source_version_id="sv-1",
        artifact_hash="a" * 64,
        locator=locator,
        acl_scope=acl,
    )


def test_evaluation_coordinator_filters_before_fusion_and_records_candidate_recall():
    candidates = [_candidate("allowed", acl="allow"), _candidate("secret", acl="secret")]
    result = EvaluationCoordinator(k_rrf=1).run(
        RerankQuery("query", "q-1"),
        {"dense": ["secret", "allowed"], "lexical": ["allowed"]},
        candidates,
        filters=EvaluationFilters(acl_scopes=frozenset({"allow"})),
        relevant_candidate_ids=["allowed"],
    )
    assert tuple(item.candidate_id for item in result.fused_candidates) == ("allowed",)
    assert result.candidate_recall == 1.0
    assert result.phase_order.index("filter") < result.phase_order.index("rrf")
    assert result.phase_order.index("candidate_recall") < result.phase_order.index("rerank")


def test_reranker_failure_policies_are_explicit_and_never_look_successful():
    candidates = [_candidate("a", text="alpha"), _candidate("b", text="beta")]
    for policy, fallback, expected_count in (("FAIL_QUERY", False, 0), ("FALLBACK_TO_FUSED_ORDER", True, 2)):
        reranker = FakeReranker(
            info=__import__("opennicf.evaluation", fromlist=["RerankerInfo"]).RerankerInfo(
                id=f"fake-{policy}", failure_policy=policy
            ),
            failure="timeout",
        )
        result = EvaluationCoordinator().run("q", {"dense": ["a", "b"]}, candidates, reranker=reranker)
        assert len(result.reranked_candidates) == expected_count
        assert result.rerank is not None
        assert result.rerank.reranker_applied is False
        assert result.rerank.fallback_invoked is fallback


def test_skip_unsupported_candidate_records_it_without_silent_loss():
    from opennicf.evaluation import RerankerInfo

    candidates = [_candidate("text"), _candidate("image", representation_type="page_image", locator="page:2")]
    reranker = FakeReranker(
        info=RerankerInfo(id="text-only", capabilities=frozenset({"text"}), failure_policy="SKIP_UNSUPPORTED_CANDIDATE"),
        scores={"text": 1.0},
    )
    result = EvaluationCoordinator().run("q", {"dense": ["text", "image"]}, candidates, reranker=reranker)
    assert result.rerank is not None
    assert result.rerank.skipped_candidate_ids == ("image",)
    assert {item.candidate_id for item in result.reranked_candidates} == {"text", "image"}


def test_context_selection_deduplicates_late_and_enforces_source_page_limits():
    first = _candidate("first", source_id="src", locator="page:1")
    duplicate = RerankCandidate(**{**first.__dict__, "candidate_id": "duplicate", "representation_id": first.representation_id})
    second_page = _candidate("second", source_id="src", locator="page:2")
    result = select_context(
        [
            __import__("opennicf.evaluation", fromlist=["RerankedCandidate"]).RerankedCandidate(first, 1.0, 1),
            __import__("opennicf.evaluation", fromlist=["RerankedCandidate"]).RerankedCandidate(duplicate, .9, 2),
            __import__("opennicf.evaluation", fromlist=["RerankedCandidate"]).RerankedCandidate(second_page, .8, 3),
        ],
        limits=ContextLimits(max_candidates=3, max_per_source=2, max_per_page=1),
    )
    assert [item.candidate_id for item in result.selected] == ["first", "second"]
    assert any(item.reason == "duplicate_representation" for item in result.suppressed)
