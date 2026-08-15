from hashlib import sha256
import json
from pathlib import Path

import pytest

from opennicf.multimodal_evaluation import (
    ArtifactFile, FeasibilityStage, HARD_NEGATIVE_CATEGORIES, Lifecycle,
    MultimodalQuery, RenderLimits, ReviewStatus, RuntimeQualification,
    SanitizedCorpusBuilder, SourceRecord, StageResult, TelemetrySample,
    VLArtifactContract, validate_artifact_inventory, validate_feasibility_ladder,
    validate_review_contract,
)


def source(name="synthetic", *, production=False):
    return SourceRecord(name, sha256(name.encode()).hexdigest(), "application/pdf", "finance", "fixture", ("eval",), production)


def test_idempotent_rendition_and_provenance_policy_propagation():
    builder = SanitizedCorpusBuilder(renderer_version="synthetic-renderer@1")
    first = builder.render(source(), modality="page_image", page=1, width=100, height=100, renderer=lambda: b"pixels")
    again = builder.render(source(), modality="page_image", page=1, width=100, height=100, renderer=lambda: b"pixels")
    assert first == again
    assert first.domain == "finance" and first.system == "fixture" and first.acl == ("eval",)
    assert first.embedded_content_is_untrusted


def test_changed_bytes_or_renderer_creates_new_representation():
    a = SanitizedCorpusBuilder(renderer_version="r1").render(source(), modality="diagram", page=1, width=10, height=10, renderer=lambda: b"a")
    b = SanitizedCorpusBuilder(renderer_version="r1").render(source(), modality="diagram", page=1, width=10, height=10, renderer=lambda: b"b")
    c = SanitizedCorpusBuilder(renderer_version="r2").render(source(), modality="diagram", page=1, width=10, height=10, renderer=lambda: b"a")
    assert len({a.representation_id, b.representation_id, c.representation_id}) == 3


def test_visual_failure_is_quarantined_without_affecting_text():
    builder = SanitizedCorpusBuilder(renderer_version="r1")
    text = builder.render(source(), modality="text", page=1, width=1, height=1, renderer=lambda: b"safe text")
    visual = builder.render(source(), modality="page_image", page=1, width=10, height=10, renderer=lambda: (_ for _ in ()).throw(RuntimeError("renderer")))
    assert text.lifecycle == Lifecycle.READY
    assert visual.lifecycle == Lifecycle.QUARANTINED and visual.retryable
    retry = builder.render(source(), modality="page_image", page=1, width=10, height=10, renderer=lambda: b"recovered")
    assert retry.lifecycle == Lifecycle.READY


@pytest.mark.parametrize("kwargs", [
    {"width": 101, "height": 100, "page": 1, "images": 1, "batch": 1},
    {"width": 1, "height": 1, "page": 3, "images": 1, "batch": 1},
    {"width": 1, "height": 1, "page": 1, "images": 3, "batch": 1},
    {"width": 1, "height": 1, "page": 1, "images": 1, "batch": 3},
])
def test_render_limits(kwargs):
    limits = RenderLimits(max_pixels=10_000, max_pages=2, max_images=2, max_batch=2)
    with pytest.raises(ValueError):
        limits.validate(**kwargs)


def test_malformed_and_production_inputs_are_rejected_or_quarantined():
    builder = SanitizedCorpusBuilder(renderer_version="r1")
    malformed = builder.render(source(), modality="image", page=1, width=10, height=10, renderer=lambda: b"MALFORMED image")
    assert malformed.lifecycle == Lifecycle.QUARANTINED
    with pytest.raises(ValueError, match="production"):
        builder.render(source(production=True), modality="image", page=1, width=10, height=10, renderer=lambda: b"x")


def test_review_contract_keeps_generated_queries_pending_and_resolves_refs():
    rep = SanitizedCorpusBuilder(renderer_version="r1").render(source(), modality="table", page=1, width=10, height=10, renderer=lambda: b"x")
    query = MultimodalQuery("q1", "Which synthetic row?", "en", "mixed", "finance", "table", (rep.representation_id,), (rep.representation_id,), (rep.representation_id,), False)
    assert query.review_status == ReviewStatus.PENDING_REVIEW
    validate_review_contract([query], [rep])


def test_telemetry_contract_and_no_claims():
    sample = TelemetrySample("render", 100, 2, 1000, None, 4, 100, 150, concurrency=4, restart_cycle=2)
    assert sample.throughput_per_second == 20
    assert sample.storage_amplification == 1.5
    with pytest.raises(ValueError):
        TelemetrySample("embed", 1, 1, 1, 1, 1, 1, 1, concurrency=3)
    assert len(HARD_NEGATIVE_CATEGORIES) == 8


def official_embedding():
    return VLArtifactContract(
        role="embedding", repository="Qwen/Qwen3-VL-Embedding-2B",
        revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda", license="apache-2.0",
        architecture="Qwen3VLForConditionalGeneration",
        files=(ArtifactFile("model.safetensors", 4_255_140_312, "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"),),
        native_dimension=2048, supported_dimensions=(64, 128, 256, 512, 768, 1024, 2048),
        output_normalized=True, processor_revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
        min_pixels=4096, max_pixels=1_310_720,
    )


def test_official_artifact_metadata_and_remote_hash_are_bound():
    artifact = official_embedding()
    key = (artifact.repository, artifact.revision, "model.safetensors")
    validate_artifact_inventory([artifact], {key: artifact.files[0].sha256})
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_artifact_inventory([artifact], {key: "0" * 64})


def test_w4_requires_weight_method_base_and_calibration():
    with pytest.raises(ValueError, match="pinned base identity"):
        VLArtifactContract(
            role="embedding", repository="example/w4", revision="a" * 40,
            license="apache-2.0", architecture="Qwen3VLForConditionalGeneration",
            files=(ArtifactFile("model.safetensors", 1, "b" * 64),),
            quantization_method="AWQ", weight_bits=4,
        ).validate_metadata()


def test_runtime_gate_distinguishes_installation_from_exact_forward_support():
    artifact = official_embedding()
    installed_only = RuntimeQualification(
        "transformers", "4.57.1", True, True, None, None, None,
        multimodal_forward_proven=False, embedding_head_proven=False,
    )
    assert not installed_only.executable_for(artifact, cuda=False)
    proven_cpu = RuntimeQualification(
        "transformers", "4.57.1", True, True, None, None, None,
        multimodal_forward_proven=True, embedding_head_proven=True,
    )
    assert proven_cpu.executable_for(artifact, cuda=False)
    assert not proven_cpu.executable_for(artifact, cuda=True)


def test_feasibility_ladder_stops_after_first_failure():
    validate_feasibility_ladder([
        StageResult(FeasibilityStage.METADATA, True, True, "pinned metadata validated"),
        StageResult(FeasibilityStage.CPU_FORWARD, False, True, "runtime absent", "RUNTIME_UNAVAILABLE"),
    ])
    with pytest.raises(ValueError, match="after a failed stage"):
        validate_feasibility_ladder([
            StageResult(FeasibilityStage.METADATA, False, True, "bad hash", "HASH_MISMATCH"),
            StageResult(FeasibilityStage.CPU_FORWARD, True, True, "must not run"),
        ])


def test_published_vl_qualification_is_pinned_and_fail_closed():
    report = json.loads((Path(__file__).parents[1] / "docs/issue90-vl-qualification.json").read_text())
    embedding = report["official"]["embedding"]
    reranker = report["official"]["reranker"]
    assert embedding["revision"] == "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    assert embedding["weight"]["remote_lfs_sha256"] == "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"
    assert embedding["evaluation_dimension"] == 768 and embedding["normalization_required"]
    assert reranker["revision"] == "4bd860ac4f15ad1897a214615cccc700f8f71818"
    assert reranker["score_contract"] == {
        "true_token_id": 9693, "false_token_id": 2152,
        "template": "reranker", "add_generation_prompt": True,
    }
    assert all(candidate["disposition"] != "EXECUTABLE" for candidate in report["w4_candidates"])
    assert report["conclusion"] == "W4_EXECUTABLE_ON_PASCAL=BLOCKED_ON_RUNTIME_EVIDENCE"
