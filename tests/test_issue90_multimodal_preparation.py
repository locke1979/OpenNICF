from hashlib import sha256

import pytest

from opennicf.multimodal_evaluation import (
    HARD_NEGATIVE_CATEGORIES, Lifecycle, MultimodalQuery, RenderLimits,
    ReviewStatus, SanitizedCorpusBuilder, SourceRecord, TelemetrySample,
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
