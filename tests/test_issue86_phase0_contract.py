"""Model-free safety tests for the issue #86 Phase 0 contract."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"


def test_phase0_manifest_is_evaluation_only_and_space_isolated():
    payload = json.loads((DOCS / "issue86-phase0-manifest.example.json").read_text())
    assert payload["issue"] == 86
    assert payload["baseline_issue"] == 85
    assert payload["production"] is False
    assert payload["production_migration_authorized"] is False
    assert payload["text_space_id"] != payload["vl_space_id"]
    assert payload["fusion"] == "rrf"
    assert payload["candidate_recall_before_rerank_required"] is True


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
