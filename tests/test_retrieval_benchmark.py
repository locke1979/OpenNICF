import json
import shutil
from pathlib import Path

import pytest

from opennicf.contracts import parse_contract
from opennicf.knowledge import (
    EmbeddingSpaceMismatch,
    GeminiEmbeddingBackend,
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    PrivacyBoundaryError,
    RetrievalFilters,
)
from opennicf.knowledge.benchmark import (
    BenchmarkManifestError,
    RetrievalBenchmark,
    load_manifest,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "retrieval_benchmark" / "v1"
MANIFEST = FIXTURE_ROOT / "manifest.json"


def _platform(dimensions=32):
    return KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=dimensions)),
    )


def _deterministic_benchmark(platform):
    return RetrievalBenchmark(platform, MANIFEST, clock=lambda: 0)


def test_versioned_fixture_manifest_is_sanitized_and_provenance_ready():
    payload, fixtures, queries = load_manifest(MANIFEST)

    assert payload["manifest_version"] == 1
    assert payload["fixture_set"].endswith("-v1")
    assert {fixture.language for fixture in fixtures} >= {"pt-BR", "en-US", "java", "sql", "powershell", "csv"}
    assert len(fixtures) == 10
    assert len(queries) == 10
    assert all(len(fixture.content_sha256) == 64 for fixture in fixtures)


def test_manifest_rejects_fixture_hash_tampering(tmp_path):
    manifest_root = tmp_path / "v1"
    shutil.copytree(FIXTURE_ROOT, manifest_root)
    manifest_path = manifest_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fixtures"][0]["content_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="fixture hash mismatch"):
        load_manifest(manifest_path)


def test_all_retrieval_modes_are_repeatable_and_emit_provenance_manifests():
    first = _deterministic_benchmark(_platform()).run()
    second = _deterministic_benchmark(_platform()).run()

    assert set(first["summaries"]) == {"lexical", "vector", "hybrid", "symbol_aware"}
    assert first["summaries"]["symbol_aware"]["query_count"] == 3
    assert first["execution"]["fallback"] is True
    assert first["execution"]["fallback_reason"] == "deterministic_cpu_backend"
    assert first["execution"]["embedding_space_id"]

    def stable(report):
        return [(row["mode"], row["query_id"], row["metrics"], row["ranked"], row["compacted_chunk_ids"]) for row in report["results"]]

    assert stable(first) == stable(second)
    for row in first["results"]:
        assert {"recall_at_k", "mrr", "latency_ms", "returned_chunks", "context_tokens_after_compaction"} <= set(row["metrics"])
        assert all(
            hit["source_version_id"] and hit["artifact_hash"] and hit["locator"] and hit["excerpt_hash"]
            for hit in row["ranked"]
        )
    assert first["summaries"]["symbol_aware"]["mrr"] == 1.0


def test_benchmark_acl_and_domain_filters_cannot_leak_decoy_evidence():
    platform = _platform()
    benchmark = _deterministic_benchmark(platform)
    benchmark.ingest_fixtures()
    platform.ingest(
        source_id="outside-domain-public",
        source_uri="outside.py",
        content="def calculateTotal():\n    return 'decoy'",
        source_type="code",
        domain="outside_domain",
        acl_scope="public",
    )

    filters = RetrievalFilters(
        principal_acl_scopes=frozenset({"internal"}),
        domain_ids=("retrieval_benchmark",),
        limit=20,
        neighbor_window=0,
    )
    hits = platform.search("calculateTotal", filters=filters)
    assert hits
    assert all(hit.domain_id == "retrieval_benchmark" for hit in hits)
    assert all(hit.source_id != "outside-domain-public" for hit in hits)
    symbols = platform.search_code_symbols("calculateTotal", filters=filters)
    assert symbols
    assert all(symbol.domain_id == "retrieval_benchmark" and symbol.acl_scope == "internal" for symbol in symbols)


def test_benchmark_rejects_cross_embedding_space_search():
    platform = _platform(dimensions=8)
    benchmark = _deterministic_benchmark(platform)
    benchmark.ingest_fixtures()
    alternate = HashingEmbeddingBackend(model="benchmark-alternate", dimensions=8)
    platform.embeddings.register(alternate)

    with pytest.raises(EmbeddingSpaceMismatch):
        platform.search(
            "artifact hash",
            filters=RetrievalFilters(
                domain_ids=("retrieval_benchmark",),
                embedding_space_id=alternate.info.embedding_space_id,
                neighbor_window=0,
            ),
        )


def test_local_only_rejects_cloud_embedding_before_transport():
    calls = []

    def transport(_model, _payload):
        calls.append(True)
        return {"embedding": {"values": [1.0] * 8}}

    google = GeminiEmbeddingBackend("gemini-embedding-001", dimension=8, transport=transport)
    service = LocalFirstEmbeddingService(
        cpu_backend=HashingEmbeddingBackend(dimensions=8),
        providers=[google],
    )

    with pytest.raises(PrivacyBoundaryError):
        service.embed(
            ["private benchmark evidence"],
            space_id=google.info.embedding_space_id,
            privacy_policy="local_only",
        )
    assert calls == []


def test_issue_52_machine_readable_contract_remains_valid():
    body = """## Scope\nbenchmark\n\n```yaml
openclaw:
  contract_version: 1
  pipeline: subagent-pipeline
  autorun: true
  depends_on: [49, 50, 51]
  deploy: true
  deployment_stage: retrieval-evaluation
```"""

    contract = parse_contract(52, body)
    assert contract.issue_number == 52
    assert contract.autorun is True
    assert contract.depends_on == (49, 50, 51)
    assert contract.deploy is True
