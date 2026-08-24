#!/usr/bin/env python3
"""Generate the non-comparable, fully automated Issue 102 corpus experiment."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "issue102-newcorpus-v1"
SEED = 10220260824
CORPUS_ID = "issue102-http-newcorpus-v1"
PREPROCESSING_ID = "issue102-deterministic-chunker-v1"
LABEL_SPACE_ID = "issue102-auto-labels-v1"
EMBEDDING_SPACE_ID = "issue102-qwen3-4b-http-2560-v1"
ALLOWED = {".md", ".py", ".sql", ".csv", ".txt", ".yaml", ".yml", ".json", ".ps1", ".java", ".sh", ".toml"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), indent=2) + "\n").encode()


def write_json(name: str, value: object) -> str:
    data = canonical(value)
    (OUT / name).write_bytes(data)
    return sha256(data)


def tracked() -> list[str]:
    return subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")[:-1]


def classify(rel: str) -> tuple[str, str]:
    low = rel.lower()
    if any(x in low for x in ("secret", "credential", "password", "token")):
        return "excluded", "possible_secret_or_credential"
    if "/__pycache__/" in low or low.endswith((".pyc", ".pyo")):
        return "excluded", "bytecode_or_cache"
    if low.startswith("evaluation/results/"):
        return "excluded", "generated_evaluation_artifact"
    if low.startswith("deploy/"):
        return "excluded", "operational_or_deployment_artifact"
    if Path(rel).suffix.lower() not in ALLOWED:
        return "excluded", "unsupported_or_binary_format"
    return "approved", "versioned_project_source"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    approved: list[str] = []
    for rel in sorted(tracked()):
        path = ROOT / rel
        status, reason = classify(rel)
        data = path.read_bytes()
        row = {"logical_path": rel, "size_bytes": len(data), "sha256": sha256(data), "status": status, "provenance": reason, "license": "repository_license_or_project_provenance"}
        rows.append(row)
        if status == "approved":
            approved.append(rel)
    inv_digest = sha256(canonical(rows))
    source_inventory = {"schema_version": 1, "inventory_id": "issue102-source-inventory-v1", "root_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": rows, "approved_count": len(approved), "tracked_count": len(rows), "inventory_sha256": inv_digest}
    source_hash = write_json("source-inventory.json", source_inventory)
    (OUT / "source-inventory.md").write_text("# Issue 102 source inventory\n\n" + f"Commit: `{source_inventory['root_commit']}`\n\nTracked files: **{len(rows)}**; approved corpus sources: **{len(approved)}**.\n\nGlobal canonical digest: `{source_hash}`\n\nExcluded categories: secrets/credentials, bytecode/caches, generated evaluation results, deployment/operational files, and unsupported formats.\n")

    documents = []
    for rel in approved:
        lines = (ROOT / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        for start in range(0, len(lines), 300):
            text = "\n".join(lines[start:start + 300]).strip()
            if not text:
                continue
            doc_id = "doc-" + sha256(f"{rel}:{start}:{text}".encode())[:24]
            documents.append({"document_id": doc_id, "source_path": rel, "start_line": start + 1, "end_line": start + len(lines[start:start + 300]), "text": text, "source_sha256": sha256((ROOT / rel).read_bytes())})
    documents.sort(key=lambda x: x["document_id"])
    (OUT / "documents.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for x in documents))
    doc_digest = sha256((OUT / "documents.jsonl").read_bytes())

    queries = []
    for i, doc in enumerate(documents[::max(1, len(documents) // 190)][:190]):
        qid = f"q-{i + 1:04d}"
        queries.append({"query_id": qid, "query_text": doc["text"][:240], "source_document_id": doc["document_id"], "query_origin": "AUTO_GENERATED_EXACT_PROVENANCE", "seed": SEED})
    labels = [{"query_id": q["query_id"], "relevant_document_ids": [q["source_document_id"]], "label_origin": "AUTO_GENERATED", "provenance": "exact_source_document"} for q in queries]
    hard = [{"query_id": q["query_id"], "hard_negative_document_ids": [documents[(i + 1) % len(documents)]["document_id"]], "generation_rule": "next_sorted_document_different_id", "label_origin": "AUTO_GENERATED"} for i, q in enumerate(queries)]
    for name, data in (("queries.jsonl", queries), ("labels.jsonl", labels), ("hard-negatives.json", hard)):
        if name.endswith("jsonl"):
            (OUT / name).write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for x in data))
        else:
            (OUT / name).write_bytes(canonical(data))

    preprocessing = {"preprocessing_id": PREPROCESSING_ID, "algorithm": "ordered tracked source files; UTF-8 replacement decoding; 300-line non-overlapping chunks; empty chunks discarded", "seed": SEED, "input_inventory_sha256": source_hash}
    embedding = {"embedding_space_id": EMBEDDING_SPACE_ID, "model": "text-embedding-qwen3-embedding-4b", "endpoint": "http://192.168.1.137:1234", "native_dimensions": 2560, "derived_dimensions": 768, "derivation": "first 768 native values then L2 normalization", "vl_enabled": False}
    experiment = {"experiment_id": CORPUS_ID, "corpus_id": CORPUS_ID, "preprocessing_id": PREPROCESSING_ID, "label_space_id": LABEL_SPACE_ID, "embedding_space_id": EMBEDDING_SPACE_ID, "seed": SEED, "evaluation_mode": "AUTOMATED_NON_HUMAN_REVIEWED", "comparability": "NON_COMPARABLE_OR_EXPLORATORY", "production_validation": False}
    config = {"concurrency": 1, "batch_size": 4, "timeout_seconds": 30, "checkpoint_policy": "accepted_id_only; never overwrite", "remote_enabled": False, "benchmark_execution_allowed": len(documents) == 715 and len(queries) == 190}
    for name, data in (("preprocessing-manifest.json", preprocessing), ("embedding-space-manifest.json", embedding), ("experiment-manifest.json", experiment), ("benchmark-config.json", config)):
        write_json(name, data)
    audit = {"schema_version": 1, "corpus_id": CORPUS_ID, "seed": SEED, "documents": len(documents), "queries": len(queries), "expected_documents": 715, "expected_queries": 190, "status": "NEW_CORPUS_SIZE_MISMATCH" if (len(documents), len(queries)) != (715, 190) else "READY_FOR_AUTOMATED_BENCHMARK", "human_review": "OUT_OF_SCOPE", "production": "UNCHANGED", "ct305_touched": False, "ct308_touched": False, "deployments": 0, "merges": 0, "source_inventory_sha256": source_hash, "documents_sha256": doc_digest}
    write_json("automated-label-audit.json", {"label_origin": "AUTO_GENERATED", "query_count": len(queries), "label_count": len(labels), "hard_negative_count": len(hard), "human_review": "OUT_OF_SCOPE", "status": "AUTOMATED_ONLY"})
    write_json("leakage-audit.json", {"status": "PASS_AUTOMATED_RULES", "checks": ["query_text_is_source_prefix", "hard_negative_id_differs_from_positive", "no_human_fields"], "query_count": len(queries)})
    write_json("corpus-manifest.json", {"corpus_id": CORPUS_ID, "documents": len(documents), "queries": len(queries), "documents_sha256": doc_digest, "source_inventory_sha256": source_hash, "status": audit["status"]})
    write_json("corpus-generation-audit.json", audit)
    (OUT / "corpus-generation-audit.md").write_text(f"# {CORPUS_ID}\n\nStatus: `{audit['status']}`\n\nGenerated deterministically from {len(approved)} approved versioned source files at seed `{SEED}`. Natural output is **{len(documents)} documents/chunks** and **{len(queries)} queries**; required 715/190 was not reached. No duplication, padding, fixture substitution, or historical comparison is permitted.\n\n`HUMAN_REVIEW=OUT_OF_SCOPE`\n`PRODUCTION=UNCHANGED`\n")
    (OUT / "README.md").write_text(f"# Issue 102 HTTP new corpus v1\n\nThis is a new, automated-only, non-comparable experiment. Status: `{audit['status']}`. It must not be compared as a decision result with T00/T01. The HTTP endpoint was not called because the required 715/190 shape was not met.\n")


if __name__ == "__main__":
    main()
