#!/usr/bin/env python3
"""Audit issue #85 inputs and emit a deterministic human review artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opennicf.text_evaluation import audit_inputs, build_review_artifact, file_sha256


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--repaired-gold-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    corpus_path = args.artifact_dir / "corpus-v2.json"
    gold_path = args.artifact_dir / "gold-set-v1.json"
    review_path = args.artifact_dir / "gold-review-queue-v2.json"
    corpus, gold, review = load(corpus_path), load(gold_path), load(review_path)
    vectors = {
        path.stem: load(path)
        for path in sorted(args.artifact_dir.glob("vectors-*.json"))
        if path.name in {"vectors-06b.json", "vectors-4b.json"}
    }
    owning_manifest = load(args.artifact_dir / "evaluation-manifest.json")
    expected = owning_manifest.get("corpus_sha256")
    audit = audit_inputs(
        corpus, gold, review, vectors,
        corpus_digest=file_sha256(corpus_path), gold_digest=file_sha256(gold_path), expected_corpus_digest=expected,
    )
    artifact = build_review_artifact(corpus, gold, review, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory = {
        "corpus_manifest_matches": expected == file_sha256(corpus_path),
        "gold_embedded_corpus_digest_matches": gold.get("corpus_manifest_sha256") == file_sha256(corpus_path),
    }
    artifact_name = owning_manifest.get("spaces", {}).get("four_b_q4", {}).get("artifact")
    artifact_path = args.artifact_dir / "artifacts" / artifact_name if artifact_name else None
    if artifact_path and artifact_path.is_file():
        actual_artifact_digest = file_sha256(artifact_path)
        expected_artifact_digest = owning_manifest["spaces"]["four_b_q4"].get("artifact_sha256")
        inventory["four_b_artifact"] = {
            "path": f"artifacts/{artifact_name}", "size_bytes": artifact_path.stat().st_size,
            "expected_sha256": expected_artifact_digest, "actual_sha256": actual_artifact_digest,
            "matches": actual_artifact_digest == expected_artifact_digest,
        }
    audit_payload = {"valid": audit.valid, "errors": audit.errors, "inventory": inventory, **audit.summary}
    if args.repaired_gold_output:
        repaired_gold = dict(gold)
        repaired_gold["corpus_manifest_sha256"] = file_sha256(corpus_path)
        repaired_gold["objective_repair"] = {
            "kind": "corpus_digest_reconciliation",
            "source_gold_sha256": file_sha256(gold_path),
            "human_labels_changed": False,
        }
        args.repaired_gold_output.parent.mkdir(parents=True, exist_ok=True)
        args.repaired_gold_output.write_text(json.dumps(repaired_gold, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit_payload["objective_repair"] = {
            "path": str(args.repaired_gold_output),
            "sha256": file_sha256(args.repaired_gold_output),
            "human_labels_changed": False,
            "repaired_embedded_corpus_digest_matches": repaired_gold["corpus_manifest_sha256"] == file_sha256(corpus_path),
        }
    args.audit_output.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
    return 0 if audit.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
