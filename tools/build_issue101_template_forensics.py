#!/usr/bin/env python3
"""Produce byte-exact, immutable Issue #101 template evidence."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def normalize_historical(text: str) -> str:
    return " ".join(text.split())


def historical_query(text: str) -> str:
    return f"Instruct: {INSTRUCTION} Query: {normalize_historical(text)}"


def official_query(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery:{text}"


def historical_document(text: str) -> str:
    return normalize_historical(text)


def official_document(text: str) -> str:
    return text


def identity(name: str, query_template: str, document_template: str, source: str) -> dict:
    binding = {
        "id": name, "encoding": "UTF-8", "instruction": INSTRUCTION,
        "query_template": query_template, "document_template": document_template,
        "source": source, "normalization": "python_whitespace_split_join" if "historical" in name else "none",
        "tokenizer_binding": "PENDING_PINNED_RUNTIME_TOKEN_FORENSICS",
    }
    binding["sha256"] = digest_bytes(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode())
    return binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample query")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample = args.sample
    historical = historical_query(sample).encode("utf-8")
    official = official_query(sample).encode("utf-8")
    report = {
        "schema_version": 1,
        "historical": identity("template_4b_historical_t01", "Instruct: {instruction} Query: {normalized_text}", "{normalized_text}", "tools/run_issue86_cuda_vectors.py@192331e97427e7d7080c3a186a4ac810a494ba1b"),
        "official": identity("template_qwen3_official", "Instruct: {instruction}\\nQuery:{text}", "{text}", "Qwen official model contract pinned by issue #101"),
        "sample": {
            "historical_utf8_hex": historical.hex(), "official_utf8_hex": official.hex(),
            "historical_sha256": digest_bytes(historical), "official_sha256": digest_bytes(official),
            "historical_bytes": len(historical), "official_bytes": len(official),
        },
        "difference_scope": "QUERIES_AND_DOCUMENT_WHITESPACE_NORMALIZATION",
        "token_forensics": "PENDING_EXACT_PINNED_TOKENIZER_RUNTIME",
        "official_input_transport": "JSON_OR_DIRECT_API_REQUIRED_TO_PRESERVE_EMBEDDED_LF",
        "reproduction_gate": "REQUIRED_BEFORE_DIMENSION_ATTRIBUTION",
        "canonical_t00_t07": "PRESERVED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
