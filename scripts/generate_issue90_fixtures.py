#!/usr/bin/env python3
"""Generate deterministic, synthetic-only Gate C review fixtures."""
import json
from hashlib import sha256
from pathlib import Path

from opennicf.multimodal_evaluation import (
    HARD_NEGATIVE_CATEGORIES, MultimodalQuery, SanitizedCorpusBuilder, SourceRecord,
    validate_review_contract,
)


def main(output: str) -> None:
    builder = SanitizedCorpusBuilder(renderer_version="opennicf-synthetic-svg@1")
    representations = []
    modalities = ("page_image", "screenshot", "diagram", "table_form", "ocr_poor", "mixed")
    for index in range(100):
        source_id = f"synthetic-source-{index:03d}"
        payload = f"SANITIZED FIXTURE {index:03d}".encode()
        source = SourceRecord(source_id, sha256(payload).hexdigest(), "image/svg+xml", f"domain-{index % 5}", "synthetic", ("evaluation",))
        representations.append(builder.render(source, modality=modalities[index % len(modalities)], page=1, width=640, height=480, renderer=lambda p=payload: p))
    queries = []
    for index in range(50):
        relevant = representations[index * 2]
        negative = representations[(index * 2 + 1) % 100]
        queries.append(MultimodalQuery(
            f"mmq-{index:03d}", f"Find the sanitized synthetic evidence number {index * 2:03d}.",
            "en" if index % 2 == 0 else "pt", "mixed", relevant.domain,
            relevant.modality, (relevant.representation_id,), (relevant.representation_id,),
            (negative.representation_id,), False,
            leakage_indicators=("synthetic_numeric_identifier",),
            hard_negative_category=HARD_NEGATIVE_CATEGORIES[index % len(HARD_NEGATIVE_CATEGORIES)],
        ))
    validate_review_contract(queries, representations)
    document = {
        "schema_version": 1,
        "safety": {"synthetic_only": True, "production_routes_changed": False},
        "review_instructions": {
            "generated_labels_are_authoritative": False,
            "allowed_decisions": ["ACCEPT", "REWRITE", "REJECT"],
            "required_for_decision": ["review_status", "review_notes"],
            "batches": [
                {"batch_id": f"issue90-{start // 10 + 1}", "query_ids": [q.query_id for q in queries[start:start + 10]]}
                for start in range(0, 50, 10)
            ],
        },
        "representations": builder.manifest(),
        "queries": [q.__dict__ for q in queries],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=lambda value: value.value) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    main(args.output)
