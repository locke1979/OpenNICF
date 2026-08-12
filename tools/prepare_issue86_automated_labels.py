#!/usr/bin/env python3
"""Emit separate automated labels from the issue #85/#90 review queues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opennicf.automated_labels import build_audit, build_multimodal_automated_labels, build_text_automated_labels


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-review", type=Path, required=True)
    parser.add_argument("--multimodal-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    text = build_text_automated_labels(_load(args.text_review))
    multimodal = build_multimodal_automated_labels(_load(args.multimodal_review))
    audit = build_audit(text, multimodal)
    _write(args.output_dir / "automated-text-labels-v1.json", text)
    _write(args.output_dir / "automated-multimodal-labels-v1.json", multimodal)
    _write(args.output_dir / "automated-label-audit-v1.json", audit)
    return 0 if audit["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
