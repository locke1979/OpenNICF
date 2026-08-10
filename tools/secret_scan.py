#!/usr/bin/env python3
"""Scan changed source files for secrets and private endpoints."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opennicf.ops.secrets import scan_text

excluded = {"tests/fixtures/secrets/fake-credentials.env"}


def _changed_paths(base: str | None) -> list[str]:
    if base:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...HEAD"],
            text=True,
        )
        return [path for path in output.splitlines() if path]
    staged = subprocess.check_output(
        ["git", "diff", "--name-only", "--cached", "--diff-filter=ACMRTUXB"],
        text=True,
    ).splitlines()
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--others", "--modified", "--exclude-standard"],
        text=False,
    )
    working_tree = [path.decode() for path in output.split(b"\0") if path]
    return sorted({path for path in staged + working_tree if path})


base = sys.argv[1] if len(sys.argv) > 1 else None
findings = []
for path in _changed_paths(base):
    if path in excluded or path.endswith((".png", ".jpg", ".pdf")):
        continue
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        continue
    findings.extend((path, finding) for finding in scan_text(text))

if findings:
    for path, finding in findings:
        print(f"{path}:{finding.line}: {finding.kind}")
    sys.exit("secret scan failed")
print("secret scan: OK")
