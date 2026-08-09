"""Helpers for source-controlled knowledge-store migrations."""

from __future__ import annotations

from hashlib import sha256
import importlib.resources as resources
from pathlib import Path


def migration_version(name: str) -> int:
    return int(name.split("_", 1)[0])


def iter_migration_files(package: str = "opennicf.knowledge.migrations") -> list[Path]:
    root = resources.files(package)
    return sorted((path for path in root.iterdir() if path.suffix == ".sql"), key=lambda path: migration_version(path.name))


def migration_checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

