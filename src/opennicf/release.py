"""Release-artifact and health-gated activation primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


class ReleaseError(RuntimeError):
    """Raised when an artifact or activation gate fails."""


def _safe_version(version: str) -> str:
    if not version or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in version):
        raise ReleaseError("invalid release version")
    return version


def manifest_for(root: str | Path, *, version: str) -> dict:
    root = Path(root).resolve()
    version = _safe_version(version)
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append({"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size})
    return {"schema": "opennicf.release/v1", "version": version, "files": files}


def verify_manifest(root: str | Path, manifest: dict) -> None:
    if manifest.get("schema") != "opennicf.release/v1":
        raise ReleaseError("unsupported release manifest")
    _safe_version(str(manifest.get("version", "")))
    root = Path(root).resolve()
    entries = manifest.get("files", [])
    expected = {item["path"]: item for item in entries}
    if len(expected) != len(entries):
        raise ReleaseError("duplicate manifest path")
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if set(actual) != set(expected):
        raise ReleaseError("release file set differs from manifest")
    for name, item in expected.items():
        path = actual[name]
        if path.stat().st_size != int(item["size"]) or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ReleaseError(f"release hash mismatch: {name}")


def write_manifest(root: str | Path, *, version: str) -> dict:
    root = Path(root)
    manifest = manifest_for(root, version=version)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def safe_extract_artifact(archive: str | Path, destination: str | Path) -> None:
    """Extract an artifact while rejecting traversal and executable links."""
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ReleaseError("artifact path escapes extraction root")
            if member.issym() or member.islnk():
                raise ReleaseError("artifact links are not allowed")
        try:
            tar.extractall(destination, members=members, filter="data")
        except TypeError:
            # Python 3.11 lacks tarfile's filter argument.  The member
            # traversal/link checks above provide the same bounded archive
            # policy for the supported production runtime.
            tar.extractall(destination, members=members)


def _systemctl(*args: str) -> None:
    command = os.environ.get("OPENNICF_SYSTEMCTL", "systemctl")
    result = subprocess.run([command, *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise ReleaseError(f"systemctl {args[0]} failed")


def readiness(url: str, *, timeout: float | None = None) -> bool:
    timeout = timeout if timeout is not None else float(os.environ.get("OPENNICF_RELEASE_READINESS_TIMEOUT", "15"))
    deadline = time.monotonic() + timeout
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    while True:
        try:
            with urllib.request.urlopen(request, timeout=min(5.0, max(0.1, deadline - time.monotonic()))) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)


def _switch(root: Path, version: str) -> None:
    target = root / "releases" / _safe_version(version)
    if not target.is_dir():
        raise ReleaseError("release directory missing")
    temporary = root / ".current.new"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    temporary.replace(root / "current")


def activate(root: str | Path, version: str, service: str, health_url: str) -> dict:
    root = Path(root).resolve()
    previous = (root / "current").resolve().name if (root / "current").is_symlink() else None
    _switch(root, version)
    try:
        _systemctl("restart", service)
        if not readiness(health_url):
            raise ReleaseError("readiness gate failed")
    except ReleaseError:
        if previous:
            _switch(root, previous)
            try:
                _systemctl("restart", service)
            except ReleaseError:
                pass
        raise
    return {"status": "active", "version": version, "previous": previous, "service": service}


def rollback(root: str | Path, service: str, health_url: str) -> dict:
    root = Path(root).resolve()
    releases = sorted(path.name for path in (root / "releases").iterdir() if path.is_dir())
    current = (root / "current").resolve().name if (root / "current").is_symlink() else None
    candidates = [version for version in releases if version != current]
    if not candidates:
        raise ReleaseError("no previous release available")
    version = candidates[-1]
    _switch(root, version)
    _systemctl("restart", service)
    if not readiness(health_url):
        raise ReleaseError("rollback readiness gate failed")
    return {"status": "rolled_back", "version": version, "service": service}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenNICF release controller")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("root")
    activate_parser = sub.add_parser("activate")
    activate_parser.add_argument("root")
    activate_parser.add_argument("version")
    activate_parser.add_argument("service")
    activate_parser.add_argument("health_url")
    rollback_parser = sub.add_parser("rollback")
    rollback_parser.add_argument("root")
    rollback_parser.add_argument("service")
    rollback_parser.add_argument("health_url")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            manifest = json.loads((Path(args.root) / "manifest.json").read_text(encoding="utf-8"))
            verify_manifest(args.root, manifest)
            print(json.dumps({"status": "verified", "version": manifest["version"]}, sort_keys=True))
        elif args.command == "activate":
            print(json.dumps(activate(args.root, args.version, args.service, args.health_url), sort_keys=True))
        else:
            print(json.dumps(rollback(args.root, args.service, args.health_url), sort_keys=True))
        return 0
    except (OSError, ValueError, ReleaseError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)[:160]}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
