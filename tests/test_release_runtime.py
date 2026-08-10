import os
import subprocess
import sys
import tarfile

import pytest

from opennicf.release import ReleaseError, activate, manifest_for, safe_extract_artifact, verify_manifest, write_manifest
from opennicf.service_runtime import checks


def test_manifest_detects_tampering(tmp_path):
    (tmp_path / "app.txt").write_text("approved", encoding="utf-8")
    manifest = write_manifest(tmp_path, version="2026.08.10-1")
    verify_manifest(tmp_path, manifest)
    (tmp_path / "app.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ReleaseError, match="hash mismatch"):
        verify_manifest(tmp_path, manifest)


def test_manifest_rejects_unexpected_file(tmp_path):
    (tmp_path / "app.txt").write_text("approved", encoding="utf-8")
    manifest = write_manifest(tmp_path, version="v1")
    (tmp_path / "unexpected").write_text("no", encoding="utf-8")
    with pytest.raises(ReleaseError, match="file set"):
        verify_manifest(tmp_path, manifest)


def test_version_rejects_path_traversal(tmp_path):
    with pytest.raises(ReleaseError):
        manifest_for(tmp_path, version="../bad")


def test_artifact_extraction_rejects_traversal_and_links(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 3
        import io
        tar.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ReleaseError, match="escapes"):
        safe_extract_artifact(archive, tmp_path / "out")


def test_ingestion_health_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENNICF_INGESTION_LANDING_ROOT", str(tmp_path / "landing"))
    result = checks("ingestion")
    assert result["ready"] is True
    assert set(result) == {"service", "release", "checks", "ready"}


def test_build_script_emits_verifiable_artifact(tmp_path):
    output = tmp_path / "release.tar.gz"
    subprocess.run(["bash", "deploy/runtime/build-release", "test-1", str(output)], check=True)
    assert output.exists() and output.stat().st_size > 0
    listing = subprocess.check_output(["tar", "-tzf", str(output)], text=True)
    assert "__pycache__" not in listing and ".pyc" not in listing


def test_install_script_verifies_into_isolated_roots(tmp_path):
    artifact = tmp_path / "release.tar.gz"
    subprocess.run(["bash", "deploy/runtime/build-release", "test-install", str(artifact)], check=True)
    runtime = tmp_path / "runtime"
    env = {
        **os.environ,
        "OPENNICF_RUNTIME_ROOT": str(runtime),
        "OPENNICF_ETC_ROOT": str(tmp_path / "etc"),
        "OPENNICF_STATE_ROOT": str(tmp_path / "state"),
        "OPENNICF_LOG_ROOT": str(tmp_path / "log"),
        "OPENNICF_SKIP_USER_SETUP": "true",
    }
    subprocess.run(["bash", "deploy/runtime/install-release", str(artifact), "test-install"], check=True, env=env)
    assert (runtime / "releases" / "test-install" / "manifest.json").exists()


def test_failed_activation_restores_previous_release(tmp_path, monkeypatch):
    releases = tmp_path / "releases"
    (releases / "v1").mkdir(parents=True)
    (releases / "v2").mkdir()
    (tmp_path / "current").symlink_to(releases / "v1")
    systemctl = tmp_path / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o755)
    monkeypatch.setenv("OPENNICF_SYSTEMCTL", str(systemctl))
    with pytest.raises(ReleaseError, match="readiness"):
        activate(tmp_path, "v2", "opennicf-test", "http://127.0.0.1:1/health/ready")
    assert (tmp_path / "current").resolve().name == "v1"
