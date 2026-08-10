import os
import json
import threading
import subprocess
import sys
import tarfile
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def test_healthy_activation_reports_candidate_and_previous_release(tmp_path, monkeypatch):
    releases = tmp_path / "releases"
    (releases / "v1").mkdir(parents=True)
    (releases / "v2").mkdir()
    (tmp_path / "current").symlink_to(releases / "v1")
    systemctl = tmp_path / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    systemctl.chmod(0o755)
    monkeypatch.setenv("OPENNICF_SYSTEMCTL", str(systemctl))

    class ReadyHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), ReadyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = activate(tmp_path, "v2", "opennicf-test", f"http://127.0.0.1:{server.server_port}/ready")
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert result == {"status": "active", "version": "v2", "previous": "v1", "service": "opennicf-test"}
    assert (tmp_path / "current").resolve().name == "v2"


def test_install_units_copies_templates_and_reloads_systemd(tmp_path):
    artifact = tmp_path / "release.tar.gz"
    subprocess.run(["bash", "deploy/runtime/build-release", "units-1", str(artifact)], check=True)
    runtime = tmp_path / "runtime"
    env = {
        **os.environ,
        "OPENNICF_RUNTIME_ROOT": str(runtime),
        "OPENNICF_ETC_ROOT": str(tmp_path / "etc"),
        "OPENNICF_STATE_ROOT": str(tmp_path / "state"),
        "OPENNICF_LOG_ROOT": str(tmp_path / "log"),
        "OPENNICF_SKIP_USER_SETUP": "true",
    }
    subprocess.run(["bash", "deploy/runtime/install-release", str(artifact), "units-1"], check=True, env=env)
    (runtime / "current").symlink_to(runtime / "releases" / "units-1")
    systemctl = tmp_path / "systemctl"
    log = tmp_path / "systemctl.log"
    systemctl.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\n", encoding="utf-8")
    systemctl.chmod(0o755)
    subprocess.run(
        ["bash", "deploy/runtime/install-units"],
        check=True,
        env={**env, "OPENNICF_SYSTEMCTL": str(systemctl), "OPENNICF_SYSTEMD_UNIT_DIR": str(tmp_path / "units")},
    )
    assert (tmp_path / "units" / "opennicf-knowledge.service").exists()
    assert (tmp_path / "units" / "opennicf-ingestion.service").exists()
    assert log.read_text(encoding="utf-8").strip() == "daemon-reload"
