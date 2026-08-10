from __future__ import annotations

import io
import tarfile
import zipfile
from xml.sax.saxutils import escape

import pytest

from opennicf import (
    DirectoryWatcher,
    IngestionJob,
    IngestionQueue,
    IngestionService,
    KnowledgePlatform,
    SFTPWatcher,
)


def _build_docx_bytes(paragraphs: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
""",
        )
        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
        )
        body = "".join(
            f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>" for paragraph in paragraphs
        )
        zf.writestr(
            "word/document.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{body}</w:body>
</w:document>
""",
        )
    return buffer.getvalue()


def test_docx_webex_attachment_is_parsed_into_provenanced_chunks():
    platform = KnowledgePlatform.in_memory()
    service = IngestionService(platform)

    service.submit_webex_message(
        room_id="room-42",
        message_id="msg-7",
        message_text="manual note",
        attachments=[("evidence.docx", _build_docx_bytes(["First doc paragraph", "Second doc paragraph"]))],
    )
    outcomes = service.run()

    assert len(outcomes) == 2
    assert any("First doc paragraph" in chunk.text for chunk in platform.store.chunks.values())
    assert any(chunk.locator.endswith("#paragraph-1") for chunk in platform.store.chunks.values())
    assert all(bundle.version.parser_version == "1" for outcome in outcomes for bundle in outcome.bundles)


def test_docx_telegram_attachment_is_parsed_into_provenanced_chunks():
    platform = KnowledgePlatform.in_memory()
    service = IngestionService(platform)

    service.submit_telegram_message(
        chat_id="chat-99",
        message_id="msg-17",
        message_text="telegram note",
        attachments=[("evidence.docx", _build_docx_bytes(["Telegram paragraph one", "Telegram paragraph two"]))],
    )
    outcomes = service.run()

    assert len(outcomes) == 2
    assert any("Telegram paragraph one" in chunk.text for chunk in platform.store.chunks.values())
    assert any(chunk.locator.endswith("#paragraph-1") for chunk in platform.store.chunks.values())
    assert all(bundle.version.parser_version == "1" for outcome in outcomes for bundle in outcome.bundles)


def test_tar_gz_archive_submission_is_supported_and_path_traversal_is_rejected(tmp_path):
    service = IngestionService(KnowledgePlatform.in_memory())

    archive_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        payload = b"archive line one\narchive line two\n"
        info = tarfile.TarInfo("evidence.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    jobs = service.submit_file(archive_path, channel="sftp")
    assert len(jobs) == 1
    service.run()
    assert any("archive line one" in chunk.text for chunk in service.platform.store.chunks.values())

    unsafe_archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../evil.txt", b"should fail")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        service.submit_file(unsafe_archive, channel="sftp")


def test_manual_submission_is_deterministic_and_idempotent():
    service = IngestionService(KnowledgePlatform.in_memory())

    first = service.submit_manual(source_uri="manual://evidence", content="stable payload")
    second = service.submit_manual(source_uri="manual://evidence", content="stable payload")

    assert first.job_id == second.job_id
    assert len(service.queue.pending) == 1


def test_queue_snapshot_restore_and_dead_letter_retry_roundtrip():
    queue = IngestionQueue()
    job = IngestionJob(
        job_id="job_fixed",
        source_id="src_fixed",
        source_uri="manual://evidence",
        channel="manual",
        mime_type="text/plain",
        content_hash="content-hash",
        raw_bytes=b"queue payload",
    )
    assert queue.enqueue(job)
    claimed = queue.claim()
    assert claimed is not None
    queue.fail(claimed, "malformed payload", retryable=False)
    assert len(queue.dead_letters) == 1
    assert queue.retry_dead_letter(job.job_id)

    retried = queue.claim()
    assert retried is not None
    assert retried.job_id == job.job_id
    assert retried.attempts == 2

    snapshot = queue.snapshot()
    restored = IngestionQueue()
    restored.restore(snapshot)
    recovered = restored.snapshot()
    assert recovered["in_progress"] == []
    assert [item["job_id"] for item in recovered["pending"]] == [job.job_id]


def test_queue_state_file_recovers_claimed_job_and_bounds_status(tmp_path):
    state_path = tmp_path / "ingestion.json"
    queue = IngestionQueue(state_path=state_path)
    job = IngestionJob("job-restart", "src", "manual://src", "manual", "text/plain", "hash", b"payload")
    assert queue.enqueue(job)
    assert queue.claim() is not None

    restored = IngestionQueue.from_state(state_path)
    assert restored.status(limit=1)["in_progress"] == 0
    assert restored.status(limit=1)["pending"] == 1
    assert restored.claim() is not None


def test_watchers_are_idempotent_and_preserve_acl_domain_metadata(tmp_path):
    platform = KnowledgePlatform.in_memory()
    service = IngestionService(platform)
    watched = tmp_path / "watched"
    watched.mkdir()
    (watched / "evidence.txt").write_text("watched evidence", encoding="utf-8")

    directory = DirectoryWatcher(service, watched, domain_id="finance", system_id="ledger", acl_scope="restricted")
    assert len(directory.scan()) == 1
    assert directory.scan() == []
    service.run()
    chunk = next(iter(platform.store.chunks.values()))
    assert chunk.domain_id == "finance"
    assert chunk.system_id == "ledger"
    assert chunk.acl_scope == "restricted"

    remote = SFTPWatcher(service, lambda: ["drop/report.txt"], lambda _: b"remote evidence")
    assert len(remote.scan()) == 1
    assert len(remote.scan()) == 0  # queue idempotency, even when the source is polled again


def test_parser_version_reprocessing_creates_a_new_version(tmp_path):
    service = IngestionService(KnowledgePlatform.in_memory())
    job = service.submit_manual(source_uri="manual://reprocess.txt", content="reparse me")
    service.run()
    reprocessed = service.reprocess(job.source_id, parser_version="2")
    assert len(reprocessed) == 1
    service.run()
    versions = [version for version in service.platform.store.source_versions.values() if version.source_id == job.source_id]
    assert {version.parser_version for version in versions} == {"1", "2"}


def test_filesystem_landing_and_lifecycle_are_restart_safe(tmp_path):
    state_path = tmp_path / "state.json"
    landing = tmp_path / "landing"
    platform = KnowledgePlatform.in_memory()
    service = IngestionService(
        platform,
        queue=IngestionQueue(state_path=state_path),
        landing_root=landing,
    )
    job = service.submit_manual(
        source_uri="manual://local-evidence",
        content="local-only evidence",
        domain_id="finance",
        system_id="ledger",
        acl_scope="restricted",
        local_only=True,
    )
    assert job.landing_object_key
    assert (landing / job.landing_object_key).read_text(encoding="utf-8") == "local-only evidence"

    # A fresh service reconstructs the queue and reads the landed bytes rather
    # than depending on an in-process parser or queue object.
    restarted = IngestionService(
        platform,
        queue=IngestionQueue.from_state(state_path),
        landing_root=landing,
    )
    outcome = restarted.process_next()
    assert outcome is not None
    states = [event.state for event in restarted.queue.lifecycle[job.job_id]]
    assert states == ["received", "validated", "stored", "parsed", "chunked", "embedded", "indexed"]
    status = restarted.status(limit=1)
    assert status["states"]["indexed"] == 1
    chunk = next(iter(platform.store.chunks.values()))
    assert (chunk.domain_id, chunk.system_id, chunk.acl_scope) == ("finance", "ledger", "restricted")
    assert platform.store.sources[job.source_id].metadata["local_only"] is True


def test_failed_jobs_are_retryable_then_quarantined_and_status_is_bounded():
    service = IngestionService(
        KnowledgePlatform.in_memory(),
        queue=IngestionQueue(max_attempts=2),
        malware_scanner=lambda _content, _metadata: (_ for _ in ()).throw(ValueError("scan failed")),
    )
    job = service.submit_manual(source_uri="manual://bad", content="bad evidence")
    assert service.run() == []
    assert service.run() == []
    assert service.queue.jobs[job.job_id].state == "quarantined"
    states = [event.state for event in service.queue.lifecycle[job.job_id]]
    assert states == ["received", "failed", "retry", "failed", "quarantined"]
    status = service.status(limit=0)
    assert len(status["jobs"]) <= 1
    assert status["states"]["quarantined"] == 1
