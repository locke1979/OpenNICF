from __future__ import annotations

import io
import tarfile
import zipfile
from xml.sax.saxutils import escape

import pytest

from opennicf import IngestionJob, IngestionQueue, IngestionService, KnowledgePlatform


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
    assert restored.snapshot() == snapshot
