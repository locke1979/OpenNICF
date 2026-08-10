"""Safe ingestion pipeline for Webex, SFTP, mail-drop and manual evidence."""

from __future__ import annotations

import base64
import csv
import io
import json
import mimetypes
import posixpath
import re
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from email import policy
from email.message import Message
from email.parser import BytesParser
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

import yaml

from .knowledge import FilesystemObjectStore, IngestBundle, KnowledgePlatform, ParsedBlock
from .knowledge.namespaces import sanitize_metadata

ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".tar.gz", ".7z"}
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".csv",
    ".tsv",
    ".log",
    ".sql",
    ".ddl",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".java",
    ".rb",
    ".sh",
    ".ps1",
}


LIFECYCLE_STATES = (
    "received",
    "validated",
    "stored",
    "parsed",
    "chunked",
    "embedded",
    "indexed",
    "failed",
    "quarantined",
    "retry",
)


@dataclass(frozen=True)
class IngestionLimits:
    max_bytes: int = 50_000_000
    max_archive_entries: int = 1_000
    max_archive_expansion_bytes: int = 200_000_000
    max_single_file_bytes: int = 20_000_000
    max_concurrency: int = 1


@dataclass(frozen=True)
class IngestionJob:
    job_id: str
    source_id: str
    source_uri: str
    channel: str
    mime_type: str
    content_hash: str
    raw_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    source_kind: str = "document"
    source_type: str = "document"
    domain: str = "general"
    system: str = "unknown"
    domain_id: str = "general"
    system_id: str = "unknown"
    component_id: str = "unknown-component"
    evidence_type: str = "document"
    environment: str = "unknown"
    acl_scope: str = "internal"
    parser_hint: str | None = None
    parser_version: str | None = None
    state: str = "received"
    attempts: int = 0
    landing_object_key: str | None = None
    local_only: bool = False


@dataclass(frozen=True)
class DeadLetterEntry:
    job: IngestionJob
    reason: str
    failed_at: str
    retryable: bool = True


@dataclass(frozen=True)
class IngestionOutcome:
    job_id: str
    created: bool
    bundles: tuple[IngestBundle, ...]
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LifecycleEvent:
    state: str
    at: str
    details: dict[str, Any] = field(default_factory=dict)


class FilesystemLandingStore:
    """Durable first landing area for received bytes.

    Landing is content addressed and immutable.  The knowledge object store is
    still the authority for stored artifacts; this small boundary makes receipt
    durable before validation/parsing starts and allows a restart to recover the
    exact bytes that were acknowledged by the application.
    """

    def __init__(self, root: str | Path):
        self._store = FilesystemObjectStore(root)

    def put(self, content: bytes, *, mime_type: str, metadata: dict[str, Any] | None = None):
        return self._store.put_bytes(content, mime_type=mime_type, metadata=metadata)

    def get(self, object_key: str) -> bytes:
        return self._store.get_bytes(object_key)


class IngestionQueue:
    """Deterministic, restartable queue with a dead-letter queue."""

    def __init__(self, *, state_path: str | Path | None = None, max_attempts: int = 3):
        self.state_path = Path(state_path) if state_path else None
        self.max_attempts = max(1, int(max_attempts))
        self.pending: list[IngestionJob] = []
        self.in_progress: dict[str, IngestionJob] = {}
        self.completed: dict[str, IngestionOutcome] = {}
        self.dead_letters: list[DeadLetterEntry] = []
        self._seen: set[str] = set()
        self.jobs: dict[str, IngestionJob] = {}
        self.lifecycle: dict[str, list[LifecycleEvent]] = {}

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.snapshot(), sort_keys=True, default=_json_default), encoding="utf-8")
        temporary.replace(self.state_path)

    def enqueue(self, job: IngestionJob) -> bool:
        if job.job_id in self._seen:
            return False
        job = replace(job, state="received")
        self.pending.append(job)
        self._seen.add(job.job_id)
        self.jobs[job.job_id] = job
        self._event(job.job_id, "received")
        self._persist()
        return True

    def _event(self, job_id: str, state: str, details: dict[str, Any] | None = None) -> None:
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown ingestion lifecycle state: {state}")
        self.lifecycle.setdefault(job_id, []).append(
            LifecycleEvent(state, datetime.now(UTC).isoformat(), dict(details or {}))
        )

    def transition(self, job_id: str, state: str, *, details: dict[str, Any] | None = None) -> IngestionJob:
        job = self.jobs[job_id]
        updated = replace(job, state=state)
        self.jobs[job_id] = updated
        self.pending = [updated if item.job_id == job_id else item for item in self.pending]
        self.in_progress = {key: (updated if key == job_id else item) for key, item in self.in_progress.items()}
        self._event(job_id, state, details)
        self._persist()
        return updated

    def claim(self) -> IngestionJob | None:
        if not self.pending:
            return None
        job = self.pending.pop(0)
        # ``running`` remains an operational queue state; lifecycle events are
        # reserved for durable evidence milestones.
        claimed = replace(job, state="running", attempts=job.attempts + 1)
        self.in_progress[job.job_id] = claimed
        self.jobs[job.job_id] = claimed
        self._persist()
        return self.in_progress[job.job_id]

    def ack(self, job: IngestionJob, outcome: IngestionOutcome) -> None:
        self.in_progress.pop(job.job_id, None)
        self.completed[job.job_id] = outcome
        self.jobs[job.job_id] = replace(job, state="indexed")
        self._persist()

    def fail(self, job: IngestionJob, reason: str, *, retryable: bool = True) -> None:
        self.in_progress.pop(job.job_id, None)
        self.jobs[job.job_id] = replace(job, state="failed")
        self._event(job.job_id, "failed", {"reason": reason[:240], "retryable": retryable})
        dead = DeadLetterEntry(
            job=replace(job, state="dead-letter"),
            reason=reason,
            failed_at=datetime.now(UTC).isoformat(),
            retryable=retryable,
        )
        self.dead_letters.append(dead)
        if retryable and job.attempts < self.max_attempts:
            retry = replace(job, state="retry")
            self.pending.append(retry)
            self.jobs[job.job_id] = retry
            self._event(job.job_id, "retry", {"attempt": job.attempts + 1})
        else:
            self.jobs[job.job_id] = replace(job, state="quarantined")
            self._event(job.job_id, "quarantined")
        self._persist()

    def retry_dead_letter(self, job_id: str) -> bool:
        for index, entry in enumerate(list(self.dead_letters)):
            if entry.job.job_id == job_id:
                self.dead_letters.pop(index)
                retry = replace(entry.job, state="retry")
                self.pending.append(retry)
                self.jobs[job_id] = retry
                self._seen.add(job_id)
                self._event(job_id, "retry", {"manual": True})
                self._persist()
                return True
        return False

    def status(self, *, limit: int = 100) -> dict[str, Any]:
        """Return a bounded operational view without exposing payload bytes."""
        bound = max(1, int(limit))
        jobs = [*self.pending, *self.in_progress.values()]
        return {
            "pending": len(self.pending),
            "in_progress": len(self.in_progress),
            "completed": len(self.completed),
            "dead_letters": len(self.dead_letters),
            "states": {state: sum(1 for job in self.jobs.values() if job.state == state) for state in LIFECYCLE_STATES},
            "jobs": [
                {"job_id": job.job_id, "source_id": job.source_id, "state": job.state, "attempts": job.attempts}
                for job in jobs[:bound]
            ],
            "dead_letter_jobs": [entry.job.job_id for entry in self.dead_letters[:bound]],
            "lifecycle": {
                job_id: [asdict(event) for event in events[-bound:]]
                for job_id, events in list(self.lifecycle.items())[:bound]
            },
        }

    def retry_dead_letters(self, *, limit: int = 10) -> int:
        retried = 0
        for entry in list(self.dead_letters):
            if retried >= max(0, int(limit)):
                break
            if self.retry_dead_letter(entry.job.job_id):
                retried += 1
        return retried

    @classmethod
    def from_state(cls, state_path: str | Path, *, max_attempts: int = 3) -> IngestionQueue:
        queue = cls(state_path=state_path, max_attempts=max_attempts)
        if queue.state_path and queue.state_path.exists():
            queue.restore(json.loads(queue.state_path.read_text(encoding="utf-8")))
        return queue

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending": [_job_to_json(job) for job in self.pending],
            "in_progress": [_job_to_json(job) for job in self.in_progress.values()],
            "completed": {job_id: _outcome_to_json(outcome) for job_id, outcome in self.completed.items()},
            "dead_letters": [_dead_letter_to_json(entry) for entry in self.dead_letters],
            "seen": sorted(self._seen),
            "jobs": {job_id: _job_to_json(job) for job_id, job in self.jobs.items()},
            "lifecycle": {job_id: [asdict(event) for event in events] for job_id, events in self.lifecycle.items()},
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        state_path, max_attempts = self.state_path, self.max_attempts
        self.__init__(state_path=state_path, max_attempts=max_attempts)
        self.pending = [_job_from_json(job) for job in snapshot.get("pending", [])]
        # A process can die after claim and before ack. Requeue those jobs so a
        # restart never leaves evidence permanently invisible.
        recovered = [replace(_job_from_json(job), state="queued") for job in snapshot.get("in_progress", [])]
        self.pending.extend(recovered)
        self.in_progress = {}
        self.completed = {
            job_id: _outcome_from_json(outcome)
            for job_id, outcome in snapshot.get("completed", {}).items()
        }
        self.dead_letters = [_dead_letter_from_json(entry) for entry in snapshot.get("dead_letters", [])]
        self._seen = set(snapshot.get("seen", []))
        self.jobs = {job_id: _job_from_json(job) for job_id, job in snapshot.get("jobs", {}).items()}
        self.lifecycle = {
            job_id: [LifecycleEvent(**event) for event in events]
            for job_id, events in snapshot.get("lifecycle", {}).items()
        }
        for job in [*self.pending, *self.in_progress.values()]:
            self.jobs.setdefault(job.job_id, job)
        for entry in self.dead_letters:
            self.jobs.setdefault(entry.job.job_id, entry.job)
        self._persist()


def _job_to_json(job: IngestionJob) -> dict[str, Any]:
    data = asdict(job)
    data["raw_bytes"] = base64.b64encode(job.raw_bytes).decode("ascii")
    return data


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported ingestion state value: {type(value).__name__}")


def _job_from_json(data: dict[str, Any]) -> IngestionJob:
    payload = dict(data)
    payload["raw_bytes"] = base64.b64decode(payload["raw_bytes"])
    payload.setdefault("domain_id", payload.get("domain", "general"))
    payload.setdefault("system_id", payload.get("system", "unknown"))
    payload.setdefault("component_id", payload.get("component_id") or payload.get("system_id") or "unknown-component")
    payload.setdefault("evidence_type", payload.get("evidence_type") or payload.get("source_type") or "document")
    return IngestionJob(**payload)


def _outcome_to_json(outcome: IngestionOutcome) -> dict[str, Any]:
    return {
        "job_id": outcome.job_id,
        "created": outcome.created,
        "bundles": [_bundle_to_json(bundle) for bundle in outcome.bundles],
        "content_hash": outcome.content_hash,
        "metadata": outcome.metadata,
    }


def _outcome_from_json(data: dict[str, Any]) -> IngestionOutcome:
    return IngestionOutcome(
        job_id=data["job_id"],
        created=bool(data["created"]),
        bundles=tuple(_bundle_from_json(bundle) for bundle in data.get("bundles", [])),
        content_hash=data["content_hash"],
        metadata=dict(data.get("metadata", {})),
    )


def _dead_letter_to_json(entry: DeadLetterEntry) -> dict[str, Any]:
    return {"job": _job_to_json(entry.job), "reason": entry.reason, "failed_at": entry.failed_at, "retryable": entry.retryable}


def _dead_letter_from_json(data: dict[str, Any]) -> DeadLetterEntry:
    return DeadLetterEntry(
        job=_job_from_json(data["job"]),
        reason=data["reason"],
        failed_at=data["failed_at"],
        retryable=bool(data.get("retryable", True)),
    )


def _bundle_to_json(bundle: IngestBundle) -> dict[str, Any]:
    return {
        "source": asdict(bundle.source),
        "version": asdict(bundle.version),
        "artifact": asdict(bundle.artifact),
        "namespaces": [asdict(namespace) for namespace in bundle.namespaces],
        "integration_edges": [asdict(edge) for edge in bundle.integration_edges],
        "chunks": [asdict(chunk) for chunk in bundle.chunks],
        "embeddings": [asdict(embedding) for embedding in bundle.embeddings],
        "object_reference": asdict(bundle.object_reference),
        "created": bundle.created,
    }


def _bundle_from_json(data: dict[str, Any]) -> IngestBundle:
    from .knowledge import (
        ArtifactRecord,
        ChunkRecord,
        EmbeddingRecord,
        KnowledgeSource,
        KnowledgeSourceVersion,
    )
    from .knowledge.namespaces import IntegrationEdgeRecord, KnowledgeNamespaceRecord
    from .knowledge.object_store import ObjectReference

    source = dict(data["source"])
    for key in ("created_at", "updated_at"):
        if isinstance(source.get(key), str):
            source[key] = datetime.fromisoformat(source[key])
    version = dict(data["version"])
    if isinstance(version.get("ingest_timestamp"), str):
        version["ingest_timestamp"] = datetime.fromisoformat(version["ingest_timestamp"])
    artifact = dict(data["artifact"])
    if isinstance(artifact.get("created_at"), str):
        artifact["created_at"] = datetime.fromisoformat(artifact["created_at"])
    embeddings = []
    for embedding in data.get("embeddings", []):
        item = dict(embedding)
        if isinstance(item.get("created_at"), str):
            item["created_at"] = datetime.fromisoformat(item["created_at"])
        item["vector"] = tuple(item.get("vector", ()))
        embeddings.append(EmbeddingRecord(**item))
    return IngestBundle(
        source=KnowledgeSource(**source),
        version=KnowledgeSourceVersion(**version),
        artifact=ArtifactRecord(**artifact),
        chunks=tuple(ChunkRecord(**chunk) for chunk in data.get("chunks", [])),
        embeddings=tuple(embeddings),
        object_reference=ObjectReference(**data["object_reference"]),
        created=bool(data["created"]),
        namespaces=tuple(KnowledgeNamespaceRecord(**namespace) for namespace in data.get("namespaces", [])),
        integration_edges=tuple(IntegrationEdgeRecord(**edge) for edge in data.get("integration_edges", [])),
    )


def _stable_hash(*parts: str) -> str:
    digest = sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_source_id(channel: str, source_uri: str, *, suffix: str = "") -> str:
    return f"src_{_stable_hash(channel, source_uri, suffix)[:32]}"


def _canonical_mime(path: Path | None, mime_type: str | None) -> str:
    if mime_type:
        return mime_type
    if path is not None:
        lower_name = path.name.lower()
        if lower_name.endswith(".docx"):
            return DOCX_MIME_TYPE
        if lower_name.endswith(".xlsx"):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if path is not None:
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed:
            return guessed
    return "application/octet-stream"


def _is_archive_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _archive_suffix(path: Path) -> str:
    lower_name = path.name.lower()
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lower_name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def _validate_relative_path(name: str) -> PurePosixPath:
    candidate = PurePosixPath(name.replace("\\", "/"))
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe archive member path: {name}")
    return candidate


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:4096]
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="surrogateescape")


def _line_blocks(text: str, *, prefix: str | None = None) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    lines = text.splitlines()
    start = 0
    while start < len(lines):
        end = min(len(lines), start + 1)
        block_lines = lines[start:end]
        while end < len(lines) and sum(len(line) for line in block_lines) < 1200:
            if not lines[end].strip():
                break
            block_lines.append(lines[end])
            end += 1
        block_text = "\n".join(block_lines).strip()
        if block_text:
            locator = f"{prefix}:L{start + 1}-L{start + len(block_lines)}" if prefix else None
            blocks.append(ParsedBlock(text=block_text, locator=locator, line_start=start + 1, line_end=start + len(block_lines)))
        start = end + 1 if end < len(lines) and not lines[end].strip() else end
    return blocks


def _csv_blocks(data: bytes, *, delimiter: str, prefix: str | None = None) -> list[ParsedBlock]:
    text = _decode_text(data)
    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    blocks: list[ParsedBlock] = []
    for index, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        pairs = [f"{name}: {value}" for name, value in zip(header, row, strict=False)]
        locator = f"{prefix or 'csv'}#row-{index}"
        timestamp_fields = [
            name
            for name in header
            if any(marker in name.lower() for marker in ("time", "timestamp", "_time", "date"))
        ]
        metadata = {"row_index": index, "headers": header, "row": row, "timestamp_fields": timestamp_fields}
        blocks.append(ParsedBlock(text="\n".join(pairs), locator=locator, metadata=metadata))
    return blocks


def _json_blocks(data: bytes, *, prefix: str | None = None) -> list[ParsedBlock]:
    parsed = json.loads(_decode_text(data))
    text = json.dumps(parsed, indent=2, sort_keys=True)
    return [ParsedBlock(text=text, locator=prefix, metadata={"json_type": type(parsed).__name__})]


def _yaml_blocks(data: bytes, *, prefix: str | None = None) -> list[ParsedBlock]:
    parsed = yaml.safe_load(_decode_text(data))
    text = json.dumps(parsed, indent=2, sort_keys=True)
    return [ParsedBlock(text=text, locator=prefix, metadata={"yaml_type": type(parsed).__name__})]


def _xml_blocks(data: bytes, *, prefix: str | None = None) -> list[ParsedBlock]:
    root = ET.fromstring(_decode_text(data))
    blocks: list[ParsedBlock] = []

    def walk(node: ET.Element, path: str, index: int) -> None:
        locator = f"{prefix or 'xml'}{path}"
        text = " ".join(part.strip() for part in node.itertext()).strip()
        if text:
            blocks.append(ParsedBlock(text=text, locator=locator, metadata={"tag": node.tag, "path": path}))
        counts: dict[str, int] = {}
        for child in list(node):
            counts[child.tag] = counts.get(child.tag, 0) + 1
            walk(child, f"{path}/{child.tag}[{counts[child.tag]}]", counts[child.tag])

    walk(root, f"/{root.tag}[1]", 1)
    return blocks


def _docx_blocks(data: bytes, *, prefix: str | None = None) -> list[ParsedBlock]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("invalid docx archive")
        root = ET.fromstring(zf.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    blocks: list[ParsedBlock] = []
    for index, paragraph in enumerate(root.findall(".//w:body/w:p", ns), start=1):
        text = "".join(node.text or "" for node in paragraph.iterfind(".//w:t", ns)).strip()
        if text:
            blocks.append(
                ParsedBlock(
                    text=text,
                    locator=f"{prefix or 'docx'}#paragraph-{index}",
                    metadata={"paragraph": index},
                )
            )
    if blocks:
        return blocks
    if _looks_like_text(data):
        return [ParsedBlock(text=_decode_text(data), locator=prefix)]
    return []


def _pdf_blocks(data: bytes, *, prefix: str | None = None) -> list[ParsedBlock]:
    text = data.decode("latin-1", errors="ignore")
    pages = re.split(r"(?i)\bendstream\b", text)
    blocks: list[ParsedBlock] = []
    for index, page_blob in enumerate(pages, start=1):
        page_texts: list[str] = []
        for match in re.finditer(r"\((.*?)\)\s*(?:Tj|TJ)", page_blob, re.DOTALL):
            raw = match.group(1)
            raw = raw.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
            if raw.strip():
                page_texts.append(raw.strip())
        if page_texts:
            blocks.append(
                ParsedBlock(
                    text="\n".join(page_texts),
                    locator=f"{prefix or 'pdf'}#page-{index}",
                    page=index,
                    metadata={"page": index},
                )
            )
    if not blocks and _looks_like_text(data):
        return [ParsedBlock(text=_decode_text(data), locator=prefix)]
    return blocks


def _xlsx_blocks(data: bytes, *, prefix: str | None = None) -> list[ParsedBlock]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall(".//{*}si"):
                shared_strings.append("".join(part.text or "" for part in item.iter()))
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
        sheets = workbook.findall(".//a:sheets/a:sheet", ns)
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        blocks: list[ParsedBlock] = []
        for sheet_index, sheet in enumerate(sheets, start=1):
            target = rel_map.get(sheet.attrib.get(f"{{{ns['r']}}}id"), f"worksheets/sheet{sheet_index}.xml")
            xml = ET.fromstring(zf.read(f"xl/{target}"))
            rows = []
            for row in xml.findall(".//a:sheetData/a:row", ns):
                values: list[str] = []
                for cell in row.findall("a:c", ns):
                    cell_type = cell.attrib.get("t")
                    value = cell.findtext("a:v", default="", namespaces=ns)
                    if cell_type == "s":
                        try:
                            value = shared_strings[int(value)]
                        except Exception:  # noqa: BLE001, S110 - malformed spreadsheet cells remain literal
                            pass
                    values.append(value)
                if values:
                    rows.append(values)
            if rows:
                text = "\n".join(" | ".join(row) for row in rows)
                blocks.append(ParsedBlock(text=text, locator=f"{prefix or 'xlsx'}#sheet-{sheet_index}", metadata={"sheet": sheet.attrib.get("name", f"Sheet{sheet_index}")}))
        return blocks


def _archive_members_from_zip(data: bytes) -> list[tuple[PurePosixPath, bytes]]:
    members: list[tuple[PurePosixPath, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        total = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_path = _validate_relative_path(info.filename)
            total += info.file_size
            if total > 200_000_000:
                raise ValueError("archive expansion limit exceeded")
            members.append((member_path, zf.read(info)))
    return members


def _archive_members_from_tar(data: bytes) -> list[tuple[PurePosixPath, bytes]]:
    members: list[tuple[PurePosixPath, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        total = 0
        for member in tf.getmembers():
            if not member.isfile():
                continue
            member_path = _validate_relative_path(member.name)
            total += member.size
            if total > 200_000_000:
                raise ValueError("archive expansion limit exceeded")
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            members.append((member_path, extracted.read()))
    return members


def _archive_members_from_7z(data: bytes) -> list[tuple[PurePosixPath, bytes]]:
    try:
        import py7zr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError("7z archives require the optional py7zr dependency") from exc
    members: list[tuple[PurePosixPath, bytes]] = []
    total = 0
    with py7zr.SevenZipFile(io.BytesIO(data), mode="r") as archive:
        for name, extracted in archive.readall().items():
            member_path = _validate_relative_path(name)
            content = extracted.read()
            total += len(content)
            if total > 200_000_000:
                raise ValueError("archive expansion limit exceeded")
            members.append((member_path, content))
    return members


def _parse_file_bytes(path: Path, data: bytes, *, prefix: str | None = None) -> tuple[str, str, list[ParsedBlock], dict[str, Any]]:
    if _is_archive_path(path):
        raise ValueError("archives must be expanded before parsing as files")
    suffix = _archive_suffix(path)
    mime_type = _canonical_mime(path, None)
    metadata: dict[str, Any] = {"path": str(path), "suffix": suffix or None}
    if suffix == ".pdf":
        return mime_type, "pdf", _pdf_blocks(data, prefix=prefix or path.name), metadata
    if suffix == ".docx":
        return mime_type, "document", _docx_blocks(data, prefix=prefix or path.name), metadata
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        metadata.update({"delimiter": delimiter, "language": "csv"})
        return mime_type, "csv", _csv_blocks(data, delimiter=delimiter, prefix=prefix or path.name), metadata
    if suffix in {".json"}:
        metadata.update({"language": "json"})
        return mime_type, "json", _json_blocks(data, prefix=prefix or path.name), metadata
    if suffix in {".yaml", ".yml"}:
        metadata.update({"language": "yaml"})
        return mime_type, "yaml", _yaml_blocks(data, prefix=prefix or path.name), metadata
    if suffix == ".xml":
        metadata.update({"language": "xml"})
        return mime_type, "xml", _xml_blocks(data, prefix=prefix or path.name), metadata
    if suffix == ".xlsx":
        metadata.update({"language": "spreadsheet"})
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "schema", _xlsx_blocks(data, prefix=prefix or path.name), metadata
    if suffix in {".zip", ".tar", ".tgz", ".tar.gz", ".7z"}:
        raise ValueError("archives must be expanded before parsing as files")
    if _looks_like_text(data):
        language = _language_from_suffix(suffix)
        metadata.update({"language": language})
        return mime_type, language, _line_blocks(_decode_text(data), prefix=prefix or path.name), metadata
    raise ValueError(f"unsupported evidence file type: {path.suffix or '<none>'}")


def _language_from_suffix(suffix: str) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".sh": "shell",
        ".ps1": "powershell",
        ".sql": "sql",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".md": "markdown",
        ".markdown": "markdown",
        ".log": "log",
        ".txt": "text",
        ".csv": "csv",
        ".tsv": "tsv",
        ".ddl": "sql",
    }
    return mapping.get(suffix, "text")


def _classify_suffix(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if path.name.lower().endswith(".docx"):
        return "document", "document"
    if suffix == ".pdf":
        return "pdf", "document"
    if suffix in {".csv", ".tsv"}:
        return "csv", "table"
    if suffix in {".json", ".yaml", ".yml", ".xml"}:
        return suffix.lstrip("."), "document"
    if suffix == ".xlsx":
        return "schema", "schema"
    if suffix in TEXT_SUFFIXES:
        if suffix in {".py", ".js", ".ts", ".go", ".java", ".rb", ".sh", ".ps1"}:
            return _language_from_suffix(suffix), "code"
        if suffix in {".sql", ".ddl"}:
            return "schema", "schema"
        if suffix in {".log"}:
            return "log", "log"
        return "document", "document"
    return "document", "document"


def _validate_mime_suffix(mime_type: str, suffix: str) -> None:
    if not mime_type:
        return
    if suffix in {".csv", ".tsv"} and "csv" not in mime_type and "tab-separated-values" not in mime_type:
        raise ValueError(f"mime type mismatch for {suffix}: {mime_type}")
    if suffix in {".json"} and "json" not in mime_type:
        raise ValueError(f"mime type mismatch for {suffix}: {mime_type}")
    if suffix in {".yaml", ".yml"} and "yaml" not in mime_type and "text" not in mime_type:
        raise ValueError(f"mime type mismatch for {suffix}: {mime_type}")
    if suffix in {".xml"} and "xml" not in mime_type and "text" not in mime_type:
        raise ValueError(f"mime type mismatch for {suffix}: {mime_type}")


def _infer_blocks_from_bytes(path: Path, data: bytes) -> tuple[str, str, list[ParsedBlock], dict[str, Any]]:
    if _is_archive_path(path):
        raise ValueError("archives must be handled via the archive ingestion path")
    mime_type, source_type, blocks, metadata = _parse_file_bytes(path, data, prefix=path.name)
    _validate_mime_suffix(mime_type, _archive_suffix(path))
    return mime_type, source_type, blocks, metadata


def _iter_directory_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


class IngestionService:
    """Queue-backed ingestion service that routes evidence into the knowledge layer."""

    def __init__(
        self,
        platform: KnowledgePlatform,
        *,
        queue: IngestionQueue | None = None,
        limits: IngestionLimits | None = None,
        malware_scanner: Callable[[bytes, dict[str, Any]], None] | None = None,
        landing_root: str | Path | None = None,
    ):
        self.platform = platform
        self.queue = queue or IngestionQueue()
        self.limits = limits or IngestionLimits()
        self.malware_scanner = malware_scanner or (lambda _content, _metadata: None)
        inferred_landing = self.queue.state_path.parent / "landing" if self.queue.state_path else None
        self.landing = FilesystemLandingStore(landing_root or inferred_landing) if (landing_root or inferred_landing) else None

    def status(self, *, limit: int = 100) -> dict[str, Any]:
        return self.queue.status(limit=limit)

    def retry_dead_letters(self, *, limit: int = 10) -> int:
        return self.queue.retry_dead_letters(limit=limit)

    def reprocess(self, source_id: str, *, parser_version: str, limit: int = 1) -> list[IngestionJob]:
        """Queue bounded parser-version work from retained immutable job bytes."""
        candidates = [job for job in self.queue.jobs.values() if job.source_id == source_id]
        candidates.sort(key=lambda job: (job.content_hash, job.job_id))
        jobs: list[IngestionJob] = []
        for original in candidates[:max(0, int(limit))]:
            job = replace(original, parser_version=parser_version, state="queued", attempts=0)
            job = replace(job, job_id=f"job_{_stable_hash(job.channel, job.source_id, job.content_hash, job.parser_hint or job.source_type, parser_version)}")
            if self.queue.enqueue(job):
                jobs.append(job)
        return jobs

    def _build_job(
        self,
        *,
        source_id: str,
        source_uri: str,
        channel: str,
        mime_type: str,
        raw_bytes: bytes,
        source_kind: str,
        source_type: str,
        domain: str,
        system: str,
        domain_id: str | None,
        system_id: str | None,
        component_id: str | None,
        evidence_type: str | None,
        environment: str,
        acl_scope: str,
        parser_hint: str | None,
        parser_version: str | None = None,
        local_only: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionJob:
        content_hash = sha256(raw_bytes).hexdigest()
        job_id = _stable_hash(channel, source_id, content_hash, parser_hint or source_type, parser_version or "detected")
        sanitized_metadata, _ = sanitize_metadata(metadata)
        local_only = local_only or bool(sanitized_metadata.get("local_only", False))
        effective_domain_id = domain_id or domain
        effective_system_id = system_id or system
        effective_component_id = component_id or sanitized_metadata.get("component_id") or f"{effective_system_id}_component"
        effective_evidence_type = evidence_type or source_type or source_kind
        return IngestionJob(
            job_id=f"job_{job_id}",
            source_id=source_id,
            source_uri=source_uri,
            channel=channel,
            mime_type=mime_type,
            content_hash=content_hash,
            raw_bytes=raw_bytes,
            metadata=sanitized_metadata,
            source_kind=source_kind,
            source_type=source_type,
            domain=domain,
            system=system,
            domain_id=effective_domain_id,
            system_id=effective_system_id,
            component_id=effective_component_id,
            evidence_type=effective_evidence_type,
            environment=environment,
            acl_scope=acl_scope,
            parser_hint=parser_hint,
            parser_version=parser_version,
            local_only=local_only,
        )

    def _enqueue(self, job: IngestionJob) -> IngestionJob:
        """Land bytes before making the job visible to the queue."""
        if self.landing is not None:
            reference = self.landing.put(
                job.raw_bytes,
                mime_type=job.mime_type,
                metadata={"job_id": job.job_id, "source_id": job.source_id, "local_only": job.local_only},
            )
            job = replace(job, landing_object_key=reference.object_key)
        self.queue.enqueue(job)
        return job

    def _validate_payload_size(self, raw_bytes: bytes, *, label: str) -> None:
        if len(raw_bytes) > self.limits.max_bytes:
            raise ValueError(f"{label} exceeds size limit")

    def submit_manual(
        self,
        *,
        source_uri: str,
        content: bytes | str,
        mime_type: str | None = None,
        source_id: str | None = None,
        channel: str = "manual",
        source_kind: str = "document",
        source_type: str = "document",
        domain: str = "general",
        system: str = "unknown",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "unknown",
        acl_scope: str = "internal",
        parser_hint: str | None = None,
        parser_version: str | None = None,
        local_only: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionJob:
        raw_bytes = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        self._validate_payload_size(raw_bytes, label="evidence payload")
        source_id = source_id or _stable_source_id(channel, source_uri)
        job = self._build_job(
            source_id=source_id,
            source_uri=source_uri,
            channel=channel,
            mime_type=mime_type or _canonical_mime(Path(source_uri), None),
            raw_bytes=raw_bytes,
            source_kind=source_kind,
            source_type=source_type,
            domain=domain,
            system=system,
            domain_id=domain_id,
            system_id=system_id,
            component_id=component_id,
            evidence_type=evidence_type,
            environment=environment,
            acl_scope=acl_scope,
            parser_hint=parser_hint,
            parser_version=parser_version,
            metadata=metadata,
            local_only=local_only,
        )
        job = self._enqueue(job)
        return job

    def submit_file(
        self,
        path: str | Path,
        *,
        channel: str,
        source_id: str | None = None,
        source_uri: str | None = None,
        domain: str = "general",
        system: str = "unknown",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "unknown",
        acl_scope: str = "internal",
        metadata: dict[str, Any] | None = None,
        parser_version: str | None = None,
        local_only: bool = False,
    ) -> list[IngestionJob]:
        file_path = Path(path)
        if file_path.is_dir():
            jobs: list[IngestionJob] = []
            for child in _iter_directory_files(file_path):
                jobs.extend(
                    self.submit_file(
                        child,
                        channel=channel,
                        source_id=None,
                        source_uri=posixpath.join(source_uri or str(file_path), child.relative_to(file_path).as_posix()),
                        domain=domain,
                        system=system,
                        domain_id=domain_id,
                        system_id=system_id,
                        component_id=component_id,
                        evidence_type=evidence_type,
                        environment=environment,
                        acl_scope=acl_scope,
                        metadata={"parent_directory": str(file_path), **dict(metadata or {})},
                        local_only=local_only,
                    )
                )
            return jobs
        raw_bytes = file_path.read_bytes()
        self._validate_payload_size(raw_bytes, label="evidence file")
        if len(raw_bytes) > self.limits.max_single_file_bytes:
            raise ValueError("evidence file exceeds size limit")
        source_uri = source_uri or str(file_path)
        source_id = source_id or _stable_source_id(channel, source_uri)
        mime_type = _canonical_mime(file_path, None)
        file_path.suffix.lower()
        if _is_archive_path(file_path):
            member_jobs: list[IngestionJob] = []
            for member_path, member_bytes in self._expand_archive(file_path, raw_bytes):
                self._validate_payload_size(member_bytes, label="archive member")
                member_uri = posixpath.join(source_uri, member_path.as_posix())
                member_source_id = _stable_source_id(channel, member_uri)
                member_path_obj = Path(member_path.name)
                member_mime, member_source_type, _, member_metadata = _infer_blocks_from_bytes(member_path_obj, member_bytes)
                _, member_source_kind = _classify_suffix(member_path_obj)
                job = self._build_job(
                    source_id=member_source_id,
                    source_uri=member_uri,
                    channel=channel,
                    mime_type=member_mime,
                    raw_bytes=member_bytes,
                    source_kind=member_source_kind,
                    source_type=member_source_type,
                    domain=domain,
                    system=system,
                    domain_id=domain_id,
                    system_id=system_id,
                    component_id=component_id,
                    evidence_type=evidence_type,
                    environment=environment,
                    acl_scope=acl_scope,
                    parser_hint=member_source_type,
                    parser_version=parser_version,
                    local_only=local_only,
                        metadata={"archive_path": str(file_path), "member_path": member_path.as_posix(), **member_metadata, **dict(metadata or {})},
                )
                job = self._enqueue(job)
                member_jobs.append(job)
            return member_jobs
        source_type, source_kind = _classify_suffix(file_path)
        mime_type, inferred_source_type, _, parsed_metadata = _infer_blocks_from_bytes(file_path, raw_bytes)
        job = self._build_job(
            source_id=source_id,
            source_uri=source_uri,
            channel=channel,
            mime_type=mime_type,
            raw_bytes=raw_bytes,
            source_kind=source_kind,
            source_type=source_type if source_type else inferred_source_type,
            domain=domain,
            system=system,
            domain_id=domain_id,
            system_id=system_id,
            component_id=component_id,
            evidence_type=evidence_type,
            environment=environment,
            acl_scope=acl_scope,
            parser_hint=source_type,
            parser_version=parser_version,
            local_only=local_only,
            metadata={**parsed_metadata, **dict(metadata or {})},
        )
        job = self._enqueue(job)
        return [job]

    def _submit_conversation_message(
        self,
        *,
        channel: str,
        conversation_kind: str,
        conversation_id: str,
        message_id: str,
        message_text: str,
        attachments: Sequence[tuple[str, bytes]] = (),
        thread_id: str | None = None,
        source_uri: str | None = None,
        domain: str = "communications",
        system: str = "unknown",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "prod",
        acl_scope: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> list[IngestionJob]:
        base_uri = source_uri or f"{channel}://{conversation_kind}s/{conversation_id}/messages/{message_id}"
        conversation_key = f"{conversation_kind}_id"
        jobs = [
            self.submit_manual(
                source_uri=f"{base_uri}#message",
                source_id=_stable_source_id(channel, f"{conversation_id}:{message_id}:message"),
                content=message_text,
                mime_type="text/plain",
                channel=channel,
                source_kind="document",
                source_type="document",
                domain=domain,
                system=system,
                domain_id=domain_id,
                system_id=system_id,
                component_id=component_id,
                evidence_type=evidence_type,
                environment=environment,
                acl_scope=acl_scope,
                parser_hint=f"{channel}-message",
                metadata={conversation_key: conversation_id, "message_id": message_id, "thread_id": thread_id, **dict(metadata or {})},
            )
        ]
        for attachment_name, attachment_bytes in attachments:
            attachment_uri = f"{base_uri}/attachments/{attachment_name}"
            attachment_path = Path(attachment_name)
            mime_type = _canonical_mime(attachment_path, None)
            jobs.append(
                self.submit_manual(
                    source_uri=attachment_uri,
                    source_id=_stable_source_id(channel, f"{conversation_id}:{message_id}:{attachment_name}"),
                    content=attachment_bytes,
                    mime_type=mime_type,
                    channel=channel,
                    source_kind="document",
                    source_type=_classify_suffix(attachment_path)[0],
                    domain=domain,
                    system=system,
                    domain_id=domain_id,
                    system_id=system_id,
                    component_id=component_id,
                    evidence_type=evidence_type,
                    environment=environment,
                    acl_scope=acl_scope,
                    parser_hint=f"{channel}-attachment",
                    metadata={
                        conversation_key: conversation_id,
                        "message_id": message_id,
                        "attachment_name": attachment_name,
                        "thread_id": thread_id,
                        **dict(metadata or {}),
                    },
                )
            )
        return jobs

    def submit_webex_message(
        self,
        *,
        room_id: str,
        message_id: str,
        message_text: str,
        attachments: Sequence[tuple[str, bytes]] = (),
        thread_id: str | None = None,
        source_uri: str | None = None,
        domain: str = "communications",
        system: str = "webex",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "prod",
        acl_scope: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> list[IngestionJob]:
        return self._submit_conversation_message(
            channel="webex",
            conversation_kind="room",
            conversation_id=room_id,
            message_id=message_id,
            message_text=message_text,
            attachments=attachments,
            thread_id=thread_id,
            source_uri=source_uri,
            domain=domain,
            system=system,
            domain_id=domain_id,
            system_id=system_id,
            component_id=component_id,
            evidence_type=evidence_type,
            environment=environment,
            acl_scope=acl_scope,
            metadata=metadata,
        )

    def submit_telegram_message(
        self,
        *,
        chat_id: str,
        message_id: str,
        message_text: str,
        attachments: Sequence[tuple[str, bytes]] = (),
        thread_id: str | None = None,
        source_uri: str | None = None,
        domain: str = "communications",
        system: str = "telegram",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "prod",
        acl_scope: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> list[IngestionJob]:
        return self._submit_conversation_message(
            channel="telegram",
            conversation_kind="chat",
            conversation_id=chat_id,
            message_id=message_id,
            message_text=message_text,
            attachments=attachments,
            thread_id=thread_id,
            source_uri=source_uri,
            domain=domain,
            system=system,
            domain_id=domain_id,
            system_id=system_id,
            component_id=component_id,
            evidence_type=evidence_type,
            environment=environment,
            acl_scope=acl_scope,
            metadata=metadata,
        )

    def submit_mail_drop(
        self,
        path: str | Path,
        *,
        channel: str = "mail-drop",
        domain: str = "communications",
        system: str = "mail",
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str = "prod",
        acl_scope: str = "internal",
        metadata: dict[str, Any] | None = None,
    ) -> list[IngestionJob]:
        path = Path(path)
        if path.is_dir():
            jobs: list[IngestionJob] = []
            for child in _iter_directory_files(path):
                jobs.extend(
                    self.submit_mail_drop(
                        child,
                        channel=channel,
                        domain=domain,
                        system=system,
                        domain_id=domain_id,
                        system_id=system_id,
                        component_id=component_id,
                        evidence_type=evidence_type,
                        environment=environment,
                        acl_scope=acl_scope,
                        metadata=metadata,
                    )
                )
            return jobs
        if path.suffix.lower() == ".eml":
            raw_bytes = path.read_bytes()
            self._validate_payload_size(raw_bytes, label="mail-drop message")
            message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
            jobs = self._jobs_from_email(
                message,
                source_uri=str(path),
                channel=channel,
                domain=domain,
                system=system,
                domain_id=domain_id,
                system_id=system_id,
                component_id=component_id,
                evidence_type=evidence_type,
                environment=environment,
                acl_scope=acl_scope,
                metadata=metadata,
            )
            for job in jobs:
                self.queue.enqueue(job)
            return jobs
        return self.submit_file(
            path,
            channel=channel,
            domain=domain,
            system=system,
            domain_id=domain_id,
            system_id=system_id,
            component_id=component_id,
            evidence_type=evidence_type,
            environment=environment,
            acl_scope=acl_scope,
            metadata={"mail_drop": True, **dict(metadata or {})},
        )

    def _jobs_from_email(
        self,
        message: Message,
        *,
        source_uri: str,
        channel: str,
        domain: str,
        system: str,
        domain_id: str | None = None,
        system_id: str | None = None,
        component_id: str | None = None,
        evidence_type: str | None = None,
        environment: str,
        acl_scope: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[IngestionJob]:
        jobs: list[IngestionJob] = []
        body_parts: list[str] = []
        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    continue
                disposition = part.get_content_disposition()
                payload = part.get_payload(decode=True) or b""
                if disposition == "attachment":
                    filename = part.get_filename() or "attachment.bin"
                    jobs.append(
                        self.submit_manual(
                            source_uri=f"{source_uri}/attachments/{filename}",
                            source_id=_stable_source_id("mail-drop", f"{source_uri}:{filename}"),
                            content=payload,
                            mime_type=part.get_content_type(),
                            channel=channel,
                            source_kind="document",
                            source_type=_classify_suffix(Path(filename))[0],
                            domain=domain,
                            system=system,
                            domain_id=domain_id,
                            system_id=system_id,
                            component_id=component_id,
                            evidence_type=evidence_type,
                            environment=environment,
                            acl_scope=acl_scope,
                            parser_hint="email-attachment",
                            metadata={"filename": filename, "message_id": message.get("Message-ID"), "mail_subject": message.get("Subject"), **dict(metadata or {})},
                        )
                    )
                else:
                    if part.get_content_type().startswith("text/"):
                        body_parts.append(_decode_text(payload))
        else:
            payload = message.get_payload(decode=True) or b""
            body_parts.append(_decode_text(payload))
        if body_parts:
            jobs.insert(
                0,
                self.submit_manual(
                    source_uri=f"{source_uri}#body",
                    source_id=_stable_source_id("mail-drop", f"{source_uri}:body"),
                    content="\n\n".join(body_parts),
                    mime_type="text/plain",
                    channel=channel,
                    source_kind="document",
                    source_type="document",
                    domain=domain,
                    system=system,
                    domain_id=domain_id,
                    system_id=system_id,
                    component_id=component_id,
                    evidence_type=evidence_type,
                    environment=environment,
                    acl_scope=acl_scope,
                    parser_hint="email-body",
                    metadata={"message_id": message.get("Message-ID"), "mail_subject": message.get("Subject"), **dict(metadata or {})},
                ),
            )
        return jobs

    def _expand_archive(self, path: Path, raw_bytes: bytes) -> list[tuple[PurePosixPath, bytes]]:
        suffix = _archive_suffix(path)
        if suffix == ".zip":
            members = _archive_members_from_zip(raw_bytes)
        elif suffix in {".tar", ".tgz", ".tar.gz"}:
            members = _archive_members_from_tar(raw_bytes)
        elif suffix == ".7z":
            members = _archive_members_from_7z(raw_bytes)
        else:
            raise ValueError(f"unsupported archive type: {path.name}")
        if len(members) > self.limits.max_archive_entries:
            raise ValueError("archive entry limit exceeded")
        total = sum(len(content) for _, content in members)
        if total > self.limits.max_archive_expansion_bytes:
            raise ValueError("archive expansion limit exceeded")
        return members

    def process_next(self) -> IngestionOutcome | None:
        job = self.queue.claim()
        if job is None:
            return None
        try:
            raw_bytes = self.landing.get(job.landing_object_key) if self.landing and job.landing_object_key else job.raw_bytes
            if sha256(raw_bytes).hexdigest() != job.content_hash:
                raise ValueError("landed content hash mismatch")
            self.malware_scanner(raw_bytes, job.metadata)
            job = self.queue.transition(job.job_id, "validated", details={"attempt": job.attempts})
            job = self.queue.transition(job.job_id, "stored", details={"landing_object_key": job.landing_object_key})
            job = replace(job, raw_bytes=raw_bytes)
            bundles = self._process_job(job)
            self.queue.transition(job.job_id, "parsed")
            self.queue.transition(job.job_id, "chunked", details={"chunks": sum(len(bundle.chunks) for bundle in bundles)})
            self.queue.transition(job.job_id, "embedded", details={"embeddings": sum(len(bundle.embeddings) for bundle in bundles)})
            self.queue.transition(job.job_id, "indexed")
            outcome = IngestionOutcome(job_id=job.job_id, created=any(bundle.created for bundle in bundles), bundles=tuple(bundles), content_hash=job.content_hash, metadata=dict(job.metadata))
            self.queue.ack(job, outcome)
            return outcome
        except Exception as exc:  # noqa: BLE001 - ingestion boundary records failure in the queue
            self.queue.fail(job, str(exc), retryable=True)
            return None

    def run(self, *, max_jobs: int | None = None) -> list[IngestionOutcome]:
        outcomes: list[IngestionOutcome] = []
        processed = 0
        while self.queue.pending and (max_jobs is None or processed < max_jobs):
            outcome = self.process_next()
            processed += 1
            if outcome is not None:
                outcomes.append(outcome)
        return outcomes

    def _process_job(self, job: IngestionJob) -> list[IngestBundle]:
        path = Path(job.source_uri)
        blocks, parser_name, detected_parser_version, source_type = self._parse_job(job, path)
        parser_version = job.parser_version or detected_parser_version
        bundle = self.platform.ingest(
            source_id=job.source_id,
            source_uri=job.source_uri,
            content=job.raw_bytes,
            blocks=blocks,
            mime_type=job.mime_type,
            source_kind=job.source_kind,
            channel=job.channel,
            domain=job.domain,
            system=job.system,
            domain_id=job.domain_id,
            system_id=job.system_id,
            component_id=job.component_id,
            evidence_type=job.evidence_type,
            environment=job.environment,
            acl_scope=job.acl_scope,
            source_type=source_type,
            parser_name=parser_name,
            parser_version=parser_version,
            metadata={**job.metadata, "content_hash": job.content_hash, "channel": job.channel, "local_only": job.local_only},
        )
        return [bundle]

    def _parse_job(self, job: IngestionJob, path: Path) -> tuple[list[ParsedBlock], str, str, str]:
        suffix = _archive_suffix(path)
        if job.parser_hint in {"webex-message", "telegram-message"}:
            return _line_blocks(_decode_text(job.raw_bytes), prefix=path.name), f"{job.parser_hint}-parser", "1", "document"
        if job.parser_hint in {"webex-attachment", "telegram-attachment"}:
            if suffix == ".pdf":
                return _pdf_blocks(job.raw_bytes, prefix=path.name), "pdf-parser", "1", "document"
            if suffix == ".docx":
                return _docx_blocks(job.raw_bytes, prefix=path.name), "docx-parser", "1", "document"
            if suffix in {".csv", ".tsv"}:
                return _csv_blocks(job.raw_bytes, delimiter="\t" if suffix == ".tsv" else ",", prefix=path.name), "csv-parser", "1", "table"
            if suffix in {".json"}:
                return _json_blocks(job.raw_bytes, prefix=path.name), "json-parser", "1", "document"
            if suffix in {".yaml", ".yml"}:
                return _yaml_blocks(job.raw_bytes, prefix=path.name), "yaml-parser", "1", "document"
            if suffix in {".xml"}:
                return _xml_blocks(job.raw_bytes, prefix=path.name), "xml-parser", "1", "document"
            return _parse_binary_or_text(path, job.raw_bytes)
        if job.parser_hint == "email-body":
            return _line_blocks(_decode_text(job.raw_bytes), prefix=path.name), "email-body-parser", "1", "document"
        if job.parser_hint == "email-attachment":
            return _parse_binary_or_text(path, job.raw_bytes)
        if suffix in {".zip", ".tar", ".tgz", ".tar.gz", ".7z"}:
            raise ValueError("archives are expanded before processing")
        return _parse_binary_or_text(path, job.raw_bytes)


def _parse_binary_or_text(path: Path, data: bytes) -> tuple[list[ParsedBlock], str, str, str]:
    suffix = _archive_suffix(path)
    if suffix == ".docx":
        return _docx_blocks(data, prefix=path.name), "docx-parser", "1", "document"
    _mime_type, source_type, blocks, _ = _infer_blocks_from_bytes(path, data)
    parser_name = f"{source_type}-parser"
    parser_version = "1"
    if not blocks and _looks_like_text(data):
        blocks = _line_blocks(_decode_text(data), prefix=path.name)
    return blocks, parser_name, parser_version, source_type


class DirectoryWatcher:
    """Poll a directory; unchanged files are submitted at most once per scan."""

    def __init__(self, service: IngestionService, root: str | Path, *, channel: str = "directory", **submit_options: Any):
        self.service, self.root, self.channel = service, Path(root), channel
        self.submit_options = submit_options
        self._observed: dict[str, tuple[int, int]] = {}

    def scan(self) -> list[IngestionJob]:
        jobs: list[IngestionJob] = []
        for path in _iter_directory_files(self.root):
            stat = path.stat()
            marker = (stat.st_size, stat.st_mtime_ns)
            key = str(path)
            if self._observed.get(key) == marker:
                continue
            jobs.extend(self.service.submit_file(path, channel=self.channel, metadata={"watcher": "directory", **self.submit_options.get("metadata", {})}, **{k: v for k, v in self.submit_options.items() if k != "metadata"}))
            self._observed[key] = marker
        return jobs


class ManualWatcher:
    """Explicit, operator-driven source watcher for files or in-memory payloads."""

    def __init__(self, service: IngestionService):
        self.service = service

    def submit(self, *, source_uri: str, content: bytes | str, **options: Any) -> IngestionJob:
        return self.service.submit_manual(source_uri=source_uri, content=content, **options)


class SFTPWatcher:
    """SFTP boundary using injected list/read callables; no credentials in this layer."""

    def __init__(self, service: IngestionService, list_files: Callable[[], Iterable[str]], read_file: Callable[[str], bytes], *, channel: str = "sftp", **submit_options: Any):
        self.service, self.list_files, self.read_file = service, list_files, read_file
        self.channel, self.submit_options = channel, submit_options
        self._observed: set[str] = set()

    def scan(self) -> list[IngestionJob]:
        jobs: list[IngestionJob] = []
        for remote_path in sorted(self.list_files()):
            if remote_path in self._observed:
                continue
            raw = self.read_file(remote_path)
            options = dict(self.submit_options)
            options["metadata"] = {"watcher": "sftp", **dict(options.get("metadata", {}))}
            jobs.append(self.service.submit_manual(source_uri=f"sftp://{remote_path}", content=raw, channel=self.channel, **options))
            self._observed.add(remote_path)
        return jobs


class MailDumpWatcher:
    """Poll a local mail dump, delegating MIME and attachment handling to the service."""

    def __init__(self, service: IngestionService, root: str | Path, **submit_options: Any):
        self.service, self.root, self.submit_options = service, Path(root), submit_options
        self._observed: set[str] = set()

    def scan(self) -> list[IngestionJob]:
        jobs: list[IngestionJob] = []
        for path in _iter_directory_files(self.root):
            if path.suffix.lower() != ".eml" or str(path) in self._observed:
                continue
            jobs.extend(self.service.submit_mail_drop(path, **self.submit_options))
            self._observed.add(str(path))
        return jobs
