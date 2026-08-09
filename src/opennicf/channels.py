"""Common Webex and Telegram conversational channel adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import PurePosixPath
import json
import mimetypes
import sqlite3
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .ingestion import IngestionService


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stable_hash(*parts: str) -> str:
    digest = sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_attachment_name(name: str) -> str:
    candidate = PurePosixPath(name.replace("\\", "/"))
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe attachment filename: {name}")
    return candidate.name


def _guess_mime_type(name: str) -> str:
    mime_type, _ = mimetypes.guess_type(name)
    return mime_type or "application/octet-stream"


def _json_payload(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, indent=2, default=str).encode("utf-8")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _conversation_source_uri(channel: str, conversation_id: str, message_id: str) -> str:
    if channel == "webex":
        return f"webex://rooms/{conversation_id}/messages/{message_id}"
    if channel == "telegram":
        return f"telegram://chats/{conversation_id}/messages/{message_id}"
    return f"{channel}://conversations/{conversation_id}/messages/{message_id}"


def _delivery_target(message: "ChannelMessage") -> tuple[str | None, str | None]:
    thread_id = message.conversation.thread_id or message.conversation.message_id
    reply_target = message.conversation.reply_target or message.conversation.message_id
    return thread_id, reply_target


@dataclass(frozen=True)
class ChannelActor:
    actor_id: str
    display_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelConversation:
    channel: str
    conversation_id: str
    message_id: str
    thread_id: str | None = None
    reply_target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelAttachment:
    filename: str
    content_type: str
    content: bytes
    content_hash: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bytes(
        cls,
        filename: str,
        content: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ChannelAttachment":
        safe_name = _safe_attachment_name(filename)
        payload = bytes(content)
        return cls(
            filename=safe_name,
            content_type=content_type or _guess_mime_type(safe_name),
            content=payload,
            content_hash=sha256(payload).hexdigest(),
            size_bytes=len(payload),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_text(
        cls,
        filename: str,
        text: str,
        *,
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> "ChannelAttachment":
        return cls.from_bytes(
            filename,
            text.encode("utf-8"),
            content_type=content_type,
            metadata=metadata,
        )

    @classmethod
    def from_json(
        cls,
        filename: str,
        payload: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> "ChannelAttachment":
        return cls.from_bytes(
            filename,
            _json_payload(payload),
            content_type="application/json",
            metadata=metadata,
        )

    def as_ingestion_tuple(self) -> tuple[str, bytes]:
        return self.filename, self.content


@dataclass(frozen=True)
class ChannelMessage:
    event_id: str
    channel: str
    actor: ChannelActor
    auth_scope: str
    conversation: ChannelConversation
    text: str
    attachments: tuple[ChannelAttachment, ...] = ()
    received_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reply_target(self) -> str | None:
        return self.conversation.reply_target or self.conversation.message_id

    @property
    def thread_id(self) -> str | None:
        return self.conversation.thread_id or self.conversation.message_id


@dataclass(frozen=True)
class ChannelDelivery:
    kind: str
    text: str
    reply_target: str | None = None
    thread_id: str | None = None
    attachments: tuple[ChannelAttachment, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelResponse:
    intent: str
    status: str
    deliveries: tuple[ChannelDelivery, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelPage:
    items: tuple[Mapping[str, Any], ...]
    next_cursor: str | None = None
    retry_after_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelAllowList:
    actor_ids: frozenset[str] = frozenset()
    conversation_ids: frozenset[str] = frozenset()
    reply_targets: frozenset[str] = frozenset()
    auth_scopes: frozenset[str] = frozenset({"internal"})

    def permits(self, message: ChannelMessage) -> bool:
        if self.actor_ids and message.actor.actor_id not in self.actor_ids:
            return False
        if self.conversation_ids and message.conversation.conversation_id not in self.conversation_ids:
            return False
        if self.reply_targets and (message.reply_target or "") not in self.reply_targets:
            return False
        if self.auth_scopes and message.auth_scope not in self.auth_scopes:
            return False
        return True


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0


class ChannelError(RuntimeError):
    pass


class ChannelTransientError(ChannelError):
    pass


class ChannelRateLimitError(ChannelTransientError):
    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ChannelEventLedger:
    """Deterministic idempotency ledger for webhook and update processing."""

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                channel TEXT NOT NULL,
                event_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                thread_id TEXT,
                reply_target TEXT,
                actor_id TEXT NOT NULL,
                auth_scope TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(channel, event_id)
            )
            """
        )
        self.db.commit()

    def record(self, message: ChannelMessage) -> bool:
        try:
            self.db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message.channel,
                    message.event_id,
                    message.conversation.message_id,
                    message.conversation.conversation_id,
                    message.conversation.thread_id,
                    message.conversation.reply_target,
                    message.actor.actor_id,
                    message.auth_scope,
                    message.received_at.isoformat(),
                ),
            )
        except sqlite3.IntegrityError:
            self.db.rollback()
            return False
        self.db.commit()
        return True

    def seen(self, message: ChannelMessage) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM events WHERE channel = ? AND event_id = ?",
            (message.channel, message.event_id),
        ).fetchone()
        return row is not None

    def snapshot(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT channel, event_id, message_id, conversation_id, thread_id, reply_target, actor_id, auth_scope, created_at FROM events ORDER BY created_at"
        ).fetchall()
        return [
            {
                "channel": row[0],
                "event_id": row[1],
                "message_id": row[2],
                "conversation_id": row[3],
                "thread_id": row[4],
                "reply_target": row[5],
                "actor_id": row[6],
                "auth_scope": row[7],
                "created_at": row[8],
            }
            for row in rows
        ]


class ConversationTransport(Protocol):
    channel: str

    def fetch_page(self, cursor: str | None = None, limit: int = 50) -> ChannelPage:
        raise NotImplementedError

    def send_delivery(self, event: ChannelMessage, delivery: ChannelDelivery) -> None:
        raise NotImplementedError


class MemoryConversationTransport:
    """Synthetic provider used by tests and deterministic local runs."""

    def __init__(self, channel: str, *, pages: Sequence[ChannelPage] | None = None):
        self.channel = channel
        self._pages = list(pages or [])
        self.sent_deliveries: list[tuple[ChannelMessage, ChannelDelivery]] = []

    def queue_page(self, page: ChannelPage) -> None:
        self._pages.append(page)

    def fetch_page(self, cursor: str | None = None, limit: int = 50) -> ChannelPage:
        if self._pages:
            return self._pages.pop(0)
        return ChannelPage(items=(), next_cursor=None)

    def send_delivery(self, event: ChannelMessage, delivery: ChannelDelivery) -> None:
        self.sent_deliveries.append((event, delivery))


@dataclass(frozen=True)
class ChannelEnvelope:
    event: ChannelMessage
    ingestion_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChannelRuntimeHandler(Protocol):
    def handle(self, envelope: ChannelEnvelope) -> ChannelResponse:
        raise NotImplementedError


RuntimeHandler = ChannelRuntimeHandler | Callable[[ChannelEnvelope], ChannelResponse]


class ConversationChannelAdapter:
    def __init__(
        self,
        transport: ConversationTransport,
        ingestion: IngestionService,
        runtime_handler: RuntimeHandler,
        *,
        allow_list: ChannelAllowList | None = None,
        ledger: ChannelEventLedger | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep=time.sleep,
    ):
        self.transport = transport
        self.ingestion = ingestion
        self.runtime_handler = runtime_handler
        self.allow_list = allow_list or ChannelAllowList()
        self.ledger = ledger or ChannelEventLedger()
        self.retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep

    def _normalize_event(self, raw: Mapping[str, Any]) -> ChannelMessage:
        raise NotImplementedError

    def _retry(self, func):
        last_error: Exception | None = None
        backoff = self.retry_policy.initial_backoff_seconds
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                return func()
            except ChannelRateLimitError as exc:
                last_error = exc
                delay = exc.retry_after_seconds if exc.retry_after_seconds is not None else backoff
                self._sleep(max(0.0, delay))
            except ChannelTransientError as exc:
                last_error = exc
                self._sleep(max(0.0, backoff))
            backoff = min(backoff * 2, self.retry_policy.max_backoff_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("retry helper exhausted without an error")

    def _send_deliveries(self, event: ChannelMessage, deliveries: Sequence[ChannelDelivery]) -> None:
        for delivery in deliveries:
            self._retry(lambda delivery=delivery: self.transport.send_delivery(event, delivery))

    def _summarize_job(self, job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "source_id": job.source_id,
            "source_uri": job.source_uri,
            "channel": job.channel,
            "mime_type": job.mime_type,
            "content_hash": job.content_hash,
            "source_kind": job.source_kind,
            "source_type": job.source_type,
            "domain": job.domain,
            "system": job.system,
            "domain_id": job.domain_id,
            "system_id": job.system_id,
            "component_id": job.component_id,
            "evidence_type": job.evidence_type,
            "environment": job.environment,
            "acl_scope": job.acl_scope,
            "parser_hint": job.parser_hint,
            "state": job.state,
            "attempts": job.attempts,
            "metadata": dict(job.metadata),
        }

    def _summarize_outcome(self, outcome) -> dict[str, Any]:
        return {
            "job_id": outcome.job_id,
            "created": outcome.created,
            "content_hash": outcome.content_hash,
            "metadata": dict(outcome.metadata),
            "bundles": [
                {
                    "source_id": bundle.source.source_id,
                    "source_uri": bundle.source.source_uri,
                    "source_kind": bundle.source.source_kind,
                    "channel": bundle.source.channel,
                    "mime_type": bundle.source.mime_type,
                    "content_hash": bundle.source.content_hash,
                    "size_bytes": bundle.source.size_bytes,
                    "source_type": bundle.source.source_type,
                    "parser_version": bundle.version.parser_version,
                    "parser_name": bundle.artifact.parser_name,
                    "artifact_hash": bundle.artifact.artifact_hash,
                    "chunk_count": len(bundle.chunks),
                    "metadata": dict(bundle.artifact.metadata),
                }
                for bundle in outcome.bundles
            ],
        }

    def _ingest_event(self, event: ChannelMessage) -> dict[str, Any]:
        attachments = tuple(attachment.as_ingestion_tuple() for attachment in event.attachments)
        metadata = {
            "event_id": event.event_id,
            "actor_id": event.actor.actor_id,
            "actor_display_name": event.actor.display_name,
            "conversation_id": event.conversation.conversation_id,
            "message_id": event.conversation.message_id,
            "thread_id": event.thread_id,
            "reply_target": event.reply_target,
            "channel_event_metadata": dict(event.metadata),
            "conversation_metadata": dict(event.conversation.metadata),
            "actor_metadata": dict(event.actor.metadata),
            "attachment_count": len(event.attachments),
            "attachments": [
                {
                    "filename": attachment.filename,
                    "content_type": attachment.content_type,
                    "content_hash": attachment.content_hash,
                    "size_bytes": attachment.size_bytes,
                    "metadata": dict(attachment.metadata),
                }
                for attachment in event.attachments
            ],
        }
        if event.channel == "webex":
            jobs = self.ingestion.submit_webex_message(
                room_id=event.conversation.conversation_id,
                message_id=event.conversation.message_id,
                message_text=event.text,
                attachments=attachments,
                thread_id=event.thread_id,
                source_uri=event.metadata.get("source_uri"),
                acl_scope=event.auth_scope,
                metadata=metadata,
            )
        elif event.channel == "telegram":
            jobs = self.ingestion.submit_telegram_message(
                chat_id=event.conversation.conversation_id,
                message_id=event.conversation.message_id,
                message_text=event.text,
                attachments=attachments,
                thread_id=event.thread_id,
                source_uri=event.metadata.get("source_uri"),
                acl_scope=event.auth_scope,
                metadata=metadata,
            )
        else:
            raise ValueError(f"unsupported channel for ingestion: {event.channel}")

        outcomes = self.ingestion.run(max_jobs=len(jobs))
        return {
            "submitted_jobs": [self._summarize_job(job) for job in jobs],
            "outcomes": [self._summarize_outcome(outcome) for outcome in outcomes],
        }

    def process_event(self, event: ChannelMessage) -> ChannelResponse:
        if self.ledger.seen(event):
            return ChannelResponse(intent="duplicate", status="duplicate", deliveries=())
        if not self.allow_list.permits(event):
            delivery = ChannelDelivery(
                kind="message",
                text="This conversation is not allow-listed for OpenNICF processing.",
                thread_id=event.thread_id,
                reply_target=event.reply_target,
                metadata={"event_id": event.event_id, "intent": "rejected"},
            )
            self._send_deliveries(event, (delivery,))
            self.ledger.record(event)
            return ChannelResponse(intent="rejected", status="rejected", deliveries=(delivery,), metadata={"event_id": event.event_id})

        ingestion_summary = self._ingest_event(event)
        envelope = ChannelEnvelope(
            event=event,
            ingestion_summary=ingestion_summary,
            metadata={
                "channel": event.channel,
                "conversation_id": event.conversation.conversation_id,
                "message_id": event.conversation.message_id,
                "thread_id": event.thread_id,
                "reply_target": event.reply_target,
            },
        )
        response = self.runtime_handler.handle(envelope) if hasattr(self.runtime_handler, "handle") else self.runtime_handler(envelope)
        self._send_deliveries(event, response.deliveries)
        self.ledger.record(event)
        return response

    def handle_webhook(self, payload: Mapping[str, Any]) -> ChannelResponse:
        return self.process_event(self._normalize_event(payload))

    def sync(self, *, cursor: str | None = None, limit: int = 50) -> list[ChannelResponse]:
        responses: list[ChannelResponse] = []
        next_cursor = cursor
        while True:
            page = self._retry(lambda next_cursor=next_cursor: self.transport.fetch_page(next_cursor, limit))
            if not page.items:
                break
            for raw in page.items:
                responses.append(self.process_event(self._normalize_event(raw)))
            if not page.next_cursor and len(page.items) < limit:
                break
            next_cursor = page.next_cursor
        return responses


class WebexConversationAdapter(ConversationChannelAdapter):
    def _normalize_event(self, raw: Mapping[str, Any]) -> ChannelMessage:
        attachments = tuple(
            ChannelAttachment.from_bytes(
                attachment.get("name") or attachment.get("filename") or "attachment.bin",
                attachment.get("content") or attachment.get("bytes") or b"",
                content_type=attachment.get("contentType") or attachment.get("mime_type"),
                metadata={"provider": "webex", "attachment_id": attachment.get("id")},
            )
            for attachment in (raw.get("attachments") or [])
        )
        conversation_id = _as_text(raw.get("roomId") or raw.get("room_id") or raw.get("conversationId") or raw.get("conversation_id"))
        message_id = _as_text(raw.get("id") or raw.get("messageId") or raw.get("message_id") or raw.get("webhook_id") or conversation_id)
        actor_id = _as_text(raw.get("personId") or raw.get("actorId") or raw.get("personEmail") or raw.get("from"))
        actor = ChannelActor(
            actor_id=actor_id,
            display_name=_as_text(raw.get("personEmail") or raw.get("personDisplayName") or raw.get("displayName") or actor_id),
            metadata={"provider": "webex"},
        )
        thread_id = _as_text(raw.get("parentId") or raw.get("threadId") or raw.get("thread_id")) or None
        reply_target = _as_text(raw.get("parentId") or raw.get("replyTarget") or raw.get("reply_target")) or None
        auth_scope = _as_text(raw.get("authScope") or raw.get("auth_scope") or "internal")
        text = _as_text(raw.get("text") or raw.get("markdown") or raw.get("body"))
        conversation = ChannelConversation(
            channel="webex",
            conversation_id=conversation_id,
            message_id=message_id,
            thread_id=thread_id,
            reply_target=reply_target,
            metadata={"source_uri": _conversation_source_uri("webex", conversation_id, message_id)},
        )
        return ChannelMessage(
            event_id=_as_text(raw.get("eventId") or raw.get("event_id") or raw.get("id") or message_id),
            channel="webex",
            actor=actor,
            auth_scope=auth_scope,
            conversation=conversation,
            text=text,
            attachments=attachments,
            metadata={
                "raw": dict(raw),
                "source_uri": _conversation_source_uri("webex", conversation_id, message_id),
            },
        )


class TelegramConversationAdapter(ConversationChannelAdapter):
    def _normalize_event(self, raw: Mapping[str, Any]) -> ChannelMessage:
        update = dict(raw)
        message = update.get("message") or update.get("edited_message") or update.get("channel_post") or update
        chat = message.get("chat") or {}
        sender = message.get("from") or update.get("from") or {}
        attachments = tuple(
            ChannelAttachment.from_bytes(
                attachment.get("filename") or attachment.get("file_name") or attachment.get("name") or "attachment.bin",
                attachment.get("content") or attachment.get("bytes") or b"",
                content_type=attachment.get("mime_type") or attachment.get("content_type"),
                metadata={"provider": "telegram", "attachment_id": attachment.get("id")},
            )
            for attachment in (message.get("attachments") or [])
        )
        if not attachments:
            document = message.get("document")
            if isinstance(document, Mapping):
                attachments = (
                    ChannelAttachment.from_bytes(
                        document.get("file_name") or document.get("filename") or "attachment.bin",
                        document.get("content") or document.get("bytes") or b"",
                        content_type=document.get("mime_type") or document.get("mimeType"),
                        metadata={"provider": "telegram", "attachment_id": document.get("file_id")},
                    ),
                )
        conversation_id = _as_text(chat.get("id") or update.get("chat_id") or update.get("conversation_id"))
        message_id = _as_text(message.get("message_id") or update.get("message_id") or update.get("update_id") or conversation_id)
        actor_id = _as_text(sender.get("id") or update.get("from_id") or sender.get("username") or "unknown")
        actor = ChannelActor(
            actor_id=actor_id,
            display_name=_as_text(sender.get("username") or sender.get("first_name") or sender.get("last_name") or actor_id),
            metadata={"provider": "telegram"},
        )
        reply_to = message.get("reply_to_message") or {}
        thread_id = _as_text(message.get("message_thread_id") or reply_to.get("message_id") or message_id) or None
        reply_target = _as_text(reply_to.get("message_id") or message.get("message_thread_id") or message_id) or None
        auth_scope = _as_text(update.get("auth_scope") or message.get("auth_scope") or "internal")
        text = _as_text(message.get("text") or message.get("caption") or "")
        conversation = ChannelConversation(
            channel="telegram",
            conversation_id=conversation_id,
            message_id=message_id,
            thread_id=thread_id,
            reply_target=reply_target,
            metadata={"source_uri": _conversation_source_uri("telegram", conversation_id, message_id)},
        )
        return ChannelMessage(
            event_id=_as_text(update.get("update_id") or message.get("update_id") or message_id),
            channel="telegram",
            actor=actor,
            auth_scope=auth_scope,
            conversation=conversation,
            text=text,
            attachments=attachments,
            metadata={
                "raw": dict(update),
                "source_uri": _conversation_source_uri("telegram", conversation_id, message_id),
            },
        )
