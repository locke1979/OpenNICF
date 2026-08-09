from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

from opennicf import (
    ChannelAllowList,
    ChannelAttachment,
    ChannelDelivery,
    ChannelEnvelope,
    ChannelEventLedger,
    ChannelPage,
    ChannelRateLimitError,
    ChannelResponse,
    IngestionService,
    KnowledgePlatform,
    MemoryConversationTransport,
    TelegramConversationAdapter,
    WebexConversationAdapter,
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


def _parse_command(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if not stripped:
        return "help", ""
    token, _, remainder = stripped.partition(" ")
    if token.startswith("/"):
        token = token[1:]
    return token.lower(), remainder.strip()


def _best_snippets(platform: KnowledgePlatform, query: str) -> list[str]:
    lowered = query.lower()
    return [
        chunk.text
        for chunk in platform.store.chunks.values()
        if not lowered or lowered in chunk.text.lower()
    ][:5]


class FakeRuntimeHandler:
    def __init__(self, platform: KnowledgePlatform):
        self.platform = platform
        self.envelopes: list[ChannelEnvelope] = []

    def handle(self, envelope: ChannelEnvelope) -> ChannelResponse:
        self.envelopes.append(envelope)
        command, query = _parse_command(envelope.event.text)
        reply_target = envelope.event.reply_target
        thread_id = envelope.event.thread_id

        if command == "search":
            snippets = _best_snippets(self.platform, query)
            attachment = ChannelAttachment.from_json(
                "search-results.json",
                {
                    "query": query,
                    "matches": snippets,
                    "ingestion": envelope.ingestion_summary,
                },
            )
            deliveries = (
                ChannelDelivery(
                    kind="message",
                    text=f"Acknowledged search request for {query or 'all evidence'}.",
                    reply_target=reply_target,
                    thread_id=thread_id,
                    metadata={"intent": "search"},
                ),
                ChannelDelivery(
                    kind="artifact",
                    text="Search results attached.",
                    reply_target=reply_target,
                    thread_id=thread_id,
                    attachments=(attachment,),
                    metadata={"intent": "search"},
                ),
            )
            return ChannelResponse(
                intent="search",
                status="completed",
                deliveries=deliveries,
                metadata={"query": query, "matches": len(snippets)},
            )

        if command == "status":
            attachment = ChannelAttachment.from_json(
                "status.json",
                {
                    "state": "ok",
                    "conversation_id": envelope.event.conversation.conversation_id,
                    "attachments": envelope.ingestion_summary["submitted_jobs"] if envelope.ingestion_summary else [],
                },
            )
            deliveries = (
                ChannelDelivery(
                    kind="message",
                    text="Acknowledged status request.",
                    reply_target=reply_target,
                    thread_id=thread_id,
                    metadata={"intent": "status"},
                ),
                ChannelDelivery(
                    kind="artifact",
                    text="Status snapshot attached.",
                    reply_target=reply_target,
                    thread_id=thread_id,
                    attachments=(attachment,),
                    metadata={"intent": "status"},
                ),
            )
            return ChannelResponse(
                intent="status",
                status="completed",
                deliveries=deliveries,
                metadata={"state": "ok"},
            )

        if command == "health":
            attachment = ChannelAttachment.from_json(
                "health.json",
                {
                    "state": "healthy",
                    "conversation_id": envelope.event.conversation.conversation_id,
                },
            )
            deliveries = (
                ChannelDelivery(
                    kind="message",
                    text="Acknowledged health request.",
                    reply_target=reply_target,
                    thread_id=thread_id,
                    metadata={"intent": "health"},
                ),
                ChannelDelivery(
                    kind="artifact",
                    text="Health report attached.",
                    reply_target=reply_target,
                    thread_id=thread_id,
                    attachments=(attachment,),
                    metadata={"intent": "health"},
                ),
            )
            return ChannelResponse(
                intent="health",
                status="completed",
                deliveries=deliveries,
                metadata={"state": "healthy"},
            )

        attachment = ChannelAttachment.from_json(
            "help.json",
            {
                "available": ["search", "status", "health"],
            },
        )
        deliveries = (
            ChannelDelivery(
                kind="message",
                text="Available commands: /search, /status, /health.",
                reply_target=reply_target,
                thread_id=thread_id,
                metadata={"intent": "help"},
            ),
            ChannelDelivery(
                kind="artifact",
                text="Help attached.",
                reply_target=reply_target,
                thread_id=thread_id,
                attachments=(attachment,),
                metadata={"intent": "help"},
            ),
        )
        return ChannelResponse(
            intent="help",
            status="completed",
            deliveries=deliveries,
            metadata={"available": ["search", "status", "health"]},
        )


def _factory() -> tuple[KnowledgePlatform, IngestionService, FakeRuntimeHandler]:
    platform = KnowledgePlatform.in_memory()
    ingestion = IngestionService(platform)
    handler = FakeRuntimeHandler(platform)
    return platform, ingestion, handler


def test_webex_adapter_ingests_attachments_and_round_trips_search_results():
    platform, ingestion, handler = _factory()
    transport = MemoryConversationTransport("webex")
    adapter = WebexConversationAdapter(
        transport,
        ingestion,
        handler,
        allow_list=ChannelAllowList(actor_ids=frozenset({"alice"}), conversation_ids=frozenset({"room-1"})),
        ledger=ChannelEventLedger(),
        sleep=lambda _: None,
    )

    response = adapter.handle_webhook(
        {
            "id": "evt-1",
            "roomId": "room-1",
            "personId": "alice",
            "personEmail": "alice@example.com",
            "text": "/search First doc paragraph",
            "parentId": "thread-1",
            "authScope": "internal",
            "attachments": [
                {
                    "id": "att-1",
                    "name": "evidence.docx",
                    "contentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "content": _build_docx_bytes(["First doc paragraph", "Second doc paragraph"]),
                }
            ],
        }
    )

    assert response.intent == "search"
    assert response.status == "completed"
    assert len(handler.envelopes) == 1
    assert handler.envelopes[0].ingestion_summary is not None
    assert handler.envelopes[0].ingestion_summary["submitted_jobs"][1]["metadata"]["attachment_name"] == "evidence.docx"
    assert len(transport.sent_deliveries) == 2
    assert any("Acknowledged search request" in delivery.text for _, delivery in transport.sent_deliveries)
    assert any("First doc paragraph" in delivery.text for _, delivery in transport.sent_deliveries)
    assert any(
        "search-results.json" in attachment.filename
        for _, delivery in transport.sent_deliveries
        for attachment in delivery.attachments
    )
    assert any("First doc paragraph" in chunk.text for chunk in platform.store.chunks.values())


def test_telegram_adapter_deduplicates_updates_and_supports_status_intent():
    platform, ingestion, handler = _factory()
    transport = MemoryConversationTransport("telegram")
    adapter = TelegramConversationAdapter(
        transport,
        ingestion,
        handler,
        allow_list=ChannelAllowList(actor_ids=frozenset({"alice"}), conversation_ids=frozenset({"chat-1"})),
        ledger=ChannelEventLedger(),
        sleep=lambda _: None,
    )

    update = {
        "update_id": 9,
        "message": {
            "message_id": 77,
            "chat": {"id": "chat-1"},
            "from": {"id": "alice", "username": "alice"},
            "text": "/status",
            "reply_to_message": {"message_id": 12},
        },
    }

    first = adapter.handle_webhook(update)
    second = adapter.handle_webhook(update)

    assert first.intent == "status"
    assert first.status == "completed"
    assert second.status == "duplicate"
    assert len(handler.envelopes) == 1
    assert handler.envelopes[0].ingestion_summary is not None
    assert len(transport.sent_deliveries) == 2
    assert any(attachment.filename == "status.json" for _, delivery in transport.sent_deliveries for attachment in delivery.attachments)


def test_channel_adapter_retries_rate_limited_page_fetches_and_paginates():
    platform, ingestion, handler = _factory()

    class FlakyTransport(MemoryConversationTransport):
        def __init__(self) -> None:
            super().__init__(
                "telegram",
                pages=[
                    ChannelPage(
                        items=(
                            {
                                "update_id": 10,
                                "message": {
                                    "message_id": 101,
                                    "chat": {"id": "chat-2"},
                                    "from": {"id": "alice", "username": "alice"},
                                    "text": "/status",
                                },
                            },
                        ),
                        next_cursor="cursor-1",
                    ),
                    ChannelPage(
                        items=(
                            {
                                "update_id": 11,
                                "message": {
                                    "message_id": 102,
                                    "chat": {"id": "chat-2"},
                                    "from": {"id": "alice", "username": "alice"},
                                    "text": "/health",
                                },
                            },
                        ),
                        next_cursor=None,
                    ),
                ],
            )
            self._calls = 0

        def fetch_page(self, cursor: str | None = None, limit: int = 50) -> ChannelPage:
            self._calls += 1
            if self._calls == 1:
                raise ChannelRateLimitError("slow down", retry_after_seconds=0)
            return super().fetch_page(cursor, limit)

    transport = FlakyTransport()
    adapter = TelegramConversationAdapter(
        transport,
        ingestion,
        handler,
        allow_list=ChannelAllowList(actor_ids=frozenset({"alice"}), conversation_ids=frozenset({"chat-2"})),
        ledger=ChannelEventLedger(),
        sleep=lambda _: None,
    )

    responses = adapter.sync(limit=1)

    assert [response.intent for response in responses] == ["status", "health"]
    assert len(handler.envelopes) == 2
    assert len(transport.sent_deliveries) == 4


def test_allow_list_blocks_unauthorized_actor_without_ingesting():
    platform, ingestion, handler = _factory()
    transport = MemoryConversationTransport("webex")
    adapter = WebexConversationAdapter(
        transport,
        ingestion,
        handler,
        allow_list=ChannelAllowList(actor_ids=frozenset({"alice"}), conversation_ids=frozenset({"room-1"})),
        ledger=ChannelEventLedger(),
        sleep=lambda _: None,
    )

    response = adapter.handle_webhook(
        {
            "id": "evt-unauth",
            "roomId": "room-1",
            "personId": "bob",
            "personEmail": "bob@example.com",
            "text": "/status",
            "authScope": "external",
        }
    )

    assert response.status == "rejected"
    assert len(handler.envelopes) == 0
    assert len(transport.sent_deliveries) == 1
    assert not platform.store.chunks
