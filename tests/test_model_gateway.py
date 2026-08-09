from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from opennicf.router import (
    GatewayError,
    ModelGateway,
    ModelRouter,
    PrivacyPolicy,
    ProviderConfig,
    RouterConfig,
)


class _MockProviderServer:
    def __init__(
        self,
        *,
        models_status: int = 200,
        models_payload: dict | None = None,
        chat_status: int = 200,
        chat_payload: dict | None = None,
        stream_events: list[dict] | None = None,
    ):
        self.behavior = {
            "models_status": models_status,
            "models_payload": models_payload
            or {"data": [{"id": "qwen-3.5-2b"}, {"id": "fallback-model"}]},
            "chat_status": chat_status,
            "chat_payload": chat_payload
            or {
                "id": "chatcmpl-1",
                "model": "mock-model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            "stream_events": stream_events
            or [
                {"choices": [{"delta": {"content": "hello "}}]},
                {"choices": [{"delta": {"content": "world"}}]},
            ],
            "calls": [],
        }

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _write_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_stream(self, events: list[dict]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for event in events:
                    body = f"data: {json.dumps(event)}\n\n".encode("utf-8")
                    self.wfile.write(body)
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                outer.behavior["calls"].append(("GET", self.path, None))
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                status = outer.behavior["models_status"]
                payload = outer.behavior["models_payload"]
                self._write_json(status, payload)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                parsed = json.loads(raw.decode("utf-8")) if raw else None
                outer.behavior["calls"].append(("POST", self.path, parsed))
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                if outer.behavior["chat_status"] >= 400:
                    self._write_json(
                        outer.behavior["chat_status"],
                        {"error": {"message": "boom", "type": "mock_error"}},
                    )
                    return
                if parsed and parsed.get("stream"):
                    self._write_stream(outer.behavior["stream_events"])
                    return
                self._write_json(outer.behavior["chat_status"], outer.behavior["chat_payload"])

            def log_message(self, *_args) -> None:  # noqa: D401
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def mock_providers():
    local = _MockProviderServer(
        chat_payload={
            "id": "local-1",
            "model": "qwen-local",
            "choices": [
                {"message": {"role": "assistant", "content": "local answer"}}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
    )
    remote = _MockProviderServer(
        chat_payload={
            "id": "remote-1",
            "model": "qwen-remote",
            "choices": [
                {"message": {"role": "assistant", "content": "remote answer"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
    )
    try:
        yield local, remote
    finally:
        local.close()
        remote.close()


def _gateway(local: _MockProviderServer, remote: _MockProviderServer, **router_kwargs) -> ModelGateway:
    router = ModelRouter(
        RouterConfig(
            local_context_limit=router_kwargs.pop("local_context_limit", 128),
            local_max_tool_complexity=router_kwargs.pop("local_max_tool_complexity", 1),
            local_max_attachment_fan_in=router_kwargs.pop("local_max_attachment_fan_in", 4),
            local_max_source_fan_in=router_kwargs.pop("local_max_source_fan_in", 4),
        )
    )
    return ModelGateway(
        router=router,
        local_provider=ProviderConfig(
            name="lmstudio",
            base_url=local.url,
            model="qwen-local",
            timeout_seconds=1.0,
            max_retries=0,
            circuit_breaker_failures=1,
            circuit_breaker_reset_seconds=60.0,
        ),
        remote_provider=ProviderConfig(
            name="litellm_oci",
            base_url=remote.url,
            model="qwen-remote",
            timeout_seconds=1.0,
            max_retries=0,
            circuit_breaker_failures=1,
            circuit_breaker_reset_seconds=60.0,
        ),
    )


def test_from_env_uses_runtime_configuration():
    gateway = ModelGateway.from_env(
        {
            "LMSTUDIO_BASE_URL": "http://127.0.0.1:11434",
            "LMSTUDIO_MODEL": "local-model",
            "LITELLM_BASE_URL": "http://127.0.0.1:4000",
            "LITELLM_MODEL": "remote-model",
            "MODEL_ROUTER_LOCAL_CONTEXT_LIMIT": "256",
            "MODEL_ROUTER_LOCAL_MAX_TOOL_COMPLEXITY": "2",
        }
    )
    assert gateway.local_provider is not None
    assert gateway.local_provider.base_url == "http://127.0.0.1:11434"
    assert gateway.local_provider.model == "local-model"
    assert gateway.remote_provider is not None
    assert gateway.remote_provider.model == "remote-model"
    assert gateway.router.config.local_context_limit == 256
    assert gateway.router.config.local_max_tool_complexity == 2


def test_local_first_simple_prompt_uses_lmstudio(mock_providers):
    local, remote = mock_providers
    gateway = _gateway(local, remote)

    result = gateway.chat(
        [{"role": "user", "content": "Classify this ticket"}],
        task_class="classification",
        privacy=PrivacyPolicy.LOCAL_PREFERRED,
        estimated_input_tokens=16,
    )

    assert result.provider == "lmstudio"
    assert result.route.provider == "lmstudio"
    assert result.content == "local answer"
    assert local.behavior["calls"][0][0] == "POST"
    assert not any(call[0] == "POST" for call in remote.behavior["calls"])
    assert gateway.audit_events[-1].reason.startswith("local-first")


def test_large_prompt_escalates_to_litellm(mock_providers):
    local, remote = mock_providers
    gateway = _gateway(local, remote, local_context_limit=32)

    result = gateway.chat(
        [{"role": "user", "content": "Audit these 50 documents"}],
        task_class="audit",
        privacy=PrivacyPolicy.REMOTE_ALLOWED,
        estimated_input_tokens=128,
        complex_task=True,
    )

    assert result.provider == "litellm_oci"
    assert result.content == "remote answer"
    assert remote.behavior["calls"][0][0] == "POST"
    assert gateway.audit_events[-1].provider == "litellm_oci"


def test_local_outage_triggers_allowed_fallback(mock_providers):
    local, remote = mock_providers
    local.behavior["chat_status"] = 503
    gateway = _gateway(local, remote)

    result = gateway.chat(
        [{"role": "user", "content": "Summarise this short note"}],
        task_class="summarisation",
        privacy=PrivacyPolicy.LOCAL_PREFERRED,
        estimated_input_tokens=24,
    )

    assert result.provider == "litellm_oci"
    assert result.content == "remote answer"
    assert len(local.behavior["calls"]) == 1
    assert len(remote.behavior["calls"]) == 1


def test_local_only_prevents_fallback(mock_providers):
    local, remote = mock_providers
    local.behavior["chat_status"] = 503
    gateway = _gateway(local, remote)

    with pytest.raises(GatewayError) as excinfo:
        gateway.chat(
            [{"role": "user", "content": "Audit this source code"}],
            task_class="audit",
            privacy=PrivacyPolicy.LOCAL_ONLY,
            complex_task=False,
        )

    assert excinfo.value.provider == "lmstudio"
    assert excinfo.value.status_code == 503
    assert excinfo.value.retriable is True
    assert not remote.behavior["calls"]


def test_circuit_breaker_skips_unhealthy_local_after_probe(mock_providers):
    local, remote = mock_providers
    local.behavior["models_status"] = 503
    gateway = _gateway(local, remote)

    health = gateway.health_check("lmstudio")
    assert not health.healthy

    result = gateway.chat(
        [{"role": "user", "content": "Classify this note"}],
        task_class="classification",
        privacy=PrivacyPolicy.REMOTE_ALLOWED,
        estimated_input_tokens=8,
    )

    assert result.provider == "litellm_oci"
    assert not any(call[0] == "POST" for call in local.behavior["calls"])
    assert any(call[0] == "POST" for call in remote.behavior["calls"])


def test_streaming_is_normalized_and_audited(mock_providers):
    local, remote = mock_providers
    gateway = _gateway(local, remote)

    chunks = list(
        gateway.stream_chat(
            [{"role": "user", "content": "Stream this answer"}],
            task_class="classification",
            privacy=PrivacyPolicy.LOCAL_PREFERRED,
        )
    )

    assert [chunk.delta for chunk in chunks] == ["hello ", "world"]
    assert chunks[0].route.provider == "lmstudio"
    assert gateway.audit_events[-1].streaming is True

