"""QwenAgent adapter backed exclusively by :class:`ModelGateway`.

QwenAgent owns orchestration and tool loops; this adapter owns only the model
boundary.  Provider URLs, credentials, and routing policy stay in the gateway
and runtime environment.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from typing import Any

from .model_gateway import ModelGateway, PrivacyPolicy


# QwenAgent may add framework-specific generation hints that are not part of
# the OpenAI-compatible provider contract.  Keep the provider payload narrow;
# in particular, clawproxy rejects QwenAgent's ``lang`` hint.
_PROVIDER_GENERATION_KEYS = frozenset(
    {
        "frequency_penalty",
        "max_tokens",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "top_p",
        "tool_choice",
        "parallel_tool_calls",
    }
)


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {
        key: getattr(message, key)
        for key in ("role", "content", "name", "function_call", "extra")
        if getattr(message, key, None) is not None
    }


class OpenNICFChatModel:
    """Minimal QwenAgent LLM protocol implemented through ``ModelGateway``.

    The class is deliberately duck-typed so importing OpenNICF does not make
    the optional qwen-agent dependency mandatory.  QwenAgent calls ``chat``;
    every request is converted into a gateway request and receives the same
    local-first, privacy, retry, and fallback policy as direct callers.
    """

    def __init__(
        self,
        gateway: ModelGateway | None = None,
        *,
        task_class: str | None = None,
        privacy: PrivacyPolicy | str | None = None,
        complex_input_chars: int = 12_000,
    ) -> None:
        self.gateway = gateway or ModelGateway.from_env()
        configured_provider = self.gateway.local_provider or self.gateway.remote_provider
        self.model = configured_provider.model if configured_provider else "opennicf-router"
        self.model_type = "opennicf_router"
        self.task_class = task_class or os.getenv("QWEN_TASK_CLASS", "simple_rag")
        configured_privacy = privacy or os.getenv("QWEN_PRIVACY", PrivacyPolicy.LOCAL_PREFERRED.value)
        self.privacy = PrivacyPolicy(configured_privacy)
        self.complex_input_chars = complex_input_chars

    @staticmethod
    def _messages(messages: list[Any]) -> list[dict[str, Any]]:
        return [_message_dict(message) for message in messages]

    def _request_options(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        size = sum(len(str(message.get("content", ""))) for message in messages)
        return {
            "task_class": self.task_class,
            "privacy": self.privacy,
            "complex_task": size >= self.complex_input_chars,
            "estimated_input_tokens": max(1, size // 4),
        }

    @staticmethod
    def _result_message(result: Any) -> Any:
        message: dict[str, Any] = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            call = result.tool_calls[0]
            message["function_call"] = {"name": call.name, "arguments": call.arguments}
            message["extra"] = {"function_id": call.id}
        try:
            from qwen_agent.llm.schema import Message
        except ImportError:
            return message
        return Message(**message)

    @staticmethod
    def _content_message(content: str) -> Any:
        try:
            from qwen_agent.llm.schema import Message
        except ImportError:
            return {"role": "assistant", "content": content}
        return Message(role="assistant", content=content)

    def chat(
        self,
        messages: list[Any],
        functions: list[dict[str, Any]] | None = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | Iterator[list[dict[str, Any]]]:
        normalized = self._messages(messages)
        options = self._request_options(normalized)
        extra_payload: dict[str, Any] = {}
        if functions:
            extra_payload["tools"] = functions
        if extra_generate_cfg:
            extra_payload.update(
                {
                    key: value
                    for key, value in extra_generate_cfg.items()
                    if key in _PROVIDER_GENERATION_KEYS
                }
            )

        if not stream:
            result = self.gateway.chat(
                normalized,
                **options,
                extra_payload=extra_payload,
            )
            return [self._result_message(result)]

        def stream_results() -> Iterator[list[dict[str, Any]]]:
            content = ""
            for chunk in self.gateway.stream_chat(
                normalized,
                **options,
                extra_payload=extra_payload,
            ):
                content += chunk.delta
                # QwenAgent's normal stream protocol expects the full message
                # accumulated so far when delta_stream is false.
                if delta_stream:
                    yield [self._content_message(chunk.delta)]
                else:
                    yield [self._content_message(content)]

        return stream_results()


__all__ = ["OpenNICFChatModel"]
