"""Compatibility wrapper for the provider-neutral model gateway."""
from .model_gateway import (
    AuditEvent,
    CircuitBreaker,
    GatewayError,
    LOCAL_FIRST_TASK_CLASSES,
    ModelGateway,
    ModelResult,
    ModelRouter,
    PrivacyPolicy,
    ProviderConfig,
    ProviderHealth,
    Route,
    RouterConfig,
    StreamChunk,
    ToolCall,
    Usage,
)
from .qwen_adapter import OpenNICFChatModel

__all__ = [
    "AuditEvent",
    "CircuitBreaker",
    "GatewayError",
    "LOCAL_FIRST_TASK_CLASSES",
    "ModelGateway",
    "ModelResult",
    "ModelRouter",
    "PrivacyPolicy",
    "ProviderConfig",
    "ProviderHealth",
    "Route",
    "RouterConfig",
    "StreamChunk",
    "ToolCall",
    "Usage",
    "OpenNICFChatModel",
]
