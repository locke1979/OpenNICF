"""Compatibility wrapper for the provider-neutral model gateway."""
# The public export block intentionally mirrors the package's established
# grouping rather than ruff's alphabetical import/export ordering.
# ruff: noqa: I001
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
from .domain_router import DomainRoutePlan, DomainRouter, create_domain_router
from .qwen_adapter import OpenNICFChatModel

__all__ = [
    "LOCAL_FIRST_TASK_CLASSES",
    "AuditEvent",
    "CircuitBreaker",
    "DomainRoutePlan",
    "DomainRouter",
    "GatewayError",
    "ModelGateway",
    "ModelResult",
    "ModelRouter",
    "OpenNICFChatModel",
    "PrivacyPolicy",
    "ProviderConfig",
    "ProviderHealth",
    "Route",
    "RouterConfig",
    "StreamChunk",
    "ToolCall",
    "Usage",
    "create_domain_router",
]
