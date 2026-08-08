"""Provider-neutral model routing used by QwenAgent."""
from dataclasses import dataclass
from enum import Enum

class PrivacyPolicy(str, Enum):
    LOCAL_ONLY = "local_only"
    LOCAL_PREFERRED = "local_preferred"
    REMOTE_ALLOWED = "remote_allowed"
    REMOTE_REQUIRED = "remote_required"

@dataclass(frozen=True)
class Route:
    provider: str
    reason: str

class ModelRouter:
    def choose(self, *, task_class: str, privacy: PrivacyPolicy, local_healthy: bool = True, complex_task: bool = False) -> Route:
        if privacy is PrivacyPolicy.LOCAL_ONLY and not local_healthy:
            raise RuntimeError("local_only task cannot be routed to OCI")
        if privacy is PrivacyPolicy.REMOTE_REQUIRED:
            return Route("litellm_oci", "policy requires remote provider")
        if local_healthy and not complex_task:
            return Route("lmstudio", f"local-first task class: {task_class}")
        if privacy is PrivacyPolicy.LOCAL_ONLY:
            raise RuntimeError("local_only task exceeds local capability")
        return Route("litellm_oci", "complex task or local provider unavailable")

