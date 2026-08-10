"""Secret/config inventory and conservative source scanning."""
from __future__ import annotations

import re
from dataclasses import dataclass

INVENTORY = (
    ("PROXMOX_API_CREDENTIALS", "external secret store", "rotate after operator change"),
    ("LM_STUDIO_ENDPOINT_CONFIG", "deployment config", "review on endpoint change"),
    ("LITELLM_OCI_ENDPOINT_TOKEN", "external secret store", "rotate on provider change"),
    ("WEBEX_TOKEN_SECRET", "external secret store", "revoke and reissue"),
    ("TELEGRAM_TOKEN", "external secret store", "revoke and reissue"),
    ("POSTGRES_OBJECT_STORE_CREDENTIALS", "external secret store", "rotate on restore"),
    ("BROKER_SIGNING_MTLS_MATERIAL", "external secret store", "revoke certificates"),
    ("GITHUB_OPENCLAW_RUNNER_CREDENTIAL", "runner secret store", "revoke runner"),
)

_PATTERNS = (
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "private-endpoint",
        re.compile(
            r"(?i)\b(?:(?:https?://)?(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})|(?:[A-Za-z0-9-]+\.)+(?:internal|local|corp|lan))(?:[:/\s]|$)"
        ),
    ),
    ("secret-assignment", re.compile(r"(?i)\b(?:token|password|secret|api[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9/+_=.-]{20,}")),
)


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    line: int


def inventory() -> list[dict[str, str]]:
    return [{"name": name, "source": source, "rotation": rotation} for name, source, rotation in INVENTORY]


def scan_text(text: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for kind, pattern in _PATTERNS:
            if pattern.search(line):
                findings.append(SecretFinding(kind, line_no))
    return findings
