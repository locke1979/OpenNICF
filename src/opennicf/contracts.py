"""Machine-readable GitHub issue execution contracts."""
from dataclasses import dataclass
import re
import yaml

_BLOCK = re.compile(r"```yaml\s*\n(?P<body>openclaw:\s*\n.*?)(?:\n```)", re.S)

@dataclass(frozen=True)
class IssueContract:
    issue_number: int
    autorun: bool
    depends_on: tuple[int, ...]
    pipeline: str
    deploy: bool

def parse_contract(issue_number: int, body: str) -> IssueContract:
    match = _BLOCK.search(body)
    if not match:
        raise ValueError("issue has no fenced openclaw YAML contract")
    data = yaml.safe_load(match.group("body")) or {}
    raw = data.get("openclaw")
    if not isinstance(raw, dict) or raw.get("contract_version") != 1:
        raise ValueError("unsupported or malformed openclaw contract")
    deps = raw.get("depends_on", [])
    if not isinstance(deps, list) or any(not isinstance(n, int) for n in deps):
        raise ValueError("depends_on must be a list of issue numbers")
    pipeline = raw.get("pipeline")
    if pipeline != "subagent-pipeline":
        raise ValueError("issue is not assigned to subagent-pipeline")
    return IssueContract(issue_number, bool(raw.get("autorun", False)), tuple(deps), pipeline, bool(raw.get("deploy", False)))

