"""Narrow adapter boundary for the discovered OpenClaw runner."""
from dataclasses import dataclass
import os, subprocess

@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str

class SubagentPipelineRunner:
    def __init__(self, command: str | None = None):
        self.command = command or os.environ.get("SUBAGENT_RUNNER_COMMAND")

    def run(self, issue_number: int, task: str, dry_run: bool = False) -> RunResult:
        if dry_run:
            return RunResult(f"dry-run-{issue_number}", "planned")
        if not self.command:
            raise RuntimeError("SUBAGENT_RUNNER_COMMAND is not configured")
        # The task is passed as an argument, never interpolated into shell text.
        proc = subprocess.run([self.command, str(issue_number), task], check=False, capture_output=True, text=True)
        return RunResult(f"issue-{issue_number}", "succeeded" if proc.returncode == 0 else "failed")

