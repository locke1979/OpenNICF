from .contracts import IssueContract, parse_contract
from .leases import LeaseStore
from .runner import SubagentPipelineRunner

def dependencies_ready(contract: IssueContract, states: dict[int, str]) -> bool:
    return all(states.get(dep) == "closed" for dep in contract.depends_on)

def dispatch(issue: dict, states: dict[int, str], leases: LeaseStore, owner: str, runner: SubagentPipelineRunner, dry_run: bool = False):
    contract = parse_contract(issue["number"], issue["body"])
    if not contract.autorun or not dependencies_ready(contract, states):
        return None
    if not dry_run and not leases.acquire(contract.issue_number, owner):
        return None
    return runner.run(contract.issue_number, issue["title"], dry_run=dry_run)

