import pytest
from opennicf.contracts import parse_contract
from opennicf.dispatcher import dependencies_ready, dispatch
from opennicf.leases import LeaseStore
from opennicf.router import ModelRouter, PrivacyPolicy
from opennicf.runner import SubagentPipelineRunner
from opennicf.rag import Evidence, EvidenceIndex
from opennicf.tools import OpenNICFTools

BODY = """```yaml\nopenclaw:\n  contract_version: 1\n  pipeline: subagent-pipeline\n  autorun: true\n  depends_on: [2]\n  deploy: false\n```"""

def test_contract_and_dependency_gate():
    c = parse_contract(3, BODY)
    assert c.depends_on == (2,)
    assert not dependencies_ready(c, {2: "open"})
    assert dependencies_ready(c, {2: "closed"})

def test_dry_run_does_not_need_runner_or_lease(tmp_path):
    result = dispatch({"number": 3, "title": "bootstrap", "body": BODY}, {2: "closed"}, LeaseStore(str(tmp_path / "x.db")), "a", SubagentPipelineRunner(), True)
    assert result.status == "planned"

def test_duplicate_lease_is_suppressed(tmp_path):
    leases = LeaseStore(str(tmp_path / "x.db"))
    assert leases.acquire(3, "a")
    assert not leases.acquire(3, "b")

def test_local_only_fails_closed():
    with pytest.raises(RuntimeError):
        ModelRouter().choose(task_class="audit", privacy=PrivacyPolicy.LOCAL_ONLY, local_healthy=False)

def test_router_paths():
    router = ModelRouter()
    assert router.choose(task_class="classification", privacy=PrivacyPolicy.LOCAL_PREFERRED).provider == "lmstudio"
    assert router.choose(task_class="audit", privacy=PrivacyPolicy.REMOTE_ALLOWED, complex_task=True).provider == "litellm_oci"

def test_qwen_tools_reach_provenance_aware_rag_and_block_dangerous_jobs():
    index = EvidenceIndex()
    index.add(Evidence("src-1", "database timeout", "logs.csv:42"))
    tools = OpenNICFTools(index)
    assert tools.search_evidence("database timeout")[0]["locator"] == "logs.csv:42"
    with pytest.raises(PermissionError):
        tools.create_diagnostic_job({"operation_class": "arbitrary_sql"})
