from opennicf.audit import FailureAuditEngine
from opennicf.knowledge import (
    HashingEmbeddingBackend,
    KnowledgePlatform,
    LocalFirstEmbeddingService,
    MemoryKnowledgeStore,
    MemoryObjectStore,
    RetrievalFilters,
)


def _platform():
    return KnowledgePlatform(
        MemoryKnowledgeStore(),
        MemoryObjectStore(),
        LocalFirstEmbeddingService(cpu_backend=HashingEmbeddingBackend(dimensions=8)),
    )


def test_code_index_covers_supported_languages_and_relationships():
    platform = _platform()
    fixtures = {
        "python": ("service.py", "class Service:\n    def route(self):\n        logging.error('failed')\n        return self.load()\n    def load(self):\n        try:\n            return os.environ['API_KEY']\n        except Exception:\n            return None\n"),
        "java": ("Service.java", "class Service { void route() { load(); } void load() {} }"),
        "sql": ("schema.sql", "CREATE TABLE accounts (id integer);\nCREATE FUNCTION refresh() RETURNS void AS $$ SELECT 1 $$ LANGUAGE SQL;"),
        "powershell": ("deploy.ps1", "function Invoke-Deploy { Write-Error $env:API_KEY }"),
    }
    for language, (uri, content) in fixtures.items():
        platform.ingest(source_id=language, source_uri=uri, content=content, source_type="code", domain="audit", acl_scope="internal")

    symbols = platform.search_code_symbols("", filters=RetrievalFilters(principal_acl_scopes=frozenset({"internal"}), domain_ids=("audit",), limit=100))
    assert {symbol.metadata["language"] for symbol in symbols} == set(fixtures)
    assert all(symbol.source_hash and symbol.parser_version and symbol.acl_scope == "internal" for symbol in symbols)
    relationships = platform.store.relationships
    assert any(item.relation_type == "calls" and item.target_name == "load" for item in relationships.values())
    assert any(item.relation_type == "logging" for item in relationships.values())
    assert any(item.relation_type == "exception_handling" for item in relationships.values())


def test_code_index_snapshot_restart_and_acl_isolation():
    platform = _platform()
    platform.ingest(source_id="private", source_uri="private.py", content="def secret():\n    return 1", source_type="code", acl_scope="private", domain="audit")
    snapshot = platform.backup()
    restored = _platform()
    restored.restore(snapshot)
    assert restored.search_code_symbols("secret", filters=RetrievalFilters(principal_acl_scopes=frozenset({"public"}))) == []
    assert restored.search_code_symbols("secret", filters=RetrievalFilters(principal_acl_scopes=frozenset({"private"})))


def test_failure_audit_uses_persistent_code_symbols():
    platform = _platform()
    platform.ingest(source_id="audit-code", source_uri="payment.py", content="def charge():\n    raise RuntimeError('failed')", source_type="code", acl_scope="internal", domain="audit")
    report = FailureAuditEngine(platform).analyze({"query": "charge", "source_types": ["code"]})
    assert any("Persistent code index resolves symbols" in finding.statement for finding in report.findings)
