import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from opennicf.knowledge import KnowledgePlatform
from opennicf.knowledge.service import KnowledgeService, KnowledgeServiceConfig, _jsonable, serve


def _running_service(tmp_path):
    platform = KnowledgePlatform.in_memory()
    platform.ingest(source_id="public", source_uri="manual://public.txt", content="public evidence", acl_scope="public", domain_id="criminal")
    platform.ingest(source_id="internal", source_uri="manual://internal.txt", content="internal evidence", acl_scope="internal", domain_id="criminal")
    config = KnowledgeServiceConfig(port=0, object_store_root=str(tmp_path), max_result_items=2, operation_limit=2)
    service = KnowledgeService(platform, config)
    service.ready = True
    server = serve(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return service, server, thread


def _request(server, path, *, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{server.server_address[1]}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


def test_health_sources_and_provenance_are_bounded_and_acl_scoped(tmp_path):
    _service, server, thread = _running_service(tmp_path)
    try:
        assert _request(server, "/health/live")[0] == 200
        assert _request(server, "/health/ready")[0] == 200
        _, body = _request(server, "/v1/sources?acl_scope=public&domain_id=criminal&limit=99")
        assert [row["source_id"] for row in body["sources"]] == ["public"]
        _, provenance = _request(server, "/v1/sources/public/provenance")
        assert provenance["artifact"]["artifact_hash"]
        assert len(provenance["chunks"]) <= 2
        _, artifact = _request(server, f"/v1/artifacts/{provenance['artifact']['artifact_hash']}/provenance")
        assert artifact["object_present"] is True
        with pytest.raises(HTTPError) as error:
            _request(server, "/v1/sources/internal/status?acl_scope=public")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_production_configuration_requires_durable_dependencies():
    with pytest.raises(RuntimeError, match="PostgreSQL DSN"):
        KnowledgeServiceConfig(environment="production", object_store_root="/tmp/objects").validate()
    with pytest.raises(RuntimeError, match="object-store root"):
        KnowledgeServiceConfig(environment="production", postgres_dsn="postgresql://db").validate()


def test_jsonable_preserves_identifiers_but_does_not_expose_raw_bytes():
    assert _jsonable({"source_id": b"source-1", "content": b"synthetic evidence"}) == {
        "source_id": "source-1",
        "content": {"size_bytes": 18, "sha256": "48736e58b409ed0241c8d7bed9dc188a59e13002f48e7492850bec15b59147b6"},
    }


def test_request_body_limit_and_status_report_storage(tmp_path):
    service, server, thread = _running_service(tmp_path)
    service.config = KnowledgeServiceConfig(port=0, object_store_root=str(tmp_path), max_body_bytes=4)
    try:
        with pytest.raises(HTTPError) as error:
            _request(server, "/v1/retry", method="POST", payload={"limit": 1})
        assert error.value.code == 400
        _, status = _request(server, "/status")
        assert status["storage"] == "postgresql"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
