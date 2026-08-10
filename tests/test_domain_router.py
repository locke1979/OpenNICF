from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from opennicf import (
    DomainRouter,
    KnowledgePlatform,
    ModelGateway,
    ModelRouter,
    ProviderConfig,
    RouterConfig,
)


class _MockProviderServer:
    def __init__(
        self,
        *,
        models_status: int = 200,
        models_payload: dict | None = None,
        chat_status: int = 200,
        chat_payload: dict | None = None,
    ) -> None:
        self.behavior = {
            "models_status": models_status,
            "models_payload": models_payload or {"data": [{"id": "mock-local"}]},
            "chat_status": chat_status,
            "chat_payload": chat_payload
            or {
                "id": "chatcmpl-1",
                "model": "mock-model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
            "calls": [],
        }
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _write_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                outer.behavior["calls"].append(("GET", self.path, None))
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                self._write_json(outer.behavior["models_status"], outer.behavior["models_payload"])

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                parsed = json.loads(raw.decode("utf-8")) if raw else None
                outer.behavior["calls"].append(("POST", self.path, parsed))
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                self._write_json(outer.behavior["chat_status"], outer.behavior["chat_payload"])

            def log_message(self, *_args) -> None:
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


@dataclass(frozen=True)
class _FakeProfile:
    domain_id: str
    name: str
    description: str
    owned_systems: tuple[str, ...] = ()
    owned_components: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    system_aliases: tuple[str, ...] = ()
    integration_boundary_aliases: tuple[str, ...] = ()


class _BarrierTools:
    def __init__(self, profile: _FakeProfile, *, barrier: threading.Barrier | None = None, events: list[tuple[str, str]] | None = None) -> None:
        self.profile = profile
        self._barrier = barrier
        self._events = events if events is not None else []

    def _search(self, request, *, limit=None):
        if self._barrier is not None:
            self._events.append(("start", self.profile.domain_id))
            self._barrier.wait(timeout=5)
            self._events.append(("end", self.profile.domain_id))
        query = request.get("query") if isinstance(request, dict) else str(request)
        evidence_ref = {
            "provenance_ref": f"{self.profile.domain_id}:{query or 'evidence'}",
            "statement": f"{self.profile.domain_id} evidence",
            "metadata": {
                "correlation_ids": list(request.get("correlation_ids", [])) if isinstance(request, dict) else [],
                "timestamps": ["2026-08-10T00:00:00Z"],
            },
        }
        return {
            "retrieval_mode": "domain_evidence",
            "filters": {},
            "package_count": 1,
            "estimated_tokens": 16,
            "estimated_bytes": 256,
            "packages": [
                {
                    "package_id": f"pkg-{self.profile.domain_id}",
                    "evidence_refs": [evidence_ref],
                    "metadata": {"integration_edges": []},
                }
            ],
        }

    def search_code_exact(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_code_symbols(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_code_semantic(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_logs(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_docs(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_schema(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_query_outputs(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def search_domain_evidence(self, request, *, limit=None):
        return self._search(request, limit=limit)

    def request_domain_diagnostic(self, request):
        return {"status": "ok", "request": request}


class _FakeAgent:
    def __init__(self, profile: _FakeProfile, *, barrier: threading.Barrier | None = None, events: list[tuple[str, str]] | None = None) -> None:
        self.profile = profile
        self.tools = _BarrierTools(profile, barrier=barrier, events=events)

    def run(self, request: str):
        return {"domain_id": self.profile.domain_id, "request": request}


class _StubDomainFactory:
    def __init__(self, profiles: dict[str, _FakeProfile], *, barrier: threading.Barrier | None = None, events: list[tuple[str, str]] | None = None) -> None:
        self.profiles = profiles
        self._barrier = barrier
        self._events = events if events is not None else []
        self.create_calls: list[tuple[str, str | None]] = []

    def available_domains(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def profile_for(self, domain_id: str) -> _FakeProfile:
        resolved = self.resolve_domain_id(domain_id)
        return self.profiles[resolved]

    def resolve_domain_id(self, domain_id: str) -> str:
        candidate = str(domain_id).strip()
        if candidate in self.profiles:
            return candidate
        lowered = candidate.lower()
        for profile in self.profiles.values():
            aliases = {
                *(alias.lower() for alias in profile.aliases),
                *(alias.lower() for alias in profile.system_aliases),
                *(alias.lower() for alias in profile.integration_boundary_aliases),
                *(system.lower() for system in profile.owned_systems),
            }
            if lowered in aliases:
                return profile.domain_id
        raise KeyError(f"unknown domain or system alias: {domain_id}")

    def create(self, domain_id: str, privacy=None):
        resolved = self.resolve_domain_id(domain_id)
        self.create_calls.append((resolved, getattr(privacy, "value", privacy)))
        return _FakeAgent(self.profiles[resolved], barrier=self._barrier, events=self._events)


def _profiles() -> dict[str, _FakeProfile]:
    return {
        "contencioso_judicial": _FakeProfile(
            domain_id="contencioso_judicial",
            name="Contencioso Judicial",
            description="Judicial litigation",
            owned_systems=("SICJUT",),
            aliases=("sicjut",),
            system_aliases=("SICJUT",),
        ),
        "contencioso_administrativo": _FakeProfile(
            domain_id="contencioso_administrativo",
            name="Contencioso Administrativo",
            description="Administrative litigation",
            owned_systems=("SICAT", "SIGEPRA"),
            aliases=("sicat", "sigepra"),
            system_aliases=("SICAT", "SIGEPRA"),
        ),
        "contraordenacional": _FakeProfile(
            domain_id="contraordenacional",
            name="Contraordenacional",
            description="Administrative infractions",
            owned_systems=("SCO",),
            aliases=("sco",),
            system_aliases=("SCO",),
        ),
        "criminal": _FakeProfile(
            domain_id="criminal",
            name="Criminal",
            description="Criminal domain",
            owned_systems=("SINQUER",),
            aliases=("sinquer",),
            system_aliases=("SINQUER",),
        ),
        "encargos_sigef": _FakeProfile(
            domain_id="encargos_sigef",
            name="Encargos SIGEF",
            description="SIGEF domain",
            owned_systems=("SIGEF",),
            aliases=("sigef",),
            system_aliases=("SIGEF",),
        ),
    }


@contextmanager
def _router(
    *,
    chat_payload: dict | None = None,
    barrier: threading.Barrier | None = None,
    events: list[tuple[str, str]] | None = None,
):
    local = _MockProviderServer(
        models_payload={"data": [{"id": "mock-local"}]},
        chat_payload=chat_payload,
    )
    remote = _MockProviderServer(
        models_payload={"data": [{"id": "mock-remote"}]},
        chat_payload={
            "id": "chatcmpl-remote",
            "model": "mock-remote",
            "choices": [{"message": {"role": "assistant", "content": "remote"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    gateway = ModelGateway(
        router=ModelRouter(RouterConfig(local_context_limit=8192)),
        local_provider=ProviderConfig(name="lmstudio", base_url=local.url, model="mock-local"),
        remote_provider=ProviderConfig(name="litellm_oci", base_url=remote.url, model="mock-remote"),
    )
    factory = _StubDomainFactory(_profiles(), barrier=barrier, events=events)
    router = DomainRouter(
        gateway=gateway,
        knowledge=KnowledgePlatform.in_memory(),
        domain_factory=factory,
    )
    try:
        yield router, gateway, local, remote, factory
    finally:
        local.close()
        remote.close()


@pytest.mark.parametrize(
    ("route_input", "expected_domain"),
    [
        ("SICJUT", "contencioso_judicial"),
        ("SICAT SIGEPRA", "contencioso_administrativo"),
        ("SCO", "contraordenacional"),
        ("SINQUER", "criminal"),
        ("SIGEF", "encargos_sigef"),
    ],
)
def test_strong_aliases_route_to_single_domains_without_model_fallback(route_input: str, expected_domain: str) -> None:
    with _router() as (router, _gateway, local, remote, factory):
        route = router.route(route_input)

    assert route["route_kind"] == "single_domain"
    assert route["route_source"] == "deterministic_alias"
    assert route["selected_domain_ids"] == [expected_domain]
    assert route["delegate_domain_id"] == expected_domain
    assert route["classification"] == {}
    assert local.behavior["calls"] == []
    assert remote.behavior["calls"] == []
    assert factory.create_calls == [(expected_domain, "local_preferred")]


def test_ambiguous_classification_uses_the_local_first_gateway_and_routes_multi_domain() -> None:
    chat_payload = {
        "id": "chatcmpl-classification",
        "model": "mock-local",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"domain_ids": ["criminal", "contraordenacional"]}),
                }
            }
        ],
        "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
    }
    with _router(chat_payload=chat_payload) as (router, _gateway, local, remote, _factory):
        route = router.route("Need a consolidated view of the unresolved matter.")

    assert route["route_source"] == "classification"
    assert route["route_kind"] == "integration_correlation"
    assert set(route["selected_domain_ids"]) == {"criminal", "contraordenacional"}
    assert route["result"]["status"] == "ok"
    assert set(route["result"]["selected_domain_ids"]) == {"criminal", "contraordenacional"}
    assert route["classification"]["provider"] == "lmstudio"
    assert not any(call[0] == "POST" and call[1] == "/v1/chat/completions" and call[2].get("model") == "mock-remote" for call in remote.behavior["calls"])
    assert any(call[0] == "POST" and call[1] == "/v1/chat/completions" for call in local.behavior["calls"])


def test_parallel_dispatch_keeps_delegate_calls_overlapping() -> None:
    barrier = threading.Barrier(2)
    events: list[tuple[str, str]] = []
    with _router(barrier=barrier, events=events) as (router, _gateway, _local, _remote, _factory):
        route = router.route({"question": "Cross-domain SCO and SINQUER matter", "systems": ["SCO", "SINQUER"]})

    assert route["route_kind"] == "integration_correlation"
    assert set(route["selected_domain_ids"]) == {"contraordenacional", "criminal"}
    assert route["result"]["status"] == "ok"
    assert {event for event, _ in events[:2]} == {"start"}
    assert {domain for _, domain in events[:2]} == {"contraordenacional", "criminal"}
    assert {event for event, _ in events[-2:]} == {"end"}


def test_three_domain_sigef_sicjut_sigepra_routes_through_the_coordinator() -> None:
    barrier = threading.Barrier(3)
    events: list[tuple[str, str]] = []
    with _router(barrier=barrier, events=events) as (router, _gateway, _local, _remote, _factory):
        route = router.route({"question": "SIGEF, SICJUT, and SIGEPRA integration", "systems": ["SIGEF", "SICJUT", "SIGEPRA"]})

    assert route["route_kind"] == "integration_correlation"
    assert set(route["selected_domain_ids"]) == {
        "encargos_sigef",
        "contencioso_judicial",
        "contencioso_administrativo",
    }
    assert route["result"]["status"] == "ok"
    assert {domain for event, domain in events if event == "start"} == {
        "encargos_sigef",
        "contencioso_judicial",
        "contencioso_administrativo",
    }
