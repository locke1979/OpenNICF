"""Provider-neutral, space-aware embedding adapters.

The public contract deliberately speaks in retrieval purposes rather than
provider prompts.  A vector's length is not an identity: every result carries
an :class:`EmbeddingSpace` and callers must select the matching space before
searching an index.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from hashlib import blake2b
from math import sqrt
from re import findall
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CANONICAL_DIMENSION = 768
RETRIEVAL_PURPOSES = {"retrieval_query", "retrieval_document"}


class EmbeddingError(RuntimeError):
    """Base error for embedding-provider failures."""


class EmbeddingSpaceMismatch(EmbeddingError):
    """Raised before comparing vectors from different semantic spaces."""


class PrivacyBoundaryError(EmbeddingError):
    """Raised when a local-only item is routed to a cloud provider."""


@dataclass(frozen=True)
class EmbeddingSpace:
    embedding_space_id: str
    provider: str
    model: str
    model_revision: str
    dimension: int = CANONICAL_DIMENSION
    normalized: bool = True
    purpose: str = "retrieval_document"
    modality: str = "text"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if self.purpose not in RETRIEVAL_PURPOSES:
            raise ValueError(f"unsupported retrieval purpose: {self.purpose}")

    def compatible_with(self, other: EmbeddingSpace) -> bool:
        return (
            self.embedding_space_id == other.embedding_space_id
            and self.dimension == other.dimension
        )


def _space_id(model: str, dimension: int, revision: str = "v1") -> str:
    return f"{model}:{dimension}:{revision}"


@dataclass(frozen=True)
class EmbeddingModelInfo:
    model: str
    dimensions: int
    device: str
    backend: str
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_space_id: str = ""
    provider: str = "LOCAL"
    model_revision: str = "v1"
    normalized: bool = True


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    dimensions: int
    device: str
    vectors: tuple[tuple[float, ...], ...]
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_space_id: str = ""
    provider: str = "LOCAL"
    model_revision: str = "v1"
    normalized: bool = True
    purpose: str = "retrieval_document"

    @property
    def space(self) -> EmbeddingSpace:
        return EmbeddingSpace(
            self.embedding_space_id
            or _space_id(self.model, self.dimensions, self.model_revision),
            self.provider,
            self.model,
            self.model_revision,
            self.dimensions,
            self.normalized,
            self.purpose,
        )


class EmbeddingBackend(Protocol):
    @property
    def info(self) -> EmbeddingModelInfo: ...

    def embed(
        self, texts: Sequence[str], *, purpose: str = "retrieval_document"
    ) -> EmbeddingResult: ...


def _normalize(vector: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values) if norm else values


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in findall(r"[A-Za-z0-9_./:-]+", text)]


@dataclass(frozen=True)
class EmbeddingBatchSizer:
    dimensions: int
    bytes_per_value: int = 4
    safety_factor: float = 4.0
    minimum_batch: int = 1

    def limit_for(
        self,
        *,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
    ) -> int:
        budget = (
            available_vram_bytes
            if available_vram_bytes is not None
            else available_memory_bytes
        )
        if budget is None or budget <= 0:
            return self.minimum_batch
        per_item = max(
            1, int(self.dimensions * self.bytes_per_value * self.safety_factor)
        )
        return max(self.minimum_batch, budget // per_item)


class HashingEmbeddingBackend:
    """Deterministic, dependency-free CPU backend used for safe fallback/tests."""

    def __init__(
        self,
        model: str | None = None,
        dimensions: int = CANONICAL_DIMENSION,
        device: str = "cpu",
    ):
        model = model or f"opennicf-hash-{dimensions}"
        self._info = EmbeddingModelInfo(
            model=model,
            dimensions=dimensions,
            device=device,
            backend="hashing",
            fallback=True,
            embedding_space_id=_space_id(model, dimensions),
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def _vector_for(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._info.dimensions
        for token in _tokenize(text):
            digest = blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._info.dimensions
            values[bucket] += 1.0 + (int.from_bytes(digest[4:8], "big") / 2**32)
        return _normalize(values)

    def embed(
        self, texts: Sequence[str], *, purpose: str = "retrieval_document"
    ) -> EmbeddingResult:
        return EmbeddingResult(
            model=self._info.model,
            dimensions=self._info.dimensions,
            device=self._info.device,
            vectors=tuple(self._vector_for(text) for text in texts),
            fallback=True,
            metadata={
                "fallback_reason": "deterministic_cpu_backend",
                "purpose": purpose,
            },
            embedding_space_id=self._info.embedding_space_id,
            provider="LOCAL",
            model_revision=self._info.model_revision,
            normalized=True,
            purpose=purpose,
        )


class QwenEmbeddingBackend:
    """Lazy Qwen3 adapter with runtime CUDA detection and injectable loader.

    The loader hook keeps CI and constrained hosts deterministic.  If the
    optional transformers/torch stack is unavailable, the service selects its
    explicit CPU fallback instead of pretending that hashing is Qwen output.
    """

    def __init__(
        self,
        *,
        dimension: int = CANONICAL_DIMENSION,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        model_revision: str = "v1",
        max_input_length: int = 512,
        batch_size: int = 2,
        loader: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.dimension = dimension
        self.max_input_length = max_input_length
        self.batch_size = max(1, batch_size)
        self._loader = loader
        self._runtime: Any = None
        self._info = EmbeddingModelInfo(
            model=model,
            dimensions=dimension,
            device=self._detect_device(),
            backend="qwen3",
            embedding_space_id=_space_id(model, dimension, model_revision),
            model_revision=model_revision,
        )

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - optional CUDA dependency detection
            return "cpu"

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def _prepare(self, texts: Sequence[str], purpose: str) -> list[str]:
        if purpose == "retrieval_query":
            instruction = "Instruct: Given a query, retrieve relevant evidence\nQuery: "
            return [instruction + text[: self.max_input_length * 8] for text in texts]
        return [text[: self.max_input_length * 8] for text in texts]

    def _load(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self._loader is not None:
            self._runtime = self._loader
            return self._runtime
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            device = self._info.device
            dtype = torch.float16 if device == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = (
                AutoModel.from_pretrained(self.model, torch_dtype=dtype)
                .to(device)
                .eval()
            )
            self._runtime = (tokenizer, model, torch, device)
            return self._runtime
        except Exception as exc:
            raise EmbeddingError("Qwen runtime is unavailable") from exc

    def embed(
        self, texts: Sequence[str], *, purpose: str = "retrieval_document"
    ) -> EmbeddingResult:
        prepared = self._prepare(texts, purpose)
        runtime = self._load()
        if callable(runtime):
            raw = runtime(
                prepared,
                purpose=purpose,
                dimension=self.dimension,
                device=self._info.device,
            )
            vectors = tuple(_normalize(vector) for vector in raw)
        else:
            tokenizer, model, torch, device = runtime
            with torch.inference_mode():
                batch = tokenizer(
                    prepared,
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vectors = tuple(
                    _normalize(row[: self.dimension].float().cpu().tolist())
                    for row in pooled
                )
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError("Qwen backend returned the wrong dimensionality")
        return EmbeddingResult(
            model=self.model,
            dimensions=self.dimension,
            device=self._info.device,
            vectors=vectors,
            embedding_space_id=self._info.embedding_space_id,
            provider="LOCAL",
            model_revision=self._info.model_revision,
            normalized=True,
            purpose=purpose,
            metadata={
                "instruction_applied": purpose == "retrieval_query",
                "max_input_length": self.max_input_length,
            },
        )


class GeminiEmbeddingBackend:
    """Google AI Studio REST adapter; credentials are read only at call time."""

    def __init__(
        self,
        model: str,
        *,
        dimension: int = CANONICAL_DIMENSION,
        model_revision: str = "v1",
        api_key: str | None = None,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ):
        if model not in {"gemini-embedding-001", "gemini-embedding-2"}:
            raise ValueError("unsupported Gemini embedding model")
        self.model, self.dimension, self._api_key, self._transport = (
            model,
            dimension,
            api_key,
            transport,
        )
        self._info = EmbeddingModelInfo(
            model=model,
            dimensions=dimension,
            device="remote",
            backend="google-gemini",
            embedding_space_id=_space_id(model, dimension, model_revision),
            provider="GOOGLE",
            model_revision=model_revision,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def _call(self, text: str, purpose: str) -> Sequence[float]:
        key = (
            self._api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not key and self._transport is None:
            raise EmbeddingError("Google embedding credentials are not configured")
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
            "output_dimensionality": self.dimension,
            "task_type": "RETRIEVAL_QUERY"
            if purpose == "retrieval_query"
            else "RETRIEVAL_DOCUMENT",
        }
        if self._transport:
            response = self._transport(self.model, payload)
        else:
            request = Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={key}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=30) as result:  # nosec B310 - fixed Google endpoint
                    response = json.loads(result.read().decode())
            except (HTTPError, URLError) as exc:
                raise EmbeddingError("Google embedding request failed") from exc
        values = response.get("embedding", {}).get("values") or response.get(
            "embeddings", [{}]
        )[0].get("values")
        if not values:
            raise EmbeddingError("Google embedding response did not contain values")
        return values

    def embed(
        self, texts: Sequence[str], *, purpose: str = "retrieval_document"
    ) -> EmbeddingResult:
        vectors = tuple(_normalize(self._call(text, purpose)) for text in texts)
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingError("Google embedding returned the wrong dimensionality")
        return EmbeddingResult(
            model=self.model,
            dimensions=self.dimension,
            device="remote",
            vectors=vectors,
            embedding_space_id=self._info.embedding_space_id,
            provider="GOOGLE",
            model_revision=self._info.model_revision,
            normalized=True,
            purpose=purpose,
            metadata={"normalization": "verified_and_enforced", "task_type": purpose},
        )


class HttpEmbeddingBackend:
    """Provider-neutral client for the OpenNICF local embedding worker."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        service_token: str | None = None,
        timeout: float = 30.0,
        model: str = "Qwen/Qwen3-Embedding-0.6B",
        dimension: int = CANONICAL_DIMENSION,
        model_revision: str = "v1",
    ):
        self.base_url = (
            base_url or os.environ.get("OPENNICF_EMBEDDING_BASE_URL", "")
        ).rstrip("/")
        self.service_token = service_token or os.environ.get(
            "OPENNICF_EMBEDDING_SERVICE_TOKEN"
        )
        self.timeout = timeout
        self.dimension = dimension
        self._info = EmbeddingModelInfo(
            model=model,
            dimensions=dimension,
            device="remote-local",
            backend="opennicf-http",
            embedding_space_id=_space_id(model, dimension, model_revision),
            provider="LOCAL",
            model_revision=model_revision,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def _request(
        self, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.base_url:
            raise EmbeddingError("OPENNICF_EMBEDDING_BASE_URL is not configured")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.service_token:
            headers["Authorization"] = f"Bearer {self.service_token}"
        request = Request(
            self.base_url + path,
            headers=headers,
            data=json.dumps(payload).encode() if payload is not None else None,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise EmbeddingError("local embedding service request failed") from exc

    def embed(
        self, texts: Sequence[str], *, purpose: str = "retrieval_document"
    ) -> EmbeddingResult:
        payload = self._request(
            "/v1/embeddings",
            {"input": list(texts), "purpose": purpose, "dimension": self.dimension},
        )
        vectors = tuple(
            _normalize(item["embedding"] if isinstance(item, dict) else item)
            for item in payload.get("data", [])
        )
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimension for vector in vectors
        ):
            raise EmbeddingError(
                "local embedding service returned the wrong result shape"
            )
        return EmbeddingResult(
            model=self._info.model,
            dimensions=self.dimension,
            device="remote-local",
            vectors=vectors,
            fallback=bool(payload.get("fallback", False)),
            metadata={"service": "opennicf-embedding-worker"},
            embedding_space_id=self._info.embedding_space_id,
            provider="LOCAL",
            model_revision=self._info.model_revision,
            normalized=True,
            purpose=purpose,
        )


class LocalFirstEmbeddingService:
    """Provider-neutral registry with explicit active-space and privacy guards."""

    def __init__(
        self,
        *,
        preferred_backend: EmbeddingBackend | None = None,
        cpu_backend: EmbeddingBackend | None = None,
        allow_cpu_fallback: bool = True,
        batch_sizer: EmbeddingBatchSizer | None = None,
        providers: Sequence[EmbeddingBackend] | None = None,
        active_space_id: str | None = None,
    ):
        self.preferred_backend = preferred_backend
        self.allow_cpu_fallback = allow_cpu_fallback
        self.cpu_backend = cpu_backend or HashingEmbeddingBackend()
        self.batch_sizer = batch_sizer or EmbeddingBatchSizer(
            dimensions=self.cpu_backend.info.dimensions
        )
        self.providers: dict[str, EmbeddingBackend] = {
            self._provider_space_id(self.cpu_backend): self.cpu_backend
        }
        for provider in providers or ():
            self.providers[self._provider_space_id(provider)] = provider
        if preferred_backend:
            self.providers[self._provider_space_id(preferred_backend)] = (
                preferred_backend
            )
        self.active_space_id = active_space_id or (
            self._provider_space_id(preferred_backend)
            if preferred_backend
            else self._provider_space_id(self.cpu_backend)
        )
        self.migration_state: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _provider_space_id(provider: EmbeddingBackend) -> str:
        info = provider.info
        return info.embedding_space_id or _space_id(
            info.model, info.dimensions, info.model_revision
        )

    def register(self, provider: EmbeddingBackend) -> EmbeddingSpace:
        space_id = self._provider_space_id(provider)
        self.providers[space_id] = provider
        return self.space(space_id)

    def space(self, space_id: str | None = None) -> EmbeddingSpace:
        provider = self.providers[space_id or self.active_space_id]
        info = provider.info
        return EmbeddingSpace(
            self._provider_space_id(provider),
            info.provider,
            info.model,
            info.model_revision,
            info.dimensions,
            info.normalized,
        )

    def switch_active(self, space_id: str, *, corpus_ready: bool = False) -> None:
        if space_id not in self.providers:
            raise EmbeddingError(f"embedding space is not registered: {space_id}")
        if not corpus_ready:
            raise EmbeddingError(
                "active embedding space cannot switch before its corpus/index is validated"
            )
        self.active_space_id = space_id

    def _choose_backend(
        self, *, space_id: str | None, prefer_gpu: bool
    ) -> EmbeddingBackend:
        if space_id and space_id in self.providers:
            return self.providers[space_id]
        if prefer_gpu and self.preferred_backend is not None:
            return self.preferred_backend
        return self.cpu_backend

    def info(
        self,
        *,
        prefer_gpu: bool = True,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
    ) -> EmbeddingModelInfo:
        return self._choose_backend(space_id=None, prefer_gpu=prefer_gpu).info

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: str = "retrieval_document",
        dimension: int | None = None,
        prefer_gpu: bool = True,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
        space_id: str | None = None,
        privacy_policy: str = "cloud_allowed",
    ) -> EmbeddingResult:
        if purpose not in RETRIEVAL_PURPOSES:
            raise ValueError(f"purpose must be one of {sorted(RETRIEVAL_PURPOSES)}")
        backend = self._choose_backend(
            space_id=space_id or self.active_space_id, prefer_gpu=prefer_gpu
        )
        if privacy_policy == "local_only" and backend.info.provider != "LOCAL":
            raise PrivacyBoundaryError(
                "local_only evidence cannot use a Google embedding space"
            )
        if dimension is not None and dimension != backend.info.dimensions:
            raise EmbeddingError(
                "provider was not configured for the requested dimension"
            )
        limit = min(
            self.batch_sizer.limit_for(
                available_memory_bytes=available_memory_bytes,
                available_vram_bytes=available_vram_bytes if prefer_gpu else None,
            ),
            max(1, getattr(backend, "batch_size", len(texts) or 1)),
        )
        vectors: list[tuple[float, ...]] = []
        fallback = False
        metadata: dict[str, Any] = {}
        for offset in range(0, len(texts), limit):
            batch = texts[offset : offset + limit]
            try:
                try:
                    result = backend.embed(batch, purpose=purpose)
                except TypeError as exc:
                    # Preserve compatibility with pre-#43 test/dry-run backends.
                    if "purpose" not in str(exc):
                        raise
                    result = backend.embed(batch)
            except (EmbeddingError, MemoryError, RuntimeError):
                if not self.allow_cpu_fallback:
                    raise
                # Retry the batch at size one; a second failure is contained by CPU fallback.
                if len(batch) > 1:
                    for text in batch:
                        try:
                            vectors.extend(
                                backend.embed([text], purpose=purpose).vectors
                            )
                        except (EmbeddingError, MemoryError, RuntimeError):
                            vectors.extend(
                                self.cpu_backend.embed([text], purpose=purpose).vectors
                            )
                            fallback = True
                else:
                    vectors.extend(
                        self.cpu_backend.embed(batch, purpose=purpose).vectors
                    )
                    fallback = True
                metadata["oom_retry"] = True
                continue
            vectors.extend(result.vectors)
            metadata.update(result.metadata)
            fallback = fallback or result.fallback
        info = backend.info if not fallback else self.cpu_backend.info
        result_space_id = (
            self._provider_space_id(backend)
            if not fallback
            else self._provider_space_id(self.cpu_backend)
        )
        if fallback:
            metadata.setdefault(
                "fallback_reason", "preferred_embedding_runtime_unavailable_or_oom"
            )
            metadata["fallback_space_id"] = result_space_id
        return EmbeddingResult(
            model=info.model,
            dimensions=info.dimensions,
            device=info.device,
            vectors=tuple(vectors),
            fallback=fallback or info.fallback,
            metadata=metadata,
            embedding_space_id=result_space_id,
            provider=info.provider,
            model_revision=info.model_revision,
            normalized=info.normalized,
            purpose=purpose,
        )

    def describe(self) -> dict[str, Any]:
        info = self.info()
        return {
            "model": info.model,
            "dimensions": info.dimensions,
            "device": info.device,
            "backend": info.backend,
            "fallback": info.fallback,
            "active_space_id": self.active_space_id,
            "spaces": sorted(self.providers),
        }


class EmbeddingMigration:
    """Small resumable/idempotent migration coordinator.

    The caller supplies a chunk source and persistence callback so migration
    never mutates immutable source/artifact/chunk records.  Activation is only
    permitted after the caller reports that the target index is validated.
    """

    def __init__(
        self,
        service: LocalFirstEmbeddingService,
        target_space_id: str,
        *,
        batch_size: int = 16,
    ):
        if target_space_id not in service.providers:
            raise EmbeddingError(
                f"target embedding space is not registered: {target_space_id}"
            )
        self.service, self.target_space_id, self.batch_size = (
            service,
            target_space_id,
            max(1, batch_size),
        )
        self.state = service.migration_state.setdefault(
            target_space_id, {"completed": [], "status": "registered"}
        )

    def run(
        self,
        chunks: Sequence[Any],
        persist: Callable[[Any, EmbeddingResult], None],
        *,
        privacy_policy: str = "cloud_allowed",
    ) -> dict[str, Any]:
        completed = set(self.state.get("completed", []))
        for offset in range(0, len(chunks), self.batch_size):
            batch = [
                chunk
                for chunk in chunks[offset : offset + self.batch_size]
                if getattr(chunk, "chunk_id", chunk) not in completed
            ]
            if not batch:
                continue
            result = self.service.embed(
                [getattr(chunk, "text", str(chunk)) for chunk in batch],
                purpose="retrieval_document",
                space_id=self.target_space_id,
                privacy_policy=privacy_policy,
            )
            for chunk, vector in zip(batch, result.vectors):
                persist(
                    chunk,
                    EmbeddingResult(
                        model=result.model,
                        dimensions=result.dimensions,
                        device=result.device,
                        vectors=(vector,),
                        fallback=result.fallback,
                        metadata=result.metadata,
                        embedding_space_id=result.embedding_space_id,
                        provider=result.provider,
                        model_revision=result.model_revision,
                        normalized=result.normalized,
                        purpose=result.purpose,
                    ),
                )
                completed.add(getattr(chunk, "chunk_id", chunk))
                self.state["completed"] = sorted(completed)
        self.state["status"] = "complete" if len(completed) == len(chunks) else "paused"
        self.state["processed"] = len(completed)
        self.state["total"] = len(chunks)
        return dict(self.state)

    def activate(self) -> None:
        if self.state.get("status") != "complete":
            raise EmbeddingError(
                "cannot switch active space before migration is complete"
            )
        self.service.switch_active(self.target_space_id, corpus_ready=True)
        self.state["status"] = "active"
