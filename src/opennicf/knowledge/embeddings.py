"""Local-first embedding backends with CPU fallback metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from math import sqrt
from re import findall
from typing import Any, Iterable, Protocol, Sequence


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in findall(r"[A-Za-z0-9_./:-]+", text)]


@dataclass(frozen=True)
class EmbeddingModelInfo:
    model: str
    dimensions: int
    device: str
    backend: str
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    dimensions: int
    device: str
    vectors: tuple[tuple[float, ...], ...]
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingBackend(Protocol):
    @property
    def info(self) -> EmbeddingModelInfo:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError


@dataclass(frozen=True)
class EmbeddingBatchSizer:
    dimensions: int
    bytes_per_value: int = 4
    safety_factor: float = 4.0
    minimum_batch: int = 1

    def limit_for(self, *, available_memory_bytes: int | None = None, available_vram_bytes: int | None = None) -> int:
        budget = available_vram_bytes if available_vram_bytes is not None else available_memory_bytes
        if budget is None or budget <= 0:
            return self.minimum_batch
        per_item = max(1, int(self.dimensions * self.bytes_per_value * self.safety_factor))
        return max(self.minimum_batch, budget // per_item)


class HashingEmbeddingBackend:
    """Deterministic CPU fallback backend that needs no model weights."""

    def __init__(self, model: str | None = None, dimensions: int = 256, device: str = "cpu"):
        model = model or f"opennicf-hash-{dimensions}"
        self._info = EmbeddingModelInfo(model=model, dimensions=dimensions, device=device, backend="hashing", fallback=True)

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def _vector_for(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._info.dimensions
        for token in _tokenize(text):
            digest = blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._info.dimensions
            weight = 1.0 + (int.from_bytes(digest[4:8], "big") / 2**32)
            values[bucket] += weight
        norm = sqrt(sum(value * value for value in values)) or 1.0
        return tuple(value / norm for value in values)

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            model=self._info.model,
            dimensions=self._info.dimensions,
            device=self._info.device,
            vectors=tuple(self._vector_for(text) for text in texts),
            fallback=self._info.fallback,
            metadata=dict(self._info.metadata),
        )


class LocalFirstEmbeddingService:
    """Chooses a preferred local backend when available, otherwise CPU fallback."""

    def __init__(
        self,
        *,
        preferred_backend: EmbeddingBackend | None = None,
        cpu_backend: EmbeddingBackend | None = None,
        batch_sizer: EmbeddingBatchSizer | None = None,
    ):
        self.preferred_backend = preferred_backend
        self.cpu_backend = cpu_backend or HashingEmbeddingBackend()
        self.batch_sizer = batch_sizer or EmbeddingBatchSizer(dimensions=self.cpu_backend.info.dimensions)

    def _choose_backend(
        self,
        *,
        prefer_gpu: bool,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
    ) -> EmbeddingBackend:
        if not prefer_gpu or self.preferred_backend is None:
            return self.cpu_backend
        if available_vram_bytes is not None and available_vram_bytes <= 0:
            return self.cpu_backend
        return self.preferred_backend

    def info(
        self,
        *,
        prefer_gpu: bool = True,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
    ) -> EmbeddingModelInfo:
        return self._choose_backend(
            prefer_gpu=prefer_gpu,
            available_memory_bytes=available_memory_bytes,
            available_vram_bytes=available_vram_bytes,
        ).info

    def embed(
        self,
        texts: Sequence[str],
        *,
        prefer_gpu: bool = True,
        available_memory_bytes: int | None = None,
        available_vram_bytes: int | None = None,
    ) -> EmbeddingResult:
        backend = self._choose_backend(
            prefer_gpu=prefer_gpu,
            available_memory_bytes=available_memory_bytes,
            available_vram_bytes=available_vram_bytes,
        )
        limit = self.batch_sizer.limit_for(
            available_memory_bytes=available_memory_bytes,
            available_vram_bytes=available_vram_bytes if prefer_gpu else available_memory_bytes,
        )
        vectors: list[tuple[float, ...]] = []
        for offset in range(0, len(texts), limit):
            batch = texts[offset : offset + limit]
            result = backend.embed(batch)
            vectors.extend(result.vectors)
        info = backend.info
        return EmbeddingResult(
            model=info.model,
            dimensions=info.dimensions,
            device=info.device,
            vectors=tuple(vectors),
            fallback=info.fallback or backend is self.cpu_backend,
            metadata=dict(info.metadata),
        )

    def describe(self) -> dict[str, Any]:
        info = self.info()
        return {
            "model": info.model,
            "dimensions": info.dimensions,
            "device": info.device,
            "backend": info.backend,
            "fallback": info.fallback,
        }
