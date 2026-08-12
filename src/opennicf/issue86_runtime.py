"""Pinned, fail-closed runtime helpers for issue #86 evaluation only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import math
from pathlib import Path
import platform
import shutil
from typing import Iterable, Sequence


QWEN_06B_REPOSITORY = "Qwen/Qwen3-Embedding-0.6B"
QWEN_06B_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_06B_WEIGHT_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
QWEN_06B_TOKENIZER_SHA256 = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_artifact(path: str | Path, expected_sha256: str, expected_size: int) -> Path:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"required artifact is absent: {artifact}")
    if artifact.stat().st_size != expected_size:
        raise ValueError(f"artifact size mismatch: {artifact}")
    if file_sha256(artifact) != expected_sha256:
        raise ValueError(f"artifact digest mismatch: {artifact}")
    return artifact


def last_token_pool(hidden_states, attention_mask):
    """Official Qwen pooling rule, kept tensor-library neutral for tests."""
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch = range(hidden_states.shape[0])
    return hidden_states[batch, sequence_lengths]


def mrl_normalize(vector: Sequence[float], dimension: int = 768) -> tuple[float, ...]:
    if not 1 <= dimension <= len(vector):
        raise ValueError("selected MRL dimension is outside the native vector")
    prefix = tuple(float(value) for value in vector[:dimension])
    if not all(math.isfinite(value) for value in prefix):
        raise ValueError("embedding contains a nonfinite value")
    norm = math.sqrt(sum(value * value for value in prefix))
    if not norm:
        raise ValueError("embedding has zero norm")
    return tuple(value / norm for value in prefix)


def query_text(text: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"


@dataclass(frozen=True)
class RuntimePreflight:
    python: str
    packages: dict[str, str | None]
    cuda_available: bool
    disk_available_bytes: int
    statuses: dict[str, str]


def preflight(artifact_root: str | Path) -> RuntimePreflight:
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "safetensors", "huggingface-hub", "tokenizers", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    root = Path(artifact_root)
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    text_runtime = packages["torch"] is not None and packages["transformers"] is not None
    statuses = {
        "T00_T04": "EXECUTABLE" if text_runtime else "BLOCKED_RUNTIME",
        "T02_T06": "EXECUTABLE" if text_runtime else "BLOCKED_RUNTIME",
        "T01_T03_T05_T07": "BLOCKED_ARTIFACT",
        "M00_M05": "BLOCKED_ARTIFACT" if text_runtime else "BLOCKED_RUNTIME",
        "CUDA_OPERATIONAL_ARMS": "EXECUTABLE" if cuda else "INVALID_FOR_HARDWARE",
    }
    return RuntimePreflight(platform.python_version(), packages, cuda, free, statuses)

