"""Pinned, fail-closed runtime helpers for issue #86 evaluation only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import math
from pathlib import Path
import platform
import shutil
from typing import Any, Sequence


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


class CudaRequiredError(RuntimeError):
    """Issue #86 must never turn a missing/misplaced CUDA path into CPU work."""


def require_cuda(torch_module: Any | None = None) -> dict[str, Any]:
    """Prove CUDA visibility and execute a synchronized CUDA tensor operation.

    Import injection is intentional: preflight behavior can be tested on CPU-only
    CI without pretending that CI has a CUDA device.
    """
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise CudaRequiredError("BLOCKED_CUDA: PyTorch is not installed") from exc
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available() or cuda.device_count() < 1:
        raise CudaRequiredError("BLOCKED_CUDA: no CUDA device is visible")
    try:
        probe = torch_module.tensor([1.0], device="cuda")
        result = probe + probe
        cuda.synchronize()
    except Exception as exc:
        raise CudaRequiredError("BLOCKED_CUDA: CUDA tensor probe failed") from exc
    device = str(getattr(result, "device", ""))
    if not device.startswith("cuda"):
        raise CudaRequiredError("CUDA_REQUIRED_CPU_FALLBACK_DETECTED: probe output is not on CUDA")
    properties = cuda.get_device_properties(0)
    return {
        "device_count": cuda.device_count(),
        "device_name": cuda.get_device_name(0),
        "device": device,
        "cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
        "vram_total_bytes": int(properties.total_memory),
    }


def require_cuda_tensors(*values: Any, role: str = "tensor") -> None:
    """Reject CPU model parameters, inputs, or outputs before evidence is kept."""
    seen = False
    stack = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif hasattr(value, "device"):
            seen = True
            if not str(value.device).startswith("cuda"):
                raise CudaRequiredError(f"CUDA_REQUIRED_CPU_FALLBACK_DETECTED: {role} is on {value.device}")
    if not seen:
        raise CudaRequiredError(f"CUDA_REQUIRED_CPU_FALLBACK_DETECTED: no {role} device was verifiable")


def require_model_on_cuda(model: Any) -> None:
    """Verify every materialized parameter is resident on CUDA."""
    parameters = list(model.parameters())
    if not parameters:
        raise CudaRequiredError("CUDA_REQUIRED_CPU_FALLBACK_DETECTED: model has no verifiable parameters")
    require_cuda_tensors(parameters, role="model parameter")


QUANTIZATION_EVIDENCE_FIELDS = (
    "weight_bits",
    "weight_quantization_method",
    "weight_quantization_scheme",
    "group_size",
    "compute_dtype",
    "runtime",
    "runtime_revision",
    "cuda_required",
    "cpu_offload_allowed",
    "verified_cuda_residency",
)


def require_quantization_evidence(record: dict[str, Any]) -> None:
    """Fail closed when an executed model lacks precision/residency evidence."""
    if record.get("status") != "EXECUTED_CUDA":
        return
    missing = [field for field in QUANTIZATION_EVIDENCE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"executed arm lacks quantization evidence: {', '.join(missing)}")
    if record["cuda_required"] is not True or record["cpu_offload_allowed"] is not False:
        raise ValueError("executed arm violates the CUDA-only/no-offload contract")
    if record["verified_cuda_residency"] is not True:
        raise ValueError("executed arm lacks verified CUDA residency")
    bits = record["weight_bits"]
    if not isinstance(bits, (int, list)) or isinstance(bits, bool):
        raise ValueError("weight_bits must describe executable weight metadata")
    if record["weight_quantization_method"] == "none" and bits not in (16, 32):
        raise ValueError("unquantized weights cannot be relabelled as low-bit")


def preflight(artifact_root: str | Path) -> RuntimePreflight:
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "safetensors", "huggingface-hub", "tokenizers", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        require_cuda()
        cuda = True
    except CudaRequiredError:
        cuda = False
    root = Path(artifact_root)
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    text_runtime = packages["torch"] is not None and packages["transformers"] is not None
    statuses = {
        "T00_T04": "EXECUTABLE" if text_runtime and cuda else ("BLOCKED_RUNTIME" if not text_runtime else "BLOCKED_CUDA"),
        "T02_T06": "EXECUTABLE" if text_runtime and cuda else ("BLOCKED_RUNTIME" if not text_runtime else "BLOCKED_CUDA"),
        "T01_T03_T05_T07": "BLOCKED_ARTIFACT",
        "M00_M05": "BLOCKED_ARTIFACT" if text_runtime and cuda else ("BLOCKED_RUNTIME" if not text_runtime else "BLOCKED_CUDA"),
        "CUDA_OPERATIONAL_ARMS": "EXECUTABLE" if cuda else "BLOCKED_CUDA",
    }
    return RuntimePreflight(platform.python_version(), packages, cuda, free, statuses)
