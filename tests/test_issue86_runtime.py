import math
from pathlib import Path

import pytest

from opennicf.issue86_runtime import (
    CudaRequiredError,
    mrl_normalize,
    query_text,
    require_artifact,
    require_cuda,
    require_cuda_tensors,
    require_model_on_cuda,
)


def test_artifact_gate_rejects_wrong_digest_and_accepts_exact(tmp_path: Path):
    artifact = tmp_path / "weight"
    artifact.write_bytes(b"trusted")
    with pytest.raises(ValueError, match="digest"):
        require_artifact(artifact, "0" * 64, 7)
    from hashlib import sha256
    assert require_artifact(artifact, sha256(b"trusted").hexdigest(), 7) == artifact


def test_mrl_is_prefix_then_normalization():
    vector = mrl_normalize((3, 4, 100), 2)
    assert vector == pytest.approx((0.6, 0.8))
    assert math.isclose(sum(value * value for value in vector), 1.0)


def test_nonfinite_and_zero_vectors_fail_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        mrl_normalize((float("nan"), 1), 2)
    with pytest.raises(ValueError, match="zero norm"):
        mrl_normalize((0, 0), 2)


def test_official_query_template_is_stable():
    assert query_text("Where is the runbook?") == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:Where is the runbook?"
    )


class _DeviceValue:
    def __init__(self, device):
        self.device = device

    def __add__(self, other):
        return _DeviceValue(self.device)


class _Cuda:
    def __init__(self, available=True):
        self.available = available
        self.synchronized = False

    def is_available(self): return self.available
    def device_count(self): return 1 if self.available else 0
    def synchronize(self): self.synchronized = True
    def get_device_name(self, index): return "Test CUDA GPU"
    def get_device_properties(self, index): return type("Properties", (), {"total_memory": 8_000_000_000})()


class _Torch:
    def __init__(self, available=True, probe_device="cuda:0"):
        self.cuda = _Cuda(available)
        self.version = type("Version", (), {"cuda": "12.6"})()
        self.probe_device = probe_device

    def tensor(self, values, device):
        return _DeviceValue(self.probe_device)


def test_cuda_preflight_executes_probe_and_records_identity():
    torch = _Torch()
    evidence = require_cuda(torch)
    assert evidence["device_name"] == "Test CUDA GPU"
    assert evidence["cuda_runtime"] == "12.6"
    assert torch.cuda.synchronized


def test_cuda_unavailable_and_probe_cpu_fallback_fail_closed():
    with pytest.raises(CudaRequiredError, match="BLOCKED_CUDA"):
        require_cuda(_Torch(available=False))
    with pytest.raises(CudaRequiredError, match="CPU_FALLBACK"):
        require_cuda(_Torch(probe_device="cpu"))


def test_model_inputs_and_outputs_must_be_verifiably_cuda_resident():
    model = type("Model", (), {"parameters": lambda self: iter([_DeviceValue("cuda:0")])})()
    require_model_on_cuda(model)
    require_cuda_tensors({"input_ids": _DeviceValue("cuda:0")}, role="input")
    require_cuda_tensors((_DeviceValue("cuda:0"),), role="output")
    cpu_model = type("Model", (), {"parameters": lambda self: iter([_DeviceValue("cpu")])})()
    with pytest.raises(CudaRequiredError, match="CPU_FALLBACK"):
        require_model_on_cuda(cpu_model)
    with pytest.raises(CudaRequiredError, match="no output device"):
        require_cuda_tensors({"scores": [1.0]}, role="output")
