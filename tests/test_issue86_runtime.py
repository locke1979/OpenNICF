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
    require_quantization_evidence,
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


def test_executed_quantization_evidence_fails_closed():
    with pytest.raises(ValueError, match="lacks quantization evidence"):
        require_quantization_evidence({"status": "EXECUTED_CUDA"})
    record = {
        "status": "EXECUTED_CUDA", "weight_bits": 16,
        "weight_quantization_method": "none", "weight_quantization_scheme": "FP16",
        "group_size": None, "compute_dtype": "float16", "runtime": "PyTorch CUDA",
        "runtime_revision": "pinned", "cuda_required": True,
        "cpu_offload_allowed": False, "verified_cuda_residency": True,
    }
    require_quantization_evidence(record)
    with pytest.raises(ValueError, match="low-bit"):
        require_quantization_evidence({**record, "weight_bits": 4})


def test_quantization_audit_preserves_reference_precision_and_spaces():
    import json
    audit = json.loads((Path(__file__).parents[1] / "docs/issue86-quantization-audit-2026-08-12.json").read_text())
    assert audit["quantization_audit"] == "FAIL"
    assert audit["all_models_4bit"] is False
    assert audit["executed_fp16_arms"] == ["T00", "T04"]
    assert audit["executed_4bit_arms"] == ["T01", "T05"]
    executed = [model for model in audit["models"] if model["status"] == "EXECUTED_CUDA"]
    assert all(model["verified_cuda_residency"] is False for model in executed)
    for model in executed:
        with pytest.raises(ValueError, match="verified CUDA residency"):
            require_quantization_evidence(model)
    q4 = next(model for model in executed if model["weight_quantization_scheme"].startswith("Q4_K_M"))
    assert q4["tensor_type_inventory"] == {"Q4_K": 216, "Q6_K": 37, "F32": 145}
    assert audit["separate_4bit_variants"]["Q00"]["embedding_space_id"] != "TEXT_QWEN3_06B_768_AUTO_V1"
