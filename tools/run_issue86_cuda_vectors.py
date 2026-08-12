#!/usr/bin/env python3
"""Generate hash-bound Issue #86 vectors on CUDA; never falls back to CPU."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time


MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
WEIGHT_SHA = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
TOKENIZER_SHA = "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def normalize_prefix(values, dimension: int = 768) -> list[float]:
    vector = [float(item) for item in values[:dimension]]
    if len(vector) != dimension or not all(math.isfinite(item) for item in vector):
        raise RuntimeError("invalid vector")
    norm = math.sqrt(sum(item * item for item in vector))
    if not norm:
        raise RuntimeError("zero vector")
    return [item / norm for item in vector]


def load_inputs(corpus_path: Path, labels_path: Path):
    corpus = json.loads(corpus_path.read_text())
    labels = json.loads(labels_path.read_text())
    chunks = corpus["chunks"]
    queries = labels["labels"]
    if len(chunks) != 715 or len(queries) != 190:
        raise RuntimeError("unexpected evaluation input count")
    return chunks, queries


def prepare(args) -> None:
    chunks, queries = load_inputs(args.corpus, args.labels)
    args.output.mkdir(parents=True, exist_ok=True)
    documents = [{"id": item["chunk_id"], "text": item["text"]} for item in chunks]
    query_rows = [{"id": item["query_id"], "text": item["automated_query_text"]} for item in queries]
    (args.output / "ordered-inputs.json").write_text(json.dumps({"documents": documents, "queries": query_rows}, ensure_ascii=False))
    for name, rows, query in (("documents", documents, False), ("queries", query_rows, True)):
        lines = []
        for row in rows:
            text = " ".join(row["text"].split())
            if query:
                text = f"Instruct: {INSTRUCTION} Query: {text}"
            lines.append(text)
        (args.output / f"{name}-4b.txt").write_text("\n".join(lines) + "\n")


def embed_06(args) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("BLOCKED_CUDA")
    inputs = json.loads((args.input / "ordered-inputs.json").read_text())
    snapshot = args.snapshot.resolve()
    if digest(snapshot / "model.safetensors") != WEIGHT_SHA or digest(snapshot / "tokenizer.json") != TOKENIZER_SHA:
        raise RuntimeError("artifact hash gate failed")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("CUDA_REQUIRED_CPU_FALLBACK_DETECTED")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    output: dict[str, object] = {}
    for group in ("documents", "queries"):
        vectors = []
        for row in inputs[group]:
            text = row["text"]
            if group == "queries":
                text = f"Instruct: {INSTRUCTION}\nQuery:{text}"
            batch = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
            batch = {key: value.to("cuda") for key, value in batch.items()}
            if any(value.device.type != "cuda" for value in batch.values()):
                raise RuntimeError("CUDA_REQUIRED_CPU_FALLBACK_DETECTED")
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
            if hidden.device.type != "cuda":
                raise RuntimeError("CUDA_REQUIRED_CPU_FALLBACK_DETECTED")
            sequence = batch["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device="cuda"), sequence]
            vector = torch.nn.functional.normalize(pooled[:, :768].float(), p=2, dim=1)[0]
            vectors.append({"id": row["id"], "vector": vector.cpu().tolist()})
        output[group] = vectors
    torch.cuda.synchronize()
    output["provenance"] = {
        "model": MODEL, "revision": REVISION, "weight_sha256": WEIGHT_SHA,
        "tokenizer_sha256": TOKENIZER_SHA, "runtime": f"torch={torch.__version__}",
        "device": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)), "dtype": "float16",
        "dimension": 768, "pooling": "last_token", "normalized": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(), "status": "EXECUTED_CUDA",
    }
    args.output.write_text(json.dumps(output, separators=(",", ":")))


def import_4b(args) -> None:
    inputs = json.loads((args.input / "ordered-inputs.json").read_text())
    document_data = json.loads(args.documents.read_text())["data"]
    query_data = json.loads(args.queries.read_text())["data"]
    if len(document_data) != 715 or len(query_data) != 190:
        raise RuntimeError("llama.cpp output count mismatch")
    output = {}
    for group, rows, raw in (("documents", inputs["documents"], document_data), ("queries", inputs["queries"], query_data)):
        output[group] = [{"id": row["id"], "vector": normalize_prefix(item["embedding"])} for row, item in zip(rows, raw)]
    output["provenance"] = {
        "model": "Qwen/Qwen3-Embedding-4B-GGUF", "revision": "f4602530db1d980e16da9d7d3a70294cf5c190be",
        "weight_sha256": "2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b",
        "runtime": "llama.cpp@a4a4c51f3d40e086b59b73b631b5c43c8fbf4504",
        "device": "NVIDIA GeForce GTX 1060 3GB", "compute_capability": [6, 1],
        "quantization": "Q4_K_M", "dimension": 768, "pooling": "model_default_last",
        "normalized": True, "status": "EXECUTED_CUDA",
    }
    args.output.write_text(json.dumps(output, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--corpus", type=Path, required=True); prep.add_argument("--labels", type=Path, required=True); prep.add_argument("--output", type=Path, required=True)
    embed = sub.add_parser("embed-06")
    embed.add_argument("--input", type=Path, required=True); embed.add_argument("--snapshot", type=Path, required=True); embed.add_argument("--output", type=Path, required=True)
    imp = sub.add_parser("import-4b")
    imp.add_argument("--input", type=Path, required=True); imp.add_argument("--documents", type=Path, required=True); imp.add_argument("--queries", type=Path, required=True); imp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "embed-06": embed_06, "import-4b": import_4b}[args.command](args)


if __name__ == "__main__":
    main()
