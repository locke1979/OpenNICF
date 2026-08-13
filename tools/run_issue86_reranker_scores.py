#!/usr/bin/env python3
"""Resumably score the frozen Issue #86 reranker pair set on CUDA.

The output is JSONL: one immutable manifest record followed by one record per
pair.  Restarting verifies the manifest and every completed record before
skipping it.  Checkpoints are made durable with flush+fsync.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import gc
import time
from pathlib import Path


MODEL = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
WEIGHT_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
ARTIFACT_HASHES = {
    "model.safetensors": WEIGHT_SHA256,
    "tokenizer.json": TOKENIZER_SHA256,
    "config.json": "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    "generation_config.json": "81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db",
    "chat_template.jinja": "6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d",
    "1_LogitScore/config.json": "73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3",
}
EXPECTED_PAIRS = 16948
EXPECTED_GPU_UUID = "GPU-cf150279-b259-0608-019f-8f51e0664ff4"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair_id(query_id, chunk_id):
    return digest_bytes(canonical([query_id, chunk_id]))


def prompt(query, document):
    return PREFIX + f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}" + SUFFIX


def load_contract(root, model_dir):
    corpus_path = root / "corpus-v2.json"
    labels_path = root / "automated-text-labels-v1.json"
    report_path = root / "text-report.json"
    corpus = json.loads(corpus_path.read_text())
    labels_doc = json.loads(labels_path.read_text())
    report = json.loads(report_path.read_text())
    documents = {x["chunk_id"]: x["text"] for x in corpus["chunks"]}
    queries = {x["query_id"]: x["automated_query_text"] for x in labels_doc["labels"]}
    arms = ("T00", "T01", "T04", "T05")
    pairs = []
    for qid in queries:  # frozen label order
        candidates = set().union(*(report["arms"][arm]["rankings"][qid] for arm in arms))
        for chunk_id in sorted(candidates):
            if chunk_id not in documents:
                raise ValueError(f"unknown chunk ID {chunk_id}")
            pairs.append({"pair_id": pair_id(qid, chunk_id), "query_id": qid, "chunk_id": chunk_id})
    if len(pairs) != EXPECTED_PAIRS or len({x["pair_id"] for x in pairs}) != EXPECTED_PAIRS:
        raise ValueError(f"pair contract mismatch: {len(pairs)}")
    contract = {
        "record_type": "manifest", "schema_version": 1, "model": MODEL, "revision": REVISION,
        "corpus_sha256": digest_file(corpus_path), "labels_sha256": digest_file(labels_path),
        "ranking_file_sha256": digest_file(report_path),
        "ordered_chunk_ids_sha256": digest_bytes(canonical([x["chunk_id"] for x in corpus["chunks"]])),
        "ordered_query_ids_sha256": digest_bytes(canonical(list(queries))),
        "ordered_pairs_sha256": digest_bytes(canonical(pairs)), "expected_pairs": EXPECTED_PAIRS,
        "artifact_hashes": {name: digest_file(model_dir / name) for name in ARTIFACT_HASHES},
        "template_sha256": digest_bytes((PREFIX + INSTRUCTION + SUFFIX).encode()),
        "candidate_depths": [10, 20, 30, 50], "rrf": {"k": 60}, "context_depths": [5, 8, 10],
        "max_length": 8192, "batch_fallback": [4, 2, 1], "precision": "float16", "cpu_offload": False,
    }
    if contract["artifact_hashes"] != ARTIFACT_HASHES:
        raise ValueError("pinned model artifact digest mismatch")
    return corpus, labels_doc, report, documents, queries, pairs, contract


def resume(path, contract):
    completed = {}
    if not path.exists():
        return completed
    with path.open() as stream:
        first = stream.readline()
        if not first or json.loads(first) != contract:
            raise ValueError("checkpoint manifest/input contract mismatch")
        for number, line in enumerate(stream, 2):
            row = json.loads(line)
            if row.get("record_type") != "score" or row["pair_id"] in completed:
                raise ValueError(f"malformed or duplicate checkpoint record at line {number}")
            if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
                raise ValueError(f"invalid score at line {number}")
            if row["pair_id"] != pair_id(row["query_id"], row["chunk_id"]):
                raise ValueError(f"pair ID mismatch at line {number}")
            for key, expected in (("ordered_input_sha256", contract["ordered_pairs_sha256"]),
                                  ("model_revision", REVISION), ("runtime", contract["runtime"]),
                                  ("artifact_hashes", contract["artifact_hashes"])):
                if row.get(key) != expected:
                    raise ValueError(f"checkpoint provenance mismatch for {key} at line {number}")
            completed[row["pair_id"]] = row
    return completed


def atomic_checkpoint(path):
    """Publish a complete point-in-time copy without exposing a partial file."""
    checkpoint = path.with_suffix(path.suffix + ".checkpoint")
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    shutil.copyfile(path, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint)
    directory_fd = os.open(checkpoint.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/var/lib/opennicf/eval"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--max-new-pairs", type=int, default=None,
                        help="bounded preflight only; omit for the full run")
    args = parser.parse_args()
    if not 250 <= args.checkpoint_every <= 500:
        parser.error("--checkpoint-every must be 250..500")
    if args.max_new_pairs is not None and args.max_new_pairs < 1:
        parser.error("--max-new-pairs must be positive")
    output = args.output or args.root / "reranker-pair-scores-v1.jsonl"
    model_dir = args.root / f"models/Qwen3-Reranker-0.6B-{REVISION}"

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.__version__ != "2.4.1+cu121" or transformers.__version__ != "4.51.3" or torch.version.cuda != "12.1":
        raise RuntimeError("runtime version mismatch")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce GTX 1060 3GB":
        raise RuntimeError("CUDA device identity mismatch")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu_uuid != EXPECTED_GPU_UUID:
        raise RuntimeError("CUDA UUID mismatch")

    _, _, _, documents, queries, pairs, contract = load_contract(args.root, model_dir)
    contract["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda}
    contract["device"] = {"name": torch.cuda.get_device_name(0), "uuid": gpu_uuid,
                          "compute_capability": list(torch.cuda.get_device_capability(0))}
    completed = resume(output, contract)
    valid_ids = {x["pair_id"] for x in pairs}
    if not set(completed) <= valid_ids:
        raise ValueError("checkpoint contains pair outside frozen input")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16).to("cuda").eval()
    if not all(x.device.type == "cuda" for x in model.parameters()):
        raise RuntimeError("CPU offload detected")
    no_id, yes_id = tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")
    pending = []
    for item in pairs:
        if item["pair_id"] not in completed:
            text = prompt(queries[item["query_id"]], documents[item["chunk_id"]])
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            pending.append((length, item, text))
    pending.sort(key=lambda x: (x[0], x[1]["pair_id"]))
    if args.max_new_pairs is not None:
        pending = pending[:args.max_new_pairs]

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output.exists() else "w"
    started = time.perf_counter(); written = 0; oom_count = 0; index = 0
    next_checkpoint = args.checkpoint_every
    with output.open(mode) as stream:
        if mode == "w":
            stream.write(canonical(contract).decode() + "\n"); stream.flush(); os.fsync(stream.fileno())
        while index < len(pending):
            succeeded = False
            for batch_size in (4, 2, 1):
                selection = pending[index:index + batch_size]
                try:
                    batch = tokenizer([x[2] for x in selection], padding=True, truncation=True, max_length=8192, return_tensors="pt")
                    batch = {k: v.to("cuda") for k, v in batch.items()}
                    batch_started = time.perf_counter()
                    with torch.inference_mode():
                        logits = model(**batch).logits[:, -1, [no_id, yes_id]]
                    values = torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
                    elapsed = time.perf_counter() - batch_started
                except torch.cuda.OutOfMemoryError:
                    oom_count += 1
                    for name in ("batch", "logits", "values"):
                        if name in locals(): del locals()[name]
                    gc.collect(); torch.cuda.empty_cache()
                    if batch_size == 1: raise
                    continue
                for (length, item, _), score in zip(selection, values):
                    if not math.isfinite(score): raise ValueError("nonfinite score")
                    row = {"record_type":"score", **item, "score":score, "token_length":length,
                           "effective_batch_size":len(selection), "batch_elapsed_seconds":elapsed,
                           "run_elapsed_seconds":time.perf_counter()-started, "oom_count":oom_count,
                           "ordered_input_sha256":contract["ordered_pairs_sha256"], "model_revision":REVISION,
                           "runtime":contract["runtime"], "artifact_hashes":contract["artifact_hashes"]}
                    stream.write(canonical(row).decode() + "\n"); written += 1
                index += len(selection); succeeded = True
                if written >= next_checkpoint or index == len(pending):
                    stream.flush(); os.fsync(stream.fileno())
                    atomic_checkpoint(output)
                    print(json.dumps({"completed":len(completed)+index,"expected":EXPECTED_PAIRS,"oom_count":oom_count}), flush=True)
                    while next_checkpoint <= written:
                        next_checkpoint += args.checkpoint_every
                break
            if not succeeded: raise RuntimeError("all batch sizes failed")


if __name__ == "__main__":
    main()
