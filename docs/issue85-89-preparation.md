# Issues 85/89 objective preparation

This work is evaluation-only. It does not authorize deployment, migration,
merging, or writes to production indexes.

## Official reranker contract candidate

- Canonical model: `Qwen/Qwen3-Reranker-0.6B`
- Source: <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- Exact repository revision: `e61197ed45024b0ed8a2d74b80b4d909f1255473`
- License: Apache-2.0
- Weight artifact: `model.safetensors`, 1,191,588,280 bytes
- Weight SHA-256 (Hugging Face LFS metadata): `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`
- Official implementation: <https://github.com/QwenLM/Qwen3-Embedding/tree/44548aa5f0a0aed1c76d64e19afe47727a325b8f>
- Runtime options documented by Qwen: `transformers>=4.51.0` and
  `vllm>=0.8.5`. The initial bounded candidate is Transformers on isolated CPU;
  neither GPU compatibility nor performance has been measured.
- Template: official system prompt plus `<Instruct>`, `<Query>`, `<Document>`;
  score is the normalized probability of the `yes` token against `no`.
- Maximum context advertised by the model card is 32K. The executable manifest
  must select a smaller fixed maximum and batch size after resource qualification.

This is an exact, licensed reference candidate, not proof that it can meet the
target latency or resource envelope. No reranker artifact was downloaded or
executed by this preparation.

## Fail-closed workflow

Run `tools/prepare_issue85_review.py` against the isolated artifact directory.
It verifies corpus/query/reference/vector counts and digests, then emits a
stable batched review file. Mechanical leakage flags and rewrite candidates are
review assistance only. They never produce an accepted label.

`docs/issue89-executable-manifest.template.json` remains `PREPARATION` until
all human-review, hard-negative, digest, vector, and reranker fields pass
`validate_executable_manifest`.

## Audited local boundary

The machine-generated audit and eight deterministic review batches live at
`docs/issue85-artifact-audit.json` and
`docs/issue85-human-review-batches.json`. The audit deliberately preserves the
original integrity failures. `docs/issue85-gold-set-objective-repair.json`
repairs only the stale embedded corpus digest and explicitly records that no
human labels changed. It is not an accepted gold set.

The source-derived workload contains many duplicated query texts. Those are
shown as leakage/rewrite indicators in the review artifact, and all 180
machine-derived queries remain pending. Mechanically proposed hard negatives
also remain pending until a reviewer confirms them.
