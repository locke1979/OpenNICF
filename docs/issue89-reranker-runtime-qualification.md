# Issue #89 text-reranker runtime qualification

Status: `PREPARATION_COMPLETE_EXECUTION_BLOCKED`

This is artifact/runtime evidence only. It contains no retrieval-quality result.
No weights were downloaded and no tensor execution occurred.

## Official identity and scoring contract

The canonical repository is
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
at immutable revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, licensed
Apache-2.0. The Hugging Face revision API and LFS metadata agree on:

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.safetensors` | 1,191,588,280 | `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b` |
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `config.json` | 727 | `d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0` |
| `tokenizer_config.json` | 9,706 | `253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0` |
| `generation_config.json` | 214 | `81051cd3f6e77013827148d0b8a6ead93f8ac390d5ab805f849199f0af6a08db` |
| `chat_template.jinja` | 741 | `6f682162495ec5b39fd9005c01b6aa2a74669379fe967039f1e2cbbe8752369d` |
| `1_LogitScore/config.json` | 57 | `73e3156450564d8a98b7e47bcf5aace0f29600828b51937da545571e84db3ff3` |

The pinned config declares `Qwen3ForCausalLM`, BF16, 40,960 native maximum
positions and vocabulary size 151,669. The official direct-Transformers example
fixes evaluation `max_length=8192`, left padding, and preserves the exact system
prefix and assistant suffix. It takes the final-position logits for tokens
`no` and `yes`; softmax probability of `yes` increases with relevance. The
default instruction is `Given a web search query, retrieve relevant passages
that answer the query`. The deterministic local adapter implements that bounded
format, truncates only the pair body, preserves prefix/suffix, batches, rejects
non-finite/malformed outputs and orders ties by original candidate index.

## Runtime feasibility on this evaluation host

The official model card documents three candidates:

- Transformers `>=4.51.0` using `AutoModelForCausalLM`;
- Sentence Transformers `CrossEncoder`;
- vLLM `>=0.8.5`.

Host inspection found Python 3.12.3, but `torch`, `transformers`,
`sentence_transformers`, `safetensors`, and `accelerate` are absent.
`nvidia-smi` is absent, so this host provides no locally evidenced CUDA device.
Package-install feasibility is not forward-execution evidence. The exact
1.19-GB weight is also absent and the issue #89 EXECUTABLE manifest remains
invalid because reviewed labels and complete vector spaces are missing.
Consequently no dependency mutation, weight download, or synthetic forward was
authorized.

`STAGE_1_METADATA = PASSED`

`STAGE_2_SYNTHETIC_FORWARD = BLOCKED_ON_EXECUTABLE_MANIFEST_AND_RUNTIME`

## Review boundary

The review helpers now include stable item digests, original and reviewed query
text, immutable relevant IDs, explicit reviewer identity/timestamp/source,
review notes, and validated hard negatives. Applying a completed batch produces
a deterministic label digest and fails closed on unknown IDs or silent relevant
ID changes. Multimodal review assistance emits only bounded `fixture://`
rendition references plus rendition hashes; it never embeds evidence bytes.
Generated rankings or suggestions remain labelled
`NON_DECISION_GRADE_REVIEW_ASSISTANCE` and cannot become accepted labels.

Current human-review counts are unchanged: issue #85 has 10 accepted and 180
pending; issue #90 has 50 pending.

`PRODUCTION = UNCHANGED`

`DEPLOYMENTS = 0`

`MERGES = 0`
