# Issue #90 VL artifact and runtime qualification

Evidence captured 2026-08-12 without downloading or executing model weights. Repository revisions are pinned; HTTP `x-linked-etag` values are remote LFS object identifiers, not locally recomputed SHA-256 claims.

## Official reference contracts

| Role | Canonical artifact | Revision | License | Weight file | Bytes | Remote LFS object ID |
|---|---|---|---|---|---:|---|
| embedding BF16/FP16 reference | `Qwen/Qwen3-VL-Embedding-2B` | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | Apache-2.0 | `model.safetensors` | 4,255,140,312 | `c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1` |
| multimodal reranker | `Qwen/Qwen3-VL-Reranker-2B` | `4bd860ac4f15ad1897a214615cccc700f8f71818` | Apache-2.0 | `model.safetensors` | 4,255,140,312 | `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9` |

Sources: the canonical Hugging Face repositories and their pinned model cards: <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B> and <https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B>. Required companion files include `config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, and the relevant pooling/logit-score configuration.

The official cards specify 2B parameters, 28 layers, 32K context, and embedding dimensions 64..2048 via MRL. The embedding card's “quantization support” explicitly means post-processing the **output embedding**, not quantized model weights. Inputs support text, images, screenshots, videos and mixed combinations. The pinned examples require `transformers>=4.57.0`, use `torch==2.8.0`, and demonstrate Sentence Transformers, Transformers, and vLLM paths. The preprocessing/template contract is the pinned processor/chat template; evaluation must preserve the exact task instruction. Pixel policy is not delegated to defaults: the executable manifest must pin `max_pixels`, page/image/batch counts, processor revision and resize behavior.

## Weight-quantized candidates (not yet executable on Pascal)

Two real third-party W4 weight artifacts exist, but neither is accepted as executable on the target GTX 1060:

* `LifetimeMistake/Qwen3-VL-Embedding-2B-AWQ-4bit` revision `c6c8b89efbd4ceef35da5369da9f8902a4369aeb`, Apache-2.0 metadata, `model.safetensors` 2,791,145,616 bytes, remote LFS object ID `2e1b04df208947055a5fd46db833460c02108fdf1859535de84e1bb8802a6e77`. This is actual AWQ 4-bit **weight** quantization with custom code. It is not official Qwen publication and must undergo code/artifact review.
* `Forturne/Qwen3-VL-Embedding-2B-W4A16` revision `dd2c68f8eaad4866d99fcdfe2b8d3659aa45e32f`, `model.safetensors` 1,565,621,536 bytes, remote LFS object ID `f00f88f33b11e7e50be11768d019c46719bdfac1b8fd0413357c71a4d502f350`. It declares compressed-tensors W4A16, but the repository API exposes no license. It is therefore rejected at the license gate.

Optional W8 references exist (GGUF and MLX), but MLX targets Apple hardware. A W8 artifact is not selected for CUDA/Pascal until the inference topology is verified.

## CUDA 11.8 / Pascal finding

GTX 1060 is Pascal `sm_61` with 3 GB VRAM. The official reference file alone is about 4.26 GB before runtime/KV/vision activations, so it cannot fit wholly in 3 GB. The official card's recommended FlashAttention 2 path is not a Pascal path. vLLM's documented NVIDIA requirement is compute capability 7.0 or higher, excluding `sm_61`. Common Marlin/compressed-tensors W4 kernels likewise do not establish a Pascal-compatible route for this VL embedding architecture. No candidate above provides reproducible evidence that its multimodal embedding or reranking head executes correctly on Pascal.

Therefore:

`W4_EXECUTABLE_ON_PASCAL = BLOCKED_ON_RUNTIME_EVIDENCE`

This is not permission to substitute another model. Bounded alternatives to evaluate later are CPU inference/offload of the exact pinned model, a separate modern-GPU evaluation worker isolated from production, or a verified llama.cpp/GGUF path **only after** the embedding/reranker pooling semantics, vision projector, and templates are proven equivalent. Downloads and execution remain gated by the executable manifest.

## Measured vs planned

Repository revisions, licenses, filenames, HTTP sizes and remote object IDs above were measured from repository metadata/headers. Runtime compatibility and memory statements are qualification findings, not benchmark results. No weights were downloaded, no models executed, and no hardware latency/VRAM result is claimed.
