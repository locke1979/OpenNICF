# Issue #101 artifact recovery note

Status: artifact-only continuation complete for the preserved evidence set.

This note records what is already recoverable from the checked-in OpenNICF
evaluation bundle and what still requires a live authorized CUDA executor.
It does not claim to have regenerated missing vectors.

## Recovered sources

- [`evaluation/issue101-execution-plan.json`](../evaluation/issue101-execution-plan.json)
- [`evaluation/results/issue101/template-forensics.json`](../evaluation/results/issue101/template-forensics.json)
- [`evaluation/results/issue86-cuda-2026-08-12/summary.json`](../evaluation/results/issue86-cuda-2026-08-12/summary.json)
- [`evaluation/results/issue86-cuda-2026-08-12/text-report.json`](../evaluation/results/issue86-cuda-2026-08-12/text-report.json)
- [`evaluation/results/issue86-cuda-2026-08-12/retrieval-rag-prechecks.json`](../evaluation/results/issue86-cuda-2026-08-12/retrieval-rag-prechecks.json)
- [`docs/issue86-reranker-cuda-conformance-2026-08-12.json`](issue86-reranker-cuda-conformance-2026-08-12.json)

## Verified historical CUDA evidence

The preserved evaluation evidence records a completed Linux CUDA run on:

- GPU: `NVIDIA GeForce GTX 1060 3GB`
- GPU UUID: `GPU-cf150279-b259-0608-019f-8f51e0664ff4`
- compute capability: `6.1`
- driver: `535.261.03`
- driver/CUDA compatibility: `12.2`
- PyTorch CUDA runtime: `12.1`
- container GPU visibility: enabled
- minimal tensor operation: pass

That evidence comes from the checked-in CUDA execution record for LXC 308 and
is not replaced by the current workspace, which does not expose a local GPU.

## Template forensics

`evaluation/results/issue101/template-forensics.json` records:

- `template_4b_historical_t01`
- `template_qwen3_official`
- UTF-8 encoding for both
- historical query template with a space before `Query:`
- official query template with a newline before `Query:`
- distinct SHA-256 values for historical and official preprocessing

The byte-level difference is preserved and the historical T01 template is not
silently corrected.

## Preserved metric evidence

The checked-in CUDA result bundle includes the following completed arms:

- `T00` - Qwen/Qwen3-Embedding-0.6B, 768-D, FP16, last-token pooling
- `T01` - Qwen/Qwen3-Embedding-4B-GGUF, 768-D, Q4_K_M, model-default-last pooling
- `T04` - Qwen/Qwen3-Embedding-0.6B, 768-D, FP16, last-token pooling
- `T05` - Qwen/Qwen3-Embedding-4B-GGUF, 768-D, Q4_K_M, model-default-last pooling

### T00

- Recall@1: `0.8789473684210526`
- Recall@5: `0.9947368421052631`
- MRR@10: `0.9166666666666666`
- nDCG@10: `0.9359093024459895`
- hard-negative FP@5: `0.08421052631578947`
- zero-hit rate: `0.0`

### T01

- Recall@1: `0.4263157894736842`
- Recall@5: `0.9052631578947369`
- MRR@10: `0.6649999999999999`
- nDCG@10: `0.7289653642411709`
- hard-negative FP@5: `0.08947368421052632`
- zero-hit rate: `0.005263157894736842`

### T05 versus T01

The preserved paired bootstrap result shows the official 4B template is better
than the historical T01 template on ranking quality:

- Recall@5 mean delta: `-0.010526315789473684`
- Recall@5 95% CI: `[-0.05263157894736842, 0.031578947368421054]`
- MRR@10 mean delta: `0.09166666666666666`
- MRR@10 95% CI: `[0.042368421052631576, 0.14096491228070174]`
- nDCG@10 mean delta: `0.07314507932736627`
- nDCG@10 95% CI: `[0.03296169373563659, 0.11470100991110853]`
- hard-negative FP@5 mean delta: `0.04736842105263158`
- hard-negative FP@5 95% CI: `[0.021052631578947368, 0.07894736842105263]`

Interpretation: the historical-versus-official 4B query-template difference is a
partial cause of T01’s poor result. It helps materially on MRR and nDCG, but it
does not eliminate the overall gap to the stronger 0.6B arm.

## T00 reproduction status

The preserved artifacts show the canonical T00 result is already established
and remains immutable. This artifact-only pass did not regenerate inference, so
the reproduction gate is not re-executed here.

## Remaining missing inference

The following issue-101 arms still require missing live inference or preserved
per-arm checkpoints outside the current workspace:

- `D06-NATIVE-1024`
- `D4-1024`
- `D4-1536`
- `D4-NATIVE-2560`

The current workspace does not contain materialized issue-101 vector checkpoints
under `/var/lib/opennicf/eval/issue101`, so these arms cannot be completed from
artifact-only evidence in this environment.

## Cost projection from the execution plan

The issue-101 plan records raw float32 vector storage for the current corpus
(`715` documents + `190` queries) as:

- `D06_768`: `2,780,160` bytes
- `D06_1024`: `3,706,880` bytes
- `D4_768`: `2,780,160` bytes
- `D4_1024`: `3,706,880` bytes
- `D4_1536`: `5,560,320` bytes
- `D4_2560`: `9,267,200` bytes

That scales linearly to future corpus sizes. For reference:

- `100,000` vectors at 768-D: `307,200,000` bytes
- `100,000` vectors at 1024-D: `409,600,000` bytes
- `100,000` vectors at 1536-D: `614,400,000` bytes
- `100,000` vectors at 2560-D: `1,024,000,000` bytes

- `1,000,000` vectors at 768-D: `3,072,000,000` bytes
- `1,000,000` vectors at 1024-D: `4,096,000,000` bytes
- `1,000,000` vectors at 1536-D: `6,144,000,000` bytes
- `1,000,000` vectors at 2560-D: `10,240,000,000` bytes

These are raw vector-storage figures only. They do not include ANN index
overhead, build time, or search latency, which still require the missing native
issue-101 execution arms.

## Current conclusion

- Historical T01 preprocessing remains immutable.
- The historical-versus-official 4B template difference is real and materially
  changes ranking quality.
- T00 remains the preferred completed text arm in the preserved evidence set.
- The native 4B / 1024-D / 1536-D / 2560-D comparisons remain pending missing
  inference.
