# Issue 85 artifact and vector forensics

This report is evaluation-only. No model tensor was executed, no production
index was read or written, and no artifact identity was rewritten.

## 4B artifact disposition

`CORRUPT_OR_UNTRUSTED`

| Field | Expected artifact | Discovered artifact |
|---|---|---|
| filename | `Qwen3-Embedding-4B-Q4_K_M.gguf` | `Qwen3-Embedding-4B-Q4_K_M.gguf` |
| authorized location | `Qwen/Qwen3-Embedding-4B-GGUF` | `OpenNICF-issue-53/evaluations/embedding-production/artifacts/` |
| source revision | `f4602530db1d980e16da9d7d3a70294cf5c190be` | no retained source/revision log |
| SHA-256 | `2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b` | `6fc91b94aa945a268991a1a94d77129d674a12cbe20c94575889d73071f30990` |
| size | 2,496,703,776 bytes | 2,496,703,776 bytes |
| modified | repository object, immutable revision | `2026-08-12T13:32:19.737086800Z` |
| format | GGUF | GGUF v3, 36 metadata entries, 398 tensors |
| embedded identity | Qwen3 Embedding 4B | `general.name=Qwen3 Embedding 4B`; `architecture=qwen3`; Apache-2.0 |
| quantization | Q4_K_M filename/LFS object | file type 15; 216 Q4_K, 37 Q6_K, 145 F32 tensors; quantization version 2 |
| native dimension | 2,560 | 2,560 |
| context | 40,960 | 40,960 |
| pooling | embedding model | GGUF pooling type 3 |
| provenance confidence | high: canonical repository revision and LFS digest | low: matching size/metadata but no source log and wrong bytes |
| prior execution evidence bound to digest | none found | none found |
| disposition | approved identity remains absent locally | quarantine; do not execute or relabel |

The authoritative repository's response at the pinned revision reports the
expected LFS digest and exact same byte size. The discovered file is therefore
not truncated, but equality of size and plausible metadata cannot establish
byte identity. No authorized evaluation storage, worktree, or repository-owned
cache contained another GGUF with the expected digest. Historical reports
describe a small runtime feasibility run but do not bind it to either digest;
they explicitly say no comparable vectors were retained. The existing 4B
vectors are not admissible evidence for either artifact.

An exact replacement is available at the pinned canonical revision, but was
not downloaded in this lane: the retained vector artifact cannot be safely
continued even after replacement because it omits ordered IDs and semantic-
space provenance. A new, separately generated vector artifact is required.

## 0.6B vector-space feasibility

The retained `vectors-06b.json` has SHA-256
`755d6fa0c1a0ddc707ea7f702d7f9bf2c888229ed10bbf140ed533cfa5ccc211`.
It contains 10 document vectors and 190 query vectors, all 768-D and finite.
It does not embed document/query IDs, model revision, model artifact digest,
runtime/revision, preprocessing, pooling, or projection. Consequently it
cannot prove how its positional vectors map to the 715-chunk corpus and cannot
be safely appended to.

Two mutually different producers are retained:

- `vectorize_06b.py` calls a configured worker and relies on the worker's
  undisclosed preprocessing/runtime identity.
- `vectorize_06b_direct.py` pins snapshot
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, uses Transformers, FP16 CUDA,
  mean pooling, a 512-token maximum, a query instruction, MRL prefix 768, and
  normalization.

The artifact does not identify which producer created it. This is a semantic-
space ambiguity, not merely missing compute. Generating 705 vectors and mixing
them with the retained ten would violate the fail-closed space contract.
Regenerating all 715 document and all 190 query vectors is the safe recovery,
but the pinned local snapshot/runtime is not present on this evaluation host.
No vector generation was attempted.

## Objective inventory

- corpus: 715 chunks; SHA-256
  `cb6f30f5f69ff14f1bb75278c2e72ce312c570152651cee30bc2f70bc5071ee1`
- ordered chunk-ID digest (canonical compact JSON):
  `a16c919436aa411948cab41cbaee4b1ef7d1ea690d06cde58e5b95f1a07eb8b0`
- queries: 190; gold-set SHA-256
  `4867311d9b92177e5d894c61819cd34c9586f0334984fd30c3af4d5b44287bde`
- 0.6B: 10/715 document, 190/190 query, 0 nonfinite; unsafe to append
- 4B: 10/715 document, 190/190 query, 0 nonfinite; inadmissible and unsafe to append
- expected 4B GGUF recovered locally: no
- discovered 4B GGUF executed: no

`PRODUCTION = UNCHANGED`  
`DEPLOYMENTS = 0`  
`MERGES = 0`
