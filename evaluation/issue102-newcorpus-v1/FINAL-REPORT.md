# Issue 102 — HTTP benchmark 326/190

Status: `COMPLETE_NEW_CORPUS`

## Methodology

This was a fully automated, independent experiment using one serial HTTP
worker against `http://192.168.1.137:1234` with model
`text-embedding-qwen3-embedding-4b`. The endpoint contract was preflighted
before inference. Native 2560-dimensional vectors were checkpointed by ID;
the evaluation space uses the first 768 values followed by L2 normalization.

## Corpus and identity

- `corpus_id=issue102-http-newcorpus-v1-326x190`
- `experiment_id=issue102-qwen3-4b-http-326x190-v1`
- `preprocessing_id=issue102-deterministic-chunker-v1`
- `embedding_space_id=issue102-qwen3-4b-http-2560-v1`
- Documents/chunks: **326/326**
- Queries: **190/190**
- `comparability=NON_COMPARABLE_TO_T00_T01`
- `HUMAN_REVIEW=OUT_OF_SCOPE`

## Automated metrics

- Recall@1: **0.6842**
- Recall@5: **0.8158**
- Recall@10: **0.8474**
- MRR@10: **0.7339**
- nDCG@10: **0.7612**
- Zero-hit: **0.1526**
- Hard-negative false-positive rate @10: **0.0368**
- Bootstrap: deterministic query-level resampling, seed `10220260824`, 2,000 replicates, 95% CIs.

## Limitations and safety

- `LATENCY_THROUGHPUT=UNAVAILABLE_NOT_CAPTURED`; the runner did not capture per-request telemetry.
- No human review or approval was performed.
- This is not a reproduction or comparison result for historical T00/T01.
- No winner or production recommendation is issued.
- `CT305_TOUCHED=false`; `CT308_TOUCHED=false`.
- `PRODUCTION=UNCHANGED`; deployments, merges, re-embedding, and index mutation are zero.

See `metrics.json` and `result-manifest.json` for canonical values and digests.
