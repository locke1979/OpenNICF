# Issue #86 automated text evaluation decision

Status: automated experimental evidence only. `EVALUATION_MODE = AUTOMATED_NON_HUMAN_REVIEWED`; `DECISION_GRADE_PRODUCTION_VALIDATION = false`.

## Immutable scoring gate

- Score file SHA-256: `5db67936e4cc02bf8148b214d1ceb26eb1464a11a2bc73a10c08f48e8b52c4f3`.
- Ordered-pair SHA-256: `74ddce2ac588900a531f3efb2cb85494b916ff2402d68c3a471474a6f4ef1b3e`.
- Coverage: 16,948/16,948 unique pairs; missing/duplicate/nonfinite: 0/0/0.
- Controlled OOM retries: 338. Effective pair distribution: batch 4=16,672, batch 2=142, batch 1=134.
- Derived report regenerates byte-identically. Full report and RAG report contain the frozen input, model, runtime, artifact and output digests.

The repository's established arm mapping is preserved: T02=T00+r, T03=T01+r, T06=T04+r, T07=T05+r. The latest phase prose transposed the T03 and T06 descriptions; silently redefining immutable arm IDs would invalidate comparisons.

## Overall metrics at candidate depth 50

All 190 automated labels have one relevant chunk, so Recall@k is also hit rate. Precision@k is relevant hits divided by k.

| Arm | Architecture | R@1 | R@3 | R@5 | R@8 | R@10 | MRR@10 | nDCG@10 | P@5 | Zero-hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T00 | 0.6B dense | .8789 | .9316 | .9947 | .9947 | .9947 | .9167 | .9359 | .1989 | .0053 |
| T01 | 4B Q4 dense | .4263 | .9000 | .9053 | .9053 | .9105 | .6650 | .7290 | .1811 | .0895 |
| T02 | 0.6B dense+r | .8684 | .9684 | 1.0000 | 1.0000 | 1.0000 | .9254 | .9444 | .2000 | 0 |
| T03 | 4B Q4 dense+r | .4368 | .9632 | .9947 | .9947 | .9947 | .7068 | .7816 | .1989 | .0053 |
| T04 | 0.6B hybrid RRF | .8263 | .8263 | .9158 | .9632 | .9895 | .8556 | .8861 | .1832 | .0105 |
| T05 | 4B Q4 hybrid RRF | .6526 | .8263 | .8947 | .9421 | .9421 | .7567 | .8021 | .1789 | .0579 |
| T06 | 0.6B hybrid+r | .8474 | .9684 | 1.0000 | 1.0000 | 1.0000 | .9149 | .9367 | .2000 | 0 |
| T07 | 4B Q4 hybrid+r | .7053 | .9632 | .9947 | .9947 | .9947 | .8386 | .8788 | .1989 | .0053 |

## Paired 95% bootstrap conclusions

- T02−T00: R@5 +.0053 [0, .0158], MRR@10 +.0088 [−.0285, .0461], nDCG@10 +.0085 [−.0192, .0371]: inconclusive. Explicit hard-negative FP@5 worsened +.0211 [.0053, .0421], supported (lower is better).
- T03−T01: R@5 +.0895 [.0474, .1316], MRR@10 +.0418 [.0019, .0847], nDCG@10 +.0526 [.0150, .0940]: supported improvements. FP@5 +.0158 [0, .0368]: inconclusive regression.
- T06−T04: R@5 +.0842 [.0474, .1263], MRR@10 +.0593 [.0126, .1081], nDCG@10 +.0505 [.0142, .0884]: supported improvements. FP@5 worsened +.2947 [.2158, .3684], supported.
- T07−T05: R@5 +.1000 [.0632, .1474], MRR@10 +.0819 [.0375, .1267], nDCG@10 +.0767 [.0404, .1152]: supported improvements. FP@5 worsened +.3000 [.2263, .3737], supported.

LOW_LEAKAGE has only nine queries. T00 R@5/MRR@10/nDCG@10=.8889/.8889/.8889 with FP@5=0. T02 reaches 1.0/.7870/.8402 but FP@5=.4444. The 34-query HARD_NEGATIVE category has FP@5=0 for all dense and reranked arms, while the explicit negative attached to all 190 queries reveals the material regressions above. Both views are retained rather than conflated.

## Candidate depth and disagreements

- T02: depth 5 nDCG@10 exceeds depth 50 by .0216 [.0088, .0359]; prefer 5 for this arm.
- T03: depths 5/10/20/30 are significantly worse than 50; prefer 50.
- T06: depth 10 exceeds 50 by .0222 [.0077, .0376]; prefer 10.
- T07: depth 20 is indistinguishable from 50 (−.0001 [−.0057, .0039]); prefer 20.
- Wins/ties/losses versus baseline: T02 18/148/24; T03 20/156/14; T06 33/136/21; T07 46/124/20. Per-query ranks, displacement, leakage and categories are in the machine report.

There is no defensible universal depth. Operational routing should bind depth to the architecture rather than defaulting every arm to 50.

## Retrieval-only RAG

At context depth 10, expected-evidence/answerable rates are T00 .9947, T01 .9105, T02 1.0, T03 .9947, T04 .9895, T05 .9421, T06 1.0 and T07 .9947. Duplicate chunk rate is zero. Explicit hard-negative context rates are .0842, .0895, .1263, .1263, .2526, .2158, .5211 and .5211 respectively. Full depth 5/8/10 context recall, precision, evidence retention, source diversity and token estimates are in the RAG artifact.

Generated-answer and judge evaluation are `NOT_EXECUTED_NO_PINNED_GENERATOR_JUDGE_CONTRACT` and no answer-quality claim is made.

## Operational and architecture verdict

- Reranker alone: `USABLE_WITH_QUEUEING`. Verified FP16 CUDA, no CPU offload, one resident instance, max batch 4 with 4→2→1 fallback. Reconstructed GPU service telemetry: batch-4 2.494 pairs/s, batch latency p50/p95/p99 1.504/3.146/4.764 s; batch-2 .425 pairs/s and 2.786/12.647/23.029 s; batch-1 .159 pairs/s and 6.326/6.334/6.337 s. Queue/request concurrency remains unproven.
- 0.6B embedding+r: `UNPROVEN` for co-residency. The arithmetic headroom is plausible but insufficient evidence; no claim of fit.
- 4B Q4 embedding+r: `REQUIRES_SEPARATE_WORKER` or serialized residency on 3 GB. Simultaneous residency is not supported by current headroom evidence.
- Representation-space isolation, deterministic RRF, ingest identity, late dedup, context budgets, model-switch safety and explicit fallback policies remain enforced by the model-free contracts and tests. Reranker validation now fails closed on missing candidates, nonfinite scores, count mismatches and noncanonical ordering.

## Automated experimental recommendation

Prefer **T00 (0.6B dense without reranking)** as the conservative experimental default. T02's small overall quality increases are statistically inconclusive while its explicit hard-negative FP@5 regression is supported; T06/T07 show still larger supported negative exposure. If a 4B Q4 path is required, T03 materially repairs T01 quality, but it remains slower/more complex and requires serialized or separate reranking infrastructure.

This is not production authorization. `PRODUCTION_DECISION = NOT_AUTHORIZED`; `PRODUCTION = UNCHANGED`; `DEPLOYMENTS = 0`; `MERGES = 0`; `PR_87 = DRAFT`.
