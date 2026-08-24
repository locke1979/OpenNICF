# Issue 102 corpus recovery audit

Status: `TEXT_HTTP_BENCHMARK=BLOCKED_CORPUS_RECOVERY`

## Required artifact

The HTTP benchmark requires the authoritative ordered evaluation inputs:

- 715 unique documents/chunks;
- 190 ordered queries;
- complete labels and source-text digests;
- authoritative corpus digest:
  `cb6f30f5f69ff14f1bb75278c2e72ce312c570152651cee30bc2f70bc5071ee1`.

The expected input schema is:

```json
{"documents":[{"id":"...","text":"..."}],"queries":[{"id":"...","text":"..."}]}
```

## Scoped locations searched

Read-only recovery searched:

- the current Issue 102 repository and its tracked paths/history;
- the main-agent workspace and preserved OpenNICF Issue 86 worktree;
- registered/preserved OpenNICF worktrees and branches;
- checked-in Issue 86/101 manifests, checkpoints, reports, and result bundles;
- Git reflogs and unreachable-but-referenced Git objects;
- GitHub repository branches, commits, PR #103/PR #87 metadata, issue #86/#101/#102
  evidence, comments, and referenced artifacts;
- project-owned durable paths `/var/lib/opennicf/eval/issue101` and
  `/var/lib/opennicf/eval/issue102` (both absent in this environment);
- explicitly owned historical paths recorded by Issue 101, including
  `/tmp/issue101_d4_resume/` (raw document/query files absent).

Search terms included the authoritative digest prefix, expected 715/190 counts,
Issue 86/101/102 identifiers, manifest/result paths, ordered-ID digests, and
native vector/checkpoint references.

## Findings

No authoritative ordered source-document corpus, complete query-input file, or
native HTTP vector checkpoint was found. The exact authoritative digest appears
only in audit/report evidence, not in a recoverable input artifact.

Available smoke data is not a substitute:

- `tests/fixtures/retrieval_benchmark/v1/manifest.json` — SHA-256
  `4e764b6a8fe18b92f92cc18e568c42087e5cd44402fdf5d2e9313bc59f3809be`;
  only 10 documents and 10 queries.
- The 190 labels are preserved, but the ordered 715 document inputs are absent.
- Historical reports and vector counts do not reconstruct source text and cannot
  be used to infer it from embeddings.

## Rejected candidates

- The 10/10 sanitized fixture: incomplete and different manifest identity.
- Retained 0.6B/4B vectors: only partial document coverage and insufficient
  provenance; unsafe to append.
- Issue 101 D4/2560 artifacts: partial (39 documents, zero accepted queries)
  and transport/tokenization-incompatible.
- Historical 4B GGUF: discovered SHA-256
  `6fc91b94aa945a268991a1a94d77129d674a12cbe20c94575889d73071f30990` differs
  from the expected artifact digest; quarantined and not executed.
- Preserved Issue 86 complete vector counts: rejected because recorded runtime
  topology/provenance is incompatible with this endpoint-bound experiment.
- D06-native-1024 and reported D06 completion: no materialized checkpoint or
  ordered source inputs exist locally.

## Required recovery action

The evaluation/corpus owner must restore the original sanitized 715-document and
190-query ordered input artifact, with its source and manifest digests. After
recovery, freeze and validate all IDs and labels before serial HTTP embedding.
Do not merge the smoke fixture with another partial corpus or regenerate records.

No endpoint preflight was repeated, no local model was loaded, and no production
or evaluation index was mutated during this audit.

