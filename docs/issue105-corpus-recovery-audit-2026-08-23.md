# Issue 105 corpus recovery audit

Status: `CORPUS=BLOCKED_CORPUS_RECOVERY`

## Evaluation policy

- `EVALUATION_MODE=AUTOMATED_NON_HUMAN_REVIEWED`
- `DECISION_GRADE_PRODUCTION_VALIDATION=false`
- `PRODUCTION_DECISION=NOT_AUTHORIZED`
- `HUMAN_REVIEW=OUT_OF_SCOPE`

No human labels, human approval, self-approval, or human-review dependency is
used by this track. The benchmark remains eligible for automated labels,
deterministic metrics, paired bootstrap confidence intervals, and machine-
checkable provenance once the source corpus is restored.

## Required artifact

The endpoint-bound text benchmark requires one preserved, ordered input artifact
containing:

- exactly 715 unique document/chunk IDs and source text;
- exactly 190 ordered query IDs and query text;
- labels and query references compatible with the automated benchmark;
- source-text and manifest digests;
- the historical preprocessing/template identity;
- authoritative corpus SHA-256:
  `cb6f30f5f69ff14f1bb75278c2e72ce312c570152651cee30bc2f70bc5071ee1`.

## Recovery search performed

Read-only search was limited to OpenNICF-owned locations:

- `/home/claw/.openclaw/workspace-opennicf`;
- the checked-out Issue 102 repository and preserved Issue 86 material;
- tracked paths, all local refs, reflogs, and unreachable Git objects;
- checked-in Issue 85/86/101/102 manifests, reports, labels, checkpoints, and
  vector provenance;
- the previously recorded durable paths
  `/var/lib/opennicf/eval/issue101`, `/var/lib/opennicf/eval/issue102`, and
  `/tmp/issue101_d4_resume`.

Search terms included the authoritative digest, 715/190 counts, issue and
evaluation roots, document/query/raw/input/checkpoint/vector manifest names,
and stable ID references.

## Findings and rejected candidates

No valid ordered source corpus or native HTTP checkpoint was found. The exact
digest occurs in audit and result provenance, not in a recoverable source-input
artifact.

- The checked-in retrieval fixture is 10 documents/10 queries, SHA-256
  `4e764b6a8fe18b92f92cc18e568c42087e5cd44402fdf5d2e9313bc59f3809be`; it is
  smoke-only and was not substituted.
- The preserved 190-query label records do not contain the missing 715 source
  texts and do not reconstruct them.
- Retained 0.6B/4B vectors are partial and cannot be combined with another
  corpus or used to infer source text.
- Issue 101 D4/2560 artifacts are partial and have incompatible historical
  transport/template provenance.
- Historical Issue 86 reports and D06 records contain counts/metrics but no
  admissible ordered source-input artifact or endpoint-native checkpoint.
- Git refs, reflogs, and unreachable objects contain no additional candidate
  source artifact.

## Required recovery action

The corpus owner must restore the original sanitized 715-document and
190-query ordered input artifact, including source and manifest digests and
preprocessing identity. After restoration, preserve the original, calculate its
SHA-256, freeze a canonical ordered manifest, and validate every ID and label
before serial HTTP inference.

Until then:

```text
CORPUS = BLOCKED_CORPUS_RECOVERY
TEXT_HTTP_BENCHMARK = NOT_EXECUTED
HUMAN_REVIEW = OUT_OF_SCOPE
```

No endpoint preflight was repeated. No local model, CT 308, or CT 305 was used;
production, re-embedding, and index state are unchanged.
