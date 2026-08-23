# CT 305 backup corpus-recovery audit

Status: `CORPUS_IN_BACKUP=NOT_FOUND`; `CORPUS_RECOVERY=BLOCKED`

This follow-up was a read-only forensic inspection of the existing backup. CT
305 was not started, stopped, restored, mounted, or modified. The archive and
production storage were not changed, and no data was copied out.

## Archive inspection

- Archive: `/mnt/raid0store/backups/local/dump/vzdump-lxc-305-2026_08_22-20_31_25.tar.zst`
- Size: `5,850,507,957` bytes
- Timestamp: `2026-08-22 20:36:27 UTC`
- Type: Zstandard-compressed vzdump archive containing a direct root-filesystem tar
- Archive listing result: `TAR_RC=0`
- Matching member names inspected: `28,403`
- Archive SHA-256: unavailable; the detached hash process disappeared without
  producing a digest. It was not restarted.

The archive was listed by streaming only. It was not restored, mounted,
extracted, or executed. No full root filesystem was unpacked.

## Candidate disposition

Matching names represented production runtime files, model-cache entries,
Python/library files, and configuration/runtime data. No member proved a
canonical corpus manifest, ordered document/query inputs, labels, or a complete
checkpoint. In particular, no candidate proved all of:

- 715 ordered unique document inputs with source text;
- 190 ordered unique query inputs with labels/references;
- preprocessing/template identity and complete provenance;
- compatibility with authoritative digest
  `cb6f30f5f69ff14f1bb75278c2e72ce312c570152651cee30bc2f70bc5071ee1`.

The 10/10 smoke fixture, vector-only material, production indexes, model
weights, and incomplete runtime artifacts are not substitutes. No selective
metadata extraction was needed because no validated small corpus candidate was
identified in the member listing.

```text
ARCHIVE_SHA256 = UNAVAILABLE
ARCHIVE_HASH_FAILURE = RECORDED
CORPUS_IN_BACKUP = NOT_FOUND
CORPUS_RECOVERY = BLOCKED
BENCHMARK_EXECUTED = false
CT305_STARTED = false
CT305_MODIFIED = false
PRODUCTION_STORAGE_MUTATED = false
BACKUP_RESTORED = false
CORPUS_COPIED = false
PRODUCTION = UNCHANGED
HUMAN_REVIEW = OUT_OF_SCOPE
```

Exact next action: obtain an externally owned authoritative corpus export or a
separately authorized controlled export of a validated sanitized artifact.
Do not restore or mount this backup for the benchmark.
