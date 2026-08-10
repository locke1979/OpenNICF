# ADR 0008: Final synthetic E2E acceptance

Issue #11's acceptance is a deterministic composition test, not a production
data export.  `tests/test_issue11_final_e2e.py` runs the #24 DomainRouter and
coordinator over synthetic logs, Splunk CSV, code, documentation, schema, and
query output.  It uses the active 768-dimensional embedding space and asserts
that a different space is rejected before search.

The audit engine emits fact, inference, hypothesis, and missing-evidence
findings.  Missing live facts are represented by a signed, read-only,
pull-based diagnostic job.  The SQLcl path is a deterministic fixture runner;
credentials and private endpoints are never present.  The worker can resume
from its spool, and the query-output provenance hook returns the result to the
same evidence graph.

The Webex and Telegram adapters share the event ledger and return a bounded
machine-readable final report attachment.  Duplicate events and attachments
are suppressed, unauthorized actors are rejected, and reply targets retain
conversation/thread correlation.  Model completion routing is tested
separately from embedding routing: local-safe work uses LM Studio, complex
work may escalate to OCI, and `local_only` fails closed when local health is
false.  Deployment health failure proves rollback to the known-good release.

Each run writes a generated, sanitized manifest below pytest's temporary
directory.  It includes hashes, IDs, route decisions, embedding-space ID, and
contract names only.
