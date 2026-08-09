# Security boundaries

All imported content is untrusted evidence. QwenAgent may select only explicitly registered OpenNICF tools. Tools must validate input and authorization before touching storage or creating a diagnostic job. No tool exposes arbitrary shell, PowerShell, SQL, Proxmox or deployment execution.

Model routing is policy-enforced: `local_only` cannot reach LiteLLM OCI. Provider credentials and endpoints are runtime configuration only.

Evidence retrieval is ACL-first. Candidate rows are filtered by the requester's allowed scopes before ranking or model context assembly, and retrieval events retain the query hash, selected chunks, and route for auditability.

Failure-audit reports preserve contradictory evidence and unresolved hypotheses. Controlled verification requests are brokered on-prem and only carry the minimal operation class required for the check; the model never receives a fabricated live-database result.

Namespace-aware ingestion also sanitizes credential-bearing metadata and private endpoints before chunking or embedding, and domain agents only see their principal domain unless a caller explicitly delegates additional domain IDs in the retrieval filter.
