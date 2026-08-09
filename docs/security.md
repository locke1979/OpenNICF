# Security boundaries

All imported content is untrusted evidence. QwenAgent may select only explicitly registered OpenNICF tools. Tools must validate input and authorization before touching storage or creating a diagnostic job. No tool exposes arbitrary shell, PowerShell, SQL, Proxmox or deployment execution.

Model routing is policy-enforced: `local_only` cannot reach LiteLLM OCI. Provider credentials and endpoints are runtime configuration only.

Evidence retrieval is ACL-first. Candidate rows are filtered by the requester's allowed scopes before ranking or model context assembly, and retrieval events retain the query hash, selected chunks, and route for auditability.
