# ADR 0009: Local embedding worker over Tailscale

The local Qwen3 embedding provider runs behind the OpenNICF provider-neutral
HTTP contract (`/health`, `/v1/models`, `/v1/embeddings`). Consumers use the
runtime-only `OPENNICF_EMBEDDING_BASE_URL` and optional service token; provider
logic does not know about Tailscale.

The worker binds to its Tailscale interface only. Tailscale is the transport
and network authorization boundary; the application token is defense in depth.
MagicDNS is preferred for the runtime URL. The public repository contains only
safe examples and no tailnet address, identity, or credentials.

`local_only` requests select the local Qwen space or a local CPU fallback and
fail closed if no local provider is available. They never select Gemini.
Gemini-001 and Gemini-2 remain cloud adapters and separate embedding spaces;
they are not proxied through Tailscale. Tailscale transport identity is
independent from `embedding_space_id`.

The service detects CUDA at runtime, uses FP16 when CUDA is available, limits
batch/input sizes for a constrained worker, retries smaller batches on OOM,
and falls back to the local CPU backend. Health metadata distinguishes the
provider, model, device, dimension, normalization, and fallback state without
exposing secrets or evidence.
