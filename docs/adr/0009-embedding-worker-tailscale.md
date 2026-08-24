# ADR 0009: Shared OpenAI-compatible embedding endpoint

The shared Qwen3 endpoint runs behind the OpenAI-compatible contract
(`/v1/models`, `/v1/embeddings`). Consumers use
`OPENNICF_EMBEDDING_BASE_URL`, `OPENNICF_TEXT_EMBEDDING_MODEL`, and
`OPENNICF_VL_EMBEDDING_MODEL`, plus an optional service token; provider logic
does not depend on the transport.

The current migration endpoint is `http://192.168.1.137:1234`. Its URL is
configuration, not a production activation claim. The explicit
`OPENNICF_EMBEDDING_REMOTE_ENABLED` gate defaults to false and production
startup fails closed while it is disabled.

Text uses the exact model ID
`text-embedding-qwen3-embedding-4b`, validates a native 2,560-dimensional
response, and derives the storage vector by prefix truncation to 768 followed
by L2 renormalization. A response model mismatch, fallback, malformed,
nonfinite, zero, duplicate, or wrong-dimension result is rejected.

The VL model ID is configured independently as
`qwen.qwen3-vl-embedding-2b`, but remains `VL_UNVERIFIED` until `/v1/models`
and a multimodal embedding probe prove the task identity and output contract.
The current endpoint returns the text model identity for that request, so VL is
not enabled.

The legacy CT305 worker and isolated CT308 local inference remain separate
assets. This migration does not restart, mutate, deploy, or re-embed either
path. Evaluation HTTP vectors use a new endpoint-bound space and never mix
with local llama.cpp vectors.
