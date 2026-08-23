# Issue 102 HTTP embedding benchmark

This is an isolated, automated benchmark for the two model IDs served at
`http://192.168.1.137:1234`. It never writes under `evaluation/results/issue101`
and does not alter production indexes.

The application and evaluation configuration use one shared endpoint with
separate explicit model IDs:

```text
OPENNICF_EMBEDDING_BASE_URL=http://192.168.1.137:1234
OPENNICF_TEXT_EMBEDDING_MODEL=text-embedding-qwen3-embedding-4b
OPENNICF_VL_EMBEDDING_MODEL=qwen.qwen3-vl-embedding-2b
```

The production activation gate is `OPENNICF_EMBEDDING_REMOTE_ENABLED=false`
until readiness, backup, and rollback gates are approved.

The runner sends one UTF-8 input in a single-item list per request with the exact configured model ID.
It rejects explicit model mismatches, fallback responses, malformed results,
non-finite vectors, and unexpected native dimensions. The 768-dimensional
evaluation vector is the first 768 native values followed by L2 normalization;
the native vector and its digest are checkpointed outside the repository.

Run from the isolated evaluation guest after copying the repository and config:

```bash
python3 tools/run_issue102_http_embeddings.py \
  --config evaluation/issue102-http-benchmark.json \
  --input-root /var/lib/opennicf/eval/issue101 \
  --output-root /var/lib/opennicf/eval/issue102 \
  --base-url http://192.168.1.137:1234 \
  --model text-embedding-qwen3-embedding-4b \
  --model qwen.qwen3-vl-embedding-2b \
  --service-token "$OPENNICF_EMBEDDING_SERVICE_TOKEN"
```

Run the read-only contract preflight first:

```bash
python3 tools/preflight_issue102_endpoint.py \
  --base-url http://192.168.1.137:1234 \
  --service-token "$OPENNICF_EMBEDDING_SERVICE_TOKEN"
```

The endpoint currently returns 2,560-dimensional vectors for both IDs. The
text response identifies itself as `text-embedding-qwen3-embedding-4b`; the VL
response identifies itself as the text model and is rejected as an alias. The
VL model remains blocked until its serving identity and native dimension are
proven.

The current observed status is `TEXT_ENDPOINT_USABLE=true` and
`VL_ENDPOINT_STATUS=ALIASED_OR_UNSUPPORTED`: the VL request returns the text
model identity and a 2,560-dimensional vector. Endpoint-host RAM and queue
capacity remain `UNVERIFIED`.

The native LM Studio audit confirms the VL key is loaded as a `qwen3vl` GGUF
generation/VLM instance (`Q5_K_S`, 2B, 2,049,980,576 bytes), not as an
embedding instance. With the text embedding instance unloaded, the OpenAI
embedding route reported no embedding model loaded; the native v0 embedding
route likewise reported no embedding model, while chat completion succeeded.
Image-only and text-plus-image embedding inputs were rejected by the endpoint.
The final endpoint state was restored to the text embedding instance only.
See `evaluation/results/issue102-lmstudio-native-audit-2026-08-23.json`.

All results are `AUTOMATED_NON_HUMAN_REVIEWED` evidence. Existing T00/T01
spaces and canonical Issue #101 artifacts remain immutable.
