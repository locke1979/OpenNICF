# Issue 102 HTTP embedding benchmark

This is an isolated, automated benchmark for the two model IDs served at
`http://192.168.1.137:1234`. It never writes under `evaluation/results/issue101`
and does not alter production indexes.

The runner sends one UTF-8 input in a single-item list per request with the exact configured model ID.
It rejects explicit model mismatches, fallback responses, malformed results,
non-finite vectors, and unexpected native dimensions. The 768-dimensional
evaluation vector is the first 768 native values followed by L2 normalization;
the raw vector is retained only by digest in the result manifest.

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

The endpoint currently returns 2,560-dimensional vectors for both IDs. The
text response identifies itself as `text-embedding-qwen3-embedding-4b`; the VL
response identifies itself as the text model and is rejected as an alias. The
VL model remains blocked until its serving identity and native dimension are
proven.

All results are `AUTOMATED_NON_HUMAN_REVIEWED` evidence. Existing T00/T01
spaces and canonical Issue #101 artifacts remain immutable.
