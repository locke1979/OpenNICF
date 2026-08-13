# OpenNICF

This repository is the implementation target for the core OpenNICF issue track. The bootstrap
contract is intentionally provider-neutral: OpenClaw discovers and invokes the
`subagent-pipeline` runner through `SUBAGENT_RUNNER_COMMAND`, while the runtime
application is orchestrated exclusively by QwenAgent.

```bash
python -m pytest
```

Production wiring must install `opennicf[qwen,knowledge]`, configure the OpenNICF model
router with `LMSTUDIO_*` and `LITELLM_*` variables, and register only validated
OpenNICF tools with QwenAgent. Failure audits use the brokered verification
boundary and evidence-grounded reports described in `docs/architecture.md`.
See the ADR and `AGENTS.md`.

The versioned retrieval evaluation workload and its provenance-backed result
manifest are documented in [`docs/retrieval-benchmark.md`](docs/retrieval-benchmark.md).
The OpenNICF infra inventory is tracked in
[`docs/infrastructure.md`](docs/infrastructure.md).
