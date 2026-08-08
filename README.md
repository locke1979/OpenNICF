# OpenNICF

This repository is the implementation target for issues #1–#11. The bootstrap
contract is intentionally provider-neutral: OpenClaw discovers and invokes the
`subagent-pipeline` runner through `SUBAGENT_RUNNER_COMMAND`, while the runtime
application is orchestrated exclusively by QwenAgent.

```bash
python -m pytest
```

Production wiring must install `opennicf[qwen]`, configure the OpenNICF model
router with `LMSTUDIO_*` and `LITELLM_*` variables, and register only validated
OpenNICF tools with QwenAgent. See `docs/architecture.md`, the ADR, and
`AGENTS.md`.

