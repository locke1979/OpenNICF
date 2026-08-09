# Development

Run `python -m pytest`. Install `pip install -e '.[test,qwen,knowledge]'` for QwenAgent and PostgreSQL-backed knowledge work. The bridge uses `SUBAGENT_RUNNER_COMMAND` as an adapter boundary; undocumented OpenClaw internals must not be embedded in OpenNICF.

Failure-audit work should be validated with mixed fixtures that cover logs, code, schema, docs, query output, and a controlled-verification gap. The expected result is a report whose JSON and human views share the same finding IDs.
