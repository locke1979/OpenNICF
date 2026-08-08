"""QwenAgent is the sole application-level orchestration authority.

The optional dependency is imported at runtime so contract and security tests do
not require model weights or a live provider. Production startup must install
the ``qwen`` extra and inject an OpenNICF model adapter.
"""
class QwenAgentRuntime:
    def __init__(self, model_adapter, tools: dict):
        try:
            from qwen_agent.agents import Assistant
        except ImportError as exc:
            raise RuntimeError("QwenAgent is required; install opennicf[qwen]") from exc
        self._agent = Assistant(llm=model_adapter, function_list=list(tools.values()))

    def run(self, request: str):
        return self._agent.run(messages=[{"role": "user", "content": request}])

