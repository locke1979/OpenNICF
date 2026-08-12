"""Model-independent qualification boundary for the pinned Qwen3 reranker.

This module deliberately does not acquire or execute weights.  A real adapter
may only be constructed after the artifact manifest is exact and executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

from opennicf.evaluation import RerankerOutOfMemory, RerankerTimeout


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"
MODEL_SIZE = 1_191_588_280
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
CONFIG_SHA256 = "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
          'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
          '<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def validate_artifact_manifest(manifest: dict[str, object]) -> tuple[str, ...]:
    """Fail closed unless the official identity and a concrete runtime are pinned."""
    expected = {
        "model": MODEL_ID, "revision": MODEL_REVISION, "license": "apache-2.0",
        "filename": "model.safetensors", "sha256": MODEL_SHA256, "size_bytes": MODEL_SIZE,
    }
    errors = [f"{key} does not match pinned artifact" for key, value in expected.items()
              if manifest.get(key) != value]
    for key in ("runtime", "runtime_version"):
        if not manifest.get(key):
            errors.append(f"{key} is required")
    for key, expected_hash in (("tokenizer_sha256", TOKENIZER_SHA256), ("config_sha256", CONFIG_SHA256)):
        if manifest.get(key) != expected_hash:
            errors.append(f"{key} does not match pinned artifact")
    if manifest.get("status") != "EXECUTABLE":
        errors.append("status must be EXECUTABLE")
    return tuple(errors)


def format_pair(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


@dataclass(frozen=True)
class AdapterResult:
    scores: tuple[float, ...]
    ordered_indices: tuple[int, ...]
    truncated: tuple[bool, ...]


class DeterministicQwenRerankerAdapter:
    """Testable adapter using injected tokenization and final yes/no logits.

    The runner receives padded token batches and returns ``(no, yes)`` logits.
    This keeps timeout/OOM classification and prompt semantics testable without
    pretending a synthetic runner establishes model quality.
    """

    def __init__(self, encode: Callable[[str], list[int]],
                 runner: Callable[[Sequence[Sequence[int]], float], Sequence[tuple[float, float]]],
                 *, max_length: int = 8192, batch_size: int = 8, timeout_s: float = 30.0) -> None:
        if min(max_length, batch_size) <= 0 or timeout_s <= 0:
            raise ValueError("limits must be positive")
        self.encode, self.runner = encode, runner
        self.max_length, self.batch_size, self.timeout_s = max_length, batch_size, timeout_s

    def score(self, pairs: Iterable[tuple[str, str]], instruction: str = DEFAULT_INSTRUCTION) -> AdapterResult:
        prefix, suffix = self.encode(PREFIX), self.encode(SUFFIX)
        budget = self.max_length - len(prefix) - len(suffix)
        if budget < 1:
            raise ValueError("max_length cannot preserve prompt prefix and suffix")
        encoded, truncated = [], []
        for query, document in pairs:
            body = self.encode(format_pair(query, document, instruction))
            truncated.append(len(body) > budget)
            encoded.append(prefix + body[:budget] + suffix)
        scores: list[float] = []
        try:
            for start in range(0, len(encoded), self.batch_size):
                logits = self.runner(encoded[start:start + self.batch_size], self.timeout_s)
                if len(logits) != len(encoded[start:start + self.batch_size]):
                    raise ValueError("runner returned wrong score count")
                for no_logit, yes_logit in logits:
                    if not math.isfinite(no_logit) or not math.isfinite(yes_logit):
                        raise ValueError("runner returned nonfinite logits")
                    # Stable two-class softmax; higher means more relevant.
                    scores.append(1.0 / (1.0 + math.exp(max(-709.0, min(709.0, no_logit - yes_logit)))))
        except TimeoutError as error:
            raise RerankerTimeout("pinned Qwen reranker timed out") from error
        except MemoryError as error:
            raise RerankerOutOfMemory("pinned Qwen reranker exhausted memory") from error
        order = tuple(sorted(range(len(scores)), key=lambda index: (-scores[index], index)))
        return AdapterResult(tuple(scores), order, tuple(truncated))
