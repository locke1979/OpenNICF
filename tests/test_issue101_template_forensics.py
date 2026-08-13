import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "tools" / "build_issue101_template_forensics.py"
SPEC = importlib.util.spec_from_file_location("template_forensics", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_historical_and_official_query_bytes_are_distinct():
    assert MODULE.historical_query("a  b") == f"Instruct: {MODULE.INSTRUCTION} Query: a b"
    assert MODULE.official_query("a  b") == f"Instruct: {MODULE.INSTRUCTION}\nQuery:a  b"
    assert MODULE.historical_query("x").encode() != MODULE.official_query("x").encode()


def test_document_difference_is_whitespace_normalization_only():
    assert MODULE.historical_document("a\n b") == "a b"
    assert MODULE.official_document("a\n b") == "a\n b"


def test_template_identity_is_embedding_space_identity():
    historical = MODULE.identity("template_4b_historical_t01", "h", "d", "source-h")
    official = MODULE.identity("template_qwen3_official", "o", "d", "source-o")
    assert historical["sha256"] != official["sha256"]
