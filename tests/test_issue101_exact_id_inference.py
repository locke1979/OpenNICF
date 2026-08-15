import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "tools" / "issue101_exact_id_inference.py"
SPEC = importlib.util.spec_from_file_location("exact_id", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def vector():
    return [0.25] * 2560


def test_one_input_one_output_and_exact_dimension():
    assert MODULE.validate_output({"data": [{"index": 0, "embedding": vector()}]}) == vector()


def test_multiple_outputs_fail_closed():
    try:
        MODULE.validate_output({"data": [{"embedding": vector()}, {"embedding": vector()}]})
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multiple embeddings must fail closed")


def test_bad_dimension_and_nonfinite_fail_closed():
    for bad in ([0.0], [float("nan")] * 2560):
        try:
            MODULE.validate_output({"data": [{"embedding": bad}]})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid vector must fail closed")


def test_source_and_vector_digests_are_deterministic():
    assert MODULE.source_digest("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert MODULE.vector_digest(vector()) == MODULE.vector_digest(vector())


def test_manifest_record_preserves_id_and_sequence():
    expected = {"id": "q-1", "sequence_index": 3, "input_sha256": "digest"}
    record = MODULE.enrich({"id": "q-1", "embedding": vector()}, expected, {"gpu_layers": 29})
    assert record["id"] == "q-1"
    assert record["sequence_index"] == 3
    assert record["gpu_layers"] == 29


def test_newline_transport_preserves_bytes_and_renders_one_prompt():
    text = "first\n\nsecond\\n\u2028third"
    prompt = MODULE.render_prompt("documents", text)
    assert prompt == text
    assert MODULE.source_digest(prompt) == MODULE.source_digest(text)
    assert MODULE.EMBEDDING_SEPARATOR not in prompt


def test_query_template_is_explicit_and_does_not_rewrite_source():
    text = "a\r\nb\n\nend"
    prompt = MODULE.render_prompt("queries", text)
    assert prompt.endswith(text)
    assert "Query:" in prompt
    assert MODULE.EMBEDDING_SEPARATOR not in prompt


def test_json_stdout_requires_exactly_one_document():
    payload = '{"data":[{"embedding":' + str(vector()).replace("'", "") + '}]} '
    assert MODULE.parse_single_json_stdout(payload)["data"]
    for bad in (payload + '{"extra":1}', "", "not-json"):
        try:
            MODULE.parse_single_json_stdout(bad)
        except (ValueError, MODULE.json.JSONDecodeError):
            pass
        else:
            raise AssertionError("ambiguous or malformed stdout must fail closed")


def test_identity_ids_isolate_old_space():
    assert MODULE.PREPROCESSING_ID != "historical-t01"
    assert MODULE.EMBEDDING_SPACE_ID != "issue101-d4-official-newline-splitting"
