def validate_return_code(actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError("producer_return_code_mismatch")


def run_sco_batch(execution_id: str) -> None:
    validate_return_code(12, 0)  # line 42 in the sanitized runtime fixture
