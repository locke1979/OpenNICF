class SinquerCaseService:
    """Sanitized fixture for Criminal/SINQUER ownership tests."""

    def load_case(self, case_id: str) -> str:
        return f"case:{case_id}"


def persist_case(case_id: str) -> str:
    return SinquerCaseService().load_case(case_id)
