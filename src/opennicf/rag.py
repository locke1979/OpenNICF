"""Small provenance-preserving RAG boundary; storage implementation is added by #5."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Evidence:
    source_id: str
    text: str
    locator: str

class EvidenceIndex:
    def __init__(self):
        self._items: list[Evidence] = []

    def add(self, evidence: Evidence) -> None:
        self._items.append(evidence)

    def search(self, query: str) -> list[Evidence]:
        terms = set(query.lower().split())
        return [e for e in self._items if terms & set(e.text.lower().split())]

