# Operational note

The ingestion pipeline should preserve immutable originals and derived chunks.

If a document is re-imported without changes, the content hash must stay stable.

If the content changes, the source keeps its identity and the version number advances.
