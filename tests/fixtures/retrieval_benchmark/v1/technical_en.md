# Technical guide: evidence compaction

The knowledge index keeps immutable document versions and records the artifact hash before compaction. Incremental indexing selects only new versions.

Retrieval applies the ACL policy before the model and keeps a source locator on every chunk. The context package is bounded by chunk count and tokens.
