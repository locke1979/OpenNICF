# Issue #86 Phase 0 — architecture delta

Status: `ARCHITECTURE_EVALUATION = PHASE_0_CONTRACT_CORRECTED`

This document is an evaluation design. It does not change the active model,
embedding space, production index, service configuration, or QwenAgent
runtime authority.

## Current main path

```text
channel / watcher
  -> IngestionService
  -> artifact validation + immutable landing/object store
  -> parser-specific ParsedBlock values
  -> KnowledgePlatform.ingest
  -> stable source/version/artifact/chunk records
  -> LocalFirstEmbeddingService / configured embedding backend
  -> knowledge_embeddings, space-scoped by embedding_space_id
  -> ACL/domain/system-filtered candidates
  -> lexical + cosine hybrid ranking, or exact/symbol/log paths
  -> bounded benchmark compaction where invoked
  -> provenance-aware DomainTools result package
  -> QwenAgentRuntime / OpenNICFChatModel / ModelGateway
  -> answer and citations
```

### Exact current boundaries

| Boundary | Current implementation | Phase 0 finding |
| --- | --- | --- |
| Ingestion lifecycle | `src/opennicf/ingestion.py`: `IngestionQueue`, `IngestionService.submit_file`, `process_next`, `_process_job` | Durable queue, landing store, retries/dead letters and hash verification exist. Visual derivations must be added as representation-scoped work, not as a second ingestion loop. |
| Validation/parsing | `ArtifactValidationError`, `_validate_payload_size`, `_validate_relative_path`, `_validate_mime_suffix`, `_expand_archive`, `_parse_job` | Archive traversal and parser limits exist. Parser failures need explicit visual retry/quarantine classification. |
| Canonical evidence | `knowledge/models.py`: `KnowledgeSource`, `KnowledgeSourceVersion`, `ArtifactRecord`, `ChunkRecord`, `ParsedBlock` | The source/version/artifact/chunk chain is the provenance anchor. It must remain immutable and unchanged for visual derivatives. |
| Text embeddings | `knowledge/embeddings.py`: `EmbeddingSpace`, `QwenEmbeddingBackend`, `HttpEmbeddingBackend`, `LocalFirstEmbeddingService`, `EmbeddingMigration` | Text and VL spaces must be distinct despite equal dimensions. The existing active production space is not an evaluation target to mutate. |
| Persistence | `knowledge/store.py`: `KnowledgePlatform`, `MemoryKnowledgeStore`, `PostgresKnowledgeStore` | Existing writes are bundle/chunk/embedding-centric; representation persistence needs a separate identity/table and idempotent upsert contract. |
| Retrieval | `knowledge/retrieval.py`: `_lexical_terms`, `_lexical_score`, `HybridRetriever.search`; `domain_agent.py`: exact, symbol, semantic, log tools | ACL and namespace filtering precede scoring. RRF must consume ranked lists, never add raw scores from unrelated spaces. |
| Symbol retrieval | `knowledge/code_index.py`; `DomainTools._symbol_search`; `search_code_symbols` | Keep as a first-stage signal in T2; do not replace it with semantic reranking. |
| Compaction | `knowledge/benchmark.py:compact_hits` | Current implementation is benchmark/context-measurement code. Phase 1 must introduce an evaluation-only post-rerank context-selection boundary and prove it is invoked before tool packaging. |
| Provenance/tool boundary | `DomainTools._package_hits`, `OpenNICFTools`, `QwenAgentRuntime`, `OpenNICFChatModel` | QwenAgent remains the only runtime orchestrator. Legacy `OpenNICFTools.search_evidence` exposes a reduced shape; evaluation results must use the full provenance package. |

## Proposed evaluation path

```text
artifact
  -> immutable original + existing source/version/artifact provenance
  -> existing parser / bounded renderer
       -> text blocks -> stable text chunks -> TEXT_4B_Q4_768
       -> page/region/mixed renditions -> VL_2B_W4_768
  -> representation registry (one identity per canonical rendition)
  -> independent embedding variants keyed by (representation_id, embedding_space_id)

query
  -> ACL/domain/system filters
  -> lexical + symbol-aware ranked list
  -> text dense ranked list
  -> optional VL dense ranked list
  -> deterministic RRF candidate fusion
  -> candidate recall measurement
  -> capability-routed reranking
       text-only -> text reranker
       visual/mixed -> VL reranker
  -> representation/source deduplication and post-rerank context budget
  -> provenance-aware QwenAgent tool result
  -> fixed generator/RAG evaluator
```

The evaluation path is selected only by an explicit non-production feature
flag and an experiment manifest. The default path remains current production
behavior. No evaluation result can write to the active ANN/index path.

## Required design decisions

1. `representation_id` identifies a canonical derived rendition independently
   of model, quantization, runtime, or semantic space. Its canonical tuple is
   `source_version_id`, representation type, stable page/region locator,
   optional parent chunk, rendition hash, render name, and render version.
   `embedding_space_id` is the compatibility boundary for attached embedding
   variants. Proposed examples are `TEXT_QWEN3_4B_Q4_K_M_768_V1`,
   `VL_QWEN3_2B_BF16_768_V1`, `VL_QWEN3_2B_W4_768_V1`, and
   `VL_QWEN3_2B_W8_768_V1`; they are never mixed in one vector comparison.
2. RRF is the initial fusion method. Lexical, text dense, VL dense, and
   reranker scores are not numerically comparable without a separately
   validated calibration study.
3. First-stage candidate recall is reported before reranking. A reranker cannot
   recover a representation absent from its candidate set.
4. Reranker inputs carry candidate ID and full provenance. Reranking may change
   order only; it cannot rewrite evidence metadata.
5. Deduplication is representation-aware and source-aware: multiple regions or
   pages from one source must not crowd out unrelated evidence solely by count.
6. Context selection occurs after reranking and enforces token, image, pixel,
   representation-count, and source-neighbor budgets.
7. Visual failure is isolated: text parsing/chunking and text embedding remain
   usable if rendering or VL embedding fails. Failed visual representations are
   retryable or quarantined with a classified reason.
8. Ingested visual/text content remains untrusted evidence and cannot alter
   workflow, tools, ACLs, limits, or model routing.

## Verified runtime/tool sequence

The current path is:

```text
DomainTools.search_* / search_domain_evidence
  -> _filters() and domain/ACL enforcement
  -> KnowledgePlatform.search / exact / symbol / log route
  -> HybridRetriever.search or bounded domain-specific candidate ranking
  -> DomainTools._package_hits()
  -> DomainAgent.permitted_qwen_tools() / OpenNICFTools.as_qwen_tools()
  -> QwenAgentRuntime.run()
  -> OpenNICFChatModel.chat()
  -> ModelGateway request to the generator
```

`DomainTools._package_hits()` groups results by source version and emits
provenance-bearing evidence references, including source/version, locator,
excerpt hash, embedding space, ACL/domain/system metadata, and bounded size
estimates. `OpenNICFTools.search_evidence()` is a legacy reduced adapter and
does not expose the complete package; issue #86 must use the domain package or
an evaluation adapter that preserves the full fields.

`knowledge/benchmark.py:compact_hits()` currently performs four-chunk/256-token
deduplication for the benchmark path. The inspected production tool path does
not prove that this compactor is invoked before QwenAgent context assembly, and
the generator adapter accepts the tool result returned by QwenAgent. Therefore
the safest Phase 1 insertion point is an evaluation-only context selector
between post-rerank result ordering and `_package_hits()`, with an explicit
result phase and budgets. It must be tested without changing the production
tool response or default route. Text truncation, image count, and pixel limits
must be recorded rather than inferred from the model gateway.

## Exact proposed code changes (later phases; not implemented in Phase 0)

| Area | Proposed change | Compatibility rule |
| --- | --- | --- |
| `knowledge/models.py` | Add frozen `RepresentationRecord`, `RepresentationStatus`, and capability metadata; extend `EvidenceHit`/result package with optional representation fields | Existing constructors and legacy text hits remain valid through defaults. |
| `knowledge/store.py` | Add `save_representation`, `get_representation`, `list_representations`, idempotent rendition-hash lookup, and representation-aware candidate retrieval | Existing `save_bundle`, `search_candidates`, snapshots, and production tables remain unchanged until a separately authorized migration. |
| `knowledge/embeddings.py` | Add modality-aware provider-neutral `RepresentationEmbeddingBackend` contract and explicit space validation | Do not alter active production backend selection or `EmbeddingMigration`. |
| `knowledge/retrieval.py` | Add evaluation-only ranked-list providers, deterministic RRF, candidate recall capture, representation/source deduplication | Existing `HybridRetriever.search` behavior remains the default. |
| new `knowledge/reranking.py` | Add bounded `Reranker` protocol, text/VL capability checks, pair/token/pixel limits, timeout, fail-closed result, and deterministic ties | No model-specific orchestration loop; adapters are services called by the retrieval path. |
| `ingestion.py` | Add bounded renderer/representation stage after existing parse/chunk stages | Text bundle commit is independent from visual derivation status; no document content can control processing. |
| `domain_agent.py` / `tools.py` | Add evaluation-only search route returning candidate/final/context phases and full provenance | QwenAgent tool allow-list and domain boundary remain authoritative. |
| context assembly | Extract post-rerank compactor from benchmark-only logic into an evaluation service | Must preserve current four-chunk/256-token default when feature flag is absent. |
| configuration | Add disabled-by-default `OPENNICF_EVAL_ARCHITECTURE`, manifest path, budgets, and space IDs | Production configuration is not changed by this branch and no flag is enabled by default. |

## Phase 0 exit criteria

- architecture and schema/API contracts reviewed;
- independent spaces and provenance identity are explicit;
- no production path or active configuration is changed;
- test matrix covers isolation, ACLs, provenance, determinism, limits, and
  failure behavior;
- blockers are recorded before model acquisition.
