# Retrieval Architecture

**Scope:** the Vector Retrieval Layer. Not a RAG agent, and not a Gateway.
**Goal:** given a project's documents and code, index them locally and return the small
set of chunks that actually answer a query.

> **Phase 4** built the single-stage layer documented below (scanner → chunker → embedder
> → FAISS + SQLite → retriever).
> **Phase 5** added a second stage — a cross-encoder reranker between candidate retrieval
> and the final selection. The indexing half of this document is unchanged; the query half
> now has two stages. See `docs/reranking-architecture.md` and
> `benchmarks/reranker-results.md`.

---

## 1. Pipeline

```text
INDEXING (Phase 4, unchanged)
files
  │
  ▼
scanner        walk, hash, classify new / changed / unchanged / deleted
  │
  ▼
chunker        structured (AST / headings / paragraphs), fixed windows as fallback
  │
  ▼
embedder       Qwen3-Embedding-0.6B on CUDA, last-token pooling, L2-normalised
  │
  ▼
FAISS          IndexFlatIP = exact cosine over unit vectors
  │
  ▼
SQLite         metadata: path, lines, symbol, hashes, project

QUERY (Phase 5 adds the second stage)
query
  │
  ▼
retriever          embedding → Top-N candidate pool      (N = 10 by default)
  │
  ▼
reranker           cross-encoder reorders the pool       (8114, optional)
  │
  ▼
final Top-K        K = 5
  │
  ▼
context pack       path / lines / dense + rerank scores / content
```

Each stage is a separate module in `services/retrieval/`, so a stage can be replaced or
measured without disturbing the others.

**The reranker is optional at runtime.** If it is unreachable the pipeline returns the
dense Top-K and reports `reranker_used=false` rather than failing the request; verified for
connection-refused, HTTP 500 and disabled-by-configuration in `test_fallback.py`.

### Why a second stage was added

Dense retrieval embeds the query and each chunk **independently**, so it can only compare
them through a single vector; a cross-encoder reads them **together**. Phase 4 left correct
chunks at rank 6 and rank 13, and Phase 5 measured that reranking promotes them — the
rank-13 case reached rank 1 — lifting strict R@1 from 35.9 % to 53.8 % and end-to-end RAG
accuracy from 66.7 % to 83.3 %.

### Candidate pool sizing

Larger pools were expected to help and do not: the *pool* recall keeps rising with N while
the *final* ranking degrades, because more candidates give the reranker more chances to
promote a wrong passage. Pool 10 beat pool 20 on quality, latency and token cost in two
independent measurements. Phase 4's implicit "retrieve more and hand it all over" is
therefore wrong once a reranker is present.

## 2. Why these choices

| Decision | Rationale |
|---|---|
| **FAISS `IndexFlatIP`, not HNSW/IVF** | Exact search over ~150 chunks. An approximate index would add error and tuning for no measurable gain at this scale. The specification says not to optimise ANN prematurely. |
| **L2-normalise + inner product** | Inner product on unit vectors *is* cosine similarity, and cosine is what the model card declares. Verified against a reference implementation to 2.98e-08. |
| **Metadata in SQLite, not FAISS** | FAISS stores vectors and ids only. SQLite is the single authority for project filtering, file removal and incremental updates — and it can be queried and inspected. |
| **sentence-transformers, not llama.cpp** | The model repo ships `modules.json`, `1_Pooling` and `config_sentence_transformers.json`, i.e. it is packaged for this library. Last-token pooling and the official query prompts work without reimplementation. The specification explicitly says not to force a unified stack. |
| **stdlib `http.server`** | Two endpoints do not justify a web-framework dependency tree. |
| **Instructions on queries only** | The model card is explicit: "No need to add instruction for retrieval documents." Applying it to both sides is a common error; measured instructions help queries (+7.7 R@5). |

## 3. Chunking

Two strategies exist so they can be compared on retrieval metrics rather than on
intuition.

**fixed** — overlapping token windows, aligned to whole lines so line numbers stay
meaningful. Token counting uses the embedding model's own tokenizer; the fallback
heuristic is CJK-aware (Chinese ≈ 0.6 tokens/char, Latin ≈ 0.25) because a flat
chars/4 rule mis-sizes Chinese by roughly 2.4×.

**structured** — the default:
* **Python** via `ast`: module-level statements, functions, classes. A class that fits is
  kept whole; an oversized class splits into methods that still carry the class name in
  `symbol`. An oversized single function splits by fixed windows but keeps its symbol.
* **Markdown** by heading section; a heading is never separated from its body; an
  oversized section splits on paragraph boundaries, not raw token windows.
* **Anything else** by paragraph blocks.
* **Fallback**: if parsing fails, fixed windows are used and the chunk's `strategy`
  records `fixed(fallback)`, so the degradation is visible rather than silent.

Measured outcome: structured retrieves better (R@3 71.8 vs 64.1 at 1024 dim) and
preserves a `symbol` naming the origin, which fixed windows cannot.

## 4. Metadata schema

One row per chunk in SQLite:

| Column | Purpose |
|---|---|
| `faiss_id` | joins to the vector; assigned by the store |
| `chunk_id` | stable content-addressed id (project, path, lines, symbol) |
| `project` | collection namespace, e.g. `local-ai` |
| `source_path` | repo-relative path |
| `content_type` | `code` / `doc` / `config` / `data` |
| `language` | from the file extension |
| `start_line`, `end_line` | 1-based inclusive range |
| `content` | the chunk text |
| `symbol` | function / class / heading name when known |
| `strategy` | which chunker produced it |
| `commit_hash` | reserved for git-aware indexing |
| `mtime` | source modification time |
| `content_hash` | hash of the chunk text |
| `file_hash` | hash of the whole file, drives incremental decisions |
| `indexed_at` | when it was embedded |

A separate `files` table maps `(project, path) → file_hash`, which is what makes
incremental rebuilds cheap.

## 5. Incremental indexing

`scanner.scan_project()` classifies every indexable file as **new / changed / unchanged
/ deleted** by comparing SHA-1 file hashes against the `files` table.
`IndexBuilder.build()` then:

* removes chunks of **deleted** files,
* removes and re-adds chunks of **changed** files,
* embeds **nothing** for unchanged files.

Measured on a 19-file slice: a no-op rebuild embeds **0 chunks in 0.01 s** versus
**99 chunks in 3.21 s** cold — a 320× saving. Editing one file re-embeds only that file.

A whitespace-only edit changes the file hash and therefore forces a re-chunk, which is
correct: line numbers may have moved even when the text looks equal.

`IndexFlatIP` has no delete, so removal rebuilds the index from surviving vectors. A
parallel `vectors.npy` is kept so a rebuild never needs the model again.

## 6. Query path

```text
query
  │
  ├─ instruction prefix (query side only)  ← official format: "Instruct: …\nQuery:…"
  ▼
embedder  → 1024-d unit vector
  │
  ▼
FAISS IndexFlatIP.search(k)   exact cosine
  │
  ▼
SQLite lookup: path, lines, symbol, content
  │
  ▼
pack() → labelled SOURCE blocks
```

`pack()` deliberately does **not** concatenate chunks bare. It emits, per source:

```text
SOURCE 1
path: docs/local-fast-api.md
lines: 144-161
score: 0.6885
chunk_id: ac8b95d14dc914bd2c90
symbol: 7. Operational notes
content:
…
```

and prefixes the whole block with an instruction that these are the **only** project
facts available, and that "not present" must be stated rather than guessed. This is what
lets a downstream reader — human or model — attribute every claim and check it against
the source, instead of mistaking retrieved text for the model's own knowledge.

## 7. Service

`POST /v1/embeddings` — OpenAI-compatible; accepts a string or a list, honours
`dimensions` for Matryoshka truncation, and offers a local `query: true` flag that
applies the retrieval instruction.

`POST /v1/retrieve` — `{project, query, top_k, instruction_type, pack}` →
`{results:[{score, source_path, start_line, end_line, content, …}], context?}`.

`GET /health`, `/v1/models`, `/stats`.

Loopback only (`127.0.0.1:8113`). Concurrency is serialised behind a lock because a torch
module is not thread-safe for concurrent `encode`, and measurement shows concurrent GPU
work *loses* throughput rather than gaining it.

## 8. Resource posture

| Configuration | Free VRAM | Verdict |
|---|---:|---|
| embedding alone | 6185 MiB | ✅ |
| **4B + embedding** | **3015 MiB** | ✅ meets the ≥2 GB target |
| 9B + embedding | 720 MiB | ❌ do not co-reside |

Inference should be **serialised** between the embedding service and a generative
worker: running both at once costs the embedder 29 % and the generator 62–65 %.

## 9. What this layer deliberately does NOT do

* No reranker — that is Phase 5, and the rank-6/rank-13 finding is exactly its use case.
* No Qdrant or other vector database — FAISS is sufficient at this scale.
* No Gateway — the service stands alone on its own port.
* No agent loop — this layer retrieves and packs; it does not decide what to do next.
* No multi-tenant isolation or authentication — loopback only, single user.

## 10. Extension points

* **Cross-encoder reranking** over the top-20 candidates (Phase 5).
* **Query decomposition** for questions spanning several files.
* **Commit-aware indexing** — `commit_hash` is already in the schema.
* **Project namespacing** — `project` is already a first-class column and the API accepts it.
