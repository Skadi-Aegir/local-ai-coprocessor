# Embedding performance — Phase 4

**Model:** `local-embedding` = `Qwen/Qwen3-Embedding-0.6B` rev `97b0c614`, Apache-2.0
**Backend:** sentence-transformers 6.1.0 on PyTorch 2.11.0+cu128, isolated venv
**Measured:** 2026-09-30 on RTX 5070 Ti Laptop (sm_120), service on `127.0.0.1:8113`
**Raw data:** `results/embedding_throughput.json`, `results/embedding_coresident_*.json`

---

## 1. Load and footprint

| Metric | Value |
|---|---|
| Load time | 1.7 s |
| Device | `cuda:0` |
| **VRAM delta over desktop baseline** | **1246 MiB** |
| VRAM attribution reported by torch | 2360.6 MiB |
| Output | float32 `(n, 1024)`, L2-normalised |
| Deployed max sequence length | 8192 (model supports 32768) |

The two VRAM figures differ because torch reports its own reserved pool while the
nvidia-smi delta measures what the card actually lost. **1246 MiB is the number to
budget with**; 2400 MiB is the conservative figure used in config.

## 2. Throughput — batch sweep

Measured through the live HTTP service, median of 5 rounds per batch size, texts drawn
from this project so token lengths are representative.

| Batch | Median wall | **Chunks/s** | ms/chunk | VRAM peak |
|---:|---:|---:|---:|---:|
| 1 | 0.0908 s | 11.0 | 90.85 | 5996 MiB |
| 8 | 0.1376 s | 58.1 | 17.20 | 6030 MiB |
| **32** | **0.1792 s** | **178.6** | **5.60** | 6267 MiB |
| 64 | 0.3318 s | 192.9 | 5.18 | 6262 MiB |

**Recommended batch size: 32.**

Batch 64 buys **+8 %** throughput for **+85 %** wall latency (0.179 s → 0.332 s). For an
indexing job the extra throughput is irrelevant; for anything interactive the extra
latency is directly felt. VRAM is flat from 32 upward, so 64 is not buying memory
headroom either.

Scaling from batch 1 to 32 is **16×** for a 32× larger batch — most of the gain is
already captured by batch 8 (5× over batch 1).

## 3. Matryoshka truncation behaviour

Truncating the 1024-dim vector and re-normalising preserves the top result down to 256,
then breaks:

| Dimension | Top-1 document correct? |
|---|---|
| 1024 | ✅ |
| 512 | ✅ |
| 256 | ✅ |
| **128** | ❌ |
| **64** | ❌ |

**Practical floor ≈ 256 dimensions.** The deployment default remains 1024 because the
retrieval sweep (see `retrieval-results.md`) shows a measurable quality cost at 256
(R@3 59.0 vs 71.8) that outweighs the storage saving at this corpus size.

## 4. Instruction effect

The query-side instruction changes the query vector materially — cosine between the
plain and instructed encoding of the same query is **≈ 0.88**, so it is genuinely
applied rather than ignored. Its effect on retrieval is quantified in
`retrieval-results.md`: **+7.7 R@5 points** on the best configuration.

Documents are encoded **without** any instruction, per the model card ("No need to add
instruction for retrieval documents").

## 5. VRAM co-residency

| Configuration | GPU used | **Free** | Verdict |
|---|---:|---:|---|
| embedding alone | ~5760 MiB | 6185 MiB | ✅ |
| **local-fast (4B) + embedding** | **8930 MiB** | **3015 MiB** | ✅ **meets the ≥2 GB target** |
| local-smart (9B) + embedding | 11225 MiB | **720 MiB** | ❌ **fails the ≥2 GB target** |

**4B + embedding is a viable default pairing; the 9B still should not be resident.**

This is consistent with Phase 3, where 4B + 9B left ~1.0 GB free. Adding the embedding
service to the 9B pushes free VRAM down to ~0.7 GB, well below the safety target.

## 6. Concurrency contention

Both services resident, generating/indexing **at the same time**:

| Workload | Alone | Concurrent | Loss |
|---|---:|---:|---:|
| embedding, batch 32 | 178.6 chunks/s | **126.3 chunks/s** | **−29 %** |
| local-fast generation | ~130 t/s | **42–49 t/s** | **−65 %** |
| local-smart generation | 74.6 t/s | 28.2 t/s | **−62 %** |

**Concurrent inference is expensive for both sides**, and the generative worker suffers
far more than the embedder. This mirrors the Phase 3 finding that the GPU serialises
the work anyway while adding contention.

**Recommendation: serialise inference across the embedding service and a generative
worker** (a GPU semaphore, or simply not issuing both at once). Do not assume that
because they fit in VRAM they can usefully run together.

## 7. Verified correctness properties

Established by `test_index_store.py` and `embedder.py` self-tests (all passing):

* FAISS `IndexFlatIP` over L2-normalised vectors reproduces reference cosine similarity
  to **2.98e-08** — the assumption the whole retrieval layer rests on.
* Vectors are unit length at 1024 / 512 / 256 dimensions. An earlier version skipped
  normalisation within a tolerance, which made dim-1024 norms inconsistent with the
  others; normalisation is now unconditional.
* `add()` **refuses** non-float32 input and non-unit-length vectors rather than silently
  ranking by magnitude.
* Zero vectors do not produce NaN.
* Index and metadata stay consistent across add, save, load and file removal.

## 8. Known problems

1. **Concurrent use costs 29–65 %.** Serialise.
2. **9B + embedding leaves only 720 MiB free** — do not co-reside them.
3. **FAISS cannot open non-ASCII Windows paths.** `faiss-cpu` 1.15.1 uses C++
   `std::fopen`, so both write and read fail on a path containing CJK characters.
   Worked around via `faiss.serialize_index`; a regression test asserts both that the
   workaround works and that the raw call still fails there.
4. Batch 64 is a latency trap for interactive use despite higher throughput.
5. `max_seq_length` is capped at 8192 although the model supports 32768; raising it
   would increase memory and slow long inputs, and no Phase 4 task needed it.

## 9. Two bugs found by running against the real corpus

Both were invisible to the small unit fixtures and only appeared once a 99-file corpus
was indexed. Both would have degraded retrieval while appearing to work.

**(a) Duplicate `chunk_id` desynchronised the index from the metadata.**
`IndexStore.add()` pushed vectors to FAISS *first* and inserted metadata *second* using
`INSERT OR REPLACE`. Two chunks sharing a `chunk_id` therefore **added two vectors but
kept one row**, so `index.ntotal` exceeded the chunk count and every `faiss_id` above the
collision pointed at the **wrong chunk** — retrieval would return confidently mislabelled
content. Observed as a 1179-vs-1173 drift. Fixed by de-duplicating on `chunk_id` before
FAISS is touched; a regression test asserts the collision collapses and the index stays
consistent.

**(b) Loading one model per configuration exhausted memory.**
A sweep that constructs an `Embedder` per (dimension, strategy) pair leaks a full model
each time, because GPU allocations live until the interpreter exits. On this 32 GB
machine the commit limit was reached and the next load failed with
`OSError: [WinError 1455] 页面文件太小` — an error that points nowhere near the cause.
`Embedder.close()` now releases the model and empties the CUDA cache, and the benchmark
reuses one model per dimension. Pagefile peak during the failing run: 23.8 GB.

A third defect, in the chunker, is described in `retrieval-results.md` §7.

