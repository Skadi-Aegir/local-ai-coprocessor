# local-embedding — Model Record (Phase 4)

**Logical alias:** `local-embedding`
**Role:** Text embedding for the Vector Retrieval Layer
**Phase:** 4
**Recorded:** 2026-09-30

> Upper layers must only ever reference the alias **`local-embedding`**. Model paths,
> repository names and dimension values must never appear in agent code or prompts.

---

## 1. Source chain

| Step | Artefact | Value |
|---|---|---|
| Model | `Qwen/Qwen3-Embedding-0.6B` | official Qwen organisation |
| Revision | commit | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Last modified | | 2026-04-20 |
| Downloads / likes | | 9,646,090 · 1,256 |
| License | | **Apache-2.0** |
| Gated | | No |
| Base model | | `Qwen/Qwen3-0.6B-Base` |
| Library declared | | `sentence-transformers` |

| File | Size | SHA-256 |
|---|---|---|
| `model.safetensors` | 1136.39 MB | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| `tokenizer.json` | 10.89 MB | `def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a` |

Both verified after download. No model file is tracked in git (they are Apache-2.0 works
of the Qwen team and are not redistributed here); `scripts/fetch_models.py` downloads every
file at the revision above and verifies it.

## 2. Official specifications — verified, not assumed

| Spec | Declared | Verified from |
|---|---|---|
| Parameters | 0.6B | model card |
| Context length | 32K | `config.json`: `max_position_embeddings: 32768` |
| Embedding dimension | up to 1024 | `config.json`: `hidden_size: 1024` |
| MRL support | Yes | model card: "user-defined output dimensions ranging from 32 to 1024" |
| Instruction aware | Yes | model card, and the shipped `config_sentence_transformers.json` |
| Layers | 28 | `config.json` |
| Attention heads / KV heads | 16 / 8 | `config.json` |
| Vocabulary | 151,669 | `config.json` |
| dtype | bfloat16 | `config.json` |

### Inference details that decide correctness

Read from the shipped configuration rather than guessed:

| Detail | Value | Why it matters |
|---|---|---|
| Pooling | **`pooling_mode_lasttoken: true`** | Not mean pooling. Using mean pooling would degrade every vector. |
| Similarity | `similarity_fn_name: "cosine"` | Sets the normalisation contract. |
| Normalisation | ships a `2_Normalize` module | `encode()` already returns unit vectors; re-normalising is harmless, skipping it is not. |
| Official query prompt | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` | Applied to **queries only**. The model card states "No need to add instruction for retrieval documents." |
| Pad token | `<|endoftext|>` | `tokenizer_config.json` |
| `model_max_length` | 131072 | tokenizer config; deployment caps at 8192 for speed |

**Instruction placement is not symmetric.** Applying the instruction to documents as well
as queries is a common mistake; the official implementation adds it to the query side only,
and the A/B measurement in `benchmarks/retrieval-results.md` shows instructions help.

## 3. Inference backend

**sentence-transformers 6.1.0 on PyTorch**, in a dedicated venv. Evaluated against the
alternatives the specification lists:

| Option | Verdict |
|---|---|
| **sentence-transformers / Transformers** | **chosen** — the repo ships `modules.json`, `config_sentence_transformers.json` and `1_Pooling`, i.e. it is packaged FOR this library. Last-token pooling and the official prompts work out of the box. |
| text-embeddings-inference | not used — a separate Rust server binary would add a moving part and a second model-loading path for no functional gain here. |
| llama.cpp | deliberately not used. The specification says not to force a unified stack, and llama.cpp's embedding support does not expose the sentence-transformers prompt/pooling configuration this model relies on. |

## 4. Environment (isolated)

`services/retrieval/venv` — created with `python -m venv`, `system_site_packages: False`.
No other venv or interpreter was modified; the global Python at `D:\Python` is untouched
(verified).

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 (`cuda_available: True`, `arch_list` includes `sm_120`) |
| sentence-transformers | 6.1.0 |
| faiss-cpu | 1.15.1 |
| numpy | 2.5.3 |
| Python | 3.13.5 |

## 5. Measured runtime

| Metric | Value |
|---|---|
| Load time | 1.7 s |
| VRAM, torch attribution | 2360.6 MiB |
| **VRAM delta on this machine** | **1246 MiB** above the desktop baseline |
| Device | `cuda:0` |
| Output | float32, `(n, 1024)`, L2-normalised |
| Max sequence length (deployed) | 8192 |
| Throughput, batch 32 | **178.6 chunks/s** |
| Throughput, batch 1 | 11.0 chunks/s |

> **Noted 2026-10-05.** "VRAM, torch attribution" is the service's `/stats vram_mb`, i.e.
> cudaMemGetInfo total − free, not PyTorch's attribution to this process. Measured on this machine
> (`benchmarks/p1_11`): it already reads 1232.6 MiB before the model loads, and the model's own
> allocator then reserves 1160 MiB - which matches the 1246 MiB delta in the next row. The
> service's footprint is `/stats memory.torch_reserved_mb`. After a bulk document embed it used to
> stay at its high-water mark (OPEN_ISSUES P1-11); the service now releases cached blocks after
> each encode.

Full throughput table: `benchmarks/embedding-results.md`.

## 6. Matryoshka behaviour (measured)

Truncating to fewer dimensions preserves the top result down to 256 but **breaks at 128**:

| Dimension | Top-1 document |
|---|---|
| 1024 | correct |
| 512 | correct |
| 256 | correct |
| **128** | **wrong** |
| **64** | **wrong** |

So the usable floor for this workload is around 256 dimensions, and the deployment default
remains **1024** until the retrieval sweep says otherwise — see `benchmarks/retrieval-results.md`.

## 7. Usage contract

```
POST http://127.0.0.1:8113/v1/embeddings     OpenAI-compatible
POST http://127.0.0.1:8113/v1/retrieve       project-scoped retrieval
GET  http://127.0.0.1:8113/health | /v1/models | /stats
```

Loopback only. Documents must be encoded **without** an instruction; queries should carry
the official retrieval instruction. The service exposes a `query: true` flag that applies it.

## 8. Known limitations

1. **FAISS cannot open non-ASCII Windows paths.** `faiss-cpu` 1.15.1 uses C++ `std::fopen`,
   which rejects UTF-8 paths; both write and read fail with an opaque error. Worked around in
   `index_store.py` by serialising through `faiss.serialize_index`. Recorded because a user
   with a CJK profile or TEMP path would otherwise hit a bewildering failure.
2. **Retrieval R@5 is 74.4 %, below the 90 % Phase 4 target.** Analysis in
   `benchmarks/retrieval-results.md`; part of the shortfall is benchmark labelling rather than
   the model, and a reranker (Phase 5) is the intended remedy.
3. Model context is 32K but the deployment caps at 8192 for throughput and memory.
4. `get_sentence_embedding_dimension()` is deprecated in sentence-transformers 6.x; the code
   handles both names.
