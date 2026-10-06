# local-reranker — Model Record (Phase 5)

**Logical alias:** `local-reranker`
**Role:** Second-stage relevance scoring for the Vector Retrieval Layer
**Phase:** 5
**Recorded:** 2026-09-30

> Upper layers must only ever reference the alias **`local-reranker`**. Model paths,
> repository names and token ids must never appear in agent code or prompts.

---

## 1. Source chain

| Step | Artefact | Value |
|---|---|---|
| Model | `Qwen/Qwen3-Reranker-0.6B` | official Qwen organisation |
| Revision | commit | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| Last modified | | 2026-04-16 |
| Downloads / likes | | 1,672,965 · 405 |
| License | | **Apache-2.0** |
| Gated | | No |
| Base model | | `Qwen/Qwen3-0.6B-Base` |
| Pipeline tag | | `text-ranking` |
| Library declared | | `transformers` (packaged as a sentence-transformers CrossEncoder) |
| Paper | | arXiv:2506.05176 |
| Files | | 14, total 1151.55 MB |

| File | Size | SHA-256 |
|---|---|---|
| `model.safetensors` | 1136.39 MB | `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b` |
| `tokenizer.json` | 10.89 MB | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

Both verified against the Hub's LFS oids after download.

> **Why a verifier exists.** These hashes were written by hand twice and were wrong both
> times — the config first carried the *embedding* model's weight hash and then its
> tokenizer hash. A copied hash is worse than no hash because it looks like verification.
> `services/reranker/verify_hashes.py` now recomputes every claimed hash and compares it
> both to the upstream oid and to what the config claims. `start-local-reranker.ps1` also
> verifies the weights before starting and refuses to start on a mismatch.

**Not downloaded, by design:** `Qwen3-Reranker-4B` and `Qwen3-Reranker-8B`.
`fetch_reranker.py` hardcodes this one repository, so no accidental upgrade is possible.
Evaluating 4B requires Phase 5's evidence to justify it — see §7.

## 2. Official specifications — verified from the repository

| Spec | Value | Source |
|---|---|---|
| Parameters | 0.6B | model card |
| Layers | 28 | `config.json` |
| Hidden size | 1024 | `config.json` |
| Attention / KV heads | 16 / 8 | `config.json` |
| Head dim | 128 | `config.json` |
| Intermediate size | 3072 | `config.json` |
| Vocab size | 151,669 | `config.json` |
| **Sequence length** | **40,960** | `max_position_embeddings` |
| dtype | bfloat16 | `config.json` |
| Architecture | `Qwen3ForCausalLM` | used as a CrossEncoder |
| Instruction aware | Yes | model card |

Note the model card's comparison table lists 32K for the reranker family; `config.json`
for the 0.6B checkpoint states `max_position_embeddings: 40960`. The config is what the
implementation must respect, and deployment caps far below either value (§4).

## 3. Official scoring method — read from the repo, not invented

The specification requires following Qwen's own procedure. The repository ships
`modules.json` declaring two modules: a `Transformer` and **`1_LogitScore`**, and the
README documents the exact prompt format. This implementation reproduces it:

```
prefix  <|im_start|>system
        Judge whether the Document meets the requirements based on the Query and the
        Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
        <|im_start|>user
body    <Instruct>: {instruction}
        <Query>: {query}
        <Document>: {doc}
suffix  <|im_end|>
        <|im_start|>assistant
        <think>

        </think>

score   softmax([logit(no), logit(yes)])[1]   at the FINAL position
```

| Detail | Value | Why it matters |
|---|---|---|
| `token_true_id` (`yes`) | **9693** | the decision token |
| `token_false_id` (`no`) | **2152** | the decision token |
| Prefix tokens | 39 | fixed cost before the body |
| Suffix tokens | 9 | fixed cost after the body |
| `padding_side` | `left` | so the final position is a real token, not a pad |
| Truncation | `longest_first`, **body only** | cutting the suffix would destroy the decision |
| Empty think block | required | without it the model emits reasoning and the final logits are not the answer |

**Truncation applies to the body only.** The prefix and suffix are concatenated after
truncation precisely so the yes/no scaffolding always survives; truncating the full
sequence would eventually remove the suffix and produce a meaningless score.

### Independent fidelity evidence

The README documents the example `"What is the capital of China?"` against a Beijing
passage and a gravity passage, with raw logit differences **7.625** and **−11.375**.
Running our implementation on the same inputs gives a raw logit difference of **7.625**
— an exact match to the published figure, and a score of 0.999542 against the README's
`sigmoid(7.625) = 0.99951`. Landing on the published numbers is evidence the scoring is
faithful rather than merely self-consistent.

## 4. Inference backend and deployment

**PyTorch / Transformers in a dedicated venv.** Both official paths work and agree:
the `sentence_transformers.CrossEncoder` wrapper and the raw Transformers procedure the
README documents as primary. The raw path is used in production because it exposes the
token-level details and the memory behaviour that the wrapper hides.

| Setting | Deployed | Rationale |
|---|---|---|
| `max_length` | **2048** | body budget 2000. See §6 — measured, and chosen for safety over speed |
| `batch_size` | **20** | matches the default candidate pool |
| `max_batch_tokens` | **8192** | attention cost grows with the square of length |
| instruction | **`general`** | best balance of R@1/MRR and R@5; see §6 |
| device | `cuda`, bfloat16 | |

### Environment (isolated)

`services/reranker/venv` — separate venv, `system_site_packages: False`, 487 MB of real
files. It differs from the embedding environment only by `faiss-cpu`, deliberately
excluded: **the reranker must not manage FAISS.**

The already-verified `torch 2.11.0+cu128` build is reused through a **directory
junction** rather than duplicated, because a full reinstall meant ~4 GB onto a drive with
19.7 GB free. Package metadata is junctioned too: `transformers` calls
`importlib.metadata.version("torch")` to decide whether torch is available, and without
`dist-info` it reports torch as missing, surfacing as the misleading error
`Could not import module 'GenerationMixin'`.

## 5. Measured runtime

| Metric | Value |
|---|---|
| Cold load | **1.22 s** |
| VRAM delta | **+1160 MiB** (1200.6 → 2360.6 MiB) |
| VRAM at rest, alone | 3453 MiB total on card |
| **VRAM peak, alone, under load** | **5248 MiB** (+1794 over rest) |
| First request | 302 ms (CUDA lazy init / autotuning) |
| Warm request | 39 ms |
| Unload releases | 2960 MiB |
| Cache (all hits) | 0.2 ms vs 3834 ms cold — **20229×** |

The +1794 MiB transient is the number to plan for when the reranker runs by itself.
When other services are already resident the incremental spike is far smaller (+6 to
+80 MiB) because the allocator reuses reserved memory.

## 6. Quality — measured, and the parameter choices justified

Full tables in `benchmarks/reranker-results.md`. Headlines:

| Measurement | Result |
|---|---|
| Candidate recall @20 (multi-source) | **92.3 %** |
| Dense → reranked, strict R@1 | 35.9 % → **53.8 %** |
| Dense → reranked, strict MRR | 0.4882 → **0.6239** (+28 %) |
| Rank-13 Phase 4 failure | promoted to **rank 1** |
| End-to-end RAG accuracy, pool 10 | 66.7 % (dense) → **83.3 %** (reranked) |
| Hard negatives, strict subset, `general` | 87.5 % |

**`max_length` is 2048 despite 512 measuring faster.** The sweep showed 512 is 2.4×
faster at identical R@5 — but 226 of 780 candidate pairs were truncated at 512 and those
are precisely the longest chunks, the ones most likely to hide an answer at the tail.
Phase 4 measured that tail truncation does lose answers (score 0.9399 → 0.000269). A
speed-up bought by truncating the benchmark is not a free saving, so 2048 stays the
default and 512 is recorded as a documented latency option with its risk stated.

## 7. Is 4B worth evaluating?

Phase 5's decision rule: if candidate recall is high but reranked R@5 is low, the ranker
is the bottleneck and a larger reranker is worth considering.

| Condition | Measured | Met? |
|---|---|---|
| Candidate recall @20 ≥ 90 % | **92.3 %** | ✅ yes |
| Reranked R@5 ≥ 90 % | 87.2 % (multi), 71.8 % (strict) | ❌ no |

Both conditions therefore hold in the direction that *permits* evaluating a larger
reranker: candidates are good, the ranker leaves ~5 points on the table. But Phase 5's
explicit instruction is **not to download 4B automatically**, and it has not been
downloaded. See the final report for the recommendation.

## 8. Usage contract

```
POST http://127.0.0.1:8114/v1/rerank
GET  http://127.0.0.1:8114/health | /v1/models | /stats
POST http://127.0.0.1:8114/v1/cache/clear

request : {model, query, documents[], top_n?, instruction_type?, instruction?, batch_size?}
response: {results:[{index, score}], count, seconds, pairs_per_s, reranker_used,
           instruction_type, truncated, batches, cache}
```

`index` refers to a position in **the caller's** `documents` array. The reranker never
manages FAISS and never sees chunk metadata; the retrieval service maps indices back to
paths and line ranges. Loopback only.

## 9. Known limitations — measured, not assumed

1. **"Why" questions favour description over rationale.** Two hard-negative cases invert
   under *every* instruction tested: the model prefers a passage that *describes* a
   mechanism over one that *explains why* a decision was made (`hn19` inverted 4 of 4,
   `hn04` 3 of 4). This is consistent and explainable, not noise — roughly 3 in 4 strict
   hard-negative cases rank correctly, not all of them.
2. **No notion of version or freshness.** The reranker picks the *stale* passage in 4 of
   5 duplicate/stale cases, at scores within a few thousandths of the current one, and
   gives identical text identical scores. This is expected: it sees text only. Freshness
   must come from metadata (`mtime`, `commit_hash`) in the retrieval pipeline, **not** from
   this model. It should not be described as failing at versioning — it was never given
   version information.
3. **The `code` instruction is a poor global default**: best on multi-source R@1 (74.4 %)
   but worst on hard negatives (68.8 %).
4. **Latency exceeds the 1 s reference at pool 20** (~1.09 s end-to-end). Pool 10 lands at
   ~0.53 s. The bottleneck is forward passes dominated by padding, since the median
   candidate chunk is only 254 tokens.
5. **Batching is not "bigger is better":** batch 4 was the throughput optimum for a fixed
   64-pair workload; larger batches were *slower* because the token budget forced the same
   number of passes while adding padding.
6. `max_batch_tokens` means a single over-long document is still attempted alone and can
   spike memory; it is allowed through deliberately so no document silently disappears.
