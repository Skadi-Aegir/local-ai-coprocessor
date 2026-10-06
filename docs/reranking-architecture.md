# Reranking Architecture — Phase 5

**Scope:** the two-stage retrieval pipeline. Not an agent, not a Gateway.
**Goal:** raise the correct chunk into the final Top-K so the model sees the right
context, and prove it improves the *answer*, not merely a retrieval metric.

---

## 1. Pipeline

```text
query
  │
  ▼
embedding service (8113)          Qwen3-Embedding-0.6B, 1024-d
  │
  ▼
Top-N candidate pool              default N = 10
  │
  ▼
reranker service (8114)           Qwen3-Reranker-0.6B, official scoring
  │
  ▼
final Top-K                       default K = 5
  │
  ▼
context pack                      path / lines / dense+rerank scores / content
  │
  ▼
local-fast  or  cloud
```

The orchestrator is `services/retrieval/rerank_pipeline.py`, which lives in the retrieval
service because that service owns FAISS and the chunk metadata.

## 2. Component boundaries — the decision that matters most

| Component | Owns | Never does |
|---|---|---|
| retrieval service (8113) | FAISS index, SQLite metadata, chunk identity, context packing | score relevance with a cross-encoder |
| reranker service (8114) | scoring `(query, document)` pairs | touch FAISS, store metadata, know what a chunk is |

The reranker receives **plain strings** and returns `{index, score}` referring to positions
in the array it was given. The retrieval service maps those indices back to paths and line
ranges. This is what the specification asked for, and it has a concrete payoff: the
reranker is a pure function of text, which is why it can be tested with three literal
strings and why the duplicate test could show that identical text receives an identical
score.

It also means the reranker has **no notion of version, path or freshness** — see §7.

## 3. Official scoring

The reranker is a Qwen3 causal LM packaged as a sentence-transformers CrossEncoder whose
second module is `1_LogitScore`. Scoring follows the repository's documented procedure:

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct
provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {doc}<|im_end|>
<|im_start|>assistant
<think>

</think>

→ softmax([logit("no"), logit("yes")])[1]
```

Details that are easy to get wrong and are asserted by `test_reranker.py`:

* the suffix ends with an **empty think block**; without it the model emits reasoning and
  the final logits are not the yes/no decision
* **truncation applies to the body only** — the prefix and suffix are concatenated after
  truncation so the decision scaffolding always survives
* `padding_side='left'`, so every row's final position is a real token
* the score is read at the **final position**, and the LM head is applied to that position
  rather than to the whole sequence (see §5)

## 4. Why two stages at all

Dense retrieval embeds the query and each chunk **independently**, so it can only compare
them through a single vector. A cross-encoder reads the query and the document **together**
and can therefore judge whether this passage answers *this* question.

Phase 5 measured exactly what that buys:

| | dense | reranked |
|---|---:|---:|
| strict R@1 | 35.9 % | **53.8 %** |
| strict MRR | 0.4882 | **0.6239** |
| end-to-end RAG accuracy (pool 10) | 66.7 % | **83.3 %** |

The Phase 4 failure modes were the target: correct chunks sitting at rank 6 and rank 13.
Both were promoted into the final Top-5, and the rank-13 case reached **rank 1**.

## 5. Memory behaviour, and the two bugs it caused

A cross-encoder is far more expensive per candidate than an embedding, so memory — not
compute — was where this phase broke twice.

**(a) The official formulation exhausted GPU memory on long documents.**
`model(**inputs).logits[:, -1, :]` materialises `(batch, seq_len, 151669)`. At a
2048-token sequence that is **1.32 GiB for a single scalar**, and it raised
`torch.OutOfMemoryError` while 8.87 GiB was reported free — the transient spike, not the
steady state, was the problem. Computing the final position through the LM head explicitly
allocates `(batch, 1, 151669)` and is **mathematically identical**; the test asserts the
two agree with a maximum difference of `0.00e+00`.

**(b) A fixed row-count batch size is unsafe.** Attention activations grow with the
**square** of sequence length, so three maximal documents in one batch attempted a 768 MiB
allocation. Batching is therefore bounded by `(rows × padded length)`, capped at 8192
tokens. *Padded*, because padding makes every row as long as the longest one — measuring
each row individually is how the first version of the planner still under-estimated. A
single row is always allowed through even if it alone exceeds the budget, so an
over-long document can never be silently dropped.

## 6. Candidate pool sizing — the counter-intuitive result

Bigger pools were expected to help. They do not.

| pool | candidate recall (multi) | R@5 | end-to-end RAG accuracy | rerank latency |
|---:|---:|---:|---:|---:|
| 5 | 84.6 % | 84.6 % | — | 0.23 s |
| **10** | 89.7 % | **89.7 %** | **83.3 %** | **0.49 s** |
| 20 | 92.3 % | 87.2 % | 75.0 % | 1.05 s |
| 50 | 100.0 % | 87.2 % | — | 1.52 s |

Recall of the *pool* keeps improving with N, but the *final* ranking gets worse: more
candidates give the reranker more opportunities to promote a wrong passage. Pool 10 wins
on quality, latency and token cost simultaneously, and two independent measurements agree.

**Default changed from the specification's suggested 20 to 10.**

## 7. Freshness is a pipeline concern, not a model concern

The reranker scores text and nothing else, so it cannot know which of two conflicting
passages is current. Measured: it picks the **stale** passage in 4 of 5 cases, at scores
within a few thousandths of the current one.

That is the expected behaviour of a pure text scorer, and the fix belongs here, in the
pipeline — `mtime` and `commit_hash` already exist in the SQLite schema for it. Phase 5
deliberately does **not** build weighted freshness ranking; it records the limitation and
the extension point.

## 8. Failure handling

If the reranker is unreachable, retrieval must not fail. The pipeline catches
**transport errors only** and degrades to the dense top-K:

```json
{ "reranker_used": false, "fallback_reason": "URLError: ..." }
```

Verified for connection-refused, HTTP-500-from-a-live-server, and reranking-disabled.
The catch is deliberately narrow: a programming error raises instead of masquerading as
"reranker unavailable", because a fallback that swallows our own bugs hides real defects.

## 9. Resource posture

Peak VRAM under concurrent load, measured with nvidia-smi sampled every 150 ms while all
services actually worked. A steady-state reading is misleading here.

| configuration | at rest | **peak** | free at peak | ≥ 2 GB |
|---|---:|---:|---:|---|
| desktop baseline | 2157 | — | — | — |
| reranker alone | 3453 | **5248** | 6979 | ✅ |
| embedding + reranker | 6570 | 6649 | 5578 | ✅ |
| local-fast + reranker | 8486 | 8482 | 3745 | ✅ |
| **all three** | 7931 | **9865** | **2362** | ✅ |

All four keep ≥ 2 GB free at peak, so three resident services are permitted. But
**fitting in memory is not the same as running fast** — see §9a.

### 9a. Serialized vs concurrent scheduling — measured, and workload-dependent

Equal work in both arms: 4 embedding batches and 4 rerank queries.

| | embedding | rerank | total wall |
|---|---:|---:|---:|
| serialized | 0.520 s | 0.740 s | 1.257 s |
| concurrent | 0.880 s (**+69 %**) | 1.075 s (**+45 %**) | **1.076 s (−14.4 %)** |

Concurrency reduces **total wall time by 14.4 %** while making **every individual
operation 45–69 % slower**. Both facts are true and they imply opposite policies:

| workload | policy | why |
|---|---|---|
| an interactive request | **serialize** | a user waiting on one request is better served by running it alone |
| independent throughput jobs (index while re-ranking a queue) | **overlap** | total wall time is what matters; the overlap wins |
| two LLM generations | **serialize** | Phase 4 measured this case and concurrency loses throughput |

So a blanket `GPU semaphore = 1` is **not** universally correct. The earlier assumption
that concurrency is always harmful came from the two-LLM case; embedding and reranking have
complementary profiles and overlap profitably.

> **Measurement note.** The first run of this comparison was meaningless because the
> reranker cache was warm — four queries finished in 0.043 s, leaving almost nothing to
> overlap. Cache hits are a legitimate, separately reported result (20229×), but they must
> never masquerade as a contention measurement. The benchmark now sends a unique query with
> `use_cache=false`.

## 10. Context pack

The Phase 4 contract is preserved and extended, because a reranking failure is only
diagnosable if you can see which stage caused it:

```text
SOURCE 1
path: docs/local-fast-api.md
lines: 144-161
score: 0.6885              <- dense score, unchanged
chunk_id: ac8b95d14dc914bd2c90
dense_rank: 13             <- where dense put it
dense_score: 0.6885
rerank_rank: 1             <- where the reranker moved it
rerank_score: 0.9971
symbol: 7. Operational notes
content:
…
```

The dense score stays in `score` so a caller that understands only one score is never
silently handed a rerank probability in the same field.

## 11. What this layer deliberately does NOT do

* **Not installed by default.** All three services can be resident, but the reranker is
  optional at runtime and the pipeline works without it.
* No freshness weighting (§7) — schema is ready, policy is not.
* No query decomposition, no multi-hop retrieval.
* No Gateway, no Qdrant, no agent loop.
* No 4B/8B reranker. §7 of `docs/models/local-reranker.md` records the evidence for and
  against evaluating one; Phase 5 stops before it.

## 12. Extension points

* **Freshness rule**: prefer the newer of two conflicting chunks on the same path when
  `commit_hash` differs — a simple filter before packing, not a weighted score.
* **Adaptive pool**: use pool 10 by default, widen only when the top rerank score is low.
* **Reranker-free confidence signal**: the top rerank score is already returned; it is a
  natural input to an escalation decision.
* **Larger reranker**: only if the evidence in the model record supports it.
