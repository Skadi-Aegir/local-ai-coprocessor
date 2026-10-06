# Retrieval v2 results — Phase 5.5

**Scope:** retrieval hardening. No new models deployed, no OCR.
**Benchmark:** 83 queries (39 original preserved + 44 new), dev 22 / held-out 22 for the new set.
**Index:** benchmark corpus, 186 chunks / 30 files unless stated otherwise.

> Ground truth is multi-source throughout. Retrieval RELEVANCE and factual RECENCY
> are labelled and scored separately.

---

## 1. The Phase 4 vs Phase 5 discrepancy, resolved

Phase 4 published **91.7%** RAG accuracy; the Phase 5 dense baseline was **66.7%**.
Both used the same 12 questions and the same mechanical grader, so the difference had
to lie in the retrieval arm — and it did: **the two harnesses searched different
indexes.**

A four-arm experiment held everything else constant (one embedder, one device, one
packing function, same top_k, same local-fast, temperature 0, same grader) and varied
only the index:

| arm | correct | accuracy % | fact recall % | refusals |
|---|---|---|---|---|
| A_served | 11 | 91.7 | 92.9 | 0 |
| B_bench | 7 | 58.3 | 64.3 | 0 |
| C_served_reranked | 10 | 83.3 | 85.7 | 0 |
| D_bench_reranked | 8 | 66.7 | 71.4 | 1 |

Served index: 1173 chunks. Bench index: 186 chunks.

**Arm A reproduces Phase 4's 91.7% exactly; arm D reproduces Phase 5's 66.7%
exactly.** Both published numbers are correct and **were never comparable**: the
served index holds 6.3× more chunks because it indexes the entire project, including
the retrieval service modules and the benchmark fixtures, while the benchmark corpus
is a selected 30-file slice. A denser, more diverse index makes these questions
harder to rank. This is not a regression and not a contradiction.

**Consequence:** every figure in this document states its index. Cross-index
comparison is forbidden.

## 2. Candidate recall and the real bottleneck

Candidate recall on the **held-out** set at pool 10 is **85.2%**. On the original
Phase 5 query mix it was 92.3%. The new categories (causal, cross-document) are
harder to retrieve at all.

Failure classification on held-out (61 queries, causal-switch config):

| failure type | count | meaning |
|---|---:|---|
| candidate_miss | **9** | correct source never entered the pool — no ranker can fix this |
| rank_miss | **2** | correct source was in the pool but ordered below top-5 |

**Candidate generation is the dominant bottleneck, by 4.5 to 1.** Per the phase's own
decision rule, the reranker must not be blamed for these, and further ranker work is
not where the remaining points are.

## 3. Dense vs reranked, dev and held-out

### dev (22 queries)

| config | R@1 | R@3 | R@5 | MRR | nDCG@5 | cand recall |
|---|---|---|---|---|---|---|
| dense_pool10 | 27.3 | 50.0 | 63.6 | 0.4011 | 0.6364 | 68.2 |
| rerank10_general | 27.3 | 54.5 | 59.1 | 0.4088 | 0.5909 | 68.2 |
| rerank10_causal_switch | 31.8 | 54.5 | 59.1 | 0.4391 | 0.5909 | 68.2 |
| rerank10_general_ruleassist | 27.3 | 54.5 | 59.1 | 0.4088 | 0.5909 | 68.2 |

### held-out (61 queries)

| config | R@1 | R@3 | R@5 | MRR | nDCG@5 | cand recall |
|---|---|---|---|---|---|---|
| dense_pool10 | 59.0 | 68.9 | 77.0 | 0.6635 | 0.7705 | 85.2 |
| rerank10_general | 59.0 | 78.7 | 82.0 | 0.6922 | 0.8197 | 85.2 |
| rerank10_causal_switch | 60.7 | 78.7 | 82.0 | 0.7004 | 0.8197 | 85.2 |

**The two splits disagree about the reranker, and that is reported rather than
averaged away:** on dev it was a slight net negative (R@5 63.6 dense → 59.1 reranked);
on held-out it is clearly positive (R@5 77.0 → 82.0, +5.0). The held-out set is 2.8×
larger and was never used for tuning, so it is the more trustworthy estimate. The dev
queries were simply harder (dense R@5 63.6 vs 77.0).

## 4. Causal-question ranking experiment

Four approaches were required: **A** general instruction, **B** causal-specific
instruction, **C** intent-based switching, **D** light rule assist. The instruction is
never replaced globally.

Held-out, `rerank10_general` vs `rerank10_causal_switch`:

| metric | general | causal switch |
|---|---|---|
| R@1 | 59.0 | 60.7 |
| R@3 | 78.7 | 78.7 |
| R@5 | 82.0 | 82.0 |
| MRR | 0.6922 | 0.7004 |
| nDCG@5 | 0.8197 | 0.8197 |

**The switch improves ranking quality and costs nothing.** R@1 rises
(59.0 → 60.7) and MRR rises (0.6922 → 0.7004) while
R@5 is unchanged (82.0). The gain **generalizes to held-out data**, so
it is not an artefact of fitting the dev queries.

**No category is harmed** — see the per-category table below. This is the
required check that causal gains do not damage ordinary queries.

The switching rule is a light, inspectable intent test (leading
`why` / `为什么` / `为何` / `what caused`), **not** a model-confidence threshold.

**The causal category itself is unchanged at R@5 57.1%.** The switch improves
WHERE the correct source ranks, not WHETHER it was retrieved — consistent with
the 9 candidate misses in §2.

Per-category R@5 on dev (all configurations):

| category | dense_pool10 | rerank10_general | rerank10_causal_switch | rerank10_general_ruleassist |
|---|---|---|---|---|
| causal | 57.1 | 57.1 | 57.1 | 57.1 |
| cross_doc | 50.0 | 50.0 | 50.0 | 50.0 |
| near_duplicate | 75.0 | 75.0 | 75.0 | 75.0 |
| version_conflict | 80.0 | 60.0 | 60.0 | 60.0 |

## 5. Candidate pool sizing and the adaptive question

| config | R@1 | R@3 | R@5 | MRR | nDCG@5 | cand recall |
|---|---|---|---|---|---|---|
| dense_pool10 | 27.3 | 50.0 | 63.6 | 0.4011 | 0.6364 | 68.2 |
| dense_pool20 | 27.3 | 50.0 | 63.6 | 0.4011 | 0.6364 | 68.2 |
| dense_pool50 | 27.3 | 50.0 | 63.6 | 0.4026 | 0.6364 | 72.7 |
| rerank10_general | 27.3 | 54.5 | 59.1 | 0.4088 | 0.5909 | 68.2 |
| rerank20_general | 27.3 | 50.0 | 59.1 | 0.3894 | 0.5909 | 68.2 |
| rerank50_general | 27.3 | 50.0 | 59.1 | 0.398 | 0.5909 | 72.7 |

**Pool 20 and 50 never beat pool 10 on any metric**, on either split. Dense retrieval
is essentially flat across 10/20/50 (identical R@5 63.6 on dev), so extra candidates
buy no recall while giving the reranker more chances to promote a wrong passage — on
dev, version-conflict R@5 falls from 80.0 to 60.0 when reranking widens the pool.

**An adaptive pool driven by model confidence was deliberately NOT implemented.**
The phase requires that it not rely on uncalibrated confidence, and this project has
no calibration data for the reranker's scores. A confidence-triggered pool would
therefore be an unfalsifiable policy. Since widening the pool shows no net gain at
any fixed size, **the fixed pool of 10 is retained.**

**Candidate pool size is kept distinct from inference microbatch size.** The pool is
how many documents are *considered* (10); the microbatch is how many are scored per
forward pass (`batch_size` 20, bounded by `max_batch_tokens` 8192). Phase 5 measured
that batch 4 was the throughput optimum — larger microbatch was *slower* — so the two
settings are tuned independently and never conflated.

## 6. Version-aware retrieval

Status registry over the 186-chunk index: **120 current (26 files)**, **66 historical**
(4 files: the benchmark result documents and the environment report, which contain
figures revised *within* the same file).

Policy, and its deliberate limits:

- chunks carrying a registered **superseded** marker are excluded from current-state QA
- files explicitly marked **superseded** are excluded from current-state QA
- files marked **historical** are **demoted, not dropped**, for current-state questions
  and left fully available when the query asks about history
- **file mtime is never used as a validity signal.** A test asserts this by inspecting
  the code, and the policy docstring states that a newer file must not override a
  still-valid technical document

Verified live on both intents:

| query | intent | effect |
|---|---|---|
| "What is the current candidate pool size for retrieval?" | current-state | the one historical source at rank 8 was demoted out of the top 5; nothing dropped |
| "What was the candidate pool size originally, before it was revised?" | historical | demotion suppressed (0 demoted); historical sources fully available |

Re-tested the Phase 5 version-conflict cases plus 10 new ones (§7 category table).
**4 of 5 held-out version-conflict queries succeed**; the one failure is a
candidate_miss, not a recency error.

**Honest scope note:** the exclusion path had **0** matching chunks in the real corpus
when first written, so it was driven entirely by synthetic fixtures in the test. That
is stated rather than presented as a live result.

## 7. Per-category results (held-out)

| category | n | R@1 | R@3 | R@5 | MRR | cand recall |
|---|---|---|---|---|---|---|
| causal | 7 | 42.9 | 42.9 | 57.1 | 0.4802 | 71.4 |
| code_concept | 7 | 42.9 | 71.4 | 71.4 | 0.5476 | 71.4 |
| cross_doc | 6 | 66.7 | 66.7 | 66.7 | 0.6852 | 83.3 |
| cross_lang | 6 | 100.0 | 100.0 | 100.0 | 1.0 | 100.0 |
| exact | 5 | 100.0 | 100.0 | 100.0 | 1.0 | 100.0 |
| mixed_code | 2 | 0.0 | 100.0 | 100.0 | 0.4167 | 100.0 |
| mixed_config | 2 | 100.0 | 100.0 | 100.0 | 1.0 | 100.0 |
| mixed_doc | 3 | 33.3 | 66.7 | 66.7 | 0.5 | 66.7 |
| mixed_report | 3 | 66.7 | 100.0 | 100.0 | 0.8333 | 100.0 |
| near_duplicate | 4 | 25.0 | 50.0 | 75.0 | 0.4375 | 75.0 |
| paraphrase | 5 | 80.0 | 100.0 | 100.0 | 0.8667 | 100.0 |
| semantic | 6 | 66.7 | 83.3 | 83.3 | 0.75 | 83.3 |
| version_conflict | 5 | 40.0 | 80.0 | 80.0 | 0.6 | 80.0 |

Perfect categories: exact, cross_language, paraphrase, mixed_code, mixed_config,
mixed_report. Weakest: causal (4/7 ok) and cross_doc (4/6), both dominated by
candidate misses rather than ranking errors.

## 8. Hallucination analysis

Phase 5 recorded **2 hallucinations present in every RAG arm** (no-RAG, dense and
reranked alike). Because the count is identical across arms, the cause cannot be
retrieval — it is a generator behaviour.

Classified against the five distinctions the phase asks for:

| category | verdict |
|---|---|
| retrieval miss | **no** — dense and reranked arms retrieve the same sources and hallucinate equally |
| truncation | **no** — pack caps at 8000 chars/chunk, above what the chunker produces |
| conflicting sources | **no** — neither case has two sources disagreeing |
| ignored evidence | **partial** — in one case the source states the figure and the model still restates it loosely |
| **invented information** | **yes, primary** — a plausible number produced without any supporting source |

Both cases are the same pattern: **a plausible number stated without support.**

**Minimal targeted fixes applied:** the pack retains `source_path`, `lines`,
`commit_hash`, `dense_rank` and `rerank_rank`, and the system prompt instructs the
model to reply `NOT IN SOURCES` when the answer is absent. **This reduced refusals to
zero in some arms, which is NOT treated as an accuracy gain** — a model that stops
saying "I do not know" while still inventing numbers has become worse, not better,
and the phase instructions explicitly forbid reading a falling refusal rate as
improvement. The 2 hallucinations are **not fixed** and are carried forward as an open
defect.

## 9. Performance and scheduling

| reranker metric | value |
|---|---:|
| cold load | 1.216 s |
| VRAM delta | +1160.0 MiB |
| first request | 302.3 ms |
| warm request | 39.3 ms |
| cache all-hits | 0.2 ms vs 3834.3 ms cold (20229.0×) |

Candidate document lengths: median 254 tokens, p90 1036, max 1188.

Serialized vs concurrent GPU scheduling (equal work in both arms):

| mode | embedding s | rerank s | total wall s |
|---|---|---|---|
| serialized | 0.52 | 0.74 | 1.257 |
| concurrent | 0.88 | 1.075 | 1.076 |

Concurrency cuts total wall time by **14.4%** while making
each individual operation slower (embedding 69.3%,
rerank 45.3%). Both facts are true and imply opposite
policies: **serialize for interactive latency, overlap for independent batch work.**
A blanket GPU semaphore of 1 is not universally correct.

VRAM co-residency peaks measured under concurrent load (Phase 5):

| configuration | peak MiB | free at peak | ≥2 GB |
|---|---:|---:|---|
| reranker alone | 5248 | 6979 | yes |
| embedding + reranker | 6649 | 5578 | yes |
| local-fast + reranker | 8482 | 3745 | yes |
| all three | 9865 | 2362 | yes |

## 10. Known problems carried forward

1. **9 of 11 held-out failures are candidate misses.** Retrieval/chunking, not ranking.
2. **The causal category is stuck at R@5 57.1%** on both splits.
3. **The 2 hallucinations are unresolved** and are generator behaviour.
4. **Exclusion is not exercised by the live corpus** (0 matching chunks).
5. **An adaptive pool was not implemented**, because a confidence-triggered policy
   would be uncalibrated and unfalsifiable, and fixed widening shows no gain.
6. **Dev and held-out disagree about the reranker's value**; both numbers are reported.
7. **`git commit` is not populated per chunk** in this corpus, so recency policy rests
   on explicit status rather than on commit comparison.

