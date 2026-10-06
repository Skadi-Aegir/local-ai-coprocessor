# Reranker results — Phase 5

**Model:** `local-reranker` = `Qwen/Qwen3-Reranker-0.6B` rev `e61197ed`, Apache-2.0
**Backend:** PyTorch 2.11.0+cu128 / Transformers 5.17.0, isolated venv, service on `127.0.0.1:8114`
**Index:** `d1024_structured`, 186 chunks from 30 files
**Benchmark:** 39 hand-labelled queries (15 doc, 14 code, 10 mixed)
**Raw data:** `benchmarks/reranker/results/*.json`

---

## 1. The question Phase 5 exists to answer

> Can the reranker promote the Phase 4 rank-6 / rank-13 correct chunks into a stable final
> Top-5, and does that measurably improve RAG and context compression?

**Yes on promotion, yes on RAG, and the improvement is largest exactly where Phase 4 failed.**

| Phase 4 failure | Dense rank | After reranking |
|---|---:|---:|
| "Where is GPU memory sampled repeatedly while a job runs?" | **13** | **1** |
| "Where are the refusal paths of the launcher tested?" | **37** | **1** |
| "Why is it a bad idea to have two local models generate at once?" | **7** | **1** |
| "How much memory does the small worker leave free?" | **6** | **2** |
| "How is the JSON adherence of the fast worker measured?" | **4** | **1** |

The rank-13 case from Phase 4 is the one promoted to rank 1.

## 2. Candidate recall — reported first, deliberately

The specification is explicit that a reranker must not be blamed for candidates it never
saw, so this comes before any ranking metric.

| Pool | strict | **multi-source** |
|---:|---:|---:|
| 5 | 66.7 % | 84.6 % |
| 10 | 66.7 % | 89.7 % |
| **20** | 76.9 % | **92.3 %** |
| 50 | 89.7 % | **100.0 %** |

**Verdict: `candidate Recall@20 = 92.3 % ≥ 90 %` under the corrected ground truth, so
candidate generation is NOT the primary bottleneck.** At Top-50 recall is 100 % — every
answer is somewhere in the pool. This is what makes evaluating the ranker legitimate
rather than misdirected.

## 3. Ground truth revision — both metrics reported

Phase 4 scored each query against a **single** file. This corpus deliberately documents
the same facts in several places, so factually correct answers from another file were
counted as misses. Phase 5 adds `VALID_SOURCES`: 33 of 39 queries now have more than one
valid source; the other 6 genuinely have one, where the metrics coincide.

Every added source was read and confirmed to answer the query. Sources that merely *mention*
the topic were **not** added — widening ground truth until everything passes is precisely
the standard-lowering this project forbids.

Effect on the dense baseline, same index and queries:

| ground truth | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| strict (Phase 4, **preserved unchanged**) | 35.9 | 56.4 | 66.7 | 0.4882 |
| multi-source (new) | 66.7 | 76.9 | 84.6 | 0.7442 |

R@5 moves 66.7 → 84.6, quantifying how much of Phase 4's apparent failure was labelling
rather than retrieval.

## 4. Dense vs reranked

| config | ground | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| dense (no rerank) | strict | 35.9 | 56.4 | 66.7 | 0.4882 | 0.6667 |
| | multi | 66.7 | 76.9 | 84.6 | 0.7442 | 0.8462 |
| rerank-10 | strict | 48.7 | 64.1 | 66.7 | 0.5620 | 0.6667 |
| | multi | 71.8 | 84.6 | 87.2 | 0.7836 | 0.8718 |
| rerank-20 | strict | 51.3 | 66.7 | 71.8 | 0.5923 | 0.7179 |
| | multi | 69.2 | 82.1 | 84.6 | 0.7646 | 0.8462 |
| rerank-50 | strict | 53.8 | 69.2 | 71.8 | 0.6239 | 0.7179 |
| | multi | 71.8 | 82.1 | 87.2 | 0.7909 | 0.8718 |

The reranker improves **ranking**: strict R@1 35.9 → 53.8, strict MRR 0.4882 → 0.6239
(**+28 %**), multi MRR 0.7442 → 0.7909.

**The 90 % R@5 target is still NOT met**: best strict R@5 71.8 %, best multi-source R@5
87.2 %. The bar was not lowered.

Promotion is honest in both directions: pool 10 promoted 7 and **demoted 6**; pool 20
promoted 8 and demoted 8; pool 50 promoted 11 and demoted 7.

## 5. Pool sizing — more is worse

| pool | candidate recall | final R@5 | end-to-end RAG | rerank latency |
|---:|---:|---:|---:|---:|
| 5 | 84.6 % | 84.6 % | — | 0.23 s |
| **10** | 89.7 % | **89.7 %** | **83.3 %** | **0.49 s** |
| 20 | 92.3 % | 87.2 % | 75.0 % | 1.05 s |
| 50 | 100.0 % | 87.2 % | — | 1.52 s |

Pool recall improves with N while the *final* ranking degrades — extra candidates give the
reranker more chances to promote a wrong passage. Two independent measurements (retrieval
R@5 and end-to-end RAG) agree that **10 is the sweet spot**.

## 6. End-to-end RAG — the decisive measurement

12 questions answerable only from this project's own files, so the no-RAG arm must fail
unless the model guesses.

**candidate_top_n = 20**

| arm | correct | accuracy | fact recall | halluc | refusals | prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| no RAG | 3/12 | 25.0 % | 21.4 % | 2 | 8 | 54 |
| dense RAG | 8/12 | 66.7 % | 71.4 % | 2 | 0 | 2011 |
| reranked RAG | 9/12 | 75.0 % | 78.6 % | 2 | 1 | 3165 |

**candidate_top_n = 10**

| arm | correct | accuracy | fact recall | halluc | refusals | prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| no RAG | 3/12 | 25.0 % | 21.4 % | 2 | 8 | 54 |
| dense RAG | 8/12 | 66.7 % | 71.4 % | 2 | 0 | 2011 |
| **reranked RAG** | **10/12** | **83.3 %** | **85.7 %** | 2 | **0** | **2438** |

**25.0 % → 66.7 % (dense) → 83.3 % (reranked)**, with fact recall 21.4 % → 71.4 % →
85.7 % and refusals 8 → 0 → 0.

**Hallucinations stay at 2 in every RAG arm and are reported rather than smoothed over.**
They are not caused by reranking — dense RAG shows the same 2.

## 7. Instruction A/B (pool 20)

| instruction | ground | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| none | strict | 56.4 | 69.2 | 71.8 | 0.6387 | 0.7179 |
| none | multi | 66.7 | 82.1 | **89.7** | 0.7588 | **0.8974** |
| default | multi | 69.2 | 84.6 | 84.6 | 0.7667 | 0.8462 |
| **general** | strict | 51.3 | 71.8 | **74.4** | 0.6228 | 0.7436 |
| **general** | multi | 69.2 | **87.2** | 87.2 | 0.7814 | 0.8718 |
| code | multi | **74.4** | 84.6 | 87.2 | **0.8054** | 0.8718 |

No single instruction dominates, which partially contradicts the model card's blanket
advice to use one. An instruction improves **ranking quality** (R@1, MRR) while no
instruction edges ahead on recall@5. **`general` is recommended** — the only setting
near-best on both axes, and the best on the strict hard-negative subset (87.5 %).

**`code` is not recommended as a global default**: it leads on multi R@1 here but is the
**worst** on hard negatives (68.8 %).

## 8. Hard negatives (20 pairs)

Cases are split by how defensible the label is, because reading the first run's failures
showed the *test* was wrong, not the model: several "negatives" genuinely answered the
query (e.g. asking *why* a mode should stay disabled while the negative explained *how* to
disable it). A blended accuracy would have overstated the model's failures.

| instruction | all | **strict** | ambiguous | inversions | mean margin |
|---|---:|---:|---:|---:|---:|
| default | 15/20 | 81.2 % | 50.0 % | 1 | 0.1933 |
| **general** | 17/20 | **87.5 %** | 75.0 % | 3 | 0.2598 |
| code | 14/20 | 68.8 % | 75.0 % | 4 | 0.2185 |
| none | 16/20 | 81.2 % | 75.0 % | 4 | 0.2629 |

**Two cases invert under every instruction**, and share a cause — the query asks WHY a
design decision was made while the negative merely DESCRIBES the mechanism:

* `hn19` "Why must the reranker not manage the FAISS index?" — inverted **4 of 4**
* `hn04` "Why should thinking mode stay disabled?" — inverted **3 of 4**

Roughly **3 in 4** strict cases rank correctly — not all of them. That is the honest
headline.

## 9. Duplicate / stale — a limitation, not a bug

| instruction | current beats stale | identical text identical score |
|---|---:|---:|
| default | **1/5** | 5/5 |
| code | 1/5 | 5/5 |

The reranker picks the **stale** passage in 4 of 5 cases, at scores within a few
thousandths of the current one (dup02: 0.9997 vs 1.0000). Identical text gets an identical
score in 5/5, confirming scoring is a pure function of text.

This is expected and is **recorded as a limitation rather than patched into the model**:
it sees text only and has no notion of time, commit or version. Freshness belongs in the
retrieval pipeline's metadata (`mtime`, `commit_hash` already exist). It should not be
described as the reranker failing at versioning — it was never given version information.

## 10. Performance

| metric | value |
|---|---:|
| Cold load | 1.22 s |
| VRAM delta | +1160 MiB |
| First request | 302 ms |
| Warm request | 39 ms |
| Unload releases | 2960 MiB |
| Cache all-hits | 0.2 ms vs 3834 ms cold (**20229×**) |

### max_length sweep (pool 20, multi-source ground truth)

| max_length | body | truncated pairs | R@1 | R@5 | MRR | mean s |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 464 | 226 | 69.2 | 87.2 | 0.7776 | **0.4446** |
| 1024 | 976 | 161 | **71.8** | 87.2 | **0.7975** | 0.9030 |
| **2048** | **2000** | **0** | 69.2 | 87.2 | 0.7814 | 1.0478 |
| 4096 | 4048 | 0 | 69.2 | 87.2 | 0.7814 | 1.0581 |

**512 is 2.4× faster at identical R@5** — yet 2048 stays the default. 226 of 780 pairs were
truncated at 512, and those are precisely the longest chunks, the ones most likely to hide
an answer at the tail. Phase 4 measured that tail truncation loses answers
(0.9399 → 0.000269). A speed-up bought by truncating the benchmark is not a free saving.

Raising max_length past 1024 buys **no quality on this corpus** while doubling latency —
pure padding cost, since the median candidate is 254 tokens.

### batch sweep (fixed 64-pair workload)

| batch | forward passes | mean s | pairs/s |
|---:|---:|---:|---:|
| 1 | 64 | 4.7488 | 13.5 |
| **4** | 16 | **3.4228** | **18.7** |
| 8 | 10 | 3.8360 | 16.7 |
| 16 | 10 | 3.8321 | 16.7 |
| 32 | 10 | 3.8311 | 16.7 |

**Batch 4 is the throughput optimum; larger batches are slower.** Past 4 the token budget
forces the same 10 passes anyway, so extra rows only add padding. This contradicts the
usual "bigger batch is faster" instinct and is recorded as measured.

## 11. VRAM co-residency — peak under load

Steady-state readings are misleading for a cross-encoder, so each configuration was driven
with real concurrent work while nvidia-smi was sampled every 150 ms, including all-long-document
re-rank rounds.

| configuration | at rest | **peak** | free at peak | ≥ 2 GB |
|---|---:|---:|---:|---|
| desktop baseline | 2157 | — | — | — |
| reranker alone | 3453 | 5248 | 6979 | ✅ |
| embedding + reranker | 6570 | 6649 | 5578 | ✅ |
| local-fast + reranker | 8486 | 8482 | 3745 | ✅ |
| **all three** | 7931 | **9865** | **2362** | ✅ |

All four keep ≥ 2 GB free, so three resident services are permitted. Note the reranker
**alone** shows the largest transient spike (+1794 MiB); with other services resident the
incremental spike is +6 to +80 MiB because the allocator reuses reserved memory.

Caveat: measurements were taken sequentially, so the desktop baseline drifted slightly
between configurations. The peak column is the reliable figure.

## 12. Token cost

| stage | tokens |
|---|---:|
| whole corpus (Phase 4 baseline) | 54,699 |
| candidate pool (internal to the reranker, never sent onward) | ~20 chunks |
| **final Top-5 sent to the model (pool 10)** | **2,438** |
| Phase 4 dense Top-5 for comparison | 2,011 |

Reranking costs **+427 prompt tokens** (2,011 → 2,438, +21 %) and buys **+16.6 accuracy
points** (66.7 % → 83.3 %). Pool 20 would have cost 3,165 tokens for *less* accuracy —
worse on both axes.

The Top-N pool is handled entirely inside the reranker; the model still sees only 5 sources,
so Phase 4's ~96 % token reduction is preserved.

## 13. Latency budget (embedding + FAISS + rerank + pack; excludes generation)

| configuration | latency | within 1 s? |
|---|---:|:--|
| dense only | ~0.04 s | ✅ |
| **rerank pool 10** | **~0.53 s** | ✅ |
| rerank pool 20 | ~1.09 s | ❌ |

Pool 20 exceeds the reference budget. The bottleneck is reranker forward passes, dominated
by **padding**: 20 documents padded toward a 2000-token budget when the median candidate is
254 tokens. Pool 10 removes most of that waste.

## 14. Known problems

1. **Reranked R@5 is 87.2 % (multi) / 71.8 % (strict) — the 90 % target is not met.**
2. **"Why" questions invert consistently** (§8) — the clearest model limitation found.
3. **No freshness awareness** (§9) — by design; belongs in the pipeline.
4. **Two hallucinations persist** in every RAG arm (2/12), unaffected by reranking.
5. **Pool 20 exceeds the 1 s latency budget**; the spec's suggested default was wrong for
   this corpus.
6. **Batch > 4 is slower**, not faster (§10).
7. **A single over-long document can still spike memory**, since it is deliberately allowed
   through alone rather than being dropped.
8. **Cross-run baselines drifted** during co-residency measurement (§11); peaks are reliable,
   small at-rest differences are not.
