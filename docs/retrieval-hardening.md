# Retrieval Hardening — Phase 5.5

**Scope:** hardening the retrieval stack built in Phases 4 and 5. **No new models deployed,
no OCR.**
**Results:** [`benchmarks/retrieval-v2-results.md`](../benchmarks/retrieval-v2-results.md) (generated from the stored JSON)
**Baseline:** [`benchmarks/retrieval-v2/baseline.json`](../benchmarks/retrieval-v2/baseline.json) (frozen, refuses to overwrite)

---

## 1. What changed

| area | before | after |
|---|---|---|
| benchmark | 39 queries, single-source-ish labels | **83 queries**, 44 new, **multi-source**, dev/held-out split |
| relevance vs recency | conflated | **labelled and scored separately** |
| causal questions | instruction was global | **intent-switched**, validated on held-out |
| freshness | none | **minimal version policy**, mtime explicitly not a signal |
| candidate pool | 10 (Phase 5) | **10 confirmed**, 20/50 measured and rejected |
| scheduling | not implemented | **prototype** with 3 strategies, 3 job classes |

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** "Held-out" here is not unexposed: 39 of its queries were published since Phase 4 and the 22 new ones were in view when the pool size and the causal instruction were chosen (E1), so no generalisation claim follows. "R@K" in this page is Hit@K, i.e. the first valid source within K (E2).
| token accounting | prompt tokens only | **internal vs LLM tokens kept separate** |

## 2. The baseline freeze

`freeze_baseline.py` records commit, all three model revisions/hashes/ports/implementations,
environment versions, the index inventory and the historical Phase 4/5 figures — **copied from
the reports rather than recomputed**. It **refuses to overwrite** without `--force`: a
baseline that can be rewritten underneath later results is not a baseline.

## 3. The Phase 4 / Phase 5 discrepancy

Phase 4 published **91.7%** RAG; Phase 5's dense baseline was **66.7%**. Same 12 questions,
same grader — so the difference had to be the retrieval arm, and it was: **the harnesses
searched different indexes** (1173 chunks vs 186).

A four-arm experiment varying **only** the index reproduced both numbers exactly
(91.7% and 66.7%). Both published figures are correct and **were never comparable**.

**Rule adopted: every number states its index. Cross-index comparison is forbidden.**

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** "Only the index" was too strong: the context-pack path also differed (Phase 4 packed over HTTP, Phase 5 locally), so the four-arm test is a controlled comparison, not a full causal account (E4). A controlled cross-index comparison IS allowed; mixing metrics from different indexes is not (AD-3). The 91.7 % is not an uncontaminated figure: all 12 questions sat verbatim in the served index (OPEN_ISSUES P1-1).

## 4. What the new queries revealed

The 44 new queries are deliberately harder and they changed the picture:

| finding | detail |
|---|---|
| **Candidate generation is the bottleneck** | held-out failures: **9 candidate misses vs 2 rank misses** |
| Reranker helps on held-out | dense R@5 77.0 → **82.0** |
| Reranker does **not** help on dev | dense R@5 63.6 → 59.1 (dev queries are simply harder) |
| Causal category is stuck | R@5 **57.1%** on both splits |
| Pool widening is useless | 10/20/50 identical for dense; reranking a wider pool is *worse* |

The dev/held-out disagreement about the reranker is **reported, not averaged away.**

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The 9-to-2 split mixes four causes: 10 of the 44 new queries name sources absent from the benchmark corpus (not retrieval failures), chunking loss is unmeasured, the rest are candidate misses, and 2 are ranking downgrades (E8). Pool widening is not useless: candidate recall rises 68.2 % -> 72.7 % at pool 50 while the final Top-5 does not improve (E3).

## 5. Causal-question experiment

| approach | outcome |
|---|---|
| A general instruction | baseline |
| B causal-specific, globally | rejected — the phase forbids replacing the instruction globally |
| **C intent-based switching** | **adopted** — R@1 59.0→60.7, MRR 0.6922→0.7004 on **held-out** |
| D light rule assist | no measurable gain over C |

The switch is a light, inspectable intent test (leading `why` / `为什么` / `为何` /
`what caused`) — **not** a model-confidence threshold. It raises *rank* without raising
*recall*, and **no category is harmed**, which was the required check. Tuned on dev only;
the gain **generalises** to the 61 held-out queries.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The 61 "held-out" queries are 39 long-published ones plus 22 that were in view when this switch was chosen. The gain is measured on exposed queries and does not show generalisation (E1).

## 6. Version-aware retrieval

Status registry: **120 chunks current / 66 historical**. Policy: superseded chunks and files
are excluded from current-state QA; `historical` files are **demoted, not dropped**, and left
fully available for explicitly historical queries.

**File mtime is never used as a validity signal** — a test asserts this by inspecting the
code, and the docstring states that a newer file must not override a still-valid technical
document.

**Honest scope note:** the exclusion path matched **0** chunks in the real corpus, so it is
covered by synthetic fixtures in `test_version_aware.py` (26 assertions) rather than by a
live case. Stated rather than presented as a live result.

## 7. Hallucinations — not fixed

Two hallucinations appear in **every** RAG arm, so the cause is not retrieval. Classified
against the five required distinctions: retrieval miss **no**, truncation **no**, conflicting
sources **no**, ignored evidence **partial**, **invented information yes (primary)**.

Both are the same pattern: a plausible number stated without support. The pack retains
`source_path`, `lines`, `commit_hash`, `dense_rank`, `rerank_rank`, and the prompt requires
`NOT IN SOURCES` when the answer is absent. **Refusals falling to zero in some arms is not
treated as an accuracy gain** — a model that stops saying "I do not know" while still
inventing numbers has become worse. **These 2 cases remain open.**

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The "invented information" diagnosis was withdrawn (E6). On the saved raw answers no case remains of a plausible number invented despite adequate evidence: the retrieval-arm case is a probable retrieval TRUNCATION (the answer says SOURCE 1 cuts the `runtime` section before the `context` value), the no-RAG pooling case is a grader false positive (the answer opens "I do not know."), and the two no-RAG embedding questions are out-of-domain guessing without context.

## 8. Scheduler

See [`docs/gpu-scheduling.md`](gpu-scheduling.md). Three strategies (`serial`, `conditional`,
`unload`), three job classes (`interactive`, `background`, `exclusive`), a live-measured
2 GiB margin, and Qwen-Image deliberately outside the registry so the prototype cannot unload
it. 24 policy assertions plus a PowerShell regression test, both passing.

---

# Review questions

### 1. Why did the Phase 4 and Phase 5 RAG baselines differ?
**Different indexes.** Phase 4 read the served `local-ai` index (1173 chunks, the whole
project); Phase 5 read the benchmark index (186 chunks, a selected 30-file slice). A
controlled four-arm test varying only the index reproduced **91.7%** and **66.7%** exactly.
Both are correct; neither was comparable to the other. Questions, top_k, packing and grader
were identical — only the searched corpus differed.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** Not only the corpus: the context-pack path also differed (E4).

### 2. Did causal-question ranking improve?
**Yes, modestly and reproducibly.** On held-out data the intent switch raised R@1 from 59.0
to **60.7** and MRR from 0.6922 to **0.7004**, with R@5 unchanged and no category harmed. It
improves *where* the right source ranks, not *whether* it was retrieved. The causal category
itself stays at R@5 57.1% because its failures are candidate misses.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** "Held-out" here is exposed (E1); see the note under §1.

### 3. How much of the version-conflict problem is solved?
**Policy yes, evidence partly.** Superseded material is excluded from current-state QA,
historical material is demoted not dropped, and historical queries retrieve the past
correctly — verified live on both intents. Of 5 held-out version-conflict queries, **4
succeed**; the failure is a candidate miss, not a recency error. The exclusion path is
exercised only by synthetic fixtures (0 live matches), and `commit_hash` is not populated per
chunk, so this rests on explicit status rather than commit comparison.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The 5 "held-out" version-conflict queries are part of the exposed set (E1).

### 4. Are the two hallucinations resolved?
**No. They remain open.** They appear in every arm including no-RAG, so they are generator
behaviour, not retrieval. Diagnosis: **invented information** — a plausible number with no
supporting source. Minimal mitigations (full provenance in the pack, an explicit
`NOT IN SOURCES` instruction) are in place but did not eliminate them, and the fall in
refusals is explicitly *not* counted as improvement.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The diagnosis "invented information" was withdrawn; see the note under §7 (E6).

### 5. Is an adaptive candidate pool worth adopting?
**No — and none was implemented.** A confidence-triggered pool would rely on uncalibrated
scores and be unfalsifiable, which the phase forbids. Widening the pool shows **no net gain**
at any fixed size (dense R@5 identical across 10/20/50; reranking a wider pool is *worse*).
A margin-based adaptive arm in the cost evaluation matched the full reranked arm with 65
fewer tokens and 2 of 12 reranker calls skipped, but the margin is too small to justify the
added complexity. **Fixed pool 10 retained.**

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** "No net gain" holds for the final ranking only: candidate recall does rise, 68.2 % -> 72.7 % at pool 50 (E3).

### 6. Which GPU scheduling strategy should be used?
**`conditional` by default.** Interactive work owns the GPU relative to background work;
background work may overlap. Use `unload` only where a service is genuinely disposable
(reload costs 1.2–1.7 s). Measured basis: concurrency cuts total wall time **14.4%** while
making each operation **45–69% slower**, so the right policy depends on whether latency or
throughput is the goal — a blanket semaphore of 1 is wrong.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The job-class policy described here was replaced in P55R-01: interactive work runs alone, waits for background units already running, and no new background unit is admitted while it waits (AD-9; docs/gpu-scheduling.md §3).

### 7. Does the 0.6B reranker still have a clear capability bottleneck?
**Its bottleneck is no longer the binding constraint.** Held-out candidate recall is 85.2%
and **9 of 11 failures are candidate misses** — the correct source never reaches the pool, so
no ranker could recover it. Where candidates *are* present the reranker is a clear positive
on held-out (R@5 77.0 → 82.0). It does **not** help on the four new categories, and it
consistently prefers passages that *describe* a mechanism over ones that *explain* a
rationale (`hn19` inverted under every instruction) — a real, explainable limitation.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** "Held-out" is exposed (E1), and the 9-of-11 count mixes four causes (E8); see the note under §4.

### 8. Is it worth testing a 4B reranker?
**Not yet, on this evidence.** The Phase 5 decision rule permits it (candidate recall ≥90%,
reranked R@5 <90%), but Phase 5.5 adds a stronger counter-argument: the dominant failure is
**candidate generation** (9:2), and **no reranker fixes a source that was never retrieved**.
Spending ~8× the model size on the ranker while the retriever is the bottleneck would be
optimising the wrong stage. **Recommendation: work on retrieval/chunking first**; revisit 4B
only if candidate recall rises substantially and ranking is then demonstrably the limit.
A 4B reranker was **not** downloaded.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** The 9:2 ratio behind this recommendation mixes four causes (E8); see the note under §4.

### 9. Is it safe to proceed to Phase 6 (OCR)?
**Not yet, and it should not start on this evidence.** Phase 5.5 leaves real, quantified
gaps: 9 held-out candidate misses dominating failures, a causal category stuck at 57.1%, two
unresolved hallucinations, a version policy resting on explicit status rather than commits,
and a scheduler that currently refuses every job class while services are resident because
the 2 GiB margin leaves only ~1.6 GiB usable. Adding OCR would layer a new modality on top of
an unresolved retrieval floor. Phase 1–5 environments are all intact and no destructive
change was made, so proceeding later is safe — it is the *sequencing* that is wrong.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** Two of the gaps listed here were later revised: the "held-out" misses are on exposed queries and mix four causes (E1, E8), and the two hallucinations were re-diagnosed (E6).

---

## Known problems carried forward

1. **Candidate generation dominates failures** (9 of 11 held-out). Retrieval/chunking, not ranking.
2. **Causal category stuck at R@5 57.1%** on both splits.
3. **Two hallucinations unresolved**; generator behaviour, not retrieval.
4. **Exclusion path has no live corpus match** — covered by synthetic fixtures only.
5. **`commit_hash` not populated per chunk**, so recency rests on explicit status.
6. **Dev and held-out disagree about the reranker's value**; both reported.
7. **Adaptive pool rejected**, but only after measuring — it is close enough that a larger
   study could revisit it.
8. **Scheduler refuses all three job classes while services are resident** at the current
   margin; the policy is correct but the machine is close to its limit.
9. **All scheduling figures come from one machine** with a 12 GB card and 32 GB RAM; no
   second configuration was available to test the policy's generality.

> **Superseded — noted 2026-10-05 from ERRATA, which is outside the QA corpus.** Items 1 and 6 rest on the exposed "held-out" set and the conflated 9:2 count (E1, E8); item 3's "generator behaviour" diagnosis was withdrawn (E6).
