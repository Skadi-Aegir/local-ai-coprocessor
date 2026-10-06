# Phase 5.5R — Evidence Consolidation

**Purpose:** one place that states which code produced which measured rounds, so rounds from
different implementations are never combined into a single figure. Written in P55R-06;
documentation only — no load test was re-run.

**Scope statement, read this first.** The RAG question used throughout is the task definition's own
question (`RAG_Q = "What context size does the local-fast worker use by default?"`), and the answer
is checked by string presence of `8192` in the generated text. That makes it a **task-definition
reference check** for the purpose of exercising the pipeline, **not an independent quality
assessment** of the model's answers. Nothing below should be quoted as an answer-quality metric.

---

## 1. Result paths, in run order, with the code version that produced each

| # | when | result path | commit | outcome |
|---:|---|---|---|---|
| 1 | P55R-04 | `results/P55R-04-staged/staged_rounds.json` | `111185d` | **2/5, margin breached** — permissive `allow_unload` never unloaded (`unloaded=[]`) |
| 2 | P55R-04 | `results/P55R-04-Brerun/residency_B_rerun.json` | `111185d` | 4/5 — B re-run, `r0-rag` refused (reproduces the lost historical B3) |
| 3 | P55R-05 | `results/P55R-05-baseline/baseline.json` | `111185d` | baseline floors; `restored_ok=true` |
| 4 | P55R-05 | `results/P55R-05-swapdet/selftest.json` | `111185d` | 37/37 deterministic |
| 5 | P55R-05 | `results/P55R-05-rag/rag_diagnostic.json` | `111185d` | **PASS**, min_free 3055 |
| 6 | P55R-05 | `results/P55R-05-five/five_rounds.json` | `111185d` | **4/5 FAIL** — `r1-4a-bg` refused connection |
| 7 | P55R-05 | `results/P55R-05-five2/five_rounds.json` | `346ada4` | **5/5 PASS**, free_min 3014 (gate round) |
| 8 | P55R-05 | `results/P55R-05-five3/five_rounds.json` | `346ada4` | **5/5 + 5/5 PASS**, free_min 3000 / 3021 |
| 9 | P55R-05 re-run | `results/P55R-05b-swapdet/selftest.json` | `543e6e7` | 37/37 deterministic |
| 10 | P55R-05 re-run | `results/P55R-05b-rag/rag_diagnostic.json` | `543e6e7` | **PASS**, min_free 3152 |
| 11 | P55R-05 re-run | `results/P55R-05b-five1/five_rounds.json` | `543e6e7` | **5/5 PASS**, free_min 3193 (gate round) |
| 12 | P55R-05 re-run | `results/P55R-05b-five2/five_rounds.json` | `543e6e7` | **5/5 + 5/5 PASS**, free_min 3368 / 3374 |

**Rounds 7, 8, 11, 12 are from a DIFFERENT implementation of the harness than round 6.** They must
not be averaged, pooled, or reported as "five rounds". Two separate three-round sets exist, one per
harness version:

* `346ada4` set — rounds at #7 and #8 — free_min **3014 / 3000 / 3021**
* `543e6e7` set — rounds at #11 and #12 — free_min **3193 / 3368 / 3374**

## 2. The 4/5 failure is preserved, and its cause is not the pipeline

`results/P55R-05-five/five_rounds.json` (commit `111185d`) records:

```
round 1: completed=4/5 refused=0 failed=1 free_min=3016 passed=False
  r1-4a-bg [failed]: URLError: <urlopen error [WinError 10061] ... refused ...>
```

Both required unloads **did** happen in that run, and the margin **was** held. The single failure is
item 4a — an embedding call — issued while item 3 had **intentionally** left the embedding model
unloaded. So the failure was an **undeclared ordering precondition in the harness**, not a pipeline
or budget failure. It is kept as the record of that harness version.

## 3. Which code change produced the 5/5 rounds

`git diff 346ada4 543e6e7 --stat` shows **`benchmarks/phase55r/p5_five_items.py  | 23 ++`** and
nothing else in code — the staged pipeline itself (`p3_swap.py`) was **not** changed between the
first diagnostic and the passing rounds.

Two harness changes, in this order:

1. **`346ada4` — the item-4 service restoration (the fix that produced the first 5/5).** Item 3
   unloads the embedding and reranker models *by design*, so it leaves them down. Item 4a is an
   embedding call. The fix restores a healthy residency between item 3 and item 4, records the
   restoration explicitly with its own timing (~20-22 s), and does **not** fold that latency into
   item 4. Every 5/5 round in both sets runs with this in place.
2. **`543e6e7` — an explicit precondition before item 3.** Item 3's stage 1 is an embedding call and
   needs `local-embedding` healthy, which the item never declared; it passed only because item 2 ran
   immediately before it. The precondition is now satisfied explicitly before item 3, timed
   separately. This is the same class of fix as (1) and closes the symmetric gap.

**Statement for reuse:** the rounds after the fix differ from the 4/5 round **only** in harness
service-lifecycle ordering, not in the staged unload pipeline, not in the margin, and not in any
budget.

## 4. What the isolated path is now measured to do

* Both unloads are **required and verified**, each on two prongs that must both hold: the port is
  free **and** the live free reading rose. Verified deltas: embedding **+1368 to +1392 MiB**,
  reranker **+2591 to +2595 MiB**, across four diagnostics.
* Full RAG completes with the 2048 MiB margin held throughout; worst observed `free_min` across all
  passing rounds and diagnostics is **3000 MiB**.
* **Latency, stated because it is the cost of this design:**
  * a full RAG cycle measured **24.4 s** (P55R-05) and **~22-24 s** in the re-run — this is the
    end-to-end stage sequence, dominated by the two model loads
  * **service recovery after the cycle** measured **~8 s per service** in P55R-05 and **22.4 s
    (reranker) / 44.6 s (embedding)** in the P55R-05 re-run under heavier machine load
  * so the reload cost is real and variable, roughly **8-45 s per service** depending on load. A
    production design must treat this as the dominant cost, not a rounding error.

## 5. Budgets and boundaries (unchanged in every round)

| item | value |
|---|---|
| safety margin | **2048 MiB**, unchanged |
| reranker default transient | **1800 MiB**, unchanged |
| 881 / 1361 MiB candidates | used **nowhere**; `candidate_881_or_1361_used=false` |
| production default routing | **unchanged** — the experiment builds the staged pipeline itself |
| models / drivers / CUDA / pagefile / security / Qwen-Image | untouched |
