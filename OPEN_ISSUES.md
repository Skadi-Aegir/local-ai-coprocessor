# OPEN_ISSUES

Open problems with evidence, priority and current status. Nothing here is closed by argument;
each states what would close it.

Priority: **P1** blocks a trustworthy conclusion or blocks work · **P2** blocks a production use ·
**P3** known limitation.

---

## P1-1 — The served index is contaminated with the answer key

**Evidence:** `benchmarks/phase55r/results/P55R-01-a2/a2_contamination.json`.
**Measured:** **12 of 12** evaluation questions appear **verbatim** in the served index, and
**10 of 12** have an expected fact sitting in a key-bearing file. The contaminated files include
`services/retrieval/rag_compare.py` (18 chunks — the questions plus their expected-fact AND
forbidden-value lists as Python literals), `retrieval_bench.py` (30), `benchmark_data.py` (13)
and three result documents (51). 112 of 1173 chunks.

**Why it matters:** a dense retriever embeds the query and can match a chunk carrying the
identical question and its answer list. The Phase 4 **91.7 %** RAG figure is therefore **not an
uncontaminated measurement**. Corroboration: a model answer quotes the harness source as
evidence ("SOURCE 2 (services/retrieval/rag_compare.py), line 39, explicitly lists this fact in
its test cases").

**Not done:** the historical index was NOT modified (index.faiss sha256 prefix `b50a3e46d094ca8f`
verified unchanged after the run). An exclusion list was produced and the paired experiment is now
**DONE — result in P1-9.**

## P1-2 — The Phase 4 / Phase 5 RAG difference is only partly attributable to the index

**Evidence:** `.../P55R-01-a1/a1_baseline_audit.json`, `a1_confounds.json`.
**Measured:** question sets, top_k and `RetrieveConfig` are **identical** across all three
generations. Indexes differ (1173 vs 186 chunks). **The context-pack PATH also differs**: Phase 4
asked the HTTP service to pack; Phase 5 packed locally.
**Why it matters:** the claim that "only the index differed" was too strong. The four-arm
reproduction is a valid **controlled comparison**, but it is not the historical configuration.
**What closes it:** a run using the historical HTTP pack path against both indexes.

## P1-3 — No unexposed held-out set exists

**Evidence:** `.../P55R-01-a3/a3_metric_audit.json`.
**Measured:** the "61 held-out" figure was 39 historical queries (published since Phase 4) plus
22 new ones. Pool size and the causal instruction were both selected with the test set in view.
**What closes it:** a new, never-inspected query set.

## P1-4 — 10 of the 44 new queries cannot be retrieved at all

**Evidence:** `.../P55R-01-a3/a3_metric_audit.json`.
**Measured:** c04, c05, c07, c13, c14, x02, x09, v04, n03, n06 name sources that are **not in the
benchmark corpus**. The harness counts them in the same `candidate_miss` bucket as genuine
retrieval misses, **conflating "the corpus never contained the answer" with "retrieval failed"**.
**What closes it:** either extend the benchmark corpus or relabel these queries.

## P1-5 — `commit_hash` is empty for every chunk

**Evidence:** `.../P55R-01-regr-pass/regression.json`.
**Uniform statement: `commit_hash` is populated for 0 of 186 chunks, i.e. 186 of 186 are UNKNOWN.**
(`mtime` is populated for 186 of 186 but is explicitly NOT used as a validity signal.)
**Why it matters:** no per-chunk commit provenance exists, so the version policy rests on explicit
status rather than commit comparison. The phase forbids filling it with the current HEAD, which
would attribute historical chunks to a commit that did not produce them. It is reported as
**unknown** rather than faked.
**What closes it:** populate commit hashes at index time from git.

## P1-6 — RESOLVED (2026-10-05) — the default three-model configuration refused all normal work

**Evidence:** `.../P55R-01-b4/b4_live.json` — 10 requests, **0 completed, 10 refused**.
**Measured:** with all three resident, `152 MiB usable < 600 MiB required` (free 2200 - margin
2048). The margin was **not** lowered to produce a pass.
**Measured alternative:** `local-fast` + `local-embedding` resident leaves **3431 MiB usable** and
admits background work; reload costs 1.0-1.5 s per service. Confirmed independently on 2026-10-05:
with 8111 and 8113 both resident, **4903-4913 MiB free**.
**Resolution (2026-10-05):** AD-12's alternative was **ADOPTED as AD-21**. The default resident set is
`local-fast` + `local-embedding`; `local-reranker` and `local-smart` are loaded **on demand** and are
not resident. Any new service must satisfy one of three conditions: hold the 2048 MiB margin beyond
the default pair, or be **CPU-only**, or be loaded by the scheduler on demand. The three-model
co-residency requirement is dropped, not worked around.
**Evidence:** `ARCHITECTURE_DECISIONS.md` AD-21; this round's `benchmarks/ocr/REPORT.md` records the
CPU-only path chosen for `local-ocr` under this rule.

## P1-7 — Residency: all three strategies complete 4/5, and the shortfall is a budget figure

**Status of the earlier blocking judgement.** An earlier revision recorded that three resident
models no longer fit alongside the operator's other workload, with the embedding service failing
on `OSError: pagefile too small (os error 1455)` at commit free 5485 MB. That was accurate for the
conditions then and the historical note is kept in ERRATA E11. It is **superseded** after the
operator closed the externally-running high-VRAM programs: the new baseline is recorded at
`benchmarks/phase55r/results/P55R-02-baseline/environment_baseline.json`, and the three-service
baseline is restored and verified (8111/8113/8114 all healthy, commit free 28 GB).

**Measured on the stable baseline. Fixed workload of 5 requests through ONE scheduler instance:**

| strategy | completed | the one failure | shortfall |
|---|---:|---|---|
| **A** three models co-resident | **4/5** | `int2` (interactive during background) | 821 vs 900 MiB — **79 MiB** |
| **B** fast + embedding, reranker on demand | **4/5** | `r0-rag` (full RAG) | 3418 vs 3560 MiB — **142 MiB** |
| **C** fast only, retrieval staged | **4/5** | `r0-rag` (full RAG) | 2791 vs 3560 MiB — **769 MiB** |

A improved from 3/5 to 4/5 once the external load was closed, so external load **did** interfere
with the earlier A run. **B and C still fail on the full-RAG request.**

**The decisive measurement — BOTH a budget component and an occupancy component.**

Declared figures are `local-fast 2981+0`, `local-embedding 1246+250`, `local-reranker 1160+1800`
= **7437 MiB** against **7464 MiB** available on a 12227 MiB card after the 2048 MiB margin —
**27 MiB of slack**.

The reranker's declared **1800 MiB** transient exceeds the measured peak. P55R-03 measured it
properly — 12 cycles, 4 load cases, 3 repeats each, sampler at 0.02 s, content-unique documents:
**observed maximum 881 MiB**, margin never breached, minimum free VRAM 2510 MiB. The declared
figure is therefore conservative by roughly 900 MiB.

**RETRACTED (P55R-03): "refusals are caused purely by the budget figure, not by insufficient
hardware."** A candidate budget of **1361 MiB** (881 x 1.4 sampling allowance + 128 MiB fixed
measurement margin, still 439 MiB below the declared value) was tested in an isolated registry and
full RAG was **STILL refused**: `2933 MiB usable < 3121 MiB required`, short by **188 MiB**. With
the reranker stopped only about 4826 MiB was free on a 12227 MiB card. **Both components
contribute, and their relative weight is not separable from what was measured.**

**RETRACTED (P55R-03): "all three strategies are short by a hair."** The numbers (79 / 142 /
769 MiB) were real, but the characterisation implied the gap was trivially closable and the
candidate-budget result shows it is not. Deficits are reported as numbers.

**The 262 MiB figure from P55R-02 is rejected as a basis** and is used nowhere in P55R-03. It came
from repeating identical documents, and `use_cache: false` is **not honoured** by the reranker
service — identical documents repeat at ~0.025 s regardless of the query, while new content costs
~0.43 s. Those repeats measured cache hits, not compute.

**Full RAG needs three services at once**, because the rerank stage consumes retrieval output.
That is why B and C fail on it: demand-loading the reranker while the embedding model is resident
requires 3560 MiB, and neither configuration can produce it.
**NOT achieved by the co-resident default.** A staged lifecycle that removes the co-residency is
**PROPOSED in `docs/staged-lifecycle-design.md`**, with its production proposal and rollback
conditions in `docs/production-integration-design.md`, and its per-round provenance in
`docs/phase55r-evidence-consolidation.md`. **It is not implemented and not enabled.**

**RESOLVED ON THE ISOLATED PATH (P55R-05).** With both unloads **required and verified** (port free
AND free VRAM rose) full RAG completes with the margin held: embedding unload **+1368 to +1392 MiB**,
reranker unload **+2591 to +2595 MiB**, worst observed `free_min` **3000 MiB** across all passing
rounds and diagnostics. Three rounds passed per harness version — `346ada4`: 3014/3000/3021 MiB;
`543e6e7`: 3193/3368/3374 MiB. An earlier 4/5 record at `111185d` is preserved and is **not** pooled
with these: its single failure was an undeclared ordering precondition in the harness, not a
pipeline or budget failure.

**Cost of the passing path, which a production decision must weigh:** full RAG **~22–24 s**, and
service reload **~8 s per service** under light load, up to **44.6 s (embedding) / 22.4 s
(reranker)** under heavier load.

**Health gate versus first inference, reported separately:** health gate **8.13 s**, first
inference **0.43 s**. A service answering `/health` is not thereby ready to serve inference.

**Margin monitoring:** no margin breach was observed within the sampling window. That is not a
continuous-time guarantee.

## P1-8 — Pipe-buffer deadlocks in my own harness, NOT external load

**Measured:** the mode C hang recorded in P55R-02 was caused by my own code. Two instances of the
same defect were found and fixed:
* `b2_unload_failure.py` — a ~15 KB transcript on a block-buffered stdout filled the pipe buffer
  and the process blocked on write. Every assertion had already passed. Fixed by flushing each
  line.
* `p2_residency.py::_run_script` — `subprocess.run(..., capture_output=True)` launched
  `start-local-reranker.ps1`; the child blocked on the full pipe while the parent blocked on the
  child. A powershell process was found alive for **700 s**. Fixed by redirecting to a file and
  reading it back, so the transcript is preserved.
After the fix the reranker starts through that same call in **8.3 s**.

**Why this matters:** the "C hung" observation was initially attributed to memory pressure near
VRAM exhaustion. It was not. A wedged child process and a wedged GPU present identically from the
outside; the difference was found only by dumping the offending process. Future hangs should be
diagnosed by identifying the blocked process before blaming capacity.

---

## P1-9 — Decontamination paired validation: contamination accounts for ONE fact, not for 91.7 %

**Evidence:** `benchmarks/phase55r/results/P55R-02-decon/` (`decon_build.json`, `arm_full.json`,
`arm_decon.json`, `pair.json`).

**MEASURED.** Frozen served index vs `decon_v1`, a copy with the 112 contaminated chunks removed
(1173 → 1061, `consistent=True`, built by reconstructing every vector). Everything else frozen: the
original 12 questions, Dense arm, top_k 5, pool 10, same instruction, same reranker, same pack, same
generation settings. 24 generation calls. Only retrievability of the contaminated material varies.

| measure | value |
|---|---:|
| facts stated in the answer — full index | **14 / 15** |
| facts stated in the answer — decontaminated | **13 / 15** |
| questions improved after decontamination | **0** |
| questions worsened after decontamination | **1** |
| answers that changed | 12 / 12 |
| questions whose answer is absent from the corpus after decontamination | **0** |
| questions whose top-5 contained a contaminated file (full arm) | **12 / 12** |

**The contaminated material WAS being retrieved in the full arm, and removing it changed one fact
out of fifteen.** On this pipeline contamination does not account for the Phase 4 91.7 % figure.
The direction is reported as measured: this shows neither that contamination was harmless nor that
accuracy improved.

**CORRECTED IN P55R-03 — which question differs, and what this run cannot support.**

The question whose fact count differs is **"How long does it take to switch between the two local
models?"** at **2 -> 1**. It is NOT the context-size question: that one is **0/1 in BOTH arms** —
its expected fact *is* present in the full arm's pack but the model did not state it, and it is
absent from the decon arm's pack. An earlier version of this entry attributed the single differing
fact to the context-size question; that was wrong and is corrected from the saved records.

**The exclusion removed three files that are NOT answer keys.** `benchmarks/local-fast-results.md`,
`local-smart-results.md` and `retrieval-results.md` are legitimate project result documents. This
run therefore **cannot separate the effect of removing the answer keys from the effect of removing
legitimate project material**, and it must **not** be used to claim the answer-key contamination was
quantified on its own.

**It also does NOT restore confidence in the Phase 4 91.7 % figure.** Nothing here re-measures that
pipeline, the 12 questions are the original exposed set, and the experiment answers only what
removing these six files changes on the *current* pipeline.

**Mechanism, which is the useful part.** For the context-size question the contaminated files were
retrieved and the expected value WAS in the pack, yet the model still did not state it — the packed
`config/local-fast.yaml` chunk is the file HEADER, not the `runtime` section at line 32 carrying
`context: 8192`, so the model answered that the value was "provisional". The binding constraint is
**chunk selection within a correctly identified file**, the same family as P2-4.

**Exclusion handling for the mixed files, as required.** Tier 1 (`rag_compare.py`,
`retrieval_bench.py`, `benchmark_data.py`) carry the questions and scoring rules, excluded
entirely. Tier 2 (`local-fast-results.md`, `local-smart-results.md`, `retrieval-results.md`) are the
project's legitimate measured output, but the questions were written FROM them, so excluding them is
the conservative choice; its cost is reported and no question became unanswerable.

**Scope limits:** not a reproduction of historical causality (the pack PATH also differed, P1-2),
not a generalisation test (the questions are exposed), and the decontaminated index is a separate
copy.

**Reuse:** `cache/retrieval_bench/indexes/decon_v1` is gitignored as a derived artifact. Rebuild with
`python benchmarks/phase55r/p2_decon_build.py --src vector/embedding/local-ai --out
cache/retrieval_bench/indexes/decon_v1 --report <path>`.

## P1-11 — RESOLVED on the deployed service (2026-10-05) — bulk document embedding inflated the embedding service's VRAM

**Status (2026-10-05, acceptance round): RESOLVED and verified on the DEPLOYED 8113**, not only in
isolation. 8113 was restarted from the merged tree carrying the fix; its `/stats` now carries the
`memory` field, which is the proof the fixed code is running (`torch_reserved_mb 1160.0`,
`release_threshold_mb 256`). `benchmarks/p1_11/p1_11_service_smoke.py --port 8113 --run-id
AB-02-deployed2` returned **`verdict = released`, `retained_mb = 0.0`,
`cache_releases_during_embed = 1`**, with `peak_above_weights_mb 1893.6` released afterwards. The
subsequent index rebuild read the service after embedding and recorded
**`retention.restart_required = false`, `retained_mb 0.0`**.
The **"restart 8113 after bulk indexing" workaround is therefore WITHDRAWN.** The
"reranker not resident for the QA path" precondition STAYS — that one is a margin fact, not a leak.
Evidence: `benchmarks/phaseAB/REPORT.md` §3, `benchmarks/p1_11/results/AB-02-deployed2.json`,
`benchmarks/phaseAB/results/AB-03-build/a_index_build.json`.
* **Cause:** PyTorch's caching allocator kept the activation blocks of long document batches. On the
  exact Phase A corpus at 16 texts per request: **5053.5 MiB** left reserved above the weights.
* **Fix:** `Embedder._release_cache()` returns cached blocks after each encode (threshold 256 MiB;
  `server.py --release-cache-mb 0` restores the old behaviour): **15.5 MiB** left, same wall time,
  bit-identical vectors. On the modified service via `/v1/embeddings`: **0.0 MiB** left.
* **Second risk, also addressed:** the build's own transient at 16 per request is ~3.75 GB, which
  breaches the margin with local-fast resident regardless of retention. The builder now sends 4 per
  request (~1 GB, and faster), refuses to embed when the margin cannot hold, samples free VRAM during
  the build and reads the service's retention afterwards.
* **Residual:** the service still does not bound the activation memory of a single request; a
  client sending many long documents at once can spike it. The builder is the only bulk client.
* **What closes it:** restart 8113 from the fixed tree; `p1_11_service_smoke.py --port 8113` must
  report `released`, and the next index build must report retention below 512 MiB with the margin
  held during the build.
* **Measurement correction:** `/stats vram_mb`, the figure all earlier P1-11 numbers use, is
  cudaMemGetInfo `total - free`; on this machine it is neither the device-wide figure nor the
  footprint (ERRATA E25). The footprint is now `/stats memory.torch_reserved_mb`.

**Evidence (original):** Phase A build and runs; `benchmarks/phaseA/REPORT.md` limitation L1. Reproduced
in Phase B: `vram_mb` 7756.6 against 2360.6 at start, free VRAM 673 MiB.
**Measured:** after embedding 520 chunks through the 8113 service, its own `/stats` reported
**`vram_mb: 7750.6`**, up from **2360.6** at service start — about **5.4 GB retained**, and not
released when the embedding work finished. With the reranker also resident, free VRAM fell to
**640-743 MiB, below the 2048 MiB margin**, and two Phase A acceptance attempts had to be discarded
for that reason. **Restarting 8113 restored `vram_mb` to 2360.6** and free VRAM to about 5.2 GB,
after which the same 10-question run held the margin at a **5105 MiB minimum**.
**Why it matters:** any future re-indexing silently degrades the margin for every later request until
the service is restarted. This is the service's own memory behaviour, not a scheduling-policy issue,
and it is the largest remaining risk to the QA path.
**NOT caused by the CLI:** the growth is inside the service process and persists after the CLI exits.
**Workaround in force:** restart 8113 after bulk indexing and keep the reranker non-resident for the
QA path; both are recorded as preconditions in the Phase A report.
**What closes it:** find and fix the retention — allocator caching, batch shape, or an un-released
per-batch buffer — so the service returns to its baseline footprint after a bulk embed.

## P2-1 — The grader counts mention as assertion, and also misses fabrications

**Evidence:** `.../P55R-01-a2g/a2_grader.json`.
**Measured, in BOTH directions:**
* FALSE POSITIVE — a **denied** wrong value is flagged as a hallucination:
  "the context size is not 4096" gives `hallucination_flags = 1`
* MISS — a fabricated extra is invisible: all required facts plus an invented number gives
  `all_facts = True`
* MISS — a correct refusal and an incorrect refusal score identically

**Consequence (corrected in P55R-02):** errors run in **both** directions, so the **net bias is
UNKNOWN**. An earlier wording said the grader "can only over-count" and that published figures are
"upper bounds"; that was itself an over-claim and is withdrawn. The figures have demonstrated
error of unmeasured net sign.
**What closes it:** adopt the corrected grader for new work; historical scoring stays unchanged.

## P2-2 — Correct and incorrect refusals are indistinguishable

**Evidence:** same file. Both graders return identical results whether the model refused with the
evidence present or absent.
**Why it matters:** a falling refusal rate cannot be evidence of improvement.

## P2-3 — Two hard-negative cases invert under every instruction

**Evidence:** `benchmarks/reranker-results.md` section 8 (Phase 5), hn19 and hn04.
**Measured:** queries asking WHY a design decision was made are consistently outranked by passages
that merely DESCRIBE the mechanism. hn19 inverted under 4 of 4 instructions, hn04 under 3 of 4.
**Status:** consistent and explainable, not noise. Roughly 3 in 4 strict hard-negative cases rank
correctly, not all.

## P2-4 — Retrieval candidate generation is the dominant failure mode

**Evidence:** `.../P55R-01-a3/a3_metric_audit.json`. On held-out, **9 candidate misses versus
2 rank misses**. Candidate recall 85.2 %.

**The 9 are NOT one cause and must not be reported as one number.** Split by what the evidence
actually supports:

| cause | count | basis |
|---|---:|---|
| **answer absent from the corpus** | **10 of the 44 new queries** | their labelled sources are not in the benchmark corpus at all. These are NOT retrieval failures and must not be counted as such |
| **chunking loss** | **unknown** | a chunk never produced cannot be distinguished from one produced and missed, because the benchmark records no chunking-independent view of each source. Evidence insufficient |
| **candidate miss** | the remainder after the above | the source exists in the corpus and is chunked, but was not in the candidate pool |
| **ranking downgrade** | **2** (the recorded `rank_miss`) | the source WAS in the candidate pool but fell outside the final Top-5 |

Only the last two are retrieval-quality signals, and only "ranking downgrade" is a reranker signal.
The earlier presentation of a single "9:2" figure conflated all four.

**Why it matters:** no ranker can fix a source that was never retrieved, so further reranker work
is aimed at the wrong stage for most of these failures.
**What closes it:** chunking or retrieval improvements measured on the same sets, plus a
chunking-independent ground-truth view that would make the "chunking loss" row measurable.

## P2-5 — Cross-document questions are the weakest category, and "multi-source" is ambiguous

**Evidence:** held-out per-category table — causal 4/7, cross_doc 4/6, both dominated by candidate
misses. Multi-source completeness: Hit@5 84.3 % but all-sources-present only **33.3 %**.

**The "multi-source" label conflates two different requirements and must be split:**
* **ALTERNATIVE sources** — any ONE of the listed sources answers the question. Scoring it by
  Hit@K is CORRECT, and the 84.3 % figure is valid for this class.
* **REQUIRED evidence groups** — every listed source carries a distinct part of the answer, so all
  must be present. Scoring this class by Hit@K is WRONG; the correct measure is
  all-sources-present, which is **33.3 %**.

The benchmark does not currently mark which class a query belongs to, so the two are mixed in one
figure. Until they are separated, neither number may be quoted without saying which class it
describes. This is what makes the 84.3 %/33.3 % pair look contradictory when it is not.

## P2-6 — The served corpus and the benchmark corpus disagree

99 files versus 32; 1173 chunks versus 186. No single index represents "the project". Every claim
must name its index (AD-3).

---

## P1-10 — BLOCKING PREREQUISITE for any staged-RAG production integration: no cross-process ledger

**Evidence:** `status()["process_boundary"]`; also P3-1.
**Measured:** the scheduler's reservation ledger and its lock are **in-process only**. A service is
protected from unload only by reservations that THIS process can see.
**Why it is blocking for the staged design:** the staged path's correctness rests on five rules —
protection ends at release; a required unload is verified on two prongs; nothing can take the freed
memory during admission; the pipeline holds at most one reservation; a required unload is refused
while the pipeline itself holds a reservation. **Every one of those depends on all participants
sharing one ledger.** Two processes driving the same services would not see each other's
reservations, so one could unload a model the other is about to use, and the margin arithmetic would
be computed against a reading that excludes the other's commitment.
**What closes it:** either serialise all staged RAG through a single process, or add a
cross-process lock/ledger. **Not implemented, and not needed while nothing uses the staged path.**
**Status:** must be resolved BEFORE enabling the integration in `docs/production-integration-design.md`.

## P3-1 — The scheduler's lock is single-process only

**Evidence:** `status()["process_boundary"]`. Separate CLI invocations do **not** share it. No
cross-process coordination exists and none should be claimed.

## P3-2 — Unload timing has been measured only on one machine

All GPU, VRAM and pagefile figures come from one 12 GB card with 32 GB RAM. The pagefile peaked at
32016 MB of 38400 MB during this work, which is close enough to the commit limit to have caused a
native crash (0xC0000409) earlier in the project, and in P55R-02 it **did** prevent a third service
from loading while other large programs were running (see ERRATA E11).

## P3-3 — Co-residency measurements were taken sequentially

Desktop baseline drifted between configurations, so at-rest columns are not strictly comparable.
Peak figures are the reliable ones.

## P3-4 — Both phases found several of their own test-design errors

Recorded because they bound how much the passes are worth:
* a scheduler policy test measured the GPU rather than the policy (memory refusals mistaken for
  policy failures) — fixed by stubbing memory for policy checks
* a B3 case held one slot while requesting another, deadlocking itself — fixed in the test
* a B2 in-use guard case never admitted the occupying job, so the guard had nothing to protect —
  fixed by asserting the precondition
* the regression suite reported CUDA OOM from resource contention as test failures — now labelled
  precondition failures
* an mtime check grepped its own docstring and failed a correct implementation
* P55R-02: a declared-dependency check refused on "not currently healthy", which made on-demand
  loading impossible — fixed to "can it be MADE available"; the test asserting the wrong behaviour
  was replaced
* P55R-02: B2's harness stubbed `port_free` but not `service_healthy`, so every fake service looked
  unhealthy under the new check and the case refused for the wrong reason
* P55R-02: two pipe-buffer deadlocks (see P1-8)

Each was fixed on the side that was actually wrong (test or code), and which side is stated in the
relevant commit.

## P3-5 — `mtime` semantics are limited

See P1-5. `mtime` is populated but is explicitly **not** used as a validity signal.

---

# Added by the review of 2026-10-05 (ERRATA E19-E28)

## P1-12 — CLOSED (2026-10-05) — Phase A accepted, with one condition on the truncation fix

**Verdict, recorded verbatim from the accepting authority (Claude, who replaced Astra as acceptance
authority on 2026-10-05):**

> **Phase A：验收通过，附一项条件。**
>
> 依据：
> - 索引直接从当前文件重建，两次构建逐位一致；
> - 回滚目标是干净的版本；
> - 引用按内容哈希校验；
> - 全新冻结题集 heldout_v1 只跑了一次：
>   - 9 道可答题全部完整，且每条陈述都有引用支撑（指挥方已逐题对照被引原文独立复核）；
>   - 3 道无答案题全部正确拒答；
>   - 无支持陈述 0 条，无效引用 0 条；
> - 余量全程守住。
>
> 条件：本轮 A 部分修复"回答被 256 token 静默截断"，并完成验证。
>
> 记录在案的局限：
> - 只有 12 题，不支持任何泛化结论；
> - 题目由修复方编写，独立性有限；
> - heldout_v1 已经暴露，以后的验收必须使用新题集；
> - h03 预登记的来源不完整，语料中另有同样有效的来源；
> - 无答案题的拒答仍依赖 4B 模型遵循 prompt（P2-7）。

**The condition is discharged in this round by AD-22** (answer limit 512, `finish_reason` and
`answer_truncated` recorded on every record, truncation stated in the CLI and in the HTTP response).
The record of the defect itself, and of the fix, is in `ARCHITECTURE_DECISIONS.md` **AD-22** and in
`benchmarks/qa_hardening/REPORT.md`.

**heldout_v1 is now EXPOSED and may never be used for acceptance again.** Its results have been read
and discussed, so any future acceptance must use a **new, unexposed frozen set** with the same
pre-registration discipline. heldout_v1 stays valid only as a regression set.

**Evidence:** `benchmarks/phaseA/heldout_v1/{questions.md,expected.json,results/run1.json,results/run1_audit.json,review.md}`,
`benchmarks/phaseAB/REPORT.md`, `ARCHITECTURE_DECISIONS.md` AD-22.

## P1-13 — RESOLVED (2026-10-05) — no clean rollback index existed

**Evidence:** ERRATA E21; `vector/qa/current.prev/manifest.json`.
**Measured (before):** the rollback slot held `a3e3c050636f838e`, built before the answer-key
exclusions: it contained `services/retrieval/rag_compare_rerank.py` (the 12 tuning questions with 12
`facts=[` lines) and three evaluation harnesses. The QA path refuses to open any index whose manifest
lacks a passing answer-key leak check, so it could not be served by accident - which also meant a
rollback had nothing to roll back to.
**Resolved by the 2026-10-05 acceptance-round switch:** the rebuilt index `488f54d755224c5b` is now
`current`, and `current.prev` is the **clean `259fae05785b433c`**. The contaminated build no longer
occupies any slot (`vector/qa` holds exactly `current` and `current.prev`).
Evidence: `benchmarks/phaseAB/REPORT.md` §4, `vector/qa/current.prev/manifest.json`.

## P2-7 — The evidence floor does not refuse answer-absent questions; the model does

**Evidence:** ERRATA E23. No record across P56, Phase A and Phase B was refused by the 0.30 floor;
all 150 recorded P56 / Phase A candidates scored 0.486 or higher, and answer-absent questions scored
as high as answerable ones.
**Why it matters:** "refuses when the corpus has nothing relevant" rests entirely on a 4B model
following the prompt. The floor is a guard against an empty or broken index, and is now documented
as such in `cli/ask.py`.
**What closes it:** a refusal signal that is measured to separate answer-absent from answerable
questions on an unexposed set - not a threshold tuned on the sets already seen.

## P2-8 — The citation gate checks identity and range, never support; any mention counts as a citation

**Evidence:** `cli/ask.py` citation parsing. The bare-form pattern registers every "SOURCE n" in the
answer, including "SOURCE 2 does not say ...", which is the mention-versus-assertion weakness of
the old grader (P2-1) inside the user-facing gate. The verifier confirms that a cited file is the
indexed version and that the lines exist; whether they support the sentence is a human judgement.
**What closes it:** a support check per claim, or the claim stated as a known limit wherever the QA
output is consumed (it is now stated in the code).

## P2-9 — Embedding vectors depend on how documents are batched

**Evidence:** `benchmarks/p1_11/results/2026-10-05-rs1.json`. Against unpadded single-text vectors,
batched requests give mean row cosine 0.99984-0.99985 and nearest-neighbour agreement of 90-94 %:
near-tied neighbours swap.
**Why it matters:** a rebuild reproduces the index bit for bit only with identical batching, and the
builder's request size changed from 16 to 4 on 2026-10-05, so the next index is a new index and its
retrieval must be validated as such. Queries are always embedded alone.
**What closes it:** nothing needs closing if every build records its request size (it is in the
builder's version inputs) and is validated on its own; noted so that a vector difference between
builds is not mistaken for a defect.

## P2-10 — The local QA interface: loopback only, no authentication, one request executing at a time

**Evidence:** `cli/ask_server.py`, Phase B L2/L3. No authentication and no TLS, which is acceptable
only because it binds loopback; it must not be exposed without both. Two requests are admitted and
one executes; the second slot is a queue of one. MAX_CONCURRENCY = 2 is a starting value, not a
measured optimum (local-fast runs a single slot, so parallel generation would queue there anyway).

## P2-11 — RETRACTED (2026-10-05, P57-C0R): the v02 "answered with no required evidence" claim was a measurement error

**The claim, as previously recorded here:** that the QA path answered v02 while holding **0 of 2**
required facts, with three verified citations, and that this was a second defect worse than
under-retrieval. **That claim is withdrawn.** It rested on a fact detector that could not see the
evidence which was in fact present.

**What the corrected measurement shows** (`benchmarks/p57c0r/results/r02_scored.json`; one model call per
measurement, full pack and full answer stored):

| k | facts in pack | marker verdict | **adjudicated verdict** |
|---|---|---|---|
| 5 | 1 / 2 | partial | **partial** — covers the margin policy, not the memory-figure meaning |
| 8 | 2 / 2 | complete | **partial** — the pack carries the config *values*, not the explanation |
| 10 | 2 / 2 | complete | **partial** — same evidence; answer still relays values |

**The marker verdicts in the middle column were too generous and are superseded** by the P57-C0R-02
offline adjudication, which reads the text rather than testing markers. Applying the rule that a
configuration value, a VRAM delta, or a bare `vram_mb` **cannot substitute** for an explanation of what
`torch_reserved` means: at k=5 that fact is **absent**, and from k=8 the pack carries
`config/local-embedding.yaml`, which states `expected_vram_mb: 2400` / `measured_vram_delta_mb: 1246`
and a comment distinguishing *"torch's own attribution ~2361 MiB"* from the measured delta — **values and
a hint, not the explanation**. The passage that explains it,
`docs/models/local-embedding.md:90-112` (*"cudaMemGetInfo total − free, not PyTorch's attribution to this
process"*), **was never supplied at any k**, because it ranks **15** for this question. The answer
therefore relays values and never explains the field.

**So v02 is `partial` at every k**, and it is a **RANKING failure**, not a depth one. The retraction
above still stands in full: v02 is **not** answered with no relevant evidence, and the original claim
remains withdrawn. See `benchmarks/p57c0r02/RESULTS.md` and `ERRATA-append.md` R10–R15.

**Three of my own errors produced the retracted claim:**
1. **Truncated carrier sets.** The facts file recorded `carriers[:12]` while reporting a separate
   `n_carriers`, so a consumer reading `carriers` saw a truncated set.
2. **Mixed requests.** The old comparison called `search()` once and `answer()` again, then reported the
   first call's candidates beside the second call's pack. Two different top-k lists; only the pack was
   ever seen by the model.
3. **A detector keyed on the wrong fact name.** The corrected script first used the key
   `vram_figure_meaning` where the frozen manifest says `torch_reserved_meaning`, so **no marker set
   applied** and the fact scored absent. Separately, the frozen manifest's own detection literals for
   that fact are `["torch_reserved", "reserved_mb"]` — the **column names** — too strict, because the
   passage that explains the figure says *"VRAM, torch attribution"* and *"cudaMemGetInfo total − free"*
   and never prints `torch_reserved`.

**Also corrected, and it changed the verdicts:** `cli/ask.py` decides `refused` by **exact match**, so an
answer written as *refusal + reasoning* looks answered. Measured: **6 of 18** answers contain the refusal
sentence. v01 at k=8 and k=10 are both *"I could not find enough evidence… **Reasoning:** …"*, and the
reasoning prose was then scanned for fact markers, so the model's own restatement of the question
registered as a covered fact. With that fixed, v01's three ks are all **incorrect refusals**.

**What survives is narrower and still real:** the exact-match refusal detector mislabels a refusal that
carries reasoning. Recorded separately as **P2-12**.

**Original text retained below, marked retracted, so both the error and its correction are on record.**

> ~~**Evidence:** `benchmarks/p57c0/results/answerable_run.json`, question **v02** of the frozen
> development set (`p57c0_devset_v1`). Reproduce with `benchmarks/p57c0/c0n_answerable_run.py` on index
> `3cd2dcda4eeb5555`. Measured: asked how the embedding service memory figures and the scheduler margin
> relate to each other, the path **answered** while **0 of the question's 2 required facts were present
> in the pack**, with **3 citations that all verified**.~~ **RETRACTED — see above.** The record it
> rested on stored only the first 400 characters of the answer and never stored the pack, so the claim
> could not be re-verified from it; the corrected measurement supersedes it.

## P2-12 — NEW (2026-10-05, P57-C0R): the refusal detector misses a refusal that carries reasoning

**Evidence:** `cli/ask.py` compares the answer to the refusal sentence by **exact match**
(`refused = text.strip().lower() == REFUSAL.lower()`). Measured on the frozen development set: **6 of 18**
answers contain the refusal sentence, and at least two (v01 at k=8 and k=10) are
*"I could not find enough evidence in the local corpus to answer this.\n\n**Reasoning:** …"* — the refusal
followed by an explanation of what was missing.

**Consequence:** `refused` is recorded `False` for those records. The model declined; the record says it
answered. Any consumer reading `refused` — the CLI output, `ask_server.py`'s `outcome`, or a grader — is
misled. It also corrupted my own scoring: the reasoning prose was scanned for fact markers, so the
model's restatement of the question (mentioning port `8111`, the port of the file the question is about)
registered as a *covered fact* and turned an incorrect refusal into a reported `partial` answer.

**Why the exact match was chosen, and why that reason does not cover this case:** it was introduced in
P56 so a *substring* test would stop flagging answers that merely **quote** the refusal sentence while
still answering (q04 did exactly that). That was correct. But the model also writes refusal **plus**
reasoning, and an exact match cannot tell "quotes the refusal, then answers" from "refuses, then
explains" — both contain the sentence.

**What closes it:** a detector that separates the two cases — e.g. treat a **leading** refusal sentence as
a refusal, while ignoring one that appears only as a quotation inside a longer answer, and score the
reasoning separately from the answer. That is a change to the QA path and is **out of scope for
P57-C0R**, which is measurement-only. Recorded so it is not lost.


**Evidence:** `benchmarks/p57c0/results/answerable_run.json`, question **v02** of the frozen
development set (`p57c0_devset_v1`). Reproduce with
`benchmarks/p57c0/c0n_answerable_run.py` on index `3cd2dcda4eeb5555`.

**Measured:** asked *"How do the embedding service memory figures and the scheduler margin relate to
each other?"*, the path **answered** while **0 of the question's 2 required facts were present in the
pack**, with **3 citations that all verified**. `torch_reserved`'s carriers were never retrieved; the
answer was assembled from the *margin* evidence that was retrieved:

> *"…relate to each other through a strict enforcement rule where a **2 GiB (2048 MiB) VRAM m…"*

**Why this is distinct from P2-7 and P2-8, and worse than the defect P57-C0 set out to fix:**
* **P2-7** says refusal rests on the model rather than the 0.30 floor. **This is a concrete,
  repeatable instance of that**, not a restatement.
* **P2-8** says the citation gate checks identity and range rather than support. Here that limit
  produces a **fully-cited answer with no required evidence behind it** — the failure mode reaching
  the user.
* The round's own subject was **under-retrieval**, which **refuses safely**. v02 is
  **over-answering**: confident, cited, and unsupported. Different cause, different fix, and not
  in P57-C0's scope.

**The asymmetry is the actionable part.** v01, v04 and v05 also lacked required facts and they
**refused**; v02 lacked **all** of them and **answered**. So the refusal decision is not driven by any
measured "are the required facts present" signal — it follows the model's judgement on whatever was
packed. Ordering the evidence differently changed the outcome, which is what makes this a defect
rather than bad luck.

**Not to be confused with the fabrication question:** the cited content is real and correctly
attributed; what is missing is the evidence the *question* required. That is precisely why citation
verification does not catch it.

**What closes it:** a check that compares the question's required evidence against what the pack
actually contains before allowing an answer, or an explicit statement of the limit wherever the output
is consumed. Either is a design decision, not a bounded retrieval tweak.

## P3-6 — The reranker service keeps its allocator cache the same way the embedding service did

**Evidence:** `services/reranker/reranker.py` releases memory only in `close()`; PROJECT_STATE records
that a repeated reranker case costs ~0 extra VRAM "because PyTorch's caching allocator retains the
block". Not changed in this round: the scheduler's declared reranker budget and every P55R
co-residency figure were measured with this behaviour, so releasing there needs its own measurement
first.

## P3-7 — The correction layer is not in the QA corpus

**Evidence:** ERRATA E27. ERRATA, OPEN_ISSUES, ARCHITECTURE_DECISIONS and PROJECT_STATE are at the
repository root and not indexed; the documents they correct are. Four `docs/` files were corrected or
annotated on 2026-10-05.
**Rule to keep it closed:** when a conclusion is retracted here, the `docs/` page that states it gets
a note in the SAME section (the chunker splits by section), in the same commit.

## P3-8 — ENVIRONMENT RISK: the antivirus can delete a DSH runtime file mid-call

**Evidence:** ERRATA **E29**. On 2026-10-05 Kaspersky deleted
`node_modules/@deepseek-ai/dsh-subprocess-local/lib/runner.js` **while the tool was using it**. The
`pwsh` tool call died with `exit code 1073741845` (`0x40000015 STATUS_FATAL_APP_EXIT`) and produced
no output. The user restored the file out of band; verified afterwards: `node --check` exits 0, the
module imports and exports `runSelectedSubprocessRunner`, and stdout capture plus a background-job
spawn both succeeded (restored sha256 `4E21528DFDB3E92BDCE2340945DC4DCB202E4A4AF0BBEB8B97585F67735D30C1`).
**Measured environment:** Kaspersky is the registered and enabled antivirus (`productState 0x41000`)
and **Windows Defender is disabled** (`0x60100`) — `Get-MpPreference` fails with `0x800106ba`. **An
exclusion can therefore only be set in Kaspersky, and only by the user.**
**Why it matters:** a re-deletion breaks the execution tool, not a project artifact, so it can look
like a project failure. The signature is distinctive — `1073741845` with no other diagnostic. The
correct first response is to check that the runtime file still exists, not to debug project code.
**Risk to this round specifically:** Phase 6 downloads a large third-party wheel set
(`paddlepaddle`, `paddleocr`) and model files. A new antivirus action during `pip install` or model
download is a **stop-and-report** condition for this round (task §6), not something to work around.
**Not done, deliberately:** no antivirus setting, exclusion or security configuration was changed —
outside this project's permitted scope (task §2.2).

### P3-8 ESCALATION (2026-10-05) — the detection type is known, and it is an architectural false positive

**Three occurrences in two rounds, now with the detection name.** The user supplied Kaspersky's own
dialogs, which identify the mechanism:

```
PDM:Exploit.Win32.Generic.nblk
```

**`PDM:` is the Proactive Defense Module — a BEHAVIOURAL, heuristic verdict, not a signature match.**
Kaspersky did not find known malicious code in the file; it judged what the file *does at runtime*.
That reframes the whole issue:

> The antivirus is not misjudging a file's **contents**. It is misjudging DSH's **execution model**,
> which makes this false positive **architectural and recurring by construction**: the more normally
> DSH works, the more it resembles what PDM is built to catch.

**Why `runner.js` in particular.** The surviving copy is a generic shell-free child-process executor:

```js
import { execFile } from 'node:child_process';
export const runNativeCommand = (command, args, signal, window) => ...
    execFile(command, [...args], { encoding: 'utf8', signal, windowsHide: ... })
```

A generic "spawn processes for other code" helper, invoked repeatedly, is a standard loader/launcher
shape. `windowsHide` deliberately hiding the child window makes it look worse to a heuristic. Each
`pwsh` tool call goes through this path, so the trigger surface is every command this session runs.

**Occurrences, all named:**

| # | file | outcome | evidence |
|---|---|---|---|
| 1 | `dsh-subprocess-local\lib\runner.js` | genuinely deleted; user restored | ERRATA **E29**; tool call died with `1073741845` |
| 2 | `dsh-subprocess-local\lib\runner.js` | deleted again; user restored again | quarantine dialog, detected 5/10/2026 下午 5:40:08 |
| 3 | *(no real file)* | **quarantine entry for a path that does not exist** | ERRATA **E31 correction** |

**Occurrence 3 is a separate finding and is now closed as a non-event.** The third quarantine entry
named `dsh-native-command\lib\runner.js`, and that path **is not part of that package and never was**:
the package manifest publishes only `lib/index.js` and `lib/types/**/*.js`, and the npm tarball —
downloaded and **verified against the lock file's sha512 integrity** — contains 22 files, none of them
`lib/runner.js`. The installed directory matches that artifact **file for file, 22 of 22, zero
differences**. So nothing was deleted and **no recovery is possible or needed**. The quarantine
dialog's own path was visibly truncated and mixed a profile path with a *different* package's
directory, so its entries should not be treated as a reliable record of what existed.

**Risk, honestly stated.** The exclusions now cover threat detection for the npx checkout, added by the
user. Whether that also covers **PDM** is not yet established — the exclusion dialog shows a
"protection components" list, and if PDM is a separate component then the behavioural verdict may
still fire. **A fourth occurrence is therefore plausible, and should be expected rather than
surprising.** If it happens, the response is unchanged and cheap: recognise `1073741845`, check the
file against the published artifact, restore from quarantine if it is genuinely gone.

**The lesson worth keeping (this is the reason the entry exists):** my own first response to
occurrence 3 was to assume the file *should* be there and report it missing — **twice**, in the tool
output and in ERRATA E31 — when the correct first step is to establish the **expected** file set from
the manifest or the integrity-checked artifact, and only then ask what is absent. In this case the
answer was "nothing". E31 carries the correction.

**Still not done by me:** no antivirus setting, exclusion or configuration was changed, and the
missing-looking file was **not** restored by me — correctly, as it turned out, since there was nothing
to restore.

**What closes it:** only the user can confirm whether the PDM component is covered by the exclusion.
Until a full round passes without a removal, this stays an open environment risk.

## P3-9 — NEW (2026-10-05): the OCR recogniser is not run-to-run deterministic

**Evidence:** `benchmarks/ocr/results/OCR-03-bench/bench.json` versus
`benchmarks/ocr/results/OCR-05-validate/b5_validation.json`; `benchmarks/ocr/REPORT.md` §4 item 4.
**Measured:** the same weights, the same image (`real_nvidia_smi`) and the same settings produced
**`memory .free`** during the B3 benchmark and **`memory.free`** when the same image was served over
HTTP in B5. The B5 output matches the frozen truth exactly; the B3 output does not. Variation is at
whitespace level.
**Why it matters:** a single-image CER therefore carries roughly **±1 percentage point** of
run-to-run noise, which is large next to the **2 %** threshold that was pre-registered for clear
text. The two B3 threshold failures (`mobile`/`mixed` 3.17 %, `server`/`ui` 5.33 %) are of the same
order as that noise, so **neither should be treated as a stable property of the tier** on 4 images per
category.
**Not hidden:** the B3 measurement of record was **not** edited. Both numbers stand and the
discrepancy is disclosed in the report and here.
**What closes it:** repeat each fixture N times and report the spread, or fix the seeds, before any
category median from this set is treated as reproducible. Then re-measure on a **new** set — this one
is now exposed.

## P3-10 — NEW (2026-10-05): one OCR fixture embeds live machine state, so it is unrepeatable

**Evidence:** `benchmarks/ocr/fixtures/manifest.json`, entry `real_nvidia_smi`.
**Measured:** the fixture is a capture of `nvidia-smi --query-gpu=name,memory.used,memory.free`, so
the memory numbers inside the image (`6558 MiB`, `5387 MiB`) are a snapshot of that moment. Re-running
the command later gives different numbers, and the drawn image cannot be regenerated from the repo.
**Why it matters:** it is not reproducible as a *command capture*. It remains perfectly valid as a
*fixed image with a fixed truth string* — which is what the benchmark actually consumes — so no B3
number is invalidated. But a future round cannot rebuild the set byte-for-byte from source.
**What closes it:** for a repeatable set, capture command output that is stable (git log of a fixed
commit) or store the captured text as the source of truth and render from that. The captured text IS
already stored under `fixtures/capture/`, so the image can be re-rendered — only the live-state
property is lost.

## Resolved on 2026-10-05 (ERRATA E19-E28), each with an offline test

| defect | fix | test |
|---|---|---|
| citations `stale` in any fresh checkout (autocrlf) | `sha256-lf-v1` hashing; live manifest migrated with raw-hash re-verification | `cli/test_qa_path.py` |
| switch deleted the rollback first; failed when the index was in use | `safe_switch`, `--switch-only` | `benchmarks/phaseA/test_a_build_index.py` |
| contaminated rollback slot servable | QA path refuses an index without a passing leak check | both tests |
| a missing index loaded as an EMPTY one and refused everything silently | `open_index` checks manifest, files and counts before loading | `cli/test_qa_path.py` |
| pack cut every chunk at 2000 characters (a regression of a decision `retriever.py` had made) and citations claimed the whole chunk | whole chunks up to 8000 characters inside a token budget counted by local-fast; citations report the lines actually shown | `cli/test_qa_path.py` |
| prompt said a bare "SOURCE n" is not counted, while the parser counts it | prompt `qa-v2`, version recorded per answer | `cli/test_qa_path.py` |
| 600 s upstream timeouts held the pipeline lock | 30 / 15 / 90 s bounds; errors classified and mapped to 503/504/502 | `cli/test_qa_path.py` |
| non-object JSON got no HTTP response; `/health` named the default index | structured 400; `/health` reports the loaded index | `cli/test_qa_path.py` |
| frozen question files read with their byte-order mark | `utf-8-sig` in the CLI and the builder | `test_a_build_index.py` |
| exclusion gaps: `test-` scripts, `cli/ask_server.py`, a dead entry; leak markers exact-format only | `test-` prefix, `cli/` rule, dead entries stop the build, whitespace-tolerant markers | `test_a_build_index.py` |
| no artifact digest, so "reproducible" was untestable | `chunks_sha256`, `vectors_sha256`, `--compare-to` | `test_a_build_index.py` |
