# ARCHITECTURE_DECISIONS

Decisions that constrain how this system is built and measured. Each records what was
decided, why, and what evidence supports it. Decisions that were **reversed** are kept with
the reason, because the reversal is itself information.

---

## AD-1 — Two-stage retrieval: dense candidates, then a cross-encoder reranker

**Decided:** keep the Phase 5 two-stage shape (embedding → Top-N → reranker → Top-K → pack),
with the reranker **optional at runtime**.

**Why:** dense retrieval embeds query and chunk independently and can only compare them through
one vector; a cross-encoder reads them together. Phase 5 measured the gain on the original
query mix (held-out dense R@5 77.0 → 82.0), and Phase 5.5R confirmed the promotion of the exact
Phase 4 failure cases (rank 13 → rank 1).

**Limits measured:** the gain does NOT extend to the four new categories (causal, cross-document,
version-conflict, near-duplicate), where reranking is currently neutral-to-negative. On dev the
reranker was a slight net negative (dense R@5 63.6 → 59.1) while on held-out it was positive; both
figures are reported.

## AD-2 — The reranker owns no index and knows nothing about chunks

**Decided:** the reranker receives plain strings and returns `{index, score}` into the caller's
array. FAISS, metadata and context packing stay in the retrieval service.

**Why:** it makes the reranker a pure function of text, testable with three literal strings, and
measurably so — identical text receives an identical score in 5/5 duplicate cases.

**Consequence accepted:** the reranker therefore has **no notion of version or freshness** and
picks a stale passage in 4 of 5 version-conflict cases. That is a property of the boundary, not a
defect, and freshness belongs in pipeline metadata.

## AD-3 — Every number states its index; cross-index comparison must be controlled, not mixed

**Decided after** the Phase 4 / Phase 5 RAG discrepancy was traced to differing indexes
(1173 chunks vs 186). Both published figures were reproduced exactly in a controlled four-arm
test that varied only the index.

**Wording corrected in P55R-02.** This decision was previously titled "cross-index comparison is
forbidden", which overstated the rule and would forbid the very experiment that resolved the
discrepancy. The actual requirement:
* a **controlled** comparison across indexes IS allowed - one index changed at a time, all other
  factors held fixed
* metrics from different indexes must **not** be mixed into one table as if comparable
* a cross-index comparison must **state the corpus difference** rather than conceal it

**Residual claim removed in P55R-02.** An earlier version of this file implied the historical
Phase 4 / Phase 5 difference was fully explained by the index. It is not: the **context-pack path
also differed** (Phase 4 used the HTTP pack path, Phase 5 packed locally). The four-arm result is
a valid controlled comparison, not a full causal account of the historical difference.

## AD-4 — Retrieval relevance and factual recency are separate labels

**Decided:** a query carries `sources` (files that answer it) and, where relevant, `superseded`
(files stating a value that no longer holds). A retriever is never credited for surfacing a
stale answer merely because the stale file is topically relevant.

## AD-5 — Superseded material is excluded; historical material is demoted, not dropped

**Decided:** explicit `superseded` markers are excluded from current-state QA. Files marked
`historical` are demoted for current-state questions and left fully available when the query is
historical (detected by a light, inspectable intent rule).

**Hard constraint:** file mtime is **never** used as a validity signal. A newer file must not
override a still-valid technical document. Asserted by code inspection.

## AD-6 — Candidate pool 10, not 20

**Decided:** fixed pool 10, Top-K 5.

**Why:** pool recall rises with N while the final ranking degrades; on dev, version-conflict R@5
falls from 80.0 to 60.0 when the pool widens under reranking. Two independent measurements
(retrieval R@5 and end-to-end RAG accuracy) agreed.

**Corrected claim:** an earlier statement that "no metric improved" with a larger pool was
**wrong**. Recomputed: candidate recall rises 68.2 % → 72.7 % at pool 50 while final Top-5 does
not. The accurate statement is "a larger pool improves candidate recall but not final ranking".

## AD-7 — No adaptive pool, because confidence is uncalibrated

**Decided:** not implemented. A confidence-triggered pool would rest on uncalibrated reranker
scores and be unfalsifiable. A margin-based arm in the cost evaluation matched the full reranked
arm with fewer tokens, but the margin was too small to justify the complexity. Fixed pool 10
retained.

## AD-8 — GPU admission is atomic, reserved, and FIFO-fair

**Decided, after four defects were reproduced deterministically:**

1. **Atomic check-and-reserve.** Planning and reservation happen under ONE lock. The previous
   version planned, released the lock, then registered — a TOCTOU window that let `serial` admit
   two jobs, let `exclusive` co-admit, and let one memory budget be spent three times.
2. **Reservation ledger.** Accepted jobs record what they committed, keyed by a unique id that
   `acquire()` returns and `release()` consumes. `usable = measured_free − reserved − margin`,
   with resident models NOT added again because live sampling already includes them.
3. **FIFO admission tickets.** Priority rules alone starved an interactive request: a background
   worker that releases and immediately re-requests re-acquires the lock before a sleeping
   interactive waiter wakes. Measured: 35 background units completed while an interactive waited
   its full 10 s timeout and was refused. With FIFO the same case completes in 0.207 s.

## AD-9 — Conservative, unified concurrency policy

**Decided:** exclusive is mutually exclusive with everything; interactive runs alone and waits
for running background units; **no new background unit is admitted while an interactive waits**;
background may overlap background within budget, capped by `max_background_units`.

**Explicitly NOT promised:** "interactive never waits for background". Without preemption that is
false. The wait is bounded by `max_background_units × unit duration` and was **measured** at
0.207 s against a 0.600 s bound.

**Reason for the change:** the previous documentation contradicted itself, claiming interactive
never shares with background while permitting background to join interactive.

## AD-10 — Jobs must declare the services they require

**Decided:** every job class carries a `needs` list; `background` declares an explicit empty one
rather than omitting the key. Unload refuses any service required by an ACTIVE job.

**Why:** an omitted `needs` previously made every model an implicit unload candidate, including
ones the job was about to use.

## AD-11 — The scheduler never terminates a running task

**Decided:** unloading happens only between jobs, through each service's own stop script, and
never to make room for a running task. The stop scripts were **inspected** rather than trusted:
all three target a specific PID, wait for the port to release, and never kill by process name.

**Stated plainly:** those scripts DO use `Stop-Process -Force` on the service's own PID. That is
the service's targeted self-termination, not the scheduler killing work.

## AD-12 — Safe refusal is necessary but is NOT a pass

**Decided:** a live validation in which all normal work is refused is a **FAIL**, not a
demonstration of safe refusal. B4 recorded exactly that: 10/10 requests refused, 0 completed,
verdict FAIL.

**Consequence:** because three resident models plus the 2048 MiB margin leave ~152 MiB usable, a
minimal-residency option was submitted for a decision rather than adopted — it is architectural.
Measured to work: `local-fast` + `local-embedding` resident leaves 3431 MiB usable and admits
background work. Reload costs 1.0–1.5 s per service.

## AD-13 — The grader's error direction is labelled, not corrected retroactively

**Decided:** the historical grader is preserved unchanged; a corrected grader is reported
alongside it.

**Error direction corrected in P55R-02.** This decision previously said the original grader
"can only OVER-count" hallucinations and correctness, and that published figures are therefore
"upper bounds". That was itself an over-claim. What the counter-examples actually demonstrate is
that **false positives AND misses both exist**, in **opposite** directions:
* FALSE POSITIVE: a denied wrong value is flagged as a hallucination ("it is not 4096")
* MISS: a fabricated extra alongside all required facts scores fully correct
* MISS: a correct refusal and an incorrect refusal are scored identically

Because errors run in both directions, **the net bias is UNKNOWN**. Published hallucination and
accuracy figures are not upper bounds; they are figures with demonstrated error in both
directions and no measured net sign. That is a weaker but accurate statement.

**Related:** a correct refusal and an incorrect refusal are indistinguishable under both graders,
which is why a falling refusal rate must never be read as improving accuracy.

## AD-14 — "Recall@K" in the Phase 4/5 harnesses is actually Hit@K

**Decided:** rename it in all new reporting. The harness scores 1 if the FIRST valid source
appears within K. Measured difference on held-out data: Hit@5 82.0 % versus mean per-source
recall@5 57.4 %. Multi-source completeness is invisible under the old name (Hit@5 84.3 % versus
all-sources-present 33.3 %).

## AD-15 — A source file's identity ignores its line endings

**Decided (2026-10-05):** index manifests hash each source with CRLF folded to LF
(`hash_scheme: sha256-lf-v1`, `services/retrieval/provenance.py`), and record the scheme.

**Why:** this repository is checked out with `core.autocrlf=true` and no `.gitattributes`, so the same
commit is LF in one working copy and CRLF in another. Raw-byte hashes matched 48/48 files where the
index was built and 0/48 in a fresh worktree of the same commit, turning every citation into a false
`stale`. A line ending changes no line's text. Manifests without the field are read as raw-byte
manifests, and a file that differs from an LF original only in line endings still verifies.

## AD-16 — An index is served only if its manifest licenses it

**Decided (2026-10-05):** the QA path (`cli/ask.py open_index`) refuses an index unless its manifest
records a passing validation INCLUDING the answer-key leak check, its files exist, and the store's
vector and chunk counts equal the manifest's `n_chunks`. It checks before touching the store.

**Why:** `IndexStore.load()` creates a missing index as an empty one, which then refused every
question as "below the evidence floor" - indistinguishable from an answer-absent question - while
`/health` reported the manifest's version. And the rollback slot held a build with the tuning answer
key in it; an index that cannot show a passing leak check must not be one rename away from serving.

## AD-17 — `index_version` names the inputs; digests decide whether two builds are the same index

**Decided (2026-10-05):** manifests record `chunks_sha256` and `vectors_sha256`; `index_version`
(scheme 2) covers commit, corpus content, configuration and code; `--compare-to` compares two
builds on chunks exactly and vectors numerically.

**Why:** "identical index_version" was offered as proof of a reproducible rebuild, but a hash of the
inputs is identical for any two builds of the same files. Vectors are not even bit-stable across
request sizes (OPEN_ISSUES P2-9), so reproduction has to be measured on what was built.

## AD-18 — The embedding service gives its allocator cache back; bulk embedding is batched small and preflighted

**Decided (2026-10-05):** `Embedder._release_cache()` empties the CUDA cache after an encode once more
than 256 MiB is cached above what is allocated. The index builder sends 4 texts per request, refuses
to embed when free VRAM minus the measured transient would breach the 2048 MiB margin, and records
the service's retention after the build.

**Why, measured** (`benchmarks/p1_11/REPORT.md`): on the Phase A corpus at 16 per request the old
service kept 5053.5 MiB above its weights after the build (OPEN_ISSUES P1-11); with the release,
15.5 MiB, at the same speed and with bit-identical vectors. The transient itself was ~3.75 GB at 16
per request and ~1 GB at 4, and 4 was faster.

**Not decided:** the reranker keeps its cache the same way (P3-6); its budget was measured with that
behaviour, so changing it needs its own measurement.

## AD-19 — The QA pack shows whole chunks inside a token budget, and a citation names what was shown

**Decided (2026-10-05):** `cli/ask.py` includes ranked chunks whole (up to 8000 characters, the value
`services/retrieval/retriever.py` already used), counts the prompt with local-fast's own `/tokenize`
against its single 8192-token slot, drops the lowest-ranked sources first - recorded, never silent -
and cuts a lone over-budget source only at a line boundary. A citation carries the lines the model
was shown, with the chunk's own range beside it.

**Why:** the P56 pack cut every chunk at 2000 characters, re-introducing the cap `retriever.py` had
removed after it cut `context: 8192` out of `config/local-fast.yaml`; on the Phase A index 48 of 520
chunks were cut, and their citations still claimed the whole chunk. Character counts cannot bound
tokens on this corpus (0.83 tokens per character at its densest), so the budget is counted by the
model's tokenizer, with a conservative estimate only when the endpoint is missing.

## AD-20 — A correction goes where the retriever will find it

**Decided (2026-10-05):** when a conclusion is retracted in ERRATA, the `docs/` page that states it gets
a dated note in the same section, in the same change.

**Why:** ERRATA and these decision records are not in the QA corpus, while the pages they correct
are; four such pages still stated superseded conclusions (ERRATA E27). The chunker splits pages by
section, so a note elsewhere on the page would not travel with the claim.

## AD-21 — The default resident set is `local-fast` + `local-embedding`; everything else is on demand

**Decided (2026-10-05):** the minimal-residency option measured under AD-12 is **ADOPTED**.
* **Resident by default:** `local-fast` (8111) + `local-embedding` (8113).
* **Not resident, loaded on demand:** `local-reranker` (8114), `local-smart` (8112).
* **A new service must satisfy at least one of three conditions:**
  1. it still holds the **2048 MiB margin** on top of the default resident pair, or
  2. it is **CPU-only** (so it consumes no VRAM at all), or
  3. it is **loaded by the scheduler on demand** rather than left resident.

**Why:** the three-model default could not do normal work at all — with all three resident the
scheduler had `152 MiB usable < 600 MiB required` and refused **10 of 10** requests (P1-6, B4). AD-12
settled that refusing everything is a FAIL, not a safe pass, and left the residency choice open. The
alternative was measured to work: `local-fast` + `local-embedding` resident leaves **3431 MiB usable**
and admits background work, with reload costing 1.0-1.5 s per service. Re-measured 2026-10-05 on the
deployed pair: **4903-4913 MiB free** with 8111 and 8113 both up.

**Consequence for the 2048 MiB margin:** the margin is **not** lowered and remains 2048 MiB. This
decision changes *what is resident*, not the threshold.

**Consequence for Phase 6:** `local-ocr` (8116) is built **CPU-only** under condition 2, so it needs
no VRAM at all and does not participate in the residency budget. That is the intended use of the
condition — a LIGHT-class service should not compete for the GPU.

**Evidence:** `OPEN_ISSUES.md` P1-6 (resolved), `.../P55R-01-b4/b4_live.json`, AD-12,
`benchmarks/phaseAB/REPORT.md`, `benchmarks/ocr/REPORT.md`.

## AD-22 — Answer limit 512 tokens, and a length truncation is always stated

**Decided (2026-10-05):**
1. `MAX_ANSWER_TOKENS = 512` (was 256), so `PROMPT_TOKEN_BUDGET = 8192 - 512 - 64 = 7616`.
2. **A length truncation must be explicit.** `choices[0].finish_reason` is read and recorded on
   **every** record as `finish_reason`, with a derived boolean `answer_truncated`
   (`finish_reason == "length"`). This applies to **all** exits, including a record downgraded to a
   refusal by the citation gate.
3. A truncated answer is **returned, not suppressed, and not retried automatically** — it is flagged.
   The CLI prints an explicit line; `cli/ask_server.py` always returns both fields, so answers and
   refusals keep the same shape.
4. **Scoring:** a truncated answer counts **at most as partial**, unless every expected fact is
   present and cited — in which case it may count as complete, with the truncation still recorded.

**Why:** the QA path sent `max_tokens: 256` and never read `finish_reason`, so a cut-off answer was
indistinguishable from a finished one. Measured in the A/B acceptance run: **h05, p05 and p07 each
consumed exactly 256 completion tokens and ended mid-sentence** while their records showed nothing
(Phase A's acceptance was granted on condition that this be fixed). A silently truncating cap is
worse than a small cap, because it makes an incomplete answer look complete to every downstream
reader, including a reviewer.

**Why 512 and not larger:** the limit is an answer-length bound, not a budget to spend. 512 doubles
the headroom past the observed truncation point while leaving the prompt budget (7616) above every
pack this corpus has produced. Raising it further would need its own measurement; it was **not**
tuned against the exposed sets.

**Evidence:** `cli/ask.py`, `cli/ask_server.py`, `cli/test_qa_path.py`,
`benchmarks/qa_hardening/results/QH-01/`.
