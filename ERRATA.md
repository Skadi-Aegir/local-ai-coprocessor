# ERRATA — conclusions from earlier reports that no longer hold

Historical reports are **kept unchanged**. This file is the correction layer: it names each
superseded conclusion, the evidence that supersedes it, and what replaces it. Read a historical
report together with this file, not on its own.

Items are marked with the phase that found them: **[P55R-01]**, **[P55R-02]**, **[P55R-03]**,
**[P55R-04]**, **[P55R-05]**, and **[R-1005]** for the review of 2026-10-05 (E19 onwards).

---

## E1 [P55R-01] "61 held-out questions" — SUPERSEDED

**Was:** 61 independent held-out questions.
**Is:** 39 historical questions (published since Phase 4, therefore exposed) + 22 new ones. Pool
size and the causal instruction were both selected with the new set in view, so it is **exposed**
too. **There is currently no unexposed held-out set**, and no independent-generalisation claim is
supported.
**Evidence:** `benchmarks/phase55r/results/P55R-01-a3/a3_metric_audit.json`

## E2 [P55R-01] "Recall@K" — MISNAMED, it is Hit@K

**Was:** Recall@K, reported as e.g. 82.0 %.
**Is:** Hit@K / success@K — 1 if the FIRST valid source appears within K. The same data gives
**mean per-source recall@5 = 57.4 %**. Multi-source completeness is invisible under the old name:
Hit@5 84.3 % versus **all-sources-present 33.3 %**.
**Evidence:** same file.

## E3 [P55R-01] "Widening the pool has no benefit" — REFUTED

**Was:** no metric improved with a larger pool.
**Is:** candidate recall **rises 68.2 % to 72.7 %** at pool 50 while final Top-5 does not improve.
The accurate statement is "a larger pool improves candidate recall but not final ranking".
**Evidence:** same file.

## E4 [P55R-01] "Only the index differed" — TOO STRONG

**Was:** the Phase 4 / Phase 5 difference was attributable to the index alone.
**Is:** the **context-pack path also differed** (Phase 4 packed over HTTP, Phase 5 locally). The
four-arm test is a valid **controlled comparison**, not a full causal account.
**Evidence:** `.../P55R-01-a1/a1_baseline_audit.json`, `a1_confounds.json`

## E5 [P55R-01] "The grader can only over-count; figures are upper bounds" — WITHDRAWN

**Was:** the grader only over-counts, so published hallucination counts and accuracy are upper
bounds.
**Is:** false positives AND misses are both demonstrated, in opposite directions, so the **net bias
is UNKNOWN**. They are not upper bounds.
**Evidence:** `.../P55R-01-a2g/a2_grader.json`

## E6 [P55R-01] "Hallucinations were invented information" — RETRACTED

**Was:** the two persistent hallucinations were classified as invented information, primary cause.
**Is:** on the saved raw answers there is **no remaining case of a plausible number invented
despite adequate evidence**:
* the retrieval-arm case is a **retrieval truncation** — the answer itself reports that SOURCE 1
  "truncates the `runtime` section before showing the `context` value"
* the no-RAG pooling case is a **grader false positive** — the answer opens "I do not know." and
  lists common pooling methods from other architectures as background
* the two no-RAG embedding questions are **out-of-domain guessing** in an arm with no context
**Evidence:** `benchmarks/phase55r/results/P55R-01-c/flagged_questions.json`
**Strength of evidence:** the truncation case rests on the model's own answer text, a **clue**, not
a packing record. The historical pack was not persisted, so it is a **PROBABLE cause**.
**The two false positives:** *pooling* = mention-as-assertion; *dimension* = missing-evidence scored
as hallucination.

## E7 [P55R-01] "Only one working copy on this machine" — WRONG

**Was:** no other working copies exist.
**Is:** `git worktree list` shows a **second worktree** at
`C:/Users/<user>/.codex/worktrees/94a6/MultiAgent`, detached at `bac96ab`. Ownership of the running
services is unaffected (ports 8111/8113/8114 are held by processes whose command lines point into
`D:\MultiAgent`).

## E8 [P55R-01] Single "9 candidate misses vs 2 rank misses" — SPLIT

**Is:** four distinct causes, only two of which are retrieval-quality signals:
answer-absent-from-corpus (10 of the 44 new queries, **not** retrieval failures), chunking loss
(**unknown**), candidate miss (the remainder), ranking downgrade (2). See OPEN_ISSUES P2-4.

## E9 [P55R-01] "Safe refusal works" as a pass for B4 — UPHELD AS A FAIL

B4 recorded 0 of 10 requests completed and was graded **FAIL**. Superseded in turn by E17.

## E10 [P55R-02] Reranker transient: the 1800 MiB declared value vs the measured peak

**Was (implicit):** refusals reported as "insufficient margin" read as a hardware limitation.
**Is:** the declared transient is **1800 MiB**; measured peak **881 MiB** (P55R-03/04), so the
declared figure is conservative by roughly 900 MiB. **This measurement stands. Its conclusion was
narrowed by E12.**
**Evidence:** `benchmarks/phase55r/results/P55R-03-peak2/reranker_peak.json`

## E11 [P55R-02] "The three-service baseline is restored" — NOT ACHIEVED (at the time)

**Was:** every earlier phase ended with all three services healthy.
**Is:** at one point in P55R-02 the state was 8111 ok, 8113 DOWN, 8114 ok, with the embedding
service failing on `OSError: pagefile too small (os error 1455)`. **Subsequently resolved** after
the operator closed externally-running high-VRAM programs. **For the current service state read
PROJECT_STATE.md, not this entry.**

## E12 [P55R-03] "Refusals are caused purely by the budget figure, not by insufficient hardware" — RETRACTED

**Was:** because the declared transient exceeded the measured peak, the margin refusals were
attributed to the budget figure rather than to the GPU.
**Is:** **both** components contribute, and their relative weight is not separable from what was
measured. A candidate budget of 1361 MiB was tested in an isolated registry and full RAG was
**still refused**: `2933 MiB usable < 3121 MiB required`, short by **188 MiB**. With the reranker
stopped only about 4826 MiB was free on a 12227 MiB card.
**Evidence:** `benchmarks/phase55r/results/P55R-03-cand/candidate_budget.json`

## E13 [P55R-03] "All three strategies are short by a hair" — RETRACTED AS A FRAMING

**Is:** the numbers (79 / 142 / 769 MiB) were real; the characterisation implied the gap was
trivially closable and it is not. Deficits are reported as numbers.

## E14 [P55R-03] The 262 MiB hot-sample figure — REJECTED AS A BASIS

**Was:** the reranker's transient peak was reported as 262 MiB.
**Is:** that figure came from repeating **identical documents**, and `use_cache: false` is **NOT
honoured** by the reranker service — identical documents repeat at ~0.025 s whether or not the query
changes, while new content costs ~0.43 s. Those repeats measured **cache hits**. The figure is used
nowhere and must not be quoted as a ceiling.
**Replacing it:** the proper measurement gives an observed maximum of **881 MiB**, with the margin
never breached. Note the shape — the FIRST call of each case spikes and repeats are ~0, because
PyTorch's caching allocator retains the block, so repeating a case does not re-exercise the peak.
**Still uncertain:** the 0.02 s sampler leaves a blind window, so 881 MiB is a **lower bound**.
**Evidence:** `benchmarks/phase55r/results/P55R-03-peak2/reranker_peak.json`

## E15 [P55R-03] "Contamination accounts for one fact, and the differing question is context size" — CORRECTED

**Is:** the differing question is **"How long does it take to switch between the two local models?"**
at 2 → 1. The context-size question is **0/1 in BOTH arms**: its expected fact is in the full arm's
pack but the model did not state it, and it is absent from the decon arm's pack.
**Also:** the exclusion removed three files that are **not** answer keys
(`local-fast-results.md`, `local-smart-results.md`, `retrieval-results.md` are legitimate project
result documents), so the run **cannot** separate answer-key removal from removal of legitimate
material, and **cannot** be used to restore confidence in the Phase 4 91.7 % figure.
**Evidence:** `benchmarks/phase55r/results/P55R-02-decon/pair.json`

## E16 [P55R-04] "The 1361 MiB candidate does not increase risk" — RETRACTED

**Was:** the candidate was described as still leaving 439 MiB of headroom under the declared
1800 MiB, hence "not an increase in risk".
**Is:** that reasoning is invalid. The candidate **loosens admission** — that is its purpose — so it
increases risk by construction. The 439 MiB is not headroom against a proven safe limit; it is
headroom against a *different unproven number*. Correct statement:
* **881 MiB is a sampling LOWER BOUND**, not a peak: the 0.02 s sampler has a blind window, and the
  spike shapes (284/384/374, 588/1/0, 881/8/3) show the allocator retains a high-water mark that a
  later sample can miss entirely.
* the **1.4x sampling allowance and the extra 128 MiB are unvalidated estimates**, not measurements.
  Together they are **not sufficient evidence of a safe upper bound**.
* adopting 1361 MiB would trade a measured-unsafe margin for an unmeasured one, the opposite of what
  the margin exists for.
**Consequence:** the candidate is **not** proposed for adoption and nothing from it is written into
any default budget. The declared 1800 MiB stands.
**Evidence:** `.../P55R-03-peak2/reranker_peak.json`, `.../P55R-03-cand/candidate_budget.json`

## E17 [P55R-04] The staged lifecycle — first attempt FAILED the gate, and why

**Was:** the P55R-03 design presented the staged lifecycle as the way to fit full RAG by avoiding
co-residency, with a `max(stage requirement)` feasibility table, and with two contradictory
statements about when the embedding reservation is released relative to the unload.
**Is:** the contradiction is **resolved** — protection ends at **RELEASE**, as a *precondition* of
the swap, not as a consequence of the unload — and a staged path was built and run in isolation.
That attempt **FAILED the margin gate**: 2/5 completed, `free_min` **1863 MiB** against a 2048 MiB
margin, 7 sub-margin samples, no OOM.
The finding that stands:
* `allow_unload` makes an unload **permitted, not required**. At swap time the budget was already
  sufficient (`1343 MiB usable >= 600 MiB required`), so `unloaded=[]` and the embedding model
  **stayed resident** — the swap delivered no memory relief. A `max()` estimate would have concealed
  this. **P55R-05 addresses exactly this by requiring both unloads to actually happen.**
**The second finding of this entry was retracted as an estimate — see E18.**
**Evidence:** `benchmarks/phase55r/results/P55R-04-staged/staged_rounds.json`,
`docs/staged-lifecycle-design.md`

**The original P55R-04 wording of this entry**, which an edit left in place as a duplicate below the corrected text. Kept for traceability; its second bullet and its last sentence ("Full RAG remains FAIL ...") are retracted by E18 and must not be quoted as current [R-1005]:

> **Was:** the P55R-03 design presented the staged lifecycle as the way to fit full RAG by avoiding
> co-residency, with a `max(stage requirement)` feasibility table, and with two contradictory
> statements about when the embedding reservation is released relative to the unload.
> **Is:** the contradiction is **resolved** — protection ends at **RELEASE**, as a *precondition* of
> the swap, not as a consequence of the unload — and the staged path was built and run in isolation.
> It **FAILED the margin gate**: 2/5 completed, `free_min` **1863 MiB** against a 2048 MiB margin,
> 7 sub-margin samples, no OOM.
> Two findings, both recorded rather than smoothed over:
> * `allow_unload` makes an unload **permitted, not required**. At swap time the budget was already
>   sufficient (`1343 MiB usable >= 600 MiB required`), so `unloaded=[]` and the embedding model
>   **stayed resident** — the swap delivered no memory relief. A `max()` estimate would have concealed
>   this.
> * even **requiring** the unload would not have sufficed: the reranker's load+transient plus the
>   generative model's requirement exceeds what is free after the swap. The staged design addresses
>   *stage-level* co-residency but **not** the fact that the reranker and the generative model must
>   both be live for pack-and-generate.
> **Consequence:** the staged path does **not** close the gap on this card; recorded as
> proposed-not-adopted. **Full RAG remains FAIL on every configuration tried across P55R-02/03/04.**
> **Evidence:** `benchmarks/phase55r/results/P55R-04-staged/staged_rounds.json`,
> `docs/staged-lifecycle-design.md`

## E18 [P55R-05] "Forced unload would not suffice / the hardware is proven unable" — RETRACTED, NOT MEASURED

**Was (E17, written in P55R-04):** "even **requiring** the unload would not have sufficed: the
reranker's load+transient plus the generative model's requirement exceeds what is free after the
swap." That was stated as a finding.
**Is:** it was an **estimate, not a measurement**, and it is now **refuted by measurement**:
* that round recorded **`unloaded=[]`** — the embedding model was never unloaded, so the
  post-forced-unload state was **never observed**
* the estimate counted a **resident** service's load budget against the **same instant's** live free
  reading. A resident model is already inside the live reading, so charging its `vram_mb +
  peak_extra_mb` again double-counts it. This is the same double-counting class that AD-8's
  watermark fix addressed for reservations.
**P55R-05 tested it and the forced-unload path PASSES.** Both unloads required and verified (port
free AND free VRAM rose): deltas **+1368 to +1392 MiB** (embedding) and **+2591 to +2595 MiB**
(reranker). Full RAG completes with the 2048 MiB margin held; worst observed `free_min` across all
passing rounds and diagnostics is **3000 MiB**.
**Status: RESOLVED AS REFUTED, not pending.** Whether full RAG fits is established on the
**isolated staged path**; it does **not** follow that the co-resident default can do it, and the
production default is unchanged.
**A related wording correction:** the earlier note here attributed the as-found 1792 MiB free VRAM
to **"stale-process inflation"**. The **observation** is solid — after using the existing safe
stop/start paths on the DSH-owned services, free VRAM was **3101 MiB** with the same three services
resident (`results/P55R-05-baseline/baseline.json`). The **cause is not established**: no
process-level evidence was captured identifying what held the earlier ~1.3 GiB, so it is recorded as
an observation of recovery without asserting the mechanism. It must not be quoted as proof that a
particular process was retaining memory.
**Evidence:** `benchmarks/phase55r/results/P55R-04-staged/staged_rounds.json` (`unloaded=[]`),
`results/P55R-05-rag/rag_diagnostic.json`, `results/P55R-05b-rag/rag_diagnostic.json`,
`results/P55R-05-baseline/baseline.json`,
`docs/phase55r-evidence-consolidation.md`, `docs/production-integration-design.md`

## E19 [R-1005] Phase A "every citation verified" — TRUE ONLY IN THE WORKING COPY THAT BUILT THE INDEX

**Was:** citations are verified against the manifest's per-file SHA256, and every citation in the
accepted run was `verified`.
**Is:** the manifest hashed raw bytes. The repository is checked out with `core.autocrlf=true` and has
no `.gitattributes`, so the same commit is LF in one copy and CRLF in another: the manifest matched
**48/48** files in `D:\MultiAgent` and **0/48** in a fresh worktree of the same commit, where every
citation - including Phase B's own `docs/models/local-fast.md:132-148` - read as `stale`, and the QA
path would have refused every answer.
**Fixed:** content is hashed with CRLF folded to LF (`sha256-lf-v1`, `services/retrieval/provenance.py`).
The live manifest was migrated without a rebuild (`benchmarks/phaseA/a_migrate_manifest.py`): each
file's raw hash was first re-verified in the copy that built it. Verified afterwards: 48/48 in the
fresh worktree. **Also established by the migration:** all 48 corpus files equal their blobs at
`854fabc`; the "dirty" build was dirty only outside the corpus.

## E20 [R-1005] Phase A "rebuild is reproducible: identical index_version" — TAUTOLOGICAL

**Was:** a second independent build produced the identical `index_version`, so the rebuild is
reproducible.
**Is:** `index_version` was a hash of the commit, the dirty flag and the source-file hashes - the
INPUTS. Any two builds of the same files share it whatever vectors they produce. The two build
reports differ only in timing and output directory, no digest of the built index was recorded, and
the second build's directory was deleted, so reproducibility cannot be checked after the fact.
Measured since: the embedding request size alone changes the vectors (mean cosine 0.99984-0.99985
against unpadded vectors, nearest-neighbour agreement 90-94 %), so bit-reproduction requires
identical batching (OPEN_ISSUES P2-9).
**Fixed:** manifests now record `chunks_sha256` and `vectors_sha256`, `index_version` (scheme 2) also
covers configuration and code, and `--compare-to` compares two builds on what they contain.

## E21 [R-1005] Phase A "atomic switch with rollback" — NOT ATOMIC, AND THE ROLLBACK SLOT IS CONTAMINATED

**Was:** the build is validated in staging and swapped in atomically, the previous index kept as
`current.prev` for rollback.
**Is:** (1) the switch deleted `current.prev` FIRST and then renamed twice. Reproduced on Windows:
while a reader holds `current/metadata.sqlite` (as `ask_server` does), the first rename fails with
WinError 5 after the rollback copy is already gone. (2) `current.prev` (`a3e3c050636f838e`, 52
files / 615 chunks) is the first Phase A build, made before `rag_compare_rerank.py` and three
evaluation harnesses were excluded: it holds the 12 tuning questions with their `facts=[` answer
lists. PROJECT_STATE named it as the rollback target.
**Fixed:** `safe_switch` deletes nothing until the new index is in place and fails cleanly when the
index is in use; `--switch-only` retries without rebuilding. The QA path refuses any index whose
manifest lacks a passing answer-key leak check, so `current.prev` cannot be served; the next
successful switch retires it. Until then there is **no clean rollback** (OPEN_ISSUES P1-13).

## E22 [R-1005] Phase A per-question table — q05 is NOT a correct refusal; q04 counted twice

**Was:** 6 complete answers (q01 q02 q03 q04 q06 q07), 1 partial (q04), 4 correct refusals
(q05 "evidence not gathered + corpus lacks 8114", q08 q09 q10), 0 incorrect refusals.
**Is:** port 8114 appears on 20 lines in **11 of the 48** Phase A corpus files; "corpus lacks 8114"
was carried over from P56's 30-file corpus without re-checking. q05's top-5 held no embedding or
retrieval document. Under the grading rule frozen before P56 ran, only q08-q10 refusals are correct,
and P56 itself said q05 must not count as an answer-absent success. Correct tally: **5 complete,
1 partial (q04), 3 correct refusals, 1 incorrect refusal (q05)**. The statements "0 unsupported
statements" and "0 invalid citations" are unaffected.

## E23 [R-1005] "The CLI refuses when retrieval finds nothing above the evidence floor" — THE FLOOR NEVER FIRED

**Was (implied by every description of the QA path):** the evidence floor is how answer-absent
questions get refused.
**Is:** all 150 candidates in the three P56 / Phase A runs scored **0.486 or higher** against a 0.30
floor, every Phase B request had 5 of 5 candidates above it, and **no record was refused by it**.
The answer-absent questions scored as high as answerable ones (Phase A top score 0.518-0.623 against
0.560-0.814). Every recorded answer-absent refusal came from the model following the prompt. No
threshold on the top score separates the two groups on this data (OPEN_ISSUES P2-7).

## E24 [R-1005] Phase B claims about the interface — FIVE CORRECTIONS

* **"All four limits VERIFIED in the run" — no.** `MAX_BODY_BYTES` was never exercised:
  `body_too_large` appears 0 times in `b_e2e.json`, and the "oversize body" step named in
  `b_client.py` did not exist. Now covered by `cli/test_qa_path.py`.
* **"Concurrency 2" — one at a time.** The lock wrapped the whole pipeline, so the two admitted
  requests ran serially (1.83 s and 3.54 s in the recorded run). `observed_max_simultaneous_200` in
  `b_e2e.json` counts admitted requests, not simultaneous ones. Now stated as admitted 2 / executing 1.
* **L4 "a hung upstream holds a slot until the client's timeout fires" — wrong.** A client that gives
  up does not free its slot; the pipeline runs on until it returns, and with the 600 s upstream
  timeouts one hung call blocked every request for up to ten minutes. Upstream calls are now bounded
  (embedding 30 s, tokenize 15 s, generation 90 s) and map to 503/504/502.
* **"Errors are structured with a real HTTP status" — not for every body.** Valid JSON that is not
  an object (`[1,2]`, `123`, `null`) closed the connection with no response at all. Now 400.
* **`/health` reported the default index whatever `--index` was given.** It now reports the index
  the process loaded, with its chunk counts.

## E25 [R-1005] "/stats vram_mb is the embedding service's footprint; the startup line is the device total" — NEITHER

**Was (PROJECT_STATE §0):** the startup line prints the device total at load time; `/stats` reports
the footprint, so use `/stats`.
**Is:** both print the same function, cudaMemGetInfo `total - free`. Measured on this WDDM machine
with only a CUDA context in the process: **1232.6 MiB, while nvidia-smi reported 3438 MiB used**; it
then rose one-for-one with the process's own allocations. It is neither the device-wide figure nor
the footprint, though its changes do track the process - which is why P1-11's 2360.6 -> 7750.6 MiB
was real retention (5053.5 MiB measured in isolation). The footprint is now `/stats
memory.torch_reserved_mb`.
**Evidence:** `benchmarks/p1_11/REPORT.md`

## E26 [R-1005] P56 and Phase A asked q01 with a byte-order mark in front

**Is:** `benchmarks/p56/frozen_questions.txt` begins with a UTF-8 byte-order mark (written by
PowerShell), and both `cli/ask.py --question-file` and the index builder read it as plain UTF-8. q01
was therefore sent to the embedder and the model as `"\ufeffWhat port ..."` in both runs (visible in
`benchmarks/phaseA/results/a_answers.json`), and the builder's check that no frozen question appears
in the corpus could never match q01. q01 was answered correctly both times; the effect is otherwise
unmeasured. Both readers now use `utf-8-sig`. The frozen file is unchanged, and its recorded hash
`4138ABDA...` is of its CRLF bytes, BOM included.

## E27 [R-1005] Corpus documents stated superseded conclusions, and the correction layer is outside the corpus

**Is:** ERRATA, OPEN_ISSUES, ARCHITECTURE_DECISIONS and PROJECT_STATE sit at the repository root and
are not indexed, while documents under `docs/` that they correct are. Found and fixed on 2026-10-05:
* `docs/gpu-scheduling.md` §2, §3 and §10 still gave the pre-P55R-01 job-class policy (interactive
  "never waits behind background work", background "may overlap an interactive job") - corrected in
  place against AD-9 and the code; §6 and §7 annotated
* `docs/staged-lifecycle-design.md` kept "2.54 s wall" for a round and "the isolated run shows it
  does not [meet the margin]" under a header saying the opposite - corrected
* `docs/retrieval-hardening.md` stated E1, E3, E4, E6 and E8 conclusions as findings - each affected
  section annotated, since the chunker splits by section
* `docs/models/local-embedding.md` labelled `vram_mb` "torch attribution" - annotated (E25)
A citation of any of these files from the index built before 2026-10-05 now reads as `stale`, which
is correct; the next build indexes the corrected text.

## E28 [R-1005] P1-11 "restart 8113 after bulk indexing" — the cause was retention, and the transient is a second risk

**Is:** the growth was PyTorch's allocator keeping its cache (5053.5 MiB above the weights after the
Phase A corpus at 16 texts per request; 15.5 MiB with the release now in the service). Separately,
the build's own transient at 16 per request is ~3.75 GB inside the service - with local-fast resident
that breaches the margin during the build whatever happens afterwards. The builder now sends 4 per
request (~1 GB, and faster) and refuses to start embedding when the margin cannot hold.
**Evidence:** `benchmarks/p1_11/REPORT.md`

## E29 [AB acceptance round] Antivirus removed a DSH runtime file mid-call, and one tool call did not run

**Is:** during the Phase A/B acceptance round one `pwsh` tool call failed with
`subprocess-local: Windows Job runner exited with exit code 1073741845 before proving its managed
range empty`. The code is `0x40000015` = **`STATUS_FATAL_APP_EXIT`** — the runner process died
abnormally; it did not return an error. The cause was external: **Kaspersky deleted
`node_modules/@deepseek-ai/dsh-subprocess-local/lib/runner.js`** while it was in use.
The user restored the file out of band. Verified afterwards: `node --check` exits 0, the module
imports and exports `runSelectedSubprocessRunner`, and a live stdout capture plus a background-job
spawn both succeed. Restored file SHA256
`4E21528DFDB3E92BDCE2340945DC4DCB202E4A4AF0BBEB8B97585F67735D30C1`.
All three `runner.js` copies are package-original and syntactically valid:
`dsh-subprocess-local`, `dsh-sandbox-windows-acl`, `dsh-native-command`.
**Consequence for this round's evidence:** the failed call was a *final verification sweep* issued
after all three checkpoints had already been committed (`8b5f00d`, `568799d`, `94f95f2`). Its
intended content was re-run successfully in the following calls, so **no result under
`benchmarks/phaseAB/` is missing because of it** — but the failed call produced no output and is
recorded here rather than left looking like a silent gap.
**Environment facts, read-only:** Kaspersky is the registered and enabled antivirus
(`productState 0x41000`) and **Windows Defender is disabled** (`0x60100`), which is why
`Get-MpPreference` fails with `0x800106ba`. Any exclusion for this runtime must be set in Kaspersky,
not Defender.
**Not done, deliberately:** no antivirus setting, exclusion or security configuration was changed —
that is outside this project's permitted scope. The path to exclude, if the user chooses to, is the
npx cache directory `C:\Users\<user>\AppData\Local\npm-cache\_npx\1e7f6d9597241db0`, which holds the
DSH checkout.
**Why it matters beyond this round:** a re-deletion breaks the `pwsh` tool rather than any project
artifact. The failure mode is distinctive — `exit code 1073741845` with no other diagnostic — and
should be recognised as "check the runtime file still exists" rather than debugged as a project fault.
**Evidence:** this file; the three restored `runner.js` copies named above.

## E30 [QA hardening] `benchmarks/phaseAB/REPORT.md`: one commit hash and one VRAM figure were wrong

Both corrections are to the report of the A/B acceptance round. The report text itself is **not**
rewritten — it stands as issued, with this erratum to be read alongside it.

**1. A commit that does not exist.** The report's header (line 6) lists the round's commits as
`8b5f00d` (steps 2-3), `568799d` (step 4), **`ee2e0be`** (steps 5-6).
`git cat-file -t ee2e0be` fails: **no such object exists in the repository.** The steps 5-6 commit is
**`94f95f2`** ("Phase A/B acceptance round complete: B re-verified, report and management files,
offline tests re-run"). A following commit, `5989730` (ERRATA E29), was made after the report was
written and is therefore not listed in it.
**How it happened:** the hash was written into the report from the round's plan rather than read back
from `git log` after the commit was created — the third checkpoint was still uncommitted when the
number was first drafted.
**Evidence:** `git log --oneline 35cf3b1..HEAD`; `git cat-file -t ee2e0be`.

**2. Phase B's minimum free VRAM is 4911 MiB, not 5175 MiB.** The report states 5175 MiB in three
places (A-8 in the conditions table, the §6 "余量" line, and the §8 per-step table).
The correct figure is **4911 MiB**, and it comes from the run's own client-side sampling in
`benchmarks/phaseAB/results/AB-05-b/b_e2e.json` (`vram_overall.min_across_scenarios = 4911`), whose
per-scenario minima are: `normal_answer 4911`, `concurrency 4913`, `answer_absent 4928`,
`stale_index 4928`, `health 4944`, `service_unavailable 6275`.
The 5175 MiB figure came from the **external** `vram.csv` (156 samples at 200 ms). That series is
shorter than the run and **does not cover the 8113 restart window** during the `service_unavailable`
scenario, so it never observed the run's lowest point. The client samples are interleaved with the
scenarios themselves and therefore do cover it.
**Both numbers exceed the 2048 MiB margin by a wide margin, so no conclusion changes** — the "margin
held" finding stands either way. The figure is corrected because a reported minimum that is 264 MiB
too high is exactly the kind of number later rounds would reuse.
**Evidence:** `benchmarks/phaseAB/results/AB-05-b/b_e2e.json`; the CSV at
`benchmarks/phaseAB/results/AB-05-b/vram.csv`.

## E31 [Phase 6 round] A second DSH runtime file was removed mid-call; one tool call did not run

**Is:** during the Phase 6 round a `pwsh` tool call failed with
`subprocess-local: Windows Job runner exited with exit code 1073741845 before proving its managed
range empty` — the same signature as **E29** (`1073741845` = `0x40000015` `STATUS_FATAL_APP_EXIT`).
Inspected immediately afterwards:
* `@deepseek-ai\dsh-subprocess-local\lib\runner.js` — **present**, 13483 bytes, sha256
  `4E21528DFDB3E92BDCE2340945DC4DCB202E4A4AF0BBEB8B97585F67735D30C1` (the same file the user restored
  after E29, unchanged)
* `@deepseek-ai\dsh-sandbox-windows-acl\lib\runner.js` — **present**, 7807 bytes
* `@deepseek-ai\dsh-native-command\lib\runner.js` — **MISSING**, and it was present when E29 was
  written
So **a different file from E29's was removed**, in the same package family. Every sibling in that
package survives, including `lib/types/runner.js` (1214 bytes), `lib/index.js` and all `.d.ts` files,
so the removal was again **targeted at one filename, not a directory or a package**.
**Impact: none observed.** The very next tool call succeeded, and tool calls continued to work
through the end of the round. Unlike the `dsh-subprocess-local` copy, the `dsh-native-command` file is
not on the runtime path this session uses — which is consistent with the failure being transient
rather than fatal. **No work was lost**: the failed call was a staging/verification step, and all
commits (`d29d2dc`, `b637590`, `0d8b175`, `6635f20`) were already on disk.
**Not restored by me.** Re-fetching a package file from the network, or copying it out of another
cache, would put an unverified artifact on the runtime path of the tool that executes every command.
That is the user's call, not mine. No antivirus setting or exclusion was changed (out of scope, task
§2.2) — see **OPEN_ISSUES P3-8**.
**Why it matters:** this is the **second** occurrence in two rounds, and it hit a *different* file
each time. The pattern is that the antivirus is still active against this checkout, so a recurrence
should be expected rather than treated as a one-off. The distinctive signature remains
`exit code 1073741845` with no other diagnostic.
**Evidence:** this entry; the two surviving `runner.js` files and the missing third, listed above.

### E31 CORRECTION (2026-10-05, same round) — the third file was never missing; my conclusion above was wrong

**What I claimed:** that `@deepseek-ai\dsh-native-command\lib\runner.js` "was **MISSING**, and it was
present when E29 was written", i.e. a second file had been deleted.

**What is actually true: that file is not part of the package and never was.** Verified two ways, both
mechanical:

1. **The package's own manifest.** `dsh-native-command\package.json` declares
   `"files": ["lib/index.js", "lib/types/**/*.js", "lib/types/**/*.d.ts"]`. `lib/runner.js` is not in
   that list, so npm never publishes it.
2. **The published artifact, integrity-checked.** `package-lock.json` records
   `resolved: https://registry.npmjs.org/@deepseek-ai/dsh-native-command/-/dsh-native-command-0.2.0-rc.2.tgz`
   and `integrity: sha512-RgjkI2avXHxJFwI7tjb42WTHHd4xL6Ybbg5I12T8qbuW4ph6gzAGcl8UKv7tSFM59yD85i2rnULTiZOzKKSdqw==`.
   The tarball was downloaded and its **sha512 matched that integrity value exactly**, and it contains
   **22 files, none of them `lib/runner.js`**.
   The installed directory was then compared **file by file**: **22 published, 22 installed, 0
   differing, 0 missing, 0 extra.** The installed package is byte-for-byte the published artifact.

**So there was nothing to restore.** The file's only real existence is
`dsh-native-command\lib\types\runner.js` (1214 bytes), which was **never deleted** and is still
present. Kaspersky's quarantine entry names a path that does not exist in this package — which is
consistent with the quarantine dialog showing a **truncated and partly rendered** path
(`C:\Users\<user>\AppData\Loca...dsh-subprocess-local\lib\`, a mix of a profile path and a *different*
package's directory). The entry is therefore best treated as **unreliable as a record of what
existed**, not as evidence of a deleted published file.

**The error's origin, stated plainly:** I inferred "missing" from *absence at a path I expected*,
without checking whether the package was ever supposed to have that path. The correct order is to
establish the expected file set first — from the manifest or the integrity-checked artifact — and
only then ask whether anything is absent. I did it backwards, and the correction here is the result
of doing it in the right order.

**What survives from E31 unchanged:** the tool-call failure with `1073741845`, the fact that
`dsh-subprocess-local\lib\runner.js` WAS genuinely deleted and restored (E29, and the user restored it
again this round), and the conclusion that the antivirus remains active against this checkout.
**What is retracted:** only the claim of a second missing file. See **OPEN_ISSUES P3-8** for the
detection-type evidence that arrived with the user's screenshots.
**Evidence:** `dsh-native-command\package.json`; `package-lock.json`; the integrity-verified tarball
(sha512 match) and its 22-file listing; the 22-vs-22 file-by-file comparison. Scratch copy deleted
afterwards; it lived under `tools/downloads/`, which is gitignored.
