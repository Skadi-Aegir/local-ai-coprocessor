# PROJECT_STATE

**Acceptance authority (current):** **Astra**, from 2026-10-05 (P57-C0). History, preserved rather
than overwritten: Astra held this role through the earlier phases; **Claude** then held it for the
QA-hardening + Phase 6 round (the A/B acceptance verdicts and AD-21/AD-22 are Claude's); and Astra
took it back for P57-C0. Verdicts belong to whoever held the role on the date they were issued.

**Last updated:** 2026-10-05, **P57-C0** (cross-document evidence gathering, diagnostic phase only —
`benchmarks/p57c0/REVIEW_REQUEST.md`). Before that: the QA-hardening + Phase 6 (local OCR) round
(`benchmarks/qa_hardening/`, `benchmarks/ocr/REPORT.md`), baseline `5989730`. This edit corrected the
acceptance-authority line and the self-contradicting status list; see the note under "Status split".

**Phase A and Phase B: ACCEPTED (2026-10-05), each with the same condition — the 256-token silent
truncation had to be fixed.** That condition is **discharged in this round by AD-22**: the limit is
now 512, `finish_reason` and `answer_truncated` are recorded on every record, and the truncation is
stated in the CLI and in the HTTP response.
* Phase A also carried these recorded limits: only 12 questions, so no generalisation; the questions
  were written by the fix author, so independence is limited; **heldout_v1 is now EXPOSED and may not
  be used for acceptance again** — future acceptance needs a new unexposed set; h03's pre-registered
  source was incomplete (the corpus holds another equally valid source); answer-absent refusal still
  relies on the 4B model following the prompt (P2-7).
* Phase B also carried: no authentication and no TLS, loopback-only binding; one request executing at
  a time; the concurrency ceiling was never tuned.
* Read the A/B round with **ERRATA E29-E30**, which correct it: the steps 5-6 commit is `94f95f2`
  (not the `ee2e0be` the report prints), and Phase B's minimum free VRAM is **4911 MiB**, not 5175.

**Residency (AD-21, adopted 2026-10-05):** resident by default = `local-fast` (8111) +
`local-embedding` (8113). On demand = `local-reranker` (8114), `local-smart` (8112), and
**`local-ocr` (8116, CPU only)**. A new service must either hold the 2048 MiB margin beyond that
pair, or be **CPU-only**, or be scheduler-loaded on demand. The **2048 MiB margin itself is
unchanged.**

**`local-ocr` (Phase 6, 2026-10-05):** `127.0.0.1:8116`, **CPU only** — under AD-21 it qualifies by
being CPU-only, so it takes no part in the VRAM budget. PP-OCRv5 (mobile tier selected by the
pre-registered rule), started on demand with `scripts/start-local-ocr.ps1`, never left running.
Models live in `models/ocr/` and are **not in Git**. See `docs/models/local-ocr.md` and
`benchmarks/ocr-results.md`.

## The QA path as it now stands

| item | value |
|---|---|
| entry points | `cli/ask.py` (CLI) and `cli/ask_server.py` (`127.0.0.1:8115`, started on demand, not left running). Dense retrieval + `local-fast`, **reranker disabled**, in-process embedder disabled, prompt `qa-v2` |
| interface limits | 2 requests admitted, **1 executing**; question 1000 chars; body 64 KiB; upstream timeouts embed 30 s / tokenize 15 s / generate 90 s, mapped to 503 / 504 / 502 |
| index | `vector/qa/current`, version **`3cd2dcda4eeb5555`**, **54 files / 595 chunks**, rebuilt 2026-10-05 from commit `6635f20` so the Phase 6 OCR material is answerable; `hash_scheme sha256-lf-v1`, `index_version_scheme 2`, `corpus_files_not_at_commit` **empty** |
| index reproducibility | a second independent build compares **bit-identical** (`same_index true`, `max_abs_diff 0.0`) - AD-17 |
| manifest | per-file SHA256 with CRLF folded to LF (`sha256-lf-v1`), per-file git status, line counts, the full corpus manifest with every exclusion reason, and the build's own validation block |
| rollback | `vector/qa/current.prev` = **`488f54d755224c5b`**. A rollback is `--switch-only --out vector/qa/current.prev`; both slots are kept |
| builder | `benchmarks/phaseA/a_build_index.py`: safe switch (nothing deleted before the new index is in place; refuses while the index is in use; `--switch-only`), 4 texts per embedding request, a margin preflight, free VRAM sampled during the build, retention read after it, `chunks_sha256` / `vectors_sha256` recorded |
| **preconditions** | keep the **reranker non-resident** for the QA path (a margin fact: with 8111 and 8113 resident, free VRAM is **4913 MiB**, and the reranker's ~2.4 GB would not fit). The old "restart 8113 after bulk indexing" precondition is **WITHDRAWN** - P1-11 is verified on the deployed service (`retained_mb 0.0`). Resident set is per **AD-21** |
| tests | `cli/test_qa_path.py` **89**, `benchmarks/phaseA/test_a_build_index.py` 59, `a_audit_heldout.py --selftest`, `services/retrieval/test_chunker.py` 38, `test_index_store.py` 42, **`services/ocr/test_ocr_server.py` 103** - offline, no GPU, all passing |
| **answer limit** | `MAX_ANSWER_TOKENS = **512**` (AD-22), `PROMPT_TOKEN_BUDGET = 7616`. `finish_reason` and `answer_truncated` are recorded on **every** record, including records downgraded to a refusal by the citation gate; the CLI prints a truncation line and `ask_server.py` always returns both fields. A truncated answer counts at most as **partial** unless every expected fact is present and cited. Previously 256 with no `finish_reason`, which cut h05, p05 and p07 mid-sentence invisibly |

**Status split:**

> **Corrected 2026-10-05 (P57-C0).** This list previously contradicted itself: **P1-11, P1-12 and
> P1-13 were named in BOTH the RESOLVED and the STILL OPEN bullets.** They are resolved and appear only
> under RESOLVED below. Also added: P1-6 (resolved via AD-21) and the issues opened since. The
> correction is recorded here rather than made silently, and no historical result file was altered.

* **RESOLVED**:
  * the staged double-unload RAG path passes on the isolated path (P55R-05);
  * the scheduler atomicity, reservation, FIFO, dependency and unload-protection defects (P55R-01/02);
  * the P56 CLI safety refusals;
  * the eleven defects tabulated at the end of OPEN_ISSUES (false `stale` citations in fresh
    checkouts, the unsafe switch, a missing index served as empty, the 2000-character pack cut,
    unbounded upstream timeouts, unstructured errors, and others);
  * **P1-11** embedding-service VRAM retention — verified on the **deployed** 8113
    (`retained_mb 0.0`), so the "restart 8113 after bulk indexing" precondition is withdrawn;
  * **P1-12** Phase A acceptance — closed by Claude's verdict on 2026-10-05; **heldout_v1 is exposed
    and may not be used for acceptance again**;
  * **P1-13** no clean rollback index — the contaminated slot was retired by the switch;
  * **P1-6** the three-model default refusing all normal work — resolved by **AD-21** (residency
    changes; the 2048 MiB margin does not);
  * the answer-truncation defect — **AD-22**: limit 512, `finish_reason` and `answer_truncated` on
    every record.
* **STILL OPEN**: P1-1 index contamination (~1 fact measured), P1-2 partial attribution, P1-3 no
  unexposed held-out set **for the retrieval benchmark**, P1-4 corpus-absent queries, P1-5
  `commit_hash` 0/186 in the benchmark index, P1-10 the cross-process ledger, P2-7 to **P2-11**, P3-6,
  P3-7, and the two opened this round — **P3-9** the OCR recogniser is not run-to-run deterministic
  (so a single-image CER carries about ±1 pp, which weakens the B3 thresholds) and **P3-10** one OCR
  fixture embeds live machine state and cannot be rebuilt byte-for-byte.
* **ENVIRONMENT RISK (user-side only)**: **P3-8** — the antivirus has removed a DSH runtime file three
  times (`PDM:Exploit.Win32.Generic.nblk`, a **behavioural** verdict, so this is an architectural
  false positive). Threat-detection exclusions are in place; whether they cover PDM is unconfirmed, so
  another occurrence should be expected. Response: recognise `1073741845`, compare the file against
  the published artifact, restore from quarantine only if it is genuinely gone.
* **PASSING ONLY IN AN ISOLATED EXPERIMENT**: the staged lifecycle. **It is NOT in the production
  routing**, and the QA path does not need it.

## 0. Live environment

**Snapshot of 2026-10-05, taken during the review:** no project service listening on 8111-8115;
GPU 12227 MiB total, 2.8-3.2 GB used by desktop applications, 8.7-9.1 GB free. The earlier figures in
this section (8632 MiB used / 3313 free, 8111/8113/8114 all ok) were a P55R-era snapshot. A snapshot
says nothing about whether a stopped service is faulty; check live before relying on any of this.

| item | value |
|---|---|
| safety margin | **2048 MiB, unchanged** |
| reranker declared transient | **1800 MiB, unchanged**; measured peak **881 MiB** (a sampling lower bound) |

**What the embedding service's memory figures mean - CORRECTED 2026-10-05 (ERRATA E25).** The startup
line and `/stats vram_mb` print the same function, cudaMemGetInfo `total - free`. On this WDDM machine
that is neither the device-wide figure (it read 1232.6 MiB while nvidia-smi showed 3438 MiB used)
nor the service's footprint. The earlier advice here to "use /stats, not the startup line" was a
misreading. The footprint is `/stats memory.torch_reserved_mb` (services carrying the P1-11 fix);
the margin is measured with nvidia-smi.

**Reranker peak, properly measured** (12 cycles, 4 load cases, 3 repeats, sampler 0.02 s,
content-unique documents): first call of each case spikes, repeats are ~0 because PyTorch's caching
allocator retains the block. Observed maximum **881 MiB**; margin never breached; minimum free
2510 MiB. This is a **lower bound** — the 0.02 s sampler has a blind window.

**`use_cache: false` is NOT honoured by the reranker service.** Identical documents repeat at
~0.025 s regardless of query; new content costs ~0.43 s. Any measurement that repeats the same
documents is measuring cache hits.

---

## 1. Deployment identity — verified from the machine, not from documentation

| item | value |
|---|---|
| repository | `D:\MultiAgent` (branch `master`, no remote configured) |
| HEAD at audit start | `bac96ab1b571` — **exactly** the baseline Astra checked |
| **working copies** | **TWO, not one.** `git worktree list` shows `D:/MultiAgent` at `master` **and** `C:/Users/<user>/.codex/worktrees/94a6/MultiAgent` at `bac96ab` on a **detached HEAD**. The P55R-01 environment snapshot reported "only one working copy on the machine"; that was **wrong**, because it was inferred from the main checkout rather than from `git worktree list`. The second worktree is a stale audit copy pinned at the baseline commit. |
| review worktree (2026-10-05) | a third worktree, `D:/MultiAgent/.claude/worktrees/workspace-content-review-be429a`, on branch `claude/workspace-content-review-be429a`. It owns no process and no port; the P1-11 measurements it ran were isolated processes that refused to start while any project port was listening, and one temporary embedding service on port 8199 that was stopped afterwards. Its fresh checkout is where the CRLF false-stale defect (ERRATA E19) was found. |
| resource ownership | the DEPLOYMENT is `D:\MultiAgent`: ports 8111/8113/8114 are served by processes whose command lines point into `D:\MultiAgent\...`. The second worktree owns no process and no port. Any future ownership question must be settled from `Win32_Process.CommandLine`, not from the existence of a directory. |
| git | 51 commits at baseline; clean tree |

### Services (ownership judged from `Win32_Process.CommandLine`)

| port | alias | process | owner | health |
|---:|---|---|---|---|
| 8111 | `local-fast` | `llama-server.exe` | project (`tools/llama.cpp/b11254-cuda13.4`) | ok |
| 8113 | `local-embedding` | `services/retrieval/venv` python | project | ok, 1173 chunks consistent |
| 8114 | `local-reranker` | `services/reranker/venv` python | project | ok, yes=9693 no=2152 |
| 3080 | DSH web GUI | `node.exe` | outside this project | responds 401 at `/` |

**A misleading measurement corrected:** `Get-Process .Path` reports the BASE interpreter for a
venv-launched process, because a venv `python.exe` is a launcher that re-executes it. The first
snapshot therefore called 8113/8114 "not project-owned", which was wrong. Ownership is judged
from the command line. Evidence: `benchmarks/phase55r/results/P55R-01-envsnap-v2/`.

### Models deployed

| alias | weights | size |
|---|---|---|
| `local-fast` | Qwen3.5-4B-Q4_K_M.gguf | 2613.96 MB |
| `local-smart` (not resident) | Qwen3.5-9B-Q4_K_M.gguf | 5417.37 MB |
| `local-embedding` | Qwen3-Embedding-0.6B safetensors | 1136.39 MB |
| `local-reranker` | Qwen3-Reranker-0.6B safetensors | 1136.39 MB |

### Hardware at audit

GPU RTX 5070 Ti Laptop, 12227 MiB total. RAM 32213 MB. Pagefile 38400 MB allocated,
32016 MB peak observed — the commit limit is a real operational constraint on this machine.

## 2. Indexes - several exist and they are NOT interchangeable

| index | chunks | distinct paths | status |
|---|---:|---:|---|
| `vector/embedding/local-ai` (served by 8113, "project" index) | 1173 | 94 | historical; carries the answer-key material (P1-1) |
| `cache/retrieval_bench/indexes/d1024_structured` (benchmark) | 186 | 30 | benchmark only |
| `vector/qa/current` (QA path) | 520 | 48 | serving; `259fae05785b433c` |
| `vector/qa/current.prev` | 615 | 52 | **contaminated, refused by the QA path** (P1-13) |
| `cache/retrieval_bench/indexes/decon_v1` | 1061 | - | derived, gitignored (P1-9) |

## 3. Benchmark sets — reported separately, never merged

| set | size | status |
|---|---:|---|
| historical regression | 39 | **published since Phase 4, therefore exposed** |
| new dev | 22 | used for tuning (pool size, causal instruction) |
| new test | 22 | **EXPOSED** — also used when choosing those settings |

There is currently **NO unexposed held-out set**, so no claim of independent generalisation is
supported. The earlier label "61 held-out" was wrong: it merged 39 long-exposed historical
queries with 22 new ones.

## 4. Environment changes made during P55R-01, and how to roll back

| change | rollback |
|---|---|
| new directory `benchmarks/phase55r/` | `Remove-Item -Recurse` |
| `services/gpu_scheduler/gpu_scheduler.py` rewritten (atomicity, reservations, FIFO) | `git revert` the P55R-01 commits |
| `services/gpu_scheduler/scheduler_cli.py` assertions updated to the unified policy | same |
| `services/retrieval/test_fallback.py` assertion widened (transport-class) | same |
| services stopped and restarted during live validation | `scripts/start-local-*.ps1` |

**No driver, CUDA, pagefile, Qwen-Image or security-software change was made.** No model was
downloaded. No historical index or result file was overwritten — all new output went to
`benchmarks/phase55r/results/<run-id>/`.

One environment note worth carrying: during B4 a killed test left 8113 unloaded. All three
services were restarted and verified healthy before this file was written.

## 5. Where the evidence lives

| topic | evidence |
|---|---|
| environment snapshot | `benchmarks/phase55r/results/P55R-01-envsnap-v2/` |
| A1 variable audit | `.../P55R-01-a1/a1_baseline_audit.json`, `a1_confounds.json` |
| A2 contamination | `.../P55R-01-a2/a2_contamination.json` |
| A2 grader | `.../P55R-01-a2g/a2_grader.json` |
| A3 metrics | `.../P55R-01-a3/a3_metric_audit.json` |
| C attribution | `.../P55R-01-c/flagged_questions.json` |
| B1 atomicity | `.../P55R-01-b1/b1_atomicity.json`, `b1_verify.json` |
| B2 unload | `.../P55R-01-b2-fixed/b2_unload_failure.json` |
| B3 semantics | `.../P55R-01-b3-final/b3_semantics.json` |
| B4 live | `.../P55R-01-b4/b4_live.json` |
| regressions | `.../P55R-01-regr-pass/regression.json` |
| frozen Phase 5 baseline | `benchmarks/retrieval-v2/baseline.json` |

Superseded run directories (earlier iterations kept for traceability, not as results):
`P55R-01-b2`, `-b3`, `-b3-diag`, `-b3-fifo`, `-b3-fixed`, `-b3-rerun`, `-regr`, `-regr-final`,
`-regr-prereq`, `-envsnap`. Only the `-final`/`-fixed`/`-pass`/`-v2` variants are the
authoritative runs.
