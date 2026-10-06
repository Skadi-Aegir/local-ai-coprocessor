# Phase A held-out question set v1

**Frozen:** yes, from creation. Do not add, edit or reorder questions after the first run; a changed
set is a new set with a new name.
**SHA256 of `frozen_questions.txt`, LF-normalised (scheme `sha256-lf-v1`):**
`fa1cf6d5871af15f4db00aaf7c2c835f07f1b1499e7c018e6ce82007af812084`
(the file is UTF-8 without a byte-order mark and LF-terminated; a CRLF checkout differs only in
line endings, which is why the LF hash is the one recorded)
**Machine-readable pre-registration:** `expected.json` (expected outcome, key facts, evidence files
and line ranges, superseded values, absent-answer reasons).

## Why this set exists

Phase A's acceptance reused the P56 frozen 10 as its quality check. That set is a regression set:
its results had been read, discussed and partly re-labelled (q04, q05) before Phase A ran. Phase A
required a NEW frozen project question set, not used for tuning, with a per-question source audit.
This is that set. It has **not been run**.

## Provenance and exposure

* Written on 2026-10-05 by Claude (Opus 5.5) in the session that made the review fixes, from
  reading the corpus. The fixes were made first; the questions were written afterwards, and no
  system change may be made in response to this set's results without freezing a new set.
* Checked against the 105 exposed questions (P56 10, `rag_compare.py` 12,
  `benchmark_data.py` 39, `benchmark_v2.py` 44): the highest token overlap with any of them is
  Jaccard 0.14. Topics already covered by those sets (ports, context sizes, pooling, dimensions,
  VRAM footprints, throughput, quantisation choice, thinking mode, pool size, PID files, the
  default escalation tier) were avoided.
* None of the 12 questions appears verbatim in the corpus the next build will index, and the index
  builder now refuses to build if one does (`FROZEN_QUESTION_FILES` in `a_build_index.py`).
* The set is **12 questions**. No percentage computed on it supports a claim about general
  accuracy; results are reported per question and as counts per category.

## Run conditions (record each one with the run)

1. The index is **rebuilt** from the tree these line ranges describe (this commit or later, with no
   change to the evidence files): its manifest must carry `hash_scheme: sha256-lf-v1`,
   `index_version_scheme: 2`, `chunks_sha256` and `vectors_sha256`. Four corpus documents were
   corrected on 2026-10-05; the index built before that still holds the superseded wording that
   h07 and h08 test.
2. Prompt `qa-v2` (`cli/ask.py PROMPT_VERSION`), dense retrieval, reranker disabled.
3. `local-embedding` (8113) started from code carrying the P1-11 fix, `local-fast` (8111) up, the
   reranker (8114) **not** resident. Record the minimum free VRAM over the run.

```bash
python benchmarks/phaseA/a_build_index.py
python cli/ask.py --question-file benchmarks/phaseA/heldout_v1/frozen_questions.txt --json --out benchmarks/phaseA/heldout_v1/results/run1.json
python benchmarks/phaseA/a_audit_heldout.py --answers benchmarks/phaseA/heldout_v1/results/run1.json
```

The audit is mechanical (outcome against expectation, expected evidence retrieved / shown / cited,
citation verification, key facts present, superseded values present). The **support verdict is a
human one**, written per question in `review.md` next to the results, as in P56.

## The questions

| id | category | evidence | question |
|---|---|---|---|
| h01 | answerable | one file | Which NVIDIA driver version and CUDA version did the Phase 0 environment check find, and was a CUDA toolkit installed? |
| h02 | answerable | one file | What did Phase 0 expect to be the binding resource constraint for this project, and how much weight storage did it estimate the planned model pool would need? |
| h03 | answerable | either of two files | How does the local reranker turn the model's output into a relevance score? |
| h04 | answerable | one file | According to the local-fast integration guide, what must a caller put in the model field of a request, and why? |
| h05 | answerable | one file | Which observable conditions would trigger a rollback of the proposed staged two-unload RAG integration? |
| h06 | answerable | one file | Which kinds of work does the escalation policy send straight to Claude or Codex, and what role do local models keep for that work? |
| h07 | cross-document | either source; **version-sensitive** | Can an interactive request in the GPU scheduler end up waiting for background work, and what bounds that wait? |
| h08 | cross-document | either source; **version-sensitive** | On the isolated staged two-unload RAG path, how much free VRAM did each required unload return, and how long does a full RAG cycle take? |
| h09 | version conflict | one file, old and new value side by side | What per-chunk character limit does the retrieval layer apply when it packs context for a model, and why was the earlier limit dropped? |
| h10 | answer-absent | — | What word error rate does the Whisper speech worker achieve on Chinese audio in this project? |
| h11 | answer-absent | — | What text-recognition accuracy did PaddleOCR reach on terminal screenshots in this project? |
| h12 | answer-absent | — | Which certificate authority issues the TLS certificate that secures the MacBook supervisor's connection to this machine? |

"Either source" marks ALTERNATIVE sources: any one of the listed files answers the question. None of
the twelve needs several files together (a required evidence group), so no score here says
anything about multi-source completeness (OPEN_ISSUES P2-5).

## Expected evidence

| id | expected answer, in short | evidence (line ranges as of 2026-10-05) |
|---|---|---|
| h01 | driver 591.86, CUDA 13.1; no toolkit (`nvcc` absent) | `docs/environment-report.md` 15-17, 67-83 |
| h02 | disk storage, not VRAM; ~15-25 GB of weights | `docs/environment-report.md` 125-141 |
| h03 | softmax over the `no`/`yes` logits at the final position; the score is P(yes) | `docs/models/local-reranker.md` 67-93, or `services/reranker/reranker.py` 275-299 |
| h04 | the alias `local-fast`, never a file name / repo id / quant label, so the weights can change unnoticed | `docs/local-fast-api.md` 10-22 |
| h05 | margin breach; unverified unload; reload failure or over-ceiling reload; latency over budget; cross-process interference; any OOM | `docs/production-integration-design.md` 99-118 |
| h06 | architecture, large refactors, security-critical code, production-affecting changes, large blast radius; local models support, not author | `docs/escalation-policy.md` 91-100 |
| h07 | yes: it waits for background units already running, no new one is admitted while it waits, and bounded background units bound the wait | `services/gpu_scheduler/gpu_scheduler.py` 37-47 and 107-111, or `docs/gpu-scheduling.md` 40-60 |
| h08 | embedding +1368..+1392 MiB, reranker +2591..+2595 MiB; ~22-24 s per cycle, reloads dominating | `docs/staged-lifecycle-design.md` 8-15, or `docs/phase55r-evidence-consolidation.md` 75-88 |
| h09 | 8000 characters now; the 2000 cap cut `context:` out of `config/local-fast.yaml` | `services/retrieval/retriever.py` 30-37 |

**Version-sensitive questions.** An answer stating the superseded value as current is WRONG:
* h07 — "an interactive request never waits behind background work" (the pre-P55R-01 policy, still
  on `docs/gpu-scheduling.md` until 2026-10-05)
* h08 — "2.54 s" for a full cycle (a P55R-04 figure from a run in which no unload happened)
* h09 — 2000 characters as the current limit (the same comment states both values)

**Why h10-h12 are answer-absent.** Whisper and PaddleOCR are not deployed and nothing measured them;
the corpus names them only in the planned model pool (`docs/environment-report.md` 133). No TLS
endpoint or supervisor connection exists; `docs/local-fast-api.md` 35 mentions mTLS only as one
possible future access layer. Each has nearby text that could tempt an invented answer.

## Grading rules, stated before any run

* An answer is **supported** only if every factual claim in it traces to a cited chunk that
  contains it. A correct answer with no citation is not supported. An answer citing a source that
  does not contain the claim is not supported. (Unchanged from P56.)
* A citation must also be `verified` against the index manifest; the CLI already refuses
  otherwise.
* **Answerable questions (h01-h09): a refusal is an INCORRECT REFUSAL**, whatever its cause, and is
  counted as one. Its cause is recorded separately (evidence not retrieved, retrieved but not shown,
  shown but not used), never used to re-label the outcome.
* **Answer-absent questions (h10-h12): refusal is the correct outcome**; any other answer is a
  failure.
* An answer that gets part of the expected content right is a **partial answer**, recorded with
  what is missing, and is not counted as complete.
* Stating a superseded value as current is an **incorrect answer**, even when cited.
* Report counts per category: complete, partial, incorrect, correct refusal, incorrect refusal,
  error. No single accuracy figure.
