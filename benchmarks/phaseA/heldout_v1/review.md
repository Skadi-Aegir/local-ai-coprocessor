# heldout_v1 — run1 per-question review

**Run:** `results/run1.json`, single run, exit 0, 38 s wall.
**Audit:** `results/run1_audit.json` (exit 0).
**Conditions met:** index `488f54d755224c5b` (rebuilt, `hash_scheme sha256-lf-v1`,
`index_version_scheme 2`), prompt `qa-v2`, dense retrieval, reranker disabled, 8113 running the
P1-11 fix, 8111 up. Frozen file `sha256-lf-v1 = fa1cf6d5…2084` verified before the run.
**Minimum free VRAM over the run: 5208 MiB** (154 samples at 200 ms; `mem_total − mem_used`).
Margin 2048 MiB held throughout. No OOM.

Scoring follows `questions.md` as pre-registered: a refusal on h01–h09 is an **incorrect refusal**;
a refusal on h10–h12 is the **correct** outcome; stating a superseded value as current is an
**incorrect answer**; a claim must be traceable to a cited chunk.

## Per-question

| id | class | cited file:lines (all `verified`) | support verdict | notes |
|---|---|---|---|---|
| h01 | complete | `docs/environment-report.md` 67-88, 219-237, 11-26 | **supported** | 591.86, CUDA 13.1, toolkit absent with the `nvcc`/directory reason — all in the cited lines |
| h02 | complete | `docs/environment-report.md` 125-144 | **supported** | disk as the binding constraint and 15–25 GB weight storage both in the cited range |
| h03 | complete | `config/local-reranker.yaml` 1-93, `docs/reranking-architecture.md` 52-82 | **supported** | softmax over the yes/no logits at the final position. The cited yaml chunk carries `model_type: CrossEncoder (LogitScore over a causal LM)`; the architecture chunk carries the scoring procedure. The audit marks the expected-evidence column `-`, i.e. the *expected file* was not the one cited; the claims themselves are supported by what was cited, so it is recorded as complete, with that difference noted |
| h04 | complete | `docs/local-fast-api.md` 10-25 | **supported** | alias `local-fast` and the quantisation-change rationale both in the cited range |
| h05 | complete | `docs/production-integration-design.md` 99-119 | **supported** | all six rollback triggers within the cited range. **Prose truncated at the 256-token cap** — see the defect section below; all six facts are present, so it stays complete |
| h06 | complete | `docs/escalation-policy.md` 91-101 | **supported** | the four work kinds and "support, not authorship" in the cited range |
| h07 | complete | `docs/retrieval-hardening.md` 161-169, `docs/gpu-scheduling.md` 40-61 | **supported** | verified directly: the cited gpu-scheduling lines carry "waits for... already running", "never queues behind background work that arrives after it", and the `max_background_units × unit duration` bound with 0.207 s / 0.600 s |
| h08 | complete | `docs/staged-lifecycle-design.md` 1-19, `docs/phase55r-evidence-consolidation.md` 75-89 | **supported** | +1368–1392 (embedding), +2591–2595 (reranker), 22–24 s — all in the cited ranges |
| h09 | complete | `services/retrieval/retriever.py` 25-40 | **supported** | verified directly: the cited lines contain "Raised from 2000 to 8000", the ~900-token chunker maximum, and the truncation reasoning |
| h10 | correct refusal | none | **correct** | Whisper is not deployed; only named in the planned model pool |
| h11 | correct refusal | none | **correct** | OCR is not deployed; PaddleOCR only in the planned model pool |
| h12 | correct refusal | none | **correct** | no TLS endpoint and no supervisor connection exist |

## Counts per category (no accuracy figure)

| category | count | ids |
|---|---:|---|
| complete | **9** | h01–h09 |
| partial | **0** | — |
| incorrect | **0** | — |
| correct refusal | **3** | h10, h11, h12 |
| incorrect refusal | **0** | — |
| error | **0** | — |
| unsupported statements | **0** | — |
| invalid citations | **0** | all citations `verified` |

## Superseded values (h09 class of error)

**No answer stated a superseded value as current.** h09 asks about the change from 2000 to 8000 and
the answer reports both values with their correct temporal roles (8000 current, 2000 dropped). The
three version-sensitive questions the set flags — h07, h08, h09 — are all answered against the
corrected wording: h07 quotes the post-P55R-01 policy (the cited file carries its own 2026-10-05
correction note), and h08 quotes the staged-path figures from the corrected documents.

## Defect found during this review: answers are cut off at MAX_ANSWER_TOKENS = 256

`cli/ask.py` sends `max_tokens: MAX_ANSWER_TOKENS` with `MAX_ANSWER_TOKENS = 256` (L82, L564) and
does **not** read back `choices[0].finish_reason`, so a length-truncated reply is indistinguishable
from a finished one in the record.

Measured on this run: **h05 used exactly 256 completion tokens and its prose stops mid-sentence** —
the visible text ends `"The rollback mechanism involves reverting the routing change"` with no
terminator. Two of the seven answered P56 regression questions are the same (`p05`, `p07`, both
exactly 256 tokens, both ending mid-word). The other 8 heldout answers and 5 regression answers end
on a terminator well below the cap.

**h05 is still recorded as complete, not partial**, because all six expected facts appear in the
text that was produced and are individually cited. The truncation is therefore a *presentation and
completeness* defect on this question rather than a missing-fact one — it removed closing prose, not
evidence. It is recorded here because a 256-token cap that silently cuts answers is a real limit on
the tool, and because "all key facts present" would otherwise hide it.

**No change was made in response.** The phase forbids changing code, prompt, corpus, thresholds or
scoring rules after seeing this set's results; `MAX_ANSWER_TOKENS` was therefore left at 256. It is
reported for Astra to decide on, and any fix belongs to a round that does not also read this set's
results.

## Audit differences

One, recorded rather than adjusted: **h03's expected-evidence column is `-`** — the answer cites the
reranker config and the reranking-architecture document rather than the file `expected.json` pins as
the primary evidence. The claims are supported by the cited chunks, so the outcome stands as
complete; the difference is noted because the audit's expected-evidence check is a useful signal
that should not be silently ignored.

No rule was changed, no question re-classified, and no post-hoc adjustment was made.
