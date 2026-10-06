# QH-01 — A2 regression (truncation fix), real machine

**This is a REGRESSION run on already-exposed sets, not an acceptance run.** Both sets have been read
and discussed, so nothing here supports a quality claim. Its purpose is to confirm that the
truncation fix (AD-22) changed no verdict and that every answer now carries `finish_reason`.

**Conditions:** index `488f54d755224c5b`, prompt `qa-v2`, dense retrieval, reranker disabled,
8111 + 8113 up, 8114 not resident. Command: `cli/ask.py --question-file <set> --json --out ...`.
VRAM sampled externally at 200 ms: **185 samples, free_min 5123 MiB, margin held, no OOM.**

Records: `p56.json`, `heldout_v1.json`, `heldout_v1_audit.json`, `vram.csv`.
Both frozen files were unchanged (neither was touched by this round).

## heldout_v1

| id | finish_reason | completion | truncated | verdict | cites | citation status |
|---|---|---:|---|---|---:|---|
| h01 | stop | 174 | no | answered | 3 | verified |
| h02 | stop | 76 | no | answered | 1 | verified |
| h03 | stop | 149 | no | answered | 2 | verified |
| h04 | stop | 118 | no | answered | 1 | verified |
| h05 | **stop** | **270** | no | answered | 1 | verified |
| h06 | stop | 144 | no | answered | 1 | verified |
| h07 | stop | 120 | no | answered | 2 | verified |
| h08 | stop | 117 | no | answered | 2 | verified |
| h09 | stop | 156 | no | answered | 1 | verified |
| h10 | stop | 15 | no | refused | 0 | — |
| h11 | stop | 15 | no | refused | 0 | — |
| h12 | stop | 15 | no | refused | 0 | — |

**h05 is the case that motivated the fix, and it resolves cleanly.** Under the 256-token cap h05 used
**exactly 256** and stopped mid-sentence. Now it finishes on its own at **270 tokens** with
`finish_reason: stop` — the model needed 270, so 256 was cutting a complete answer in half, not
merely trimming prose.

**No `finish_reason: length` remains in this set.**

**Verdicts are identical to `results/run1.json`: 12 of 12 unchanged** (h01-h09 answered, h10-h12
refused) — the required pattern holds. **No citation is other than `verified`.** The mechanical audit
re-ran with exit 0 and reproduced the same per-category counts as run1:
`answerable 6 answered_needs_support_review, cross_document 2, version_conflict 1, answer_absent 3
correct_refusal`.

## P56 regression

| id | finish_reason | completion | truncated | verdict | cites |
|---|---|---:|---|---|---:|
| q01 | stop | 184 | no | answered | 3 |
| q02 | stop | 132 | no | answered | 2 |
| q03 | stop | 195 | no | answered | 2 |
| q04 | stop | 175 | no | answered | 2 |
| **q05** | **length** | **512** | **YES** | answered | 3 |
| q06 | stop | 243 | no | answered | 2 |
| q07 | **stop** | **366** | no | answered | 3 |
| q08 | stop | 15 | no | refused | 0 |
| q09 | stop | 15 | no | refused | 0 |
| q10 | stop | 15 | no | refused | 0 |

**q05 is now correctly flagged as truncated, and it should be.** It used the full 512 tokens and its
visible text stops mid-sentence: `"... 4. **Model Instance**: Loads \`"`. Under the old cap this same
question was ALSO cut (at 256 tokens) but the record said nothing, so this is exactly the
failure mode the fix removes — the cut is now visible instead of hidden. **q05's answer must be read
as incomplete.** Its 3 citations are all `verified`, so it is traceable; it is not a complete trace of
the pipeline.

**q07 now finishes at 366 tokens** (`stop`). Under the old 256 cap it was cut at 256 — so the old cap
was also truncating q07 silently.

## What changed and what did not

| | before (256) | now (512) |
|---|---|---|
| h05 | cut at 256, silent | **finishes at 270, `stop`** |
| q05 | cut at 256, silent | cut at 512, **flagged `length`** |
| q07 | cut at 256, silent | **finishes at 366, `stop`** |
| verdicts | 6/4 (p56), 9/3 (heldout) | **unchanged** |
| fields on every record | none | `finish_reason`, `answer_truncated` |
| prompt budget | 7872 | **7616** (observed pack 1768-2466, well inside) |
| citations | all verified | **all verified** |
| latency (heldout) | run1 0.494-18.799 s | **0.407-5.346 s** |
| min free VRAM | 5208 MiB (run1) | **5123 MiB** |

**Two of the three known cuts are cured outright (h05, q07); the third (q05) is a genuinely long
answer that now announces itself.** No verdict regressed, so no stop condition was triggered and
nothing was changed after seeing these results.

**Not claimed:** that the answers are now correct or complete in a grading sense. h05/q05/q07 support
is a human judgement and is not re-reviewed here — this is a truncation regression, and the point is
only that the record now tells the truth about whether the text was cut.
