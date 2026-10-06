# P57-C0R — corrected per-question results

**Scope:** correct the evidence statistics and conclusions of P57-C0. **No retrieval implementation and
no change to the QA production path.** Baseline `a96190a` verified at start. Index `3cd2dcda4eeb5555`.

| | |
|---|---|
| corrected raw measurement | `results/r01_measured.json` (full answers, full packs, citations, line numbers) |
| corrected verdict layer | `results/r02_scored.json` |
| superseded first corrected attempt | `results/r01_measured.first-attempt.json` (kept) |
| corrections | `ERRATA.md` items **R1–R9** |
| consistency check | `r06_consistency.py` — **42 checks, 0 failures** |

---

## 1. What the task asked for, and the four defects it exposed

**A. `carriers[:12]` was published as the carrier set (R1).** `n_carriers` and `len(carriers)`
disagreed silently — 13 vs 12 for one fact, 29 vs 12 for another. The corrected record writes the
**full** set under `carrier_chunks_full`.

**B. Two requests were reported as one measurement (R2).** The old comparison ran `search()` and then
`answer()` and printed the first call's candidates beside the second call's pack — two different top-k
lists, only one of which the model ever saw. The corrected script makes **one `answer()` call per
(question, k)** and reports no candidate list at all.

**C. Records too small to re-verify their own claim (R3).** `answerable_run.json` stored
`answer_head` — the first **400 characters** — and a boolean, with no pack. Its central claim could not
be re-checked from it. The corrected record stores the full answer, the full pack with every supplied
source's file and line range, citations, index version, token counts, latency and VRAM.

**D. A detector keyed on a fact name that does not exist (R4).** The corrected script looked up v02's
markers under `vram_figure_meaning` while the frozen manifest calls the fact `torch_reserved_meaning`,
so **no marker set applied** and the fact scored absent. Separately the manifest's own literals are
`["torch_reserved", "reserved_mb"]` — the **column names** — too strict, because the passage that
explains the figure says *"VRAM, torch attribution"* and *"cudaMemGetInfo total − free"* and never
prints `torch_reserved`.

**E. Found while re-scoring: refusals carrying reasoning were graded as answers (R5).**
`cli/ask.py` decides `refused` by **exact match**, so *refusal + **Reasoning:** …* is recorded as
answered. Measured: **6 of 18** answers contain the refusal sentence. Worse for scoring, the reasoning
was then mined for fact markers, so the model's restatement of the question (naming port `8111`, the
port of the file the question is about) registered as a *covered fact* and turned an incorrect refusal
into a reported `partial`.

---

## 2. v02, re-checked as the task asked

| k | memory-figure fact in pack | margin fact in pack | verdict |
|---|---|---|---|
| 5 | **absent** | present | **partial** |
| 8 | **present** — markers `vram_mb`, `attribution` | present | **complete** |
| 10 | **present** | present | **complete** |

**The retracted statement does not hold.** v02 is **not** answered with no required evidence at k=5:
the margin fact **is** in the pack and **is** covered by the answer. What is missing at k=5 is the
meaning of the memory figure, and the answer is therefore a **partial answer, not an unsupported one**.
At k=8 the evidence arrives — `config/local-embedding.yaml` and `services/retrieval/embedder.py` enter
the pack — and the verdict is **complete**.

**Recorded honestly:** the verdicts here are **marker-based**. A marker match is evidence about content,
never proof that a source *supports* a sentence; that remains a human judgement, and `P2-8`'s
mention-versus-assertion limit still applies.

---

## 3. Per-question results, corrected

| id | k=5 | k=8 | k=10 | facts in pack at k=10 |
|---|---|---|---|---|
| **v01** | incorrect_refusal | incorrect_refusal | **incorrect_refusal** | **1 / 2** |
| v02 | partial | **complete** | **complete** | 2 / 2 |
| v03 | complete | complete | complete | 2 / 2 |
| v04 | incorrect_refusal | incorrect_refusal | **partial** | 2 / 2 |
| v05 | incorrect_refusal | **complete** | **complete** | 2 / 2 |
| v06 | complete | complete | complete | 1 / 1 |

**Overall facts in the pack: `2/6 → 4/6 → 5/6`.** The previously reported `2/6 → 3/6 → 4/6` was wrong
(R7) — the middle figure was dragged down by v02, which defect D had mis-scored.

**The "no squeezing" finding STANDS:** `n_omitted` is **0 for every question at every k**, on both the
old and the corrected measurement.

---

## 4. Which questions are still ranking problems

* **v01 — YES, and it is the round's surviving finding.** `8113` never reaches the pack at any k while
  `8111` does. Its carriers rank **38, 51, 52, 58** for this wording — beyond any bounded k. Confirmed
  at all three ks in the corrected data.
* **v02 — NO.** It was my measurement error. Corrected, it completes at k=8.
* **v04 — depth-then-generation**, not pure ranking: pack **2/2** at k=10 but the answer still covers
  **1/2**.
* **v05 — a depth case**, fixed at k=8 (carrier at rank 6).
* **v03, v06 — regression controls**, complete at every k.

---

## 5. What this does to the pending decision

* **Weakens (C) "fix over-answering first"**: the over-answering reported for v02 **does not exist**.
  What survives of (C) is the refusal-detector mislabel (**P2-12**), which is real but
  retrieval-independent and out of scope for a measurement-only round.
* **Strengthens the case against (B) "k=8 alone"**: after correction k=8 already reaches 4/6 with v02
  and v05 complete, while **v01 is untouched at every k**. k=8 buys less than it appeared to and does
  not address the one question depth cannot reach.
* **(A) remains untested.** Nothing here measures a fact-oriented second pass.

---

## 6. Self-assessment: **CONDITIONAL PASS**

**PASS** on the task as scoped — the statistics are corrected, the four defects (plus a fifth found while
correcting) are documented with measurements, the retraction is explicit and complete, and the frozen set,
the P57-C0 raw results and the historical reports are all preserved. Consistency check: **42 checks,
0 failures**.

**CONDITIONAL** because of what the exercise revealed about my own work:

* **Five separate errors, all mine, were needed to produce one wrong headline claim.** The claim was
  published as a *second defect worse than the one under investigation*. It was not a small slip: it
  rested on a truncated carrier list, two mixed requests, a record too short to re-check, and a detector
  keyed on a name that does not exist. **The lesson is that my fact detectors were never validated
  against the content they claimed to detect** — the fix applied here is to test markers against the
  actual pack and answer text and to allow "undetermined" rather than forcing a verdict.
* **The corrected verdicts are marker-based and therefore provisional.** Where a fact's markers are
  semantic rather than literal (v02's first fact), the verdict is an inference, not a proof.
* **The one thing not fixed is a real product defect:** `cli/ask.py` reports `refused: False` for a
  refusal that carries reasoning (P2-12). It affects any consumer of that field. It is out of scope this
  round by instruction, and it is filed rather than quietly worked around.
* **What survives is narrower than what was claimed:** one question (v01) is a genuine ranking failure;
  the rest of the round's headline was measurement error.

**Not claimed:** that any retrieval change works, that the corrected verdicts are final, or that marker
presence proves support.
