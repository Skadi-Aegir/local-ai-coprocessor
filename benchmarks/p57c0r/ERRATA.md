# P57-C0R — ERRATA

Corrections to the evidence statistics and conclusions of **P57-C0**. Nothing here changes code: no
file under `cli/`, `services/` or `config/` was modified, and the QA production path is untouched. The
frozen `devset_v1`, the existing raw results and the historical reports are all preserved as issued;
what changes is the interpretation, and the errors are kept on record rather than overwritten.

Baseline: `a96190a` (verified before starting). Index: `3cd2dcda4eeb5555`.
Corrected evidence: `results/r02_scored.json` (verdict layer) over `results/r01_measured.json` (raw).

---

## R1 — `carriers[:12]` was published as if it were the carrier set

**Was:** the facts file wrote `"carriers": [...][:12]` while reporting a separate `"n_carriers"`. A
consumer reading `carriers` therefore saw a truncated set — and `n_carriers` and `len(carriers)`
disagreed silently.

**Measured:** for `torch_reserved_meaning`, `n_carriers = 13` while `carriers` held **12**; for
`margin_policy`, `n_carriers = 29` while `carriers` held **12**.

**Now:** the corrected measurement writes the **full** carrier set for every fact, and records it under
`carrier_chunks_full` so it cannot be mistaken for a sample.

## R2 — two different requests were reported as one measurement

**Was:** the pack comparison called `r.search(question, top_k=k)` once, then `ASK.answer(...)` again,
and printed the **first** call's candidate list beside the **second** call's pack. Those are two
independent top-k lists. Only the pack is what the model actually saw.

**Why it matters:** any conclusion drawn by reading the two side by side — in particular "the carrier
was retrieved but the pack dropped it" — is unsupported, because the two lists need not agree.

**Now:** **one `ASK.answer()` call per (question, k)**. Everything reported — pack, supplied sources
with file and line range, token counts, facts, answer, citations, latency, VRAM — comes from that single
call. No candidate list is reported at all, because it is not what the model saw.

## R3 — records too small to re-verify their own central claim

**Was:** `answerable_run.json` stored `answer_head` (**first 400 characters**) and a boolean
`facts_in_pack`. No pack, no full answer, no line numbers.

**Consequence:** the v02 claim could not be re-checked from the record it was based on. When re-checking
began, the stored data could not settle it and a fresh request was required — the record was
insufficient for its own conclusion.

**Now:** the **full answer**, the **full pack** (every supplied source with `source_path`,
`start_line`, `end_line`, `chunk_id`), the citations with verification status, the index version, token
counts, latency and VRAM are all stored.

## R4 — a fact detector keyed on a name that does not exist, plus literals that were the column names

**Was (two layered errors):**

* the corrected script looked up v02's marker set under `vram_figure_meaning`, but the frozen manifest
  names that fact **`torch_reserved_meaning`**. No marker set was found, `markers_hit` returned
  `None` ("no markers defined"), and the fact was scored **absent**.
* the frozen manifest's own detection literals for that fact are `["torch_reserved", "reserved_mb"]` —
  the **column names**. Measured, those are too strict: the passage that explains the figure reads
  *"VRAM, torch attribution"* and *"cudaMemGetInfo total − free"* and **never prints `torch_reserved`**.

**Why this is the root of the retracted v02 claim:** one detector was too strict (missed the fact when
present) and the replacement was mis-keyed (so it silently did nothing), and between them the fact was
reported absent in every run. The evidence was there.

**Not fixed by editing the frozen set.** `devset_v1` keeps its bytes and its hashes; the semantic marker
set lives in the measurement script and the difference is recorded here.

## R5 — "refusal + reasoning" scored as an answer, and its reasoning mined for facts

**Was:** `cli/ask.py` sets `refused` by **exact match** against the refusal sentence, so an answer
written as *"I could not find enough evidence in the local corpus to answer this.\n\n**Reasoning:** …"*
is recorded as **answered**. Measured: **6 of 18** answers contain the refusal sentence.

**Worse:** with the record marked "answered", the reasoning prose was scanned for fact markers. The
model restates the question in its reasoning, so mentions of port `8111` — the port of the file the
question is about — registered as a **covered fact**, turning an incorrect refusal into a reported
`partial` answer. Measured: v01 at k=8 and k=10 are both refusal-plus-reasoning.

**Fixed in the scorer:** an answer **containing** the refusal sentence is treated as a refusal, and its
reasoning is not counted as covering any fact. Verdict changes: **v01 k=8 `partial → incorrect_refusal`**
and **v01 k=10 `incorrect → incorrect_refusal`**.

**NOT fixed in the product.** The underlying mislabel is a real defect in `cli/ask.py` and would need a
QA-path change, which this round forbids. It is filed as **OPEN_ISSUES P2-12**.

## R6 — the retracted v02 claim, stated plainly

**Claim as issued:** "v02 is answered while holding 0 of its 2 required facts", with three verified
citations, presented as a second defect worse than under-retrieval.

**Retracted.** It was the product of R1–R5. Corrected:

| k | facts in pack | verdict |
|---|---|---|
| 5 | 1 / 2 | **partial** — covers the margin policy, not what the memory figure is |
| 8 | **2 / 2** | **complete** |
| 10 | **2 / 2** | **complete** |

At k=5 the answer is a **partial answer, not an unsupported one**. No over-answering is demonstrated.

**Where the retraction is recorded:** `OPEN_ISSUES.md` P2-11 is marked **RETRACTED** with the original
text preserved; `REVIEW_REQUEST.md` §4e is marked superseded and §4g carries the correction.

## R7 — the overall progression was wrong

**Was:** `2/6 → 3/6 → 4/6`.
**Corrected:** **`2/6 → 4/6 → 5/6`**.

The old middle figure was dragged down by v02, which R4 had mis-scored. Per k, corrected:

| k | facts all in pack | still incomplete |
|---|---|---|
| 5 | 2/6 — v03, v06 | v01, v02, v04, v05 |
| 8 | **4/6** — v02, v03, v05, v06 | v01, v04 |
| 10 | **5/6** — v02, v03, v04, v05, v06 | **v01** |

**The "no squeezing" finding stands unchanged:** `n_omitted` is 0 for every question at every k, on
both the old and the corrected measurement. Raising k displaced no evidence.

## R8 — which questions are still ranking problems

* **v01 — YES.** `8113` never reaches the pack at any k while `8111` does; its carriers rank **38, 51,
  52, 58** for this wording, beyond any bounded k. **The round's central finding survives.**
* **v02 — NO.** It was R4. Corrected, it completes at k=8.
* **v04 — depth-then-generation**, not pure ranking: pack 2/2 at k=10 but the answer still covers 1/2.
* **v05 — a depth case**, fixed at k=8 (carrier at rank 6).
* **v03, v06 — regression controls**, complete at every k.

## R9 — what this does to the pending decision

* It **weakens the argument for (C) "fix over-answering first"**, because the over-answering I reported
  for v02 does not exist. The surviving content of (C) is the refusal-detector mislabel (P2-12), which
  is real but retrieval-independent and out of scope this round.
* It **strengthens the case against (B) "k=8 alone"**, because after correction k=8 already reaches 4/6
  with v02 and v05 complete, while **v01 is untouched at every k** — so k=8 buys less than it appeared
  to, and does not address the one question that is genuinely unreachable by depth.
* **(A) remains untested.** Nothing here measures a fact-oriented second pass.

---

## Discipline kept

* The frozen set was **not modified**: `manifest.json` and `questions.txt` are byte-identical, and the
  manifest, `questions.txt` and set-level hashes all still verify.
* The superseded first attempt at the corrected measurement is **kept**, not deleted:
  `results/r01_measured.first-attempt.json`.
* Historical results and the P57-C0 reports are **preserved**; corrections are additive and the
  original wording is retained wherever it was retracted.
* No change to the QA production path, no reranker, no OCR in QA, no new model. The 2048 MiB margin,
  the 1800 MiB reranker transient, the 8192 context and the 512 answer tokens are untouched.
  `agentvoice/` was not read, written, staged or committed.
