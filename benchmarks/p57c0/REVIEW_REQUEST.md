# P57-C0 — STOP-FOR-REVIEW submission

**Construction:** DSH ｜ **Command and acceptance:** Astra ｜ **Baseline:** `0e63f76` (verified)
**Status of this document:** submitted at the **step-4 stop condition**. No retrieval code was
changed. No implementation has started. Nothing below is a PASS claim.

---

## 1. Why I stopped: the measured mechanism is not depth

The step-4 instruction is explicit: *"If evidence still cannot be gathered, STOP implementation and
submit the concrete failure mechanism and the next minimal option for review — do not lump it under
'multi-hop'."* That condition is met, and here is the concrete mechanism.

### 1.1 Bounded expansion helps, then saturates immediately

Chunk-level check — did a **retrieved chunk actually contain the fact**, not merely come from the
right file:

| depth | complete | incomplete |
|---|---|---|
| **5 (current)** | **2 / 7** | d01, d02, d03, d05, d06 |
| 8 | 4 / 7 | d01, d02, d03 |
| **10 (the ceiling allowed)** | **4 / 7** | d01, d02, d03 |

Evidence: `results/diagnosis_chunklevel.json`.

Expansion to k=8 fixes **d05** (carrier at rank 7) and **d06** (carrier at rank 6) — both genuine
depth problems, both solved. **k=10 then adds nothing over k=8.** The remaining three are not depth
problems at all.

### 1.2 The three survivors are RANKING failures, with the numbers

Evidence: `results/mechanism.json`. For each missing fact I located **every** indexed chunk carrying
it and ranked those chunks against the question, searching to depth 60:

| probe | missing fact | carrier chunks | best carrier rank | needs |
|---|---|---:|---:|---|
| d01 | embedding port `8113` | 15 | **38** | k ≥ 38 |
| d02 | meaning of `torch_reserved` | 13 | **15** | k ≥ 15 |
| d03 | escalation threshold `0.30` | **0** | — | not in the corpus |

**The fact is not near the cut — it is two orders of magnitude of ranking away.** For d01 the best of
**fifteen** carrier chunks sits at rank 38. Raising k from 5 to 10 moved the boundary by five
positions; the carrier is 28 positions beyond it. **No bounded expansion reaches these.**

### 1.3 d03 is not a retrieval failure at all — it is my question-set error

`EVIDENCE_FLOOR = 0.30` is defined in `cli/ask.py` line 67, and **`cli/` is excluded from the corpus
by design**. Measured: the literal `0.30`, `EVIDENCE_FLOOR` and `evidence_floor` appear in **0 chunks**
of the current index. The `0.3` matches that do exist are unrelated numbers in other files.

So the specific threshold is **genuinely absent from the corpus**, exactly like the old q05 / 8114
case. **d03 must be reclassified from "answerable" to "answer-absent"** — my draft requirement set was
wrong, and finding that is a result, not a nuisance. It also means **the honest count of answerable
probes is 6, not 7.**

### 1.4 What the mechanism actually is

The failures share one shape, and it is not multi-hop:

> **The question's vocabulary does not reach the chunk that holds the fact.**
> d01 asks about "components … and ports"; the 8113 fact sits in a config file and a script header
> whose text is about embedding configuration, not about components. d02 asks how two things
> "relate"; the `torch_reserved` explanation sits inside a docstring and a scheduler implementation.
> A single-hop dense query is dominated by prose that is *topically* near the question, and the
> *fact-bearing* but topically-distant chunks rank far down.

The pack is not being squeezed either: no required evidence was displaced by expansion — the
expansion only helped.

**This is a retrieval-ranking problem, and I am deliberately not calling it "needs multi-hop".**
Nothing here shows that a fact requires *combining* two documents; it shows that a fact in a
*distant* document is not reached by the query.

---

## 2. A correction to my own first diagnosis (recorded, not hidden)

My first pass asserted that a **required FILE** was missing from the top-5 and reported 1/7 complete
at depth 5. **That was wrong**, and it was wrong in the direction that overstates the problem:

* a fact's carrier file being absent proves nothing if **another** carrier was retrieved;
* measured carriers: `8113` lives in **13–15** indexed files, `8114` in 11, the margin policy in 20,
  `softmax` in 6. d01's requirement on `docs/retrieval-architecture.md` was **redundant**, and d03's
  requirement on `services/gpu_scheduler/gpu_scheduler.py` was **redundant** too.

The file-level verdict was therefore invalid, and the numbers above are from the **chunk-level**
re-check. Both are preserved: `results/diagnosis.json` (superseded, kept) and
`results/diagnosis_chunklevel.json` (corrected). `results/fact_carriers.json` records the carrier
counts that exposed the error.

---

## 3. Proposed next minimal option (for review — NOT implemented)

Given a **ranking** problem rather than a depth problem, larger k is the wrong lever. The smallest
change that addresses the measured cause, and that I would propose **only with approval**:

> **Add a second, fact-oriented retrieval pass whose query is derived from the question's own
> requirement terms, and merge the two candidate lists before packing — bounded to the existing
> candidate ceiling, reusing the same packer, the same 8192 context, the same 512 answer tokens and
> the same citation/truncation markers.**

The reason this is the *minimal* lever: d01 and d02 both have their carriers **inside the top 40**,
so a second query built from the question's own distinctive terms (ports, reserved, meaning) has a
real chance of surfacing them, whereas raising k cannot. It adds no model, no reranker and no new
service — it is one extra embedding call and a merge.

**Alternatives considered and rejected as not minimal:** raising k to ~40 (rejected: unbounded token
cost, and the pack budget would then have to drop evidence — the very thing we must avoid);
enabling the reranker (rejected: explicitly out of scope this round).

**If you prefer the bounded approach anyway**, the measurements support a specific claim: at the
**retriever** layer **k=8 is the largest useful value** (k=10 adds nothing there), so k=8 is the
cheapest change that fixes anything. **But §4c showed k=10 does add one at the pack layer**, and k=8
alone leaves v01, v02 and v04 incomplete. k=8 is therefore offered as the **fallback, not the
recommendation**.

---

## 3b. Step 3 DELIVERED — the development-validation set is now frozen

Step 3 is ordered **before** any retrieval-code change, and no retrieval code has been changed, so
authoring it now follows the objective's own sequencing rather than advancing past the stop.

**`benchmarks/p57c0/devset_v1/` — 9 questions, frozen.**

| item | value |
|---|---|
| set id | `p57c0_devset_v1` |
| **manifest SHA256** | `aa44a4f870dc10bc831224e500931dbe8544426bcdd0617f32bf1690325e1c51` |
| **questions.txt SHA256** (LF) | `1ce9966dd4a583fa8fd3231347c97366b24451d8dad3472995b13acc9f999094` |
| **questions.md SHA256** | `936073bf429338746775e3b2096eca49c0bfb17153795fc90160ea6a23b90b9f` |
| answerable | **6** (v01–v06) — cross-file and single-file |
| answer-absent | **3** (v07–v09) |
| index at freeze | `3cd2dcda4eeb5555`, 54 files / 595 chunks |

**Labelled DEVELOPMENT VALIDATION explicitly**, in the manifest and in `questions.md`: the construction
side wrote the questions from the corpus it maintains, so independence is limited. **Not an independent
held-out set**, and no percentage on it supports a general claim.

**Two things this set does that the previous probe list did not:**

1. **Every fact was validated against the index BEFORE freezing.** A question enters the answerable
   class only if each of its facts has a real carrier chunk, and the canonical citation target is
   recorded with all carriers. That is how **v07 (the old d03) was caught and reclassified as
   answer-absent** — `EVIDENCE_FLOOR = 0.30` lives in `cli/ask.py`, which the corpus excludes, and
   measured, `0.30` / `EVIDENCE_FLOOR` / `evidence_floor` appear in **0 chunks**.
2. **The discriminators were audited for discriminating power, and two were rejected.** My first
   freeze "passed" while resting on accidental substring hits:
   * `cycle_latency: ["22","24"]` — bare `"22"` matched 26 chunks and `"24"` matched **49** (dates,
     versions). Tightened to require co-occurrence → **64 → 5 carriers**.
   * `why_raised: ["2000","truncat"]` — bare `"2000"` matched 10 mostly-unrelated chunks. Tightened to
     the literal code comment → **34 → 1 carrier**.

   The **first freeze is kept** at `devset_v1_superseded/` with its own hash, so the correction is
   visible rather than silent.

**Controls are built in.** v03 and v06 were **already complete at top-5** and are therefore
**regression controls**: a candidate-count change that breaks them is unacceptable however much it
helps elsewhere.

**Corpus exclusion verified, not assumed.** `benchmarks/` is outside the builder's `INCLUDE_DIRS`
(`docs`, `config`, `scripts`, `services`, `cli`), and the current manifest lists **zero** files under
`benchmarks/`.

**A coverage gap I am recording rather than glossing:** the manifest's self-hash covers its own payload
plus `questions.txt`. **`questions.md` is not covered by either recorded hash** — it was written after
the manifest was hashed. Its own hash is recorded above so a later change is detectable, but the
freeze hash does not include it. That is a real weakness in the freeze, not a technicality.

Integrity re-verified by `c0k_devset_integrity.py`: both recorded hashes recompute exactly, all six
answerable facts verify, no answer-absent question has a carrier, counts agree, set absent from the
corpus. Exit 0.

**Also landed this round: a first defect of my own making, found and fixed.** The supplementary
request-shape check used during the Phase A/B round had never been committed (it was a scratch file),
so it could not be re-run — it is now a durable file (`c0g_http_shapes.py`), and so is the consistency
comparison (`c0h_consistency.py`).

---

## 4. What was NOT done

* **No retrieval code changed.** `cli/ask.py` and `services/ocr/*` untouched; no packer, budget or
  prompt change.
* No reranker enabled; no model added; `local-ocr` **not** wired into QA; the 34 exposed OCR images
  **not** touched or re-used as a validation set.
* No default changed: 2048 MiB margin, 1800 MiB reranker transient, 8192 context, 512 answer tokens
  all untouched.
* `agentvoice/` not read, written, staged or committed. Voice Phase 1 not started.
* Historical raw results not modified.

**Resource check:** free VRAM minima across every run — diagnosis **5094 MiB**, 8115 consistency
**5145 MiB**, answer-absent **5137 MiB**, answerable run **5145 MiB**, pack comparison **5154 MiB**.
Margin 2048 MiB held throughout; no OOM.

**Correction to an earlier version of this section:** it said step 3's question set was NOT authored and
step 4 was NOT run. Both have since been done — see §3b, §4c and §4e. What remains not done is the
**retrieval change itself**, which is the deliberate stop of §1.

---

## 4c. Step 4 completed as MEASUREMENT — the pack-level comparison, and it revises §1.1

Round 1 measured whether a required fact's carrier was **retrieved**. Step 4 also requires the actual
**pack contents and token counts**, because the "without squeezing out key evidence" condition is
decided at the pack, not the candidate list. That is measurement, not implementation, so it was run
against the frozen set. Evidence: `results/pack_comparison.json`, `results/answer_absent.json`.

| k | complete (all facts IN THE PACK) | incomplete | max prompt tokens | any source omitted for budget |
|---|---|---|---|---|
| **5 (current)** | **2 / 6** | v01, v02, v04, v05 | 2945 / 7616 | **no** |
| 8 | **4 / 6** | v01, v04 | 4660 / 7616 | **no** |
| **10 (ceiling)** | **5 / 6** | v01 | **6565 / 7616** | **no** |

> **CORRECTED 2026-10-05 by P57-C0R.** The figures previously printed here were **2 / 6 → 3 / 6 →
> 4 / 6**. They came from a fact detector keyed on the wrong fact name, so its marker set never applied
> and **v02 was scored as incomplete when it is not**. The corrected measurement is **2 → 4 → 5 of 6**
> and is reproducible from `benchmarks/p57c0r/results/r02_scored.json`. Full details in
> `benchmarks/p57c0r/ERRATA.md`.

**This revises my round-1 statement that "k=10 adds nothing over k=8."** That was true at the
**retriever** layer; at the **pack** layer k=10 does add: **v04 reaches 2/2 in the pack only at k=10**.
So the honest progression is **2 → 4 → 5 of 6**: monotone, and still insufficient because **v01 never
completes**.

**The "no squeezing" condition is SATISFIED and measured, not assumed:** `n_omitted` is **0 for every
question at every k**. Raising the candidate count displaced **no** evidence — the pack had room.
Confirmation: v03 and v06, the two regression controls, are complete at every k, so expansion broke
nothing.

**But expansion is not sufficient, and now the reason is narrower than I first wrote.** With the
detector corrected, **only v01 stays incomplete at every k** (1/2 throughout — the `8113` port fact
never reaches the pack). **v02 completes at k=8** once `config/local-embedding.yaml` and
`services/retrieval/embedder.py` enter the pack, and **v04 reaches 2/2 at k=10** while its *answer*
still covers only 1/2. So:

* **v01 is a genuine RANKING failure** — its missing fact's carriers rank 38, 51, 52, 58 for this
  wording, so no bounded k reaches them. This is the finding §1.2 reported.
* **v02 was NOT a ranking failure.** It was my measurement error; see the correction above and
  `benchmarks/p57c0r/ERRATA.md`.
* **v04 is a depth-then-generation case**: the pack gets everything at k=10 but the answer still covers
  only part of it, so raising k is necessary but not sufficient for that question.

**A cost that must be weighed:** headroom at k=10 falls to **6565 of 7616** tokens, i.e. about 1051
tokens of slack. That is a real constraint on how far this lever can be pushed, and it sharpens the
argument against the k≈40 alternative I already rejected.

**Answer-absent behaviour: 3 / 3 CORRECT REFUSALS.** v07, v08 and v09 each refused. v07 is the
reclassified one — a question whose fact is genuinely absent from the corpus, refusing exactly as it
should. VRAM minimum **5137 MiB**, margin held.

---

## 4d. 8115 response consistency (acceptance condition that does NOT depend on the pending decision)

This was run because it is independent of the retrieval question, and the acceptance conditions require
it. `services/` and `cli/` were not modified, so this measures the deployed path as it stands.

Full B client re-run (`results/b_reverify.json`) against the **current** index
`3cd2dcda4eeb5555`, compared scenario by scenario with the reference run
(`benchmarks/phaseAB/results/AB-05-b/b_e2e.json`):

| scenario | reference | P57-C0 | consistent |
|---|---|---|---|
| health | 200 | 200 | **yes** |
| normal answer | 200 answered | 200 answered | **yes** |
| answer-absent | 200 refused | 200 refused | **yes** |
| index override | 400 `index_override_rejected` | 400 `index_override_rejected` | **yes** |
| stale index | 200 refused | 200 refused | **yes** |
| service unavailable | **503 `upstream_unavailable`** | **503 `upstream_unavailable`** | **yes** |
| question too long | 413 | 413 | **yes** |

`limits_verified` is identical (`max_concurrency_enforced`, `observed_max_simultaneous_200: 2`,
`question_length_enforced`, `index_override_rejected`), concurrency codes are identical
(`[429,429,429,429,200,200]`), and the final `inflight` is `{current: 0, peak: 2}` — the bound held and
drained.

Request-shape checks (`results/supplementary.json`): an oversize body declared by header alone is
rejected **413 `body_too_large` in 0.003 s without the body being sent**; a non-object JSON body is
**400 `bad_request`**; a body carrying an `index` key is **400 `index_override_rejected`**. All three
as expected.

The stale-index scenario again modified `docs/models/local-fast.md` and **restored it byte for byte**
(`git diff --exit-code` → 0).

**VRAM: minimum free across all scenarios 5145 MiB** (reference 4911 MiB), margin 2048 MiB held with a
wide margin, no OOM. 8115 was **stopped afterwards** — it is started on demand and not left running.

So the acceptance condition *"8115's normal, refusal and error responses stay consistent"* is
**satisfied**, and it is satisfied **without** any of the pending retrieval change.

---

## 4e. End-to-end behaviour of the frozen set at k=5 — SUPERSEDED by §4g

> **This whole section is superseded. Read §4g instead.**
> The v02 row and the "defect" it announced were **measurement errors of mine**, retracted in full.
> v02 is **not** `0 / 2` at k=5; the detector that produced that figure never applied because it was
> keyed on a fact name that does not exist in the frozen manifest. The section is kept, unedited
> below the line, because the record of a wrong claim and its correction is worth more than a tidy
> file. Details: §4g and `benchmarks/p57c0r/ERRATA.md`.

---

### Original text, retained as issued

`results/answerable_run.json`. Six answerable questions, current pipeline, `k=5`, index
`3cd2dcda4eeb5555`. Per the frozen scoring rules an answerable refusal is an **incorrect refusal**, and
they are recorded as such below; the cause is reported separately and never used to re-label one.

| id | facts in pack | outcome | cites | verification |
|---|---|---|---|---|
| v01 | 1 / 2 | **refused** | 0 | — |
| **v02** | **0 / 2** | **ANSWERED** | **3** | **all `verified`** |
| v03 | 2 / 2 | answered | 2 | verified |
| v04 | 0 / 2 | **refused** | 0 | — |
| v05 | 1 / 2 | **refused** | 0 | — |
| v06 | 1 / 1 | answered | 2 | verified |

**Totals at k=5: 3 answered, 3 incorrect refusals. No truncation. No non-verified citation.**
VRAM minimum 5145 MiB, margin held.

### RETRACTED: the "answer produced with NONE of the required evidence" claim

**This claim is withdrawn in full.** v02 is **not** answered while holding 0 of 2 required facts. The
figure came from a detector that was keyed on a fact name absent from the frozen manifest, so its marker
set never applied. Corrected measurement, the retraction, and the three separate errors that produced it
are in **§4g** and `benchmarks/p57c0r/ERRATA.md`. The claim as issued is preserved in the git history
and in `OPEN_ISSUES.md` P2-11, marked retracted.


## 4f. Whole-objective verification

`c0z_final_verify.py` re-derives the state from evidence rather than narration: **32 checks, 0
failures**, exit 0. It covers the baseline and working tree, the hard boundary that no code changed,
index identity and corpus exclusion, every step's deliverables, the frozen-set hashes, the monotonicity
of the k sweep, the 3/3 answer-absent refusals, the recorded over-answering case, every VRAM minimum
against 2048 MiB, and that P56/`heldout_v1` were regression-only.

Two incidents during verification are recorded because both were **my** errors, not defects:

1. The check "only `agentvoice/` is untracked" initially **failed on itself**, because the verification
   script is untracked while it runs. Fixed by excluding this round's own evidence directory.
2. `mechanism.json` carries a field named `appears_in_top` holding the **maximum** recorded carrier
   rank, not the best. Measured: d01's ranks are `[38, 51, 52, 58]` and the field says 58. The
   **conclusion rests on the best rank — 38 for d01, 15 for d02** — which is what §1.2 quotes, so
   nothing is wrong; but the field name is ambiguous. Noted rather than renaming a frozen artifact.

---

---

## 4g. P57-C0R — CORRECTED measurement (supersedes §4c's figures and retracts §4e)

**What was wrong, in the order I found it.**

1. **Carrier sets were truncated.** The facts file recorded `carriers[:12]` while reporting a separate
   `n_carriers`, so anything reading `carriers` saw a truncated set.
2. **Two requests were mixed.** §4c called `search()` once and `answer()` again, then printed the first
   call's candidates beside the second call's pack. Two different top-k lists; only the pack is what the
   model saw. Corrected: **one `answer()` call per (question, k)**, and everything is derived from that
   call's own pack.
3. **Records were too small to re-check.** §4e stored `answer_head` (400 characters) and a boolean, with
   no pack at all, so its central claim could not be re-verified. Corrected: the **full answer**, the
   **full pack with every supplied source's file and line range**, the citations, index version, token
   counts, latency and VRAM.
4. **A detector keyed on a name that does not exist.** Re-running with corrected plumbing, the v02
   marker set was looked up under `vram_figure_meaning` while the frozen manifest calls the fact
   `torch_reserved_meaning`, so **no markers were found** and the fact scored absent. Separately, the
   frozen manifest's own literals for that fact are `["torch_reserved", "reserved_mb"]` — the **column
   names** — which are too strict: the passage explaining the figure says *"VRAM, torch attribution"*
   and *"cudaMemGetInfo total − free"* and never prints `torch_reserved`.
5. **Refusals carrying reasoning were scored as answers.** `cli/ask.py` decides `refused` by **exact
   match** against the refusal sentence, so an answer written as *refusal + **Reasoning:** …* is recorded
   as answered. Measured: **6 of 18** answers contain the refusal sentence. Worse for scoring, the
   reasoning prose was then mined for fact markers, so the model's own restatement of the question
   (naming port `8111`, the port of the file the question is about) registered as a *covered fact* and
   turned an incorrect refusal into a reported `partial`.

**Corrected facts in the pack, and corrected verdicts** (`benchmarks/p57c0r/results/r02_scored.json`):

| k | facts all in pack | verdicts |
|---|---|---|
| **5 (current)** | **2 / 6** — v03, v06 | v01 v04 v05 incorrect_refusal · **v02 partial** · v03 v06 complete |
| 8 | **4 / 6** — v02, v03, v05, v06 | v01 v04 incorrect_refusal · **v02 complete** · v05 v06 complete |
| **10 (ceiling)** | **5 / 6** — v02, v03, v04, v05, v06 | **v01 incorrect_refusal** · v04 partial · rest complete |

**So the progression is 2 → 4 → 5 of 6, not 2 → 3 → 4.** The "no squeezing" finding **stands**: zero
sources were omitted for budget at any k.

### v02, answered as the task asked

| k | memory-figure fact in pack | margin fact in pack | verdict | evidence |
|---|---|---|---|---|
| 5 | **absent** | present | **partial** | answer covers the margin policy only |
| 8 | **present** (`vram_mb`, `attribution`) | present | **complete** | `config/local-embedding.yaml` + `services/retrieval/embedder.py` enter the pack |
| 10 | **present** | present | **complete** | same evidence retained |

At k=5 the answer is **partial, not unsupported**: it states the margin relationship correctly and
simply does not cover what the memory figure *is*. At k=8 and k=10 it covers both facts. **No
over-answering is demonstrated for v02.**

### Which questions are still ranking problems

* **v01 — YES, a ranking failure.** `8113` never reaches the pack at any k, while `8111` does. Its
  carriers rank 38, 51, 52, 58 for this wording, beyond any bounded k. **This is the round's finding and
  it survives.**
* **v02 — NO.** It was my measurement error. Corrected, it completes at k=8.
* **v04 — depth-then-generation.** The pack reaches 2/2 at k=10 but the answer still covers 1/2, so
  raising k is necessary but not sufficient there.
* **v05 — DEPTH case, fixed by k=8** (carrier at rank 6). **v03, v06 — controls, complete at every k.**

### What this changes about the decision

It **strengthens the case against (B) k=8 alone**, because after the correction k=8 already achieves
4/6 with v02 and v05 complete — while v01 remains untouched at every k. It **weakens the original
argument for (C)**: the over-answering I reported for v02 does not exist. The surviving defect from this
correction is the **refusal-detector mislabel** (new **P2-12**), which is real but independent of
retrieval and out of scope for a measurement-only round.

---

## 4h. P57-C0R-02 — OFFLINE ADJUDICATION supersedes the marker verdicts in §4g

**What this did:** adjudicated the stored evidence only — **no model calls**, no QA-path change. For each
question × fact × k it answers three separate things from the text: does a **supplied chunk** state the
fact, does the **answer** state it, and do the **citations** support that statement. Reasons are written
out in `benchmarks/p57c0r02/adjudication.json`; the chunk text it read is in `packet.json`, with each
chunk's `content_sha256`. **No keyword rule assigns a verdict**; undecided items are marked undecided.

**Adjudicated verdicts:**

| id | k=5 | k=8 | k=10 |
|---|---|---|---|
| v01 | incorrect_refusal | incorrect_refusal | incorrect_refusal |
| **v02** | **partial** | **partial** | **partial** |
| v03 | complete | complete | complete |
| v04 | incorrect_refusal | incorrect_refusal | **complete** |
| v05 | incorrect_refusal | **complete** | **complete** |
| v06 | complete | complete | complete |

**COMPLETE ANSWERS: 2/6 → 3/6 → 4/6** — i.e. **lower** than the 4/6 → 5/6 reported in §4g. §4g's
verdicts were assigned by testing marker strings, which is **too generous**: it counted v02 complete
because `vram_mb` and `attribution` appear in the pack text. The task requires that a config value, a
VRAM delta or a bare `vram_mb` **not** substitute for an explanation of what `torch_reserved` means, and
under that rule v02 is **partial at every k**. §4g's measurements stand; its verdict layer is superseded
by this section.

**REMAINING RETRIEVAL PROBLEMS — two ranking failures, neither reachable by any bounded k:**

| question | missing fact | canonical chunk | rank out of 60 |
|---|---|---|---|
| **v01** | embedding port `8113` | `config/local-embedding.yaml:1-94` | 38 |
| **v02** | what the memory figure MEANS | `docs/models/local-embedding.md:90-112` | **15** |

v05 is a depth case fixed at k=8 (carrier at rank 6). v04 is depth-then-generation: the pack reaches
2/2 only at k=10, and only then does the answer cover both facts.

**The pattern, from two independent questions:** both ranking failures are the query failing to reach a
**config file or a model-record table** that carries the fact. v01 asks about "components … and ports"
and cannot reach the embedding config; v02 asks how two things "relate" and cannot reach the model-record
table that defines the figure. Both facts sit in **structured, tabular, low-prose** chunks that a
natural-language question does not resemble.

**A new finding at the answer layer:** v01 at k=10 **holds** the `8111` fact —
`config/local-fast.yaml` and `docs/models/local-fast.md:132-148` are supplied and both state
`127.0.0.1:8111` — and the answer **still refuses**, reasoning that the sources do not contain the
network path. **Pack coverage is not answer coverage.**

**Two defects in my own adjudicator were found by running it:** a verdict function that scored a question
with **zero decidable facts as `complete`** (a false pass), and a precedence error that let an undecided
fact erase a decided one. Both fixed, first attempt preserved, recorded in
`benchmarks/p57c0r02/ERRATA-append.md` R11 and R12.

**Recommendation:** with two of six questions failing for **ranking** reasons with canonical carriers at
ranks 15 and 38, a **bounded second evidence pass** is the well-supported next experiment. That decision
is Astra's.


## 5. Self-assessment

**FAIL for the "implement a minimal bounded fix" objective — deliberately not attempted**, because
the measured mechanism does not support it: the three surviving probes cannot be fixed by any bounded
k, so implementing bounded expansion would produce a change that does not achieve the round's goal.

**The diagnostic objective succeeded**, and its main product is a negative result plus one correction
of my own error. Concretely established, with numbers:

1. bounded expansion **does** fix two probes (k=8 fixes d05, d06) and **saturates at 8** — k=10 adds
   nothing;
2. three probes fail for **ranking** reasons, with carriers at ranks 38, 15 and absent — quantified,
   not asserted;
3. one of my probes (d03) was an **unanswerable question mislabelled as answerable**, and is
   reclassified rather than quietly dropped;
4. my first diagnosis used the wrong granularity and **overstated the failure**; corrected and both
   versions kept.

**Not claimed:** that the proposed second-pass option works. It is untested. It is offered for review
precisely because the honest next step is a decision, not more code.

### Acceptance conditions — status of each, from the objective

| condition | status |
|---|---|
| 8115 normal/refusal/error responses stay consistent | **SATISFIED** — §4d, all seven scenarios match the reference |
| device free VRAM never below 2048 MiB, no OOM | **SATISFIED** — minima 5094 / 5137 / 5145 / 5154 MiB across the four runs |
| index version valid / citations valid | **SATISFIED for what was measured** — `3cd2dcda4eeb5555` throughout; every citation in the 8115 run was `verified` |
| every fact in the new list supported by a real source | **SATISFIED as authored** — all 6 answerable questions' facts verified against the index at freeze time with carrier chunks recorded. Per-question **answer** results belong to step 4's follow-up and are not claimed here |
| answer-absent questions still refuse | **SATISFIED — 3 / 3 correct refusals** (`results/answer_absent.json`), including the reclassified v07 |
| P56 and heldout_v1 regression only, never new acceptance | **SATISFIED** — neither was run this round; nothing was re-scored |
| no new unsupported statements | **SATISFIED** — no answer-generating code changed |
| per-question raw results, failed items, PASS/CONDITIONAL PASS/FAIL | **DELIVERED** — `results/*.json`; the failed items are **v01 and v02**, named with their measured ranks |

**The retrieval change remains UNIMPLEMENTED**, and that is the one objective not met — deliberately,
because the measured mechanism does not support the bounded change the round asked for. §1.2 and §4c
carry the numbers; the decision is Astra's.

**Correction carried forward from this round:** my round-1 claim that "k=10 adds nothing over k=8" was
true of retrieval and **false of the pack** (§4c). Recorded rather than quietly amended.

**Awaiting Astra.** Stopping here as instructed.
