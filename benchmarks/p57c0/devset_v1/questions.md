# P57-C0 development-validation question set (`p57c0_devset_v1`)

**Classification: DEVELOPMENT VALIDATION — explicit, and not a formality.** The construction side
wrote these questions from the same corpus it maintains, so independence is limited. **This is NOT an
independent held-out set**, no percentage computed on it supports a general claim, and it must not be
presented as one. P56 and `heldout_v1` remain **regression-only** and were not run for this set.

| item | value |
|---|---|
| set id | `p57c0_devset_v1` |
| **manifest SHA256** | `aa44a4f870dc10bc831224e500931dbe8544426bcdd0617f32bf1690325e1c51` |
| **questions.txt SHA256** (LF) | `1ce9966dd4a583fa8fd3231347c97366b24451d8dad3472995b13acc9f999094` |
| **SET SHA256 — covers all three files** | see `SET_HASH.json` (authoritative) |
| index at freeze | `3cd2dcda4eeb5555`, 54 files / 595 chunks |
| answerable | **6** (v01–v06) |
| answer-absent | **3** (v07–v09) |
| files | `manifest.json`, `questions.txt`, `questions.md`, `SET_HASH.json` |

**Freeze coverage, and the gap that was closed.** The manifest's self-hash covers its own payload plus
`questions.txt`. **`questions.md` was written after that hash and was therefore not covered by it** —
a real weakness, since this file carries the classification, the scoring rules and the discriminator
audit. It is now covered by **`SET_HASH.json`**, which hashes all three files (LF-normalised) and is
recorded above. The addition is **additive**: no frozen artifact was modified to produce it, so the
manifest hash and the `questions.txt` hash are unchanged and nothing already published is invalidated.
Verify all of it with `c0k_devset_integrity.py`.

**Frozen before any retrieval-code change, as step 3 requires.** No file under `cli/`, `services/` or
`config/` was modified to produce this set. Changing a question, a fact, or a class means a **new set
under a new name** — not an edit.

**Excluded from the index.** `benchmarks/` is outside the builder's `INCLUDE_DIRS`
(`docs`, `config`, `scripts`, `services`, `cli`), and it is **verified** rather than assumed: the
current manifest lists **zero** files under `benchmarks/`.

---

## Why every fact here was VALIDATED before freezing

The previous round's probe set was partly wrong, and the way it was wrong is worth encoding:

* **d03 asked for a value that is not in the corpus at all.** `EVIDENCE_FLOOR = 0.30` lives in
  `cli/ask.py`, and `cli/` is excluded from the corpus by design — `0.30`, `EVIDENCE_FLOOR` and
  `evidence_floor` each appear in **0 chunks**. It was mislabelled as answerable.
* A fact's **carrier file** being absent from a retrieval result proves nothing if **another** carrier
  was retrieved — `8113` alone lives in 15 indexed chunks.

So this set was built the other way round: each fact was **searched for in the index first**, and a
question entered the answerable class only if every one of its facts had at least one real carrier
chunk. The canonical citation target for each fact is recorded in the manifest, with all carriers.

**The discriminators were then audited for discriminating power**, and two were too loose to be sound
(`c0j_audit_facts.py`):

| fact | first attempt | problem | tightened to | carriers before → after |
|---|---|---|---|---|
| `cycle_latency` | `["22","24"]` | bare `"22"` matched 26 chunks and `"24"` matched **49** — dates and versions | must co-occur, or `24.4` / `22-24` | **64 → 5** |
| `why_raised` | `["2000","truncat"]` | bare `"2000"` matched 10 chunks, mostly unrelated years | `"Raised from 2000 to 8000"`, `"capped every chunk at 2000"` | **34 → 1** |

A fact that matches 64 chunks does not identify evidence. Without this audit the set would have
"passed" while resting on accidental substring hits. The first freeze is kept at
`devset_v1_superseded/` with its hash, so the correction is visible rather than silent.

---

## The questions

### Answerable (v01–v06)

| id | class | question | required facts → canonical source |
|---|---|---|---|
| v01 | cross-file | Which components does a question pass through before reaching local-fast, and which ports do they use? | `fast_port` → `config/local-fast.yaml:1` (11 carriers) ｜ `embed_port` → `config/local-embedding.yaml:1` (15) |
| v02 | cross-file | How do the embedding service memory figures and the scheduler margin relate to each other? | `torch_reserved_meaning` → `docs/models/local-embedding.md:90` (13) ｜ `margin_policy` → `config/local-fast.yaml:1` (29) |
| v03 | cross-file | What are the two unload figures on the staged RAG path, and what latency does a full cycle take? | `unload_figures` → `docs/models/local-embedding.md:13` (3) ｜ `cycle_latency` → `docs/phase55r-evidence-consolidation.md:75` (5) |
| v04 | single-file | What per-chunk character cap does the pack apply, and why was it raised? | `char_cap` → `services/retrieval/retriever.py:25` (2) ｜ `why_raised` → `services/retrieval/retriever.py:25` (1) |
| v05 | cross-file | Which ports are reserved for the reranker and the OCR service? | `reranker_port` → `config/local-reranker.yaml:1` (14) ｜ `ocr_port` → `config/local-ocr.yaml:1` (9) |
| v06 | single-file | How does the reranker turn the model's output into a relevance score? | `score_method` → `config/local-reranker.yaml:1` (13) |

**Carry-over and controls.** v01 is d01, v02 is d02, v03 is d04, v04 is d05, v05 is d06, v06 is d07
from the diagnostic. v03 and v06 were **already complete at top-5** and are therefore **regression
controls**: if a candidate-count change breaks them, the change is not acceptable however much it
helps elsewhere. v04 is a **depth** case (carrier at rank 7) and v05 is too (carrier at rank 6); v01
and v02 are the **ranking** cases that bounded k could not reach (carriers at ranks 38 and 15).

### Answer-absent (v07–v09) — a correct system REFUSES

| id | question | why the corpus cannot answer it |
|---|---|---|
| v07 | What numeric evidence-floor value does the QA path use to reject weak retrieval, and where is it configured? | `EVIDENCE_FLOOR = 0.30` is in `cli/ask.py`, and `cli/` is excluded from the corpus. Measured: **0 chunks** contain `0.30`, `EVIDENCE_FLOOR` or `evidence_floor`. **Reclassified from answerable** — this was my error. |
| v08 | What monthly cost in RMB does running local-fast incur on this machine? | No cost or billing figure exists anywhere in the corpus; the project is local-only and records no monetary data. |
| v09 | Which certificate authority issues the TLS certificate for the OCR service? | No TLS endpoint exists for `local-ocr`: it binds `127.0.0.1`, and the corpus records no certificate. |

---

## Pre-registered scoring

* **Answerable**: a question passes only when **every** fact is supported by a **cited** chunk whose
  citation verifies against the index. **A refusal is an INCORRECT refusal**, whatever its cause; the
  cause is recorded separately and never used to re-label the outcome.
* **Answer-absent**: **refusal is the correct outcome**; any answer at all is a failure.
* **Partial**: some facts supported and others not → recorded as **partial**, not as complete.
* **No accuracy figure.** Report counts per class. Nothing here may be combined with the P56,
  `heldout_v1` or OCR numbers.

## What this set can and cannot support

* It is **9 questions**. It supports per-question verdicts and per-class counts, and **no** general
  claim about cross-document retrieval.
* It is written by the construction side. **Independence is limited and is stated, not implied.**
* It is now **exposed**. Any future acceptance needs a new unexposed set; this one becomes
  regression-only from the moment its results are read.
