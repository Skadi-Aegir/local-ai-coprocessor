# P57-C0R-02 — ERRATA (append)

Corrections arising from the **offline evidence adjudication**. Appended to the P57-C0R errata rather
than replacing it: R1–R9 stand, and these are additional. No model was called, no QA-path file changed,
and the frozen set is untouched.

---

## R10 — the marker-based verdicts were too generous, and strict adjudication lowers them

**Was (P57-C0R):** verdicts assigned by testing marker strings against pack and answer text, giving
`v02 complete` at k=8/10 and **complete answers `2/6 → 4/6 → 5/6`**.

**Now (P57-C0R-02):** adjudication by reading the text, with a config value or a bare `vram_mb`
explicitly **not** accepted as an explanation (as the task requires), gives
**complete answers `2/6 → 3/6 → 4/6`**.

**What changed:** **v02 becomes `partial` at every k**, because the fact "what the embedding memory
figure means" is never satisfied — at k=5 it is absent, and from k=8 the pack carries the config values
and a comment distinguishing two figures but never states what the field **is**. The previously reported
`5/6` at k=10 counted v02 as complete on the strength of `vram_mb` and `attribution` appearing in the
pack text, which is precisely the substitution the task rules out.

**Not a reversal of the ranking finding.** The canonical explanation chunk
(`docs/models/local-embedding.md:90-112`) ranks **15** for v02's question, so v02 is a ranking failure.

## R11 — my adjudicator produced a FALSE PASS on its first run

**Was:** `verdict_for` computed `states = [v[1] for v in per_k.values() if isinstance(v, tuple)]`. By
that point `per_k` holds **dicts**, so the comprehension produced an **empty list** and
`yes == len(states)` became **`0 == 0` → `complete`**.

**Measured effect:** **v02 was reported `complete` at all three ks** with both of its facts undecided —
a question with **zero decidable facts scored as fully answered**. This is the same failure category the
round exists to remove, appearing inside the correction.

**Fixed:** reads the dicts; an empty fact set now returns `error_no_facts` rather than passing
vacuously. First attempt preserved as `adjudication.first-attempt-false-pass.json`.

## R12 — wrong precedence between `partial` and `uncertain`

**Was:** with one fact `yes` and another `uncertain`, the first fix returned `uncertain` — letting an
undecided fact **erase** a decided one.

**Now:** explicit precedence. A definite `yes` beside an undecided fact is **`partial`**; `uncertain` is
reserved for the case where **nothing** was decided either way. Consequence: v02 is `partial`, not
`uncertain`, at k=8 and k=10.

## R13 — my first v04 adjudication was wrong because I searched a phrase instead of reading

**Was:** `why_raised` at k=10 recorded as `answer_states_fact = no`, on the strength of a literal phrase
search that missed the wording.

**Reading the answer:** *"It was raised from 2000 to prevent **silent truncation** of answers within
retrieved chunks, which previously converted retrieval successes into misses."* That **is** the reason,
and it is supported by the cited chunk. Corrected to **`yes`**, which makes **v04 `complete` at k=10**.

## R14 — a fact CAN be in the pack and the answer still deny it

**New finding, not a correction:** **v01 at k=10**. `config/local-fast.yaml:1-119` and
`docs/models/local-fast.md:132-148` are both supplied and both state `127.0.0.1:8111`, yet the answer
refuses and reasons that the sources *"do not contain information about … the network path (other than
the loopback address 127.0.0.1)"*. The model held the fact and reported it absent, and it cited
`local-fast-api.md:1-9` and `start-local-fast.ps1` instead of the two chunks carrying the port.

**Why it matters:** "the fact is in the pack" is not the same as "the answer uses it", so a
pack-coverage metric alone cannot stand in for an answer-level verdict.

## R15 — the disagreement between P57-C0R and P57-C0R-02 is the point

P57-C0R reported `2/6 → 4/6 → 5/6`; P57-C0R-02 reports `2/6 → 3/6 → 4/6`. **Neither is a reversal of
the other's measurements** — they apply different standards to the same stored evidence. P57-C0R asked
"do the markers appear"; P57-C0R-02 asks "does the text state the fact, and does a citation support it".
The second is what the task requires, and it is stricter. **Where they disagree, this document is
authoritative.**

---

## What did NOT change

* The frozen set: `manifest.json` and `questions.txt` byte-identical, all three recorded hashes verify.
* The stored records: the 18 answers, their supplied source IDs and their citations are the inputs, not
  modified. `packet.json` records each chunk's `content_sha256` for independent checking.
* The QA production path: nothing under `cli/`, `services/` or `config/` was touched.
* `agentvoice/`, the OCR work, the reranker and every default are untouched.
