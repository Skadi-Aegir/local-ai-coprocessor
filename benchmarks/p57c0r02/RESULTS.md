# P57-C0R-02 — offline evidence adjudication

**Scope:** adjudicate the stored evidence only. **No model calls**, no change to the QA path, no
retrieval change. Baseline `f3f1509`. Fixed index `3cd2dcda4eeb5555`, frozen set `p57c0_devset_v1`.

**Inputs:** the 18 stored records (full answers, same-request supplied source IDs, citations) from
`benchmarks/p57c0r/results/r02_scored.json`, plus each supplied chunk's **text fetched from the fixed
index by ID**. Assembled into `packet.json` (each chunk's `content_sha256` is recorded there, so any
reader can confirm the text this adjudication read).

**Method:** for every question × fact × k, three separate questions are answered from the text —
does a **supplied chunk** state the fact, does the **answer** state it, and do the **citations** support
that statement — with the reason written out in `adjudication.json`. **No keyword rule assigns a
verdict.** Where the text does not settle it, the verdict is `uncertain` and says so.

---

## 1. Two defects in my adjudicator, found while running it

**F1 — a false pass, of exactly the kind this round exists to prevent.** The first `verdict_for` did
`states = [v[1] for v in per_k.values() if isinstance(v, tuple)]`, but by that point `per_k` holds
**dicts**, so the comprehension yielded an **empty list** and `yes == len(states)` became `0 == 0` →
**`complete`**. **A question whose facts were all undecided was scored complete.** Measured effect:
**v02 was reported `complete` at all three ks.** Fixed, and an empty fact set now returns
`error_no_facts` instead of passing vacuously. First attempt kept:
`adjudication.first-attempt-false-pass.json`.

**F2 — wrong precedence between `partial` and `uncertain`.** With one fact decided `yes` and another
`uncertain`, the first fix returned `uncertain`. That lets an undecided fact **erase** a decided one.
Precedence is now explicit: a definite `yes` alongside an undecided fact is **`partial`**; `uncertain`
is reserved for the case where **nothing** was decided either way.

**Also corrected:** my first pass at v04 recorded `answer_states_fact = no` for `why_raised` on the
strength of a phrase search. Reading the answer, it says *"It was raised from 2000 to prevent **silent
truncation** of answers within retrieved chunks, which previously converted retrieval successes into
misses"* — that **is** the reason. Corrected to `yes`. A phrase search is not a reading.

---

## 2. v02, re-adjudicated as instructed

The instruction is explicit: **a configuration value, a VRAM delta, or `vram_mb` appearing on its own
does not substitute for an explanation of what `torch_reserved` means.** Applying that:

| k | supplied | memory-figure fact | margin fact | verdict |
|---|---|---|---|---|
| 5 | 5 chunks | **absent** — none of the five mentions `torch_reserved`, `reserved_mb`, `vram_mb`, `cudaMemGetInfo`, `attribution` or "VRAM delta" | present & stated | **partial** |
| 8 | 8 chunks | **uncertain** — `config/local-embedding.yaml` supplies the values and a comment distinguishing "torch's own attribution ~2361 MiB" from "measured delta 1246 MiB", but nothing says what the field **is** | present & stated | **partial** |
| 10 | 10 chunks | **uncertain** — same evidence; k=10 adds `services/retrieval/embedder.py`, a carrier, but the answer is materially unchanged and still relays values rather than explaining the field | present & stated | **partial** |

**Evidence spans.** At k=5 the fact is absent outright. From k=8 the pack carries:

> `config/local-embedding.yaml` — *"…needs on top of the desktop baseline. Measured delta 1246 MiB;
> # torch's own attribution is ~2361 MiB. Keep both in mind for scheduling. expected_vram_mb: 2400,
> measured_vram_delta_mb: 1246"* (chunk sha `8f99f6aa4695134c`)

and the answer states *"a configured `expected_vram_mb` of 2400 MiB and a measured delta of 1246 MiB
[SOURCE 7]"*.

**Why `uncertain` and not `yes`:** that comment distinguishes two figures. It does not explain what the
`/stats vram_mb` field **is**. The passage that does —
`docs/models/local-embedding.md:90-112`, *"`vram_mb` is cudaMemGetInfo total − free, not PyTorch's
attribution to this process"* — **was never supplied at any k**. So the pack holds values and a hint,
not the explanation, and the answer relays values without explaining the field.

**Why `uncertain` and not `no`:** a config comment that explicitly separates *torch attribution* from
*measured delta* and says to keep both in mind is not nothing — it is partial semantic grounding. The
task forbids forcing this either way, so it is recorded as undecided with the reason.

**And the reason it never arrives is RANKING, not depth.** Measured: the canonical explanation chunk
`docs/models/local-embedding.md:90-112` (id `d4318e9908902794db33`) ranks **15** for v02's question out
of 60. Beyond any bounded k.

---

## 3. Per-question adjudicated results

| id | question | k=5 | k=8 | k=10 |
|---|---|---|---|---|
| **v01** | components and ports to `local-fast` | incorrect_refusal | incorrect_refusal | **incorrect_refusal** |
| **v02** | embedding memory figures ↔ scheduler margin | **partial** | **partial** | **partial** |
| v03 | staged-RAG unload figures and cycle latency | complete | complete | complete |
| v04 | pack character cap and why raised | incorrect_refusal | incorrect_refusal | **complete** |
| v05 | ports for reranker and OCR | incorrect_refusal | **complete** | **complete** |
| v06 | how the reranker scores | complete | complete | complete |

**Counts, complete answers:** `2/6 → 3/6 → 4/6`.
*(v02 is partial rather than complete at every k under strict adjudication, so this is **lower** than
the 4/6→5/6 the previous round reported from a marker rule. The marker rule was too generous.)*

### Fact-level findings worth keeping

* **v01 at k=10 contains the `8111` fact and the answer still denies it.** `config/local-fast.yaml:1-119`
  and `docs/models/local-fast.md:132-148` are supplied and both state `127.0.0.1:8111` — yet the answer
  refuses, reasoning that the sources *"do not contain information about … the network path (other than
  the loopback address 127.0.0.1)"*. The model held the fact and reported it absent. It also cited
  `local-fast-api.md:1-9` and `start-local-fast.ps1` rather than the two chunks carrying the port.
* **v01's `8113` never reaches the pack at any k** while `8111` does. Carriers at ranks **38/51/52/58**.
* **v05 at k=5 contains `8114` and still refuses** — a pure depth case that k=8 fixes.
* **v04 is depth-then-generation in reverse:** the pack only reaches the fact at k=10, and at k=10 the
  answer covers both facts, so k=10 resolves it.
* **v03 and v06 are clean controls** at every k, with chunk/answer/citation aligned.

---

## 4. Remaining retrieval problems

**Two questions are RANKING failures, and neither is reachable by any bounded k:**

| question | missing fact | canonical chunk | rank out of 60 |
|---|---|---|---|
| **v01** | embedding port `8113` | `config/local-embedding.yaml:1-94` | 38 (carriers 38/51/52/58) |
| **v02** | what the memory figure *means* | `docs/models/local-embedding.md:90-112` | **15** |

**One is a depth case:** v05's `8116` (carrier at rank 6), fixed at k=8.

**One is depth-then-generation:** v04 reaches 2/2 in the pack only at k=10, and only then does the
answer cover both facts.

**The pattern, now supported by two independent questions:** in both ranking failures the query's
*wording* fails to reach a **config file or a model-record table** that carries the fact. v01 asks about
"components … and ports" and cannot reach the embedding config; v02 asks how two things "relate" and
cannot reach the model-record table that defines the figure. Both facts sit in **structured, tabular,
low-prose** chunks that a natural-language question does not resemble.

**Also standing, independent of retrieval:** `cli/ask.py` records `refused: False` when the model
writes *refusal + reasoning* — measured, 6 of 18 answers carry the refusal sentence, including v01 at
k=8 and k=10. Kept as **P2-12**, not touched this round.

---

## 5. Self-assessment: **CONDITIONAL PASS**

**PASS** on the task as scoped. The adjudication is offline, uses the frozen set and the fixed index,
records chunk text with hashes, answers, citation mappings and written reasons, and marks undecided
items undecided rather than forcing them. Every claim is reproducible from `packet.json` and
`adjudication.json`. **No model was called and no QA-path file changed.**

**CONDITIONAL**, for three reasons I am recording rather than hiding:

1. **My adjudicator produced a false pass on its first run** — a question with no decidable facts scored
   `complete` (F1). That is the same category of error this whole round is correcting, appearing inside
   the correction itself.
2. **My first v04 adjudication was wrong because I searched for a phrase instead of reading the
   answer** (F2). Two of my adjudication errors were caught only by re-reading text I had already
   processed once.
3. **v02's undecided fact is a genuine judgement call, not a measurement.** I have explained why
   `uncertain` rather than `yes` or `no`, but a human reviewer may decide the config comment is
   sufficient, or plainly insufficient. **It is not settled by this document.**

**Not claimed:** that the marker-based figures of P57-C0R were wrong in direction — they were wrong in
*generosity*, and this round is stricter. Nothing here measures a second retrieval pass.

**Recommendation:** with two of six questions failing for **ranking** reasons with canonical carriers at
ranks 15 and 38, a **bounded second evidence pass** is now the well-supported next experiment — it is
the only remaining lever that addresses facts in low-prose structured chunks. That decision is Astra's.
