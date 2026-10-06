# P56 per-question review

**Run:** one pass over the frozen 10 questions, `benchmarks/p56/frozen_questions.txt`.
**Entry point:** `cli/ask.py` (dense retrieval + local-fast, reranker OFF).
**Raw records:** `results/p56_answers.json`. **Log:** `logs/p56-run.txt`.

## Run-level results

| measure | value |
|---|---|
| questions | 10 |
| answered | **6** |
| refused | **4** (q05, q08, q09, q10) |
| errors / failures | **0** |
| answered with at least one parsed citation | **6 of 6** |
| reranker used | **no** (off by default) |
| end-to-end latency | min **0.687 s**, max **4.722 s** |
| free VRAM | **min 3275 MiB** over 75 samples @200 ms, **margin (2048 MiB) held throughout** |

**How "refused" is counted, and the limitation.** The detector is a substring check for the refusal
sentence. All four refusals are **exactly 68 characters** — the bare sentence and nothing else — so
the classification is unambiguous here. The weakness is that a reply which *quotes* the refusal
sentence inside a real answer would be flagged; q04 quotes it in prose, and was verified by hand
rather than by the flag. A future revision should require the refusal to be the whole reply.

## Human verification of support

Each answer was checked against the chunks it cites. "Supported" means every factual claim in the
answer is traceable to a cited chunk.

| id | category | outcome | supported? | note |
|---|---|---|---|---|
| q01 | answerable | answered | **yes** | 8111 and 8192, cites `docs/models/local-fast.md:132-148`, which contains both |
| q02 | answerable | answered | **yes** | `last_token` pooling + dimension 1024, cites `config/local-embedding.yaml` and `services/retrieval/embedder.py` |
| q03 | answerable | answered | **yes** | strategy `structured` and AST-based meaning, cites `services/retrieval/chunker.py` and `docs/retrieval-architecture.md` |
| q04 | answerable | **partly answered** | **partly** | port 8113 correct and cited; the answer explicitly says the **response format is not defined in the sources** and cites 1, 3 and 5. Recorded as a **PARTIAL ANSWER**, not a pass — see caveat 1 |
| q05 | cross-document | **refused** | n/a — refusal correct | top-5 did not gather the evidence, and the corpus lacks port 8114 — see caveat 2 |
| q06 | cross-document | answered | **yes** | escalation to DeepSeek per `docs/escalation-policy.md:78-90`; the answer honestly notes the policy does not name a specific DeepSeek instance |
| q07 | answerable | answered | **yes** | 1:1 `faiss_id` mapping and renumbering on rebuild, cites two ranges of `services/retrieval/index_store.py` |
| q08 | answer-absent | **refused** | n/a — **correct** | no monetary cost data exists in the corpus |
| q09 | answer-absent | **refused** | n/a — **correct** | no temperature reading was ever persisted |
| q10 | answer-absent | **refused** | n/a — **correct** | `local-smart` is locally served; no cloud provider is named |

**Answer-absent behaviour: 3/3 correct refusals, 0 fabrications.** This is the most important
result in the run — the entry point declined rather than inventing a cost, a temperature, or a cloud
provider.

**Overall: 6 answered (all 6 with parsed citations and hand-verified support), 4 refused of which
3 are correct answer-absent refusals and 1 (q05) is a correct refusal of a question that needs
multi-hop synthesis.** No fabrication occurred anywhere in the run.

## Caveats, stated rather than smoothed over

**1 — q04 is a PARTIAL ANSWER, and the answer says so.** The corpus documents the `/health`
endpoint path and port but never the response body. The answer states the return value is "not
explicitly defined in the provided sources" and cites sources 1, 3 and 5. Because the question asked
for two things and only one is in the corpus, this is recorded as a **partial answer**, not as a
clean pass. The question is kept unchanged (the set is frozen); the partial outcome is the finding.

**2 — q05's refusal is correct, and the CAUSE is more specific than "multi-hop".** Verified
directly:
* port **8111** appears in 9 chunks across 7 files; port **8113** in 9 chunks across 8 files;
  **no chunk contains both**, and no file does either
* **port 8114 appears in ZERO chunks** — the reranker's port is not in the clean corpus at all
* the **top-5 pack did not gather the evidence**: the two useful chunks
  (`config/local-fast.yaml` for 8111, `docs/retrieval-architecture.md:151-165` for 8113) exist and
  are individually retrievable, but the returned top-5 did not contain both

**Correct attribution, stated precisely:** this is a **top-5 evidence-gathering shortfall on this
question combined with a corpus gap for port 8114**. It is **NOT** recorded as a proven multi-hop
defect. The earlier P56 review label "requires multi-hop synthesis" over-attributed the cause: the
refusal is correct and safe, but what is established is that *this pipeline's top-k did not gather
the needed evidence*, plus *the corpus lacks 8114*, not that a multi-hop capability was
demonstrated to be the missing mechanism. Whether a larger k, a different chunking, or genuine
multi-hop retrieval would fix it is **untested and unknown**. It is listed as a known limitation.

**Refusal-matching defect found in this same review, and fixed.** The review initially reported
"7 answered / 3 refused" by counting a substring flag, while the records showed 6/4. The detector was
a substring test and q04 both *quotes* the refusal sentence and answers, so it was one class away
from being miscounted. A bare refusal is exactly 68 characters and equals the sentence, so the true
count is **6 answered / 4 refused**, and the entry point now uses an **exact match** rather than a
substring test.

**New safety rule added in the P56 wrap-up:** an answer with **no valid in-range source number is now
refused** rather than returned. In the regression run no record triggered it
(`unsupported_draft` empty) and no out-of-range source numbers were reported, so the rule changed no
verdict — it is a guard against returning an untraceable answer, not a behaviour change on this set.

**3 — the CLI's first version loaded a second embedding model in-process**, which dropped free VRAM
to **1942 MiB**, below the margin. Fixed by reusing the running 8113 service for query embedding
(`--embedder service`, now the default). The measured run above uses the service path and never left
the margin. `--embedder in-process` still exists and is documented as margin-breaching.

**4 — the RAG check is a task-definition reference check.** The check is citation presence plus
manual support verification, not an automated quality metric. No answer-quality claim is made here
beyond what the support column states.

## P56 wrap-up regression (after the safety patch)

Re-ran the **same frozen 10 questions** against the patched entry point. Records:
`results/p56_answers_regression.json`; log: `logs/p56b-run.txt`.

| | original run | regression after patch |
|---|---|---|
| answered | 6 | **6** |
| refused | 4 | **4** |
| errors | 0 | **0** |
| verdicts changed | — | **NONE** (q01–q10 identical) |
| records with ≥1 citation | 6 | **6** |
| unsupported drafts refused for missing citation | n/a | **0** |
| out-of-range source numbers | n/a | **none** |
| reranker used | no | **no** |
| embedder | service | **service** |
| end-to-end latency | 0.687–4.722 s | **0.389–4.390 s** |
| free VRAM minimum | 3275 MiB (75 samples) | **3248 MiB (62 samples)** |

**The safety patch changed no verdict**, which is the intended outcome: it removes two unsafe
options and adds a guard against returning untraceable answers, without altering what the pipeline
answers on this set. The margin (2048 MiB) held throughout both runs.

**Smoke check** (separate from the set): one answerable question returned 8111 and 8192 with
`docs/models/local-fast.md:132-148` cited, 5.67 s, free VRAM 3259 MiB. Both disabled flags were
verified to refuse with exit code 2.

## What was NOT fixed this round

Only defects this question set exposed were addressed (the in-process embedder and the citation
parser). The q05 multi-hop boundary is **recorded, not implemented** — it would require a
retrieval change, which is outside a minimal entry point. No P5 benchmark was reopened, no model
selection was revisited, and no scheduling policy was changed.
