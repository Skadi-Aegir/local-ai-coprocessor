# P56 frozen question set

**Frozen:** yes. Do not add, edit, or reorder questions after the first run.
**SHA256 of the frozen file:** `4138ABDA5DE21DC1CFE5374703763073D70BF5BA82FC24F8C3585FFDA69EBA6C`
**Purpose:** a minimal end-to-end check of the local CLI entry point on a clean corpus.

**Post-run note on q05.** q05 was written as a *cross-document* question. The run showed its top-5
pack did **not gather the evidence needed** — the two relevant chunks are individually retrievable
but were not both returned — **and** the clean corpus contains **no mention of port 8114** at all.
The question is left unchanged (the set is frozen) and the outcome is recorded in `review.md` as a
**top-5 evidence-gathering shortfall plus a corpus gap**, explicitly **not** as a proven multi-hop
defect.

## Provenance and exclusion check

These questions are **newly written for P56** and were **not** used for any tuning. The two existing
question sets — `benchmarks/retrieval-v2/benchmark_v2.py` (83 questions) and
`services/retrieval/rag_compare.py` (12 questions) — are the tuning/exposed sets, and **none of their
questions appears here**. Both are also **excluded from the clean corpus**, so they cannot be
retrieved either.

The categories below satisfy the brief's requirement to include answerable, cross-document and
answer-absent cases.

| id | category | question |
|---|---|---|
| q01 | answerable | What port does the local-fast service listen on, and what context size does it use by default? |
| q02 | answerable | Which pooling method and output dimension does the embedding service use? |
| q03 | answerable | What chunking strategy does the index builder record for python code, and what does the strategy name signify? |
| q04 | answerable | What is the retrieval service's health endpoint expected to return, and on which port does it serve? |
| q05 | cross-document | Trace how a question reaches the local-fast model: which components are involved, in what order, and what ports do they use? |
| q06 | cross-document | If the local-fast model cannot answer well enough, what does the escalation policy say should happen, and which model is the escalation target? |
| q07 | cross-document | How does the index store keep FAISS vector positions and chunk metadata consistent with each other? |
| q08 | answer-absent | What is the monthly monetary cost of running local-fast on this machine? |
| q09 | answer-absent | What is the exact GPU die temperature in degrees Celsius while local-fast is generating? |
| q10 | answer-absent | Which commercial cloud provider hosts the local-smart model? |

## Why the answer-absent questions are genuinely absent

* **q08** — the corpus records latency, throughput and token counts, but no monetary price is
  recorded anywhere; there is no billing data in this project.
* **q09** — `nvidia-smi` output in the corpus reports memory, not temperature; no temperature
  reading was ever persisted.
* **q10** — `local-smart` is a locally served model; the corpus describes it as local and no cloud
  provider is named.

These three exist to test that the entry point **refuses** rather than inventing an answer.

## Grading rule stated in advance

An answer is **supported** only if every factual claim in it can be traced to a cited chunk. A
correct answer with **no citation** is not supported. An answer that cites a source which does not
contain the claim is **not supported**. Refusal on q08–q10 is the **correct** outcome, not a failure.
