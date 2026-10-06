# Escalation Policy

**Derived from:** Phase 2 (local-fast) and Phase 3 (local-smart) measurements on the
RTX 5070 Ti Laptop, 12 GB. Every threshold below is backed by a benchmark in
`benchmarks/local-fast-results.md` or `benchmarks/local-smart-results.md`.
**Status:** initial policy — expected to be revised as more task evidence accumulates.

---

## 1. The rule

```text
cheap and deterministic        -> local-fast      (4B, ~130 tok/s)
needs real analysis            -> local-smart     (9B,  ~74 tok/s)
hard reasoning / long context  -> DeepSeek
complex engineering / security -> Claude or Codex
irreversible or dangerous      -> human
```

Plus the routed local specialists once they exist: `local-ocr` (text in images),
`local-asr` (speech), `local-vision` (image/UI/chart understanding),
`local-image` (generation), `local-embedding` / `local-reranker` (retrieval).

---

## 2. Send to local-fast (4B)

Measured: 130.1 tok/s, 0.13 s mean TTFT on short structured prompts, **JSON 20/20 (100 % parse
and schema)** with no grammar constraint, 100 % over 800 calls in the stability run.

- classification, tagging, routing assistance
- JSON / structured field extraction
- log and file summarisation
- commit messages, changelogs from diffs
- simple translation
- prompt / context compression at small sizes
- traceback triage ("what type of error, which file")
- simple Python / Shell / SQL
- generating a small unit test or fixture
- test-result interpretation
- mechanical text transforms (timestamps, renames, formatting)
- final formatting/validation of another worker's output

**At temperature 0, prefer this model for anything a program will parse.**

## 3. Send to local-smart (9B)

Measured: 74.6 tok/s, 0.39 s TTFT at 16K, VRAM 5527 MiB at 16K, 100 % over 344 steps in a
6-minute stability run.

- multi-file code reading where the answer depends on how files interact
- traceback root-cause analysis rather than triage
- larger context compression (12–16K project history into a handoff)
- drafting a handoff another agent will resume from
- first-pass code review on a diff
- second opinion on a local result before escalating

### The measured caveat — read this before routing to the 9B

In the Phase 3 head-to-head, **the 4B matched or beat the 9B on every scored task**:

| Task | 4B | 9B |
|---|---|---|
| Bug repair pass@1 | **100 %** | 66.7 % |
| Code review recall | **58.3 %** | 41.7 % |
| Summary facts recalled | **10/10** | 9/10 |
| Structured schema match | **75 %** | 50 % |
| Routing accuracy | **73.1 %** | 65.4 % |
| Routing under-escalations | 5 | **3** |

**Therefore: do not route to the 9B by default on the assumption that "bigger is better".**
The 9B is justified only when (a) the task needs more than 8K of context, or (b) the 4B has
already failed or produced something you doubt, so the 9B serves as a second opinion.
The single measured 9B advantage was fewer under-escalations in routing (3 vs 5), which is a
reason to ask the 9B for a routing opinion on genuinely ambiguous items — not to make it the
default router.

## 4. Go straight to DeepSeek

Do not burn local attempts first when the task is recognisably of this shape:

- several-step reasoning where the intermediate steps must be correct, not merely plausible
- mathematical or formal proof work
- deep analysis of concurrency, memory or performance behaviour under specific conditions
- long-context work beyond 32K
- anything where the 4B and 9B both produce confident but *contradictory* answers

Skipping straight to DeepSeek is correct: a local model that is confidently wrong costs more
than the API call, because it must be detected and undone.

## 5. Go straight to Claude / Codex

- architecture and cross-system design
- large refactors (multi-module, multi-service)
- security-critical code: auth, permissions, crypto, payment, data deletion
- production-affecting changes
- anything where the blast radius of a subtle error is large

Local models' role for these is **support, not authorship**: gather context, retrieve relevant
files, run tests, verify the resulting patch, compress history into a handoff.

## 6. Go to a human

- irreversible or destructive operations (dropping data, deleting production resources)
- decisions with legal, financial or safety consequences
- anything the agent itself cannot verify after acting

## 7. Worked examples

| Task | Route | Why |
|---|---|---|
| "Summarise this 3000-line access log into 5 bullets" | local-fast | mechanical, 4B does this well |
| "Extract task_id/project/priority as JSON" | local-fast | 100 % measured JSON adherence |
| "Rename `user_list` to `users` across this file" | local-fast | trivial |
| "Why does page=1 return the second page?" | local-smart (or local-fast first) | 4B solved this 3/3; escalate only on doubt |
| "Compress this 14K project history into a handoff" | local-smart | exceeds the 8K fast-worker context |
| "Review this patch for concurrency bugs" | local-smart, then Claude if money/production | recall is only ~58 %, so treat as a filter |
| "Prove this distributed transaction cannot deadlock" | DeepSeek | formal reasoning |
| "Refactor the monolith into services" | Codex | large refactor |
| "Redesign the RBAC model and audit logging" | Claude | security architecture |
| "Delete the production database — proceed?" | human | irreversible |

## 8. Escalation is not a fixed ladder

The list in §1 is a **priority ordering, not a pipeline**. Walking every level costs latency and
tokens for no gain. Escalate directly to the level the task's shape implies, and treat the local
tiers as filters that reduce — not eliminate — cloud calls.

## 9. What would change this policy

- a Phase 4–8 capability (OCR, ASR, VL) making a route actually available
- a larger or better-tuned local model measured against these same benchmarks
- evidence that the 9B wins on task families this suite did not cover

Re-measure before changing a threshold; do not change it because a newer model exists.
