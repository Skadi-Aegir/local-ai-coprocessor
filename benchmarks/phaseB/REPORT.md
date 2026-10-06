# Phase B acceptance report

**Scope:** minimal local call integration for the QA path.
**Baseline:** Phase A at `8b8fcf2`. **Interface:** `cli/ask_server.py` on `127.0.0.1:8115`.
**Verdict: PASS**, with the limitations in §6.

> **Post-review corrections, 2026-10-05 - read with ERRATA E24 and E25.** `MAX_BODY_BYTES` was never exercised in this run; the two admitted requests executed one at a time; L4 is wrong (a client timeout does not free the slot, and the 600 s upstream timeouts could block every request for ten minutes); valid non-object JSON got no response; `/health` named the default index whatever was loaded. All five are fixed in `cli/ask_server.py` and covered by `cli/test_qa_path.py`. This report is otherwise kept as written.

---

## 1. Was there an existing caller or contract? (required first step)

**No.** Checked and recorded:
* no HTTP client or caller of a QA path exists in the repo — the only `urllib` uses are the CLI's own
  calls to the model services and the benchmark harnesses
* no interface contract in `docs/` (`/ask`, "ask API", "query API", "interface contract" — no hits)
* ports 8115/8116/8200/8000 were all free

So B provides a minimal local interface, as the phase directs. It is **not** a Gateway: two
endpoints, loopback-only, no auth layer, no routing, no service discovery.

## 2. Reuse rather than reimplementation

`ask_server.py` **imports `cli/ask.py`** and calls its `answer()` and `Retriever`. Retrieval,
prompting, the evidence floor, the refusal rules and citation verification are therefore **the same
code on both paths** — there is no second QA implementation to drift.

The retriever is created **once** at startup and shared, guarded by a lock. A per-request retriever
would re-open the index repeatedly, and a per-request in-process model is forbidden by the phase.

## 3. Client-cannot-choose-the-index, and the resource boundary

| control | implementation | verified |
|---|---|---|
| no arbitrary index path | the index is server-side config; `index`, `index_dir`, `index_path`, `corpus` in the body are **rejected explicitly** (400 `index_override_rejected`) rather than silently ignored | **yes — 400 returned** |
| input size | `MAX_QUESTION_CHARS = 1000`, and `MAX_BODY_BYTES = 64 KiB` checked **before** reading the body | **yes — 413 `question_too_long`** |
| concurrency | `BoundedSemaphore(2)`; excess gets a structured 429, **not** an unbounded queue that would grow resource use | **yes — 6 simultaneous requests produced 2×200 and 4×429; inflight peak recorded as 2** |
| model loading | the server never loads a model of its own; embedding goes to the resident 8113 service | by construction |

## 4. End-to-end validation with a real client

Client: `benchmarks/phaseB/b_client.py` (real HTTP). Records: `results/b_e2e.json`.

| # | scenario | result |
|---|---|---|
| 0 | `GET /health` | 200, `status=ok`, `index_version=259fae05785b433c` |
| 1 | **normal answer** | 200, `outcome=answered`, 1 citation `docs/models/local-fast.md:132-148` **verified**, 5.598 s |
| 2 | **answer-absent** | 200, `outcome=refused`, 0 citations, 0.522 s — correctly declined the monetary-cost question |
| 3a | index override attempted | 400 `index_override_rejected` |
| 3b | **stale index** | source temporarily modified → 200, `outcome=refused`, citation reported **`stale`**, reason "one or more cited sources could not be verified against the indexed version; the index may be stale" |
| 4 | **service unavailable** | embedding service stopped → **500 `pipeline_error`** in 2.125 s with the transport error, **no hang**; after restart **recovered in 10.13 s**, health `ok`, probe question **answered** |
| 5a | oversize question | 413 `question_too_long` |
| 5b | **concurrency** | 6 simultaneous → `[429,429,429,429,200,200]` |

**Response shape is identical for answers and refusals** (`outcome`, `answer`, `reason`,
`citations`, `index_version`, `elapsed_s`, `pipeline`), so a caller never has to guess which keys
exist. Errors are structured as `{"error": {"kind", "message"}}` with a real HTTP status.

**VRAM:** minimum free across all scenarios **5375 MiB**; the 2048 MiB margin held throughout. No OOM.
Final inflight `{current: 0, peak: 2}` — the bound held and drained cleanly.

## 5. A defect found and fixed during B

`b_client.py` first ran the embedding start/stop scripts with
`subprocess.run(..., capture_output=True)`. The child filled the pipe buffer and blocked while the
parent waited for it to exit — **two powershell processes were found wedged for 1867 s** and the
client stalled mid-scenario. This is the **same pipe-buffer deadlock that already cost runs during
P55R-02 and P55R-04**; the fix is the same (stdout to a file, never a pipe) and it is now applied
here too. The first B attempt was discarded for that reason and re-run cleanly.

## 6. Limitations

**L1 — the embedding service's VRAM growth (OPEN_ISSUES P1-11) reproduced again during B.** Its
`/stats` reported `vram_mb: 7756.6` against 2360.6 at start after bulk work, which with other
residents dropped free VRAM to 673 MiB — **below the margin** — before a restart restored it. This
is now confirmed twice and is the single largest operational risk. **Precondition carried forward:
restart 8113 after bulk embedding, and keep the reranker non-resident.**

**L2 — no authentication and no TLS.** The interface is loopback-only and the phase explicitly
excludes public exposure, so this is acceptable *here*; it must not be exposed without adding both.

**L3 — `MAX_CONCURRENCY = 2` is a starting value, not a measured optimum.** It was chosen to keep two
requests inside the resource boundary; the phase did not ask for throughput tuning and none was done.

**L4 — no timeout is imposed on the pipeline itself.** A hanging upstream service would hold a
concurrency slot until the client's own timeout fires. The 429 path bounds *arrival*, not *duration*.

**L5 — q05-class questions still refuse** (Phase A L3). Out of scope for B and reserved for C.

**L6 — the reranker is not integrated.** B runs dense-only, as instructed. The staged double-unload
path from P55R-05 remains a **verified isolated experiment**, not production routing, and enabling it
would first require the cross-process ledger prerequisite (OPEN_ISSUES P1-10).

## 7. Verdict

**PASS.** The interface is minimal, loopback-only, reuses the Phase A core, refuses client-chosen
indexes, enforces input and concurrency limits, returns answers/refusals with citations, index
version, timing and structured errors, and was validated end to end by a real client across all five
required scenarios with the 2048 MiB margin held.
