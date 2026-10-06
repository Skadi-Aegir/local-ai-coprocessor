# Staged Lifecycle Design — retrieve, then swap models

**Status: MEASURED PASSING ON THE ISOLATED PATH. NOT ADOPTED.**
Production routing is unchanged. **For the production proposal and its rollback conditions see
`docs/production-integration-design.md`.** For which code produced which round see
`docs/phase55r-evidence-consolidation.md`.

**RESULT (P55R-05, and re-verified in the P55R-05 re-run):** with **both unloads required and
verified**, full RAG completes inside the existing 2048 MiB margin. Worst observed `free_min` across
all passing rounds and diagnostics is **3000 MiB**. Verified unload deltas: embedding
**+1368 to +1392 MiB**, reranker **+2591 to +2595 MiB**.

**Cost, stated because it is the point of the trade:** a full RAG cycle measured **~22–24 s**, and
**service recovery after the cycle** measured **~8 s per service** under light load and up to
**44.6 s (embedding) / 22.4 s (reranker)** under heavier machine load. The reload dominates.

Supersedes the P55R-03 version of this file, which contained a contradiction and an unmeasured
conclusion (see "Contradiction resolved" and E18).

## Contradiction resolved (it was real)

P55R-03 said two things that cannot both hold:

* **(a)** "release the embedding reservation, then unload local-embedding"
* **(b)** "the embedding reservation is released only after its unload has completed"

(a) is required by the protection model: `_unload_service` refuses to unload a service named by an
**active** reservation, so while stage 1's reservation is live the unload **cannot** succeed.
(b) was written as if release follows unload, which would deadlock. **Both were wrong about which
side of the lock each action sits on.**

**Resolution — a two-phase transfer inside one critical section:**

```
phase 1  BEFORE the lock   stage 1 has COMPLETED and its reservation is CLOSED.
                           This is when dependency protection ends: at RELEASE, as a
                           precondition of phase 2 - NOT as a consequence of the unload.
phase 2  UNDER ONE lock    plan for the reranker with allow_unload=True;
                           unload local-embedding; RE-CONFIRM the live reading;
                           close the old reservation; reserve the new one.
```

Because the whole of phase 2 happens inside one lock acquisition, no other request can be admitted
between the unload and the reservation, so the freed memory cannot be taken. This is AD-8's atomic
check-and-reserve reused, not reinvented. **Correct ordering: CLOSE stage 1 → (atomic: unload →
re-confirm → reserve stage 4).**

## State machine

```
IDLE ──stage1_admit──▶ EMBEDDING ──stage1_done──▶ CANDIDATES_ON_CPU
                             │                            │
                        stage1_refuse/timeout        stage3_swap
                             ▼                            ▼
                          FAILED ◀──swap_refuse──── SWAPPING ──swap_ok──▶ RERANKING
                             ▲                                              │
                             │                                         stage4_done
                        stage5_refuse                                       ▼
                             │                                     CANDIDATES_ON_CPU
                             │                                              │
                          GENERATING ◀────────stage5_admit─────────────────┘
                             │
                        stage5_done ──▶ DONE
```

### Invariants, and where each is enforced

| id | invariant | enforcement |
|---|---|---|
| **I1** | at most one stage of this pipeline holds a reservation at a time | `StagedPipeline.live_own_reservations()`; each stage closes in its own `finally` |
| **I2** | `SWAPPING` is entered with **zero** live reservations for this pipeline | asserted before the swap; the swap aborts if violated |
| **I3** | the unload is refused while the embedding reservation is live | enforced by the scheduler, verified in deterministic test 2 |
| **I4** | after any failure, no reservation is left behind | `cleanup()` plus per-stage `finally`; asserted in tests 4, 5, 6 |
| **I5** | CPU candidates survive every failure after stage 2 | candidates are plain data; asserted in tests 4 and 5 |

## Real residency accounting — not `max(stage requirements)`

Admission is decided by the scheduler's own `_plan_locked`, from the **real** residency list and the
**real** live free reading at each step. The P55R-03 `max()` table was a feasibility estimate and is
**not** used to decide anything. The live run below shows exactly why that matters: at swap time the
budget was already sufficient (`1343 MiB usable >= 600 MiB required`), so **no unload was performed
at all** — a `max()` estimate would have hidden that.

## What the first isolated run measured, and what changed — both kept

**FIRST ATTEMPT (P55R-04, commit `111185d`) FAILED the gate:** 2/5 completed, `free_min`
**1863 MiB** against a 2048 MiB margin, 7 sub-margin samples, no OOM
(`benchmarks/phase55r/results/P55R-04-staged/staged_rounds.json`).
Cause: `allow_unload` makes an unload **permitted, not required**. At swap time the budget was
already sufficient (`1343 MiB usable >= 600 MiB required`), so `unloaded=[]`, the embedding model
**stayed resident**, and the swap delivered no memory relief.

**FIX:** the unload became an explicit operation that is **required and verified**, on two prongs
that must both hold — the port is free **and** the live free reading rose. A port that frees while
memory does not return is a FAILURE.

**SECOND ATTEMPT (P55R-05, commits `346ada4` and `543e6e7`) PASSES.** Three rounds per harness
version, all 5/5, no OOM, both unloads performed, margin held:
* `346ada4` — `free_min` **3014 / 3000 / 3021**
* `543e6e7` — `free_min` **3193 / 3368 / 3374**

The first five-item run at `111185d` recorded **4/5** (`results/P55R-05-five/five_rounds.json`),
failing on `r1-4a-bg` with a refused connection — an **undeclared ordering precondition in the
harness**, since item 3 intentionally leaves the embedding model unloaded and item 4a is an
embedding call. Both unloads had succeeded in that run and the margin was held. The fix restored a
healthy residency between item 3 and item 4, recorded separately. That failure record is preserved
and is **not** pooled with the later 5/5 rounds.

**Note on the earlier "it would not suffice" claim:** it was an estimate, and it was wrong — see
E18. The estimate also charged a resident service's load budget against the same instant's live free
reading, double-counting it.

## Failure recovery (implemented and verified)

| failure | behaviour | verified |
|---|---|---|
| stage 1 refused / times out | nothing reserved, no state to clean | test 6 |
| stage 1 exception | reservation closed in `finally` | test 1/6 |
| swap: unload fails | stage 4 not admitted; **candidates retained**; state returns to `CANDIDATES_ON_CPU` so the swap is retryable | test 4 |
| swap: reranker admission fails | same; no reservation leaked | test 5 |
| stage 4 exception | reranker reservation released; candidates retained | test 4 |
| stage 5 refused/fails | reported; candidates retained; retrieval does not need repeating | live run, item 3 |
| process dies mid-pipeline | the ledger is in-process only (P3-1), so the next process re-measures rather than inheriting a stale reservation | by construction |

## Estimated latency (measured)

| step | measured |
|---|---|
| local-reranker cold load | **8.0–8.6 s** (P55R-03); 1.9 s warm-cache start observed |
| other-service cold load | ~1–3 s |
| rerank inference | 0.43–0.51 s (max body), 2.0–2.1 s (large batch) |
| generate, short answer | ~0.4 s |
| whole staged RAG round | **~22–24 s** with both unloads required and verified (P55R-05); the reloads dominate. The earlier 2.54 s was the P55R-04 attempt, in which no unload actually happened (`unloaded=[]`) |

The staged design pays a reranker load per swap unless the reranker is kept warm.

> **Corrected 2026-10-05.** This table gave "2.54 s wall (when it completes)" for a whole round, and
> the list below said the isolated run shows the staged design does not meet the margin. Both were
> P55R-04 statements left in place when the header was updated for P55R-05, and they contradicted
> the header on this same page.

## What this does NOT claim

* it does **not** claim the co-resident default meets the margin, and it claims the staged path
  meets it **only on the isolated path** (P55R-05); the first isolated attempt (P55R-04) did not
* it does **not** change production routing
* it does **not** resolve the gap by weakening the margin or the 1800 MiB default
* the 881 MiB measurement and the 1361 MiB candidate are **not** written into any default
