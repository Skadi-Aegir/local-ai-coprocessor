# Production Integration Design — staged two-unload RAG

**Status: PROPOSED FOR REVIEW. NOT IMPLEMENTED. NOT ENABLED.**
Nothing in this round activates it. Production default routing is unchanged.

Written in P55R-06. The measured basis is `docs/phase55r-evidence-consolidation.md`; this document
does not re-run anything.

---

## 1. What would change, and what would not

**Would change:** the RAG request path only. Instead of one request declaring embedding + reranker +
generative simultaneously, a RAG request becomes a *sequence* of stages, each its own scheduler
request with its own `requires`, with the embedding and reranker models unloaded between stages.

**Would NOT change:** the 2048 MiB margin, the 1800 MiB reranker transient, the service start/stop
scripts, the retrieval service, the reranker service, the model set, and every non-RAG request path.

## 2. Stage sequence and per-stage interface

| stage | job class | `requires` | holds a reservation | ends by |
|---|---|---|---|---|
| 1 embed + retrieve | `background` | `["local-embedding"]` | yes | releasing; candidates copied to CPU |
| 2 handoff to CPU | — | — | no | candidates are plain data |
| 3 unload embedding, admit reranker | `background` | `["local-reranker"]` | yes | rerank completes |
| 4 rerank | (within 3) | — | yes | releasing; ranked results copied to CPU |
| 5 unload reranker | — | — | no | verified unload |
| 6 pack + generate | `interactive` | `["local-fast"]` | yes | releasing |

Stages 2 and 5 hold no scheduler reservation at all: in stage 2 the candidates are plain Python
data, and in stage 5 the pipeline deliberately holds nothing so the unload cannot be refused.

## 3. Concurrent protection for the double unload

This is the part that must be right, so it is stated as rules rather than as prose.

**P1 — protection ends at RELEASE, never at unload.** A service named by an ACTIVE reservation
cannot be unloaded; the scheduler refuses. Therefore the ordering is: the stage that used the
service **completes and releases**, and only then may the next stage unload it. The superseded
P55R-03 wording ("release after the unload completes") would deadlock and is not implementable.

**P2 — a required unload is verified, not assumed.** `allow_unload=True` makes an unload
*permitted*; measured in P55R-04, that resulted in `unloaded=[]` and the model staying resident. The
production operation must therefore be *required* and must verify **both** prongs: the port is free
**and** the live free reading rose. A port that frees while memory does not return is a FAILURE.
This case is covered by deterministic tests (`port_still_bound`, `memory_not_returned`).

**P3 — nothing can take the freed memory during admission.** The reranker admission is performed by
the scheduler under its single lock (AD-8's atomic check-and-reserve), so between the unload and the
reservation no other request can be admitted against the same budget.

**P4 — the pipeline holds at most one reservation (invariant I1).** Each stage closes in its own
`finally`, so a crash mid-stage cannot leave two.

**P5 — a required unload is refused while the pipeline itself holds a reservation (invariant I2).**
Enforced and tested; this is what stops the pipeline from trying to unload the service it is using.

**Residual risk that must be stated:** the scheduler's reservation ledger is **in-process only**
(OPEN_ISSUES P3-1). Two processes driving the same services would not see each other's
reservations, and P1-P5 would not hold across them. Any production integration must either
serialise RAG through one process or add a cross-process lock. **This is a blocking prerequisite,
not a nicety.**

## 4. Service recovery

The staged path unloads models *by design*, so recovery is part of the normal path, not an error
path. Measured reload costs:

| service | measured reload |
|---|---|
| local-embedding | ~8 s light load, **up to 44.6 s** under heavier machine load |
| local-reranker | ~8 s light load, **up to 22.4 s** under heavier machine load |

**Design position:** the reload is the dominant cost of this design, so the integration should keep
the reranker **warm across requests whenever the budget allows**, and treat the cold path as the
fallback rather than the norm. Recovery must be health-gated: the next stage is only entered once
the service reports healthy, and a failed reload fails that stage rather than proceeding.

## 5. Failure degradation

| failure | behaviour | already verified |
|---|---|---|
| stage 1 refused / times out | plain refusal, nothing reserved | yes (deterministic) |
| unload refused because a reservation is live | reported as a refusal, not skipped; stage fails | yes |
| unload succeeds but memory does not return | reported `memory_not_returned`; stage fails | yes |
| unload leaves the port bound | reported `port_still_bound`; stage fails | yes |
| reranker admission refused after the unload | candidates retained on CPU; swap retryable | yes |
| rerank fails | reranker reservation released; candidates retained | yes |
| reranker unload fails after reranking | **ranked results retained**; generate can be retried | yes |
| generate refused | ranked results retained; no retrieval repeat needed | yes |
| process death mid-pipeline | ledger is in-process, so the next process re-measures rather than inheriting stale state | by construction |

**Degradation decision to make at integration time:** when the reranker cannot be admitted, the
safest degradation is to **serve the dense-only top-k** (the reranker is optional by design,
AD-1/AD-2) rather than fail the request. This is not implemented here and would need its own
measurement, because a dense-only answer is a *different quality point* from the reranked one.

## 6. Rollback conditions

Roll back to the co-resident path (single request, `requires=[embedding, reranker, fast]`) if ANY of
the following holds. These are stated as observable triggers, not as judgement calls:

1. **Margin breach.** Any sample of free VRAM below 2048 MiB during a staged RAG cycle.
2. **Unverified unload.** Either required unload reports anything other than port-free AND
   free-VRAM-rose.
3. **Reload failure.** A stage cannot reach a healthy dependency, or a reload exceeds a configured
   ceiling (suggested: 3x the observed p95, i.e. well above 45 s).
4. **Latency regression beyond a set budget.** Staged RAG wall time must stay within an agreed
   ceiling; the measured cold path is already ~22-24 s plus reload, so if the product cannot afford
   that, the design is wrong for it.
5. **Cross-process interference.** Any evidence of two processes driving the same services without a
   shared ledger.
6. **Any OOM.** Immediate rollback; no retry.

**Rollback mechanism:** revert the routing change. Because nothing in this design alters the margin,
the budget, the services or the scripts, rollback is a routing revert with **no data migration and no
service reconfiguration** — this is the main reason the design keeps all policy values untouched.

## 7. What this design does not claim

* it does not claim the staged path is *better* than co-residency — it is *possible* where
  co-residency measured 4/5 or worse; the trade is ~22-24 s of latency and a reload per cycle
* it does not claim answer quality improved; the RAG check is a task-definition reference check, not
  an independent quality assessment (see the consolidation document)
* it does not authorise implementation. Implementation requires a separate decision, and the
  cross-process prerequisite in P5-residual must be resolved first.
