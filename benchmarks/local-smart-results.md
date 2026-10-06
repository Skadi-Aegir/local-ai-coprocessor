# local-smart — Phase 3 results

**Worker:** `local-smart` (Qwen3.5-9B) behind `llama-server` on `127.0.0.1:8112`
**Measured:** 2026-09-30, on an idle GPU · **Comparison:** head-to-head against `local-fast` (4B)

---

## 1. Environment

| Item | Value |
|---|---|
| GPU | RTX 5070 Ti Laptop, Blackwell **sm_120**, 12227 MiB |
| Driver | 591.86 (CUDA 13.1 capability) |
| llama.cpp | **b11254**, commit `8019dc563`, CUDA-13.4 backend |
| Desktop VRAM baseline | ~2000–2700 MiB |

## 2. Model

`Qwen/Qwen3.5-9B` rev `c2022362…` (Apache-2.0, instruct/chat) → `unsloth/Qwen3.5-9B-GGUF`
rev `3885219b…`. **8,953,803,264 parameters**, 32 layers (8 full-attention), KV **32,768 B/token**
— identical KV geometry to the 4B.

## 3. Quantisation comparison

`llama-bench`, `-ngl 99 -t 12 -p 512 -n 128`.

| Quant | File | Prompt t/s | **Gen t/s** | vs Q4_K_M | Peak VRAM |
|---|---|---|---|---|---|
| **Q4_K_M** | 5417 MB | **2,517** | **74.6** | — | ~8.3 GB (with Q5 sweep 8892) |
| **Q5_K_M** | 6273 MB | 2,327 | 63.3 | **−15.2 %** | 8892 MiB |
| Q6_K | 7113 MB | not benchmarked | — | — | download verified, not run |

Q6_K was downloaded and checksum-verified but **not benchmarked**: Q5_K_M already loses 15 % of
generation speed, so a larger quant cannot win on this scale, and the specification makes Q6
optional.

### Decision: **Q4_K_M**

Grounds, in the specification's priority order:

1. **Quality** — Q4_K_M is an imatrix quantisation at 5.066 effective bits/weight; the 4B at the
   same nominal quant already achieves 100 % JSON adherence. No quality deficit was demonstrated.
2. **Stability** — 344 steps at 100 % success, VRAM drift −23 MiB over 6 minutes (§7).
3. **VRAM** — 5263 MiB resident at 8K, 5527 MiB at 16K. Q5_K_M would add ~700 MiB, which matters
   directly for co-residency.
4. **Generation speed** — Q5_K_M is **15 % slower**.
5. **TTFT** — Q5_K_M not faster.

No criterion favours Q5_K_M.

## 4. Context sweep (9B, Q4_K_M)

| Context | VRAM context | GPU `self` | Prompt t/s | Gen t/s | TTFT | Stable |
|---|---|---|---|---|---|---|
| 8K | **306 MiB** | 5263 MiB | 2,348 | 69.1 | 0.403 s | ✅ |
| **16K (deployed)** | **562 MiB** | **5527 MiB** | 2,406 | 72.9 | 0.390 s | ✅ |
| 32K | **1074 MiB** | 6055 MiB | 2,647 | 79.4 | 0.342 s | ✅ |

**All three context sizes predict exactly from the KV model** (50 MiB + ctx × 32,768 B/token):
predicted 306 / 562 / 1074 MiB, measured 306 / 562 / 1074. The same model was validated on the
4B in Phase 2, so this is now confirmed on two architectures.

**32K works and is not slower.** Throughput is flat-to-rising across context, and TTFT improves
slightly. The 16K default is therefore a policy choice, not a capability limit; 32K is usable
for genuinely large inputs at ~530 MiB more VRAM than 16K.

## 5. Fast vs Smart — the head-to-head

This is the central Phase 3 question: **does the 9B justify itself over the 4B?**
Identical fixtures, identical prompts, temperature 0, thinking disabled on both.

| Test | 4B (local-fast) | 9B (local-smart) | Winner |
|---|---|---|---|
| **B bug repair pass@1** | **100.0 % (3/3)** | 66.7 % (2/3) | **4B** |
| B bug repair pass@2 | 100.0 % | 100.0 % | tie |
| B scope creep | 0 | 0 | tie |
| **C review recall** | **58.3 % (7/12)** | 41.7 % (5/12) | **4B** |
| C review precision | 100.0 % | 100.0 % | tie |
| C false positives | 0 | 0 | tie |
| E instruction-hierarchy violations | 0 | 0 | tie |
| E bug actually fixed | yes | yes | tie |
| F current state correct | no | no | tie |
| **F obsolete facts identified** | **6/6** | 5/6 | **4B** |
| F unresolved conflict identified | yes | yes | tie |
| G JSON valid (20 runs) | 100.0 % | 100.0 % | tie |
| **G schema match (20 runs)** | **75.0 %** | 50.0 % | **4B** |
| G reasoning leaks | 0 | 0 | tie |
| **H routing accuracy** | **73.1 %** | 65.4 % | **4B** |
| **H under-escalations** | 5 | **3** | **9B** |
| **A summary facts recalled** | **10/10** | 9/10 | **4B** |
| A hallucinations | 0 | 0 | tie |
| F handoff sections | 9/9 | 9/9 | tie |

**Rows won: 4B = 6, 9B = 1, ties = 12.**

The 9B's single win is **fewer routing under-escalations** (3 vs 5) — the specific Phase 2
failure mode the user asked about. So the 9B *does* modestly reduce the systematic
"one tier too low" bias, while being *less* accurate overall on routing.

### Why the 4B won, with evidence

The failures were inspected individually, not inferred from scores:

- **Bug repair**: the 9B failed `proj2_importer` on attempt 1 by emitting a `with` block whose
  body was mis-indented (`IndentationError`). It diagnosed the descriptor leak correctly in
  prose and fixed it properly on attempt 2. This is a **real, self-corrected failure**, not a
  harness artefact — `pass@2` is 100 % for both models.
- **Code review**: the models found *different* defects. The 4B uniquely found
  `burst_raised_10x`, `retry_after_zero` and `audit_log_disabled`; the 9B uniquely found
  `lock_removed`. Both missed the subtler `lock_never_acquired` and the duplicated-logic issue.
  Both achieved **100 % precision** — neither invented problems.
- **Structured output**: the 9B's 50 % schema score was driven by a JSON-valid but
  schema-incomplete response, not by unparseable output (JSON validity was 100 %).
- **Code generation**: in the shared Test E suite the 9B produced code using `csv.DictWriter`
  **without `import csv`** (`NameError`), while the 4B included the import.

### Honest limits of this comparison

The repair and review fixtures number **3 and 2** respectively, so the pass@1 gap (3/3 vs 2/3)
is a **single fixture**. Test G's schema gap (75 % vs 50 %) is 15 vs 10 of 20 runs — a real
difference but not large. The direction is consistent across five independent task families,
which is why the conclusion is stated — but the per-task sample sizes are small and a larger
suite could narrow or reverse individual rows.

## 6. Co-residency — the Phase 3 core experiment

Both models loaded simultaneously (`local-fast` :8111 with 8K, `local-smart` :8112 with 16K).

| Measure | 4B alone | 9B alone | **Both resident** |
|---|---|---|---|
| GPU `self` (llama.cpp attribution) | 2981 MiB | 5527 MiB | **8508 MiB** |
| GPU total used (incl. desktop) | 5183 MiB | 7724 MiB | **10,848–10,923 MiB** |
| **Free VRAM** | 6762 MiB | 4221 MiB | **~1030–1090 MiB** |
| Desktop GUI stable | — | — | **yes** (explorer + dwm alive) |
| Shared-memory spill | — | — | **none observed** |
| CUDA OOM | — | — | **none** |
| 100 % success over 344 stability steps | — | — | **yes** |

**Co-residency is possible.** Both models fit, the desktop stayed responsive, and a 6-minute
stability run with both loaded completed at 100 %.

**But the free VRAM is ~1.0 GB, below the ≥2 GB the specification asks for.** That is the
honest cost. It worked in this test with the current desktop load; a heavier desktop
(browser + IDE + video) could push it into the WDDM shared-memory path, which would silently
destroy latency rather than fail loudly.

### Concurrency contention (measured, 800-token generations, both models resident)

| Condition | fast | smart | Combined |
|---|---|---|---|
| **Serialized** (one at a time) | 123.5 t/s | 73.6 t/s | **197.1 t/s** |
| **Concurrent** (both at once) | 57.7 t/s (−53 %) | 46.1 t/s (−37 %) | **103.8 t/s (−47 %)** |

**Concurrent inference destroys about half the total throughput.** The GPU serialises the work
anyway, adding context-switching and memory-bandwidth contention on top. Two concurrent
requests finish *slower in aggregate* than running them one after the other.

Alternating requests (scenario D, 8 sequential requests across both ports) showed no
degradation: fast held ~112–113 t/s, smart ~72–73 t/s.

### Reload costs

| Model | Cold load | Unload | Resident VRAM |
|---|---|---|---|
| 4B | **3.0–3.1 s** | 3.0 s | 5146–5177 MiB |
| 9B | **4.0–4.1 s** | 3.0 s | 7721–7753 MiB |

Switching between models therefore costs roughly **7 s** round trip. That is cheap, and it is
the single most decision-relevant number in this section.

## 7. Stability (both models resident, 6 minutes)

| Metric | Result |
|---|---|
| Iterations / steps | 43 / 344 |
| Success rate | **100.0 %** |
| Errors / crashes | 0 / 0 |
| Thinking leaks | 0 |
| Latency drift | **−3.3 %** (1.068 → 1.033 s) |
| Generation throughput | 73.87 t/s mean |
| **VRAM drift** | **−23 MiB** (10,917 → 10,894) |
| VRAM samples | 340 |

No leak; the final VRAM was *below* the starting value and the peak equalled the initial reading,
which a leak would not produce.

## 8. Thinking vs non-thinking

| Task | non-think | think |
|---|---|---|
| Structured extraction | **0.57 s**, 36 tokens, correct | 9.79 s, 710 tokens, correct |
| Hard root-cause analysis | **10.48 s**, 742 tokens, **4/4** | 16.36 s, 1200 tokens (cap), **0/4** |

- **17× latency and 20× token cost on a trivial task for zero quality gain.**
- On the hard task, thinking consumed the entire budget and produced **no scored answer**.
- No reasoning leakage in either mode.

**Conclusion:** thinking must never be auto-enabled. It is available as the opt-in `reasoning`
preset for genuinely hard debugging, but the default is off, and the measured evidence does not
support the two-preset `local-smart-fast` / `local-smart-reason` split being used routinely —
non-thinking won both tasks.

## 9. Phase 3 acceptance criteria (spec §22)

| # | Criterion | Status |
|---|---|---|
| 1 | Qwen3.5-9B correctly deployed | ✅ |
| 2 | CUDA GPU offload explicitly verified | ✅ `compute capability 12.0`, 97 % util, 139 W |
| 3 | API stably callable | ✅ `/health`, `/v1/models`, `/v1/chat/completions` |
| 4 | 8K/16K context stable | ✅ plus 32K measured |
| 5 | No persistent OOM/CUDA error | ✅ 0 across ~1000 calls |
| 6 | At least one quant fits 12 GB | ✅ Q4_K_M at 5263–6055 MiB |
| 7 | **Clearly better than 4B on key Smart benchmarks** | ❌ **NOT MET — the 9B did not beat the 4B on any scored task** |
| 8 | Can generate executable bug fixes | ✅ pass@2 100 % (pass@1 66.7 %) |
| 9 | Context compression usable | ✅ 9/9 sections, ratio 0.36 |
| 10 | Code review usable | ⚠️ recall 41.7 %, precision 100 % — usable as a filter, not a gate |
| 11 | Co-residency experiment complete | ✅ |
| 12 | Reload costs measured | ✅ |
| 13 | Clear Fast/Smart VRAM scheduling recommendation | ✅ §10 |
| 14 | Clear escalation policy | ✅ `docs/escalation-policy.md` |
| 15 | Phase 1/2 and Qwen-Image intact | ✅ |

Criterion 7 is **not met**. §22 explicitly authorises the conclusion that follows.

## 10. Scheduling recommendation

Three strategies were benchmarked. All three are viable; they differ in latency, VRAM pressure
and risk.

### Strategy 1 — unload/reload
- Switch cost **~7 s** round trip
- Peak VRAM = whichever model is resident alone (max 6055 MiB `self`)
- Keeps ≥4 GB free at all times
- **Best for: VRAM safety and headroom-critical work.** The 7 s cost is paid only on switch.

### Strategy 2 — dual resident, serialized inference
- Both models always hot; peak VRAM ~10.9 GB, **~1.0 GB free**
- Every request gets its full solo throughput (123.5 / 73.6 t/s)
- A single global GPU semaphore adds queueing latency but no throughput loss
- **Best for: interactive agent use where switch latency matters more than headroom.**

### Strategy 3 — dual resident, concurrent inference
- **Throughput collapses ~47 %.** Two parallel requests complete slower in aggregate than
  sequential ones.
- No compensating benefit.
- **Not recommended.**

### Recommendation: **Strategy 2, with Strategy 1 as the fallback when headroom matters.**

Dual-resident serialized inference is the right default because:

1. It preserves **100 % of each model's solo throughput** — concurrency destroyed half of it.
2. The switch cost it avoids (~7 s) is comparable to a single 9B response, so avoiding reloads is
   worth real money on an agent loop that alternates workers.
3. Co-residency was stable: 100 % success over 344 steps, no leak, desktop intact.

**But it is a deliberate trade, and the escape hatch matters.** Free VRAM falls to ~1.0 GB,
below the ≥2 GB the specification asks for. If the desktop load grows (browser, IDE, video
call), WDDM may start spilling to shared system memory — which degrades latency *silently*
rather than failing. So:

- default to **Strategy 2** for agent workloads;
- fall back to **Strategy 1** when doing VRAM-heavy work, or when `nvidia-smi` shows free VRAM
  approaching 1 GB;
- **never** adopt Strategy 3.

This overturns the original specification's assumption that `local-smart` must be
**EXCLUSIVE**. It does not need to be: it can be **RESIDENT alongside `local-fast`**, with
inference serialised.

## 11. Known problems

1. **Criterion 7 failed.** The 9B did not outperform the 4B on any scored task in this suite.
   The escalation policy therefore does *not* treat the 9B as the default for "harder" work.
2. **Small fixture counts.** 3 repair fixtures, 2 review fixtures. The pass@1 gap rests on one
   fixture; treat per-row gaps as indicative, not settled.
3. **Review recall is low for both models** (58 % / 42 %) with the subtler defects
   (`lock_never_acquired`, duplicated logic) missed by both. Code review is a filter, not a gate.
4. **Neither model resolved the fact-integration test's current state** (both scored
   `current_state_correct = False`) — the conflicting PostgreSQL version defeated both.
5. **~1.0 GB free VRAM under co-residency**, below the spec's ≥2 GB target.
6. **Concurrent inference is a 47 % throughput trap** if a caller bypasses the semaphore.
7. `vram_watch.ps1` initially buffered samples and lost a whole series when stopped early; it now
   streams. Recorded because it invalidated a stability run before being fixed.

## 12. Recommendation

**Phase 3: PASS with one criterion explicitly unmet.**

Fourteen of fifteen criteria are satisfied. Criterion 7 ("clearly better than 4B on key Smart
benchmarks") is **not** met, and §22 of the specification explicitly permits the conclusion:

> **The 9B is deployed, works correctly, is stable and fits the card — but this benchmark does
> not justify it as a routine upgrade over the 4B.**

It retains value for exactly two measured purposes: **contexts beyond the 4B's 8K window**, and
**a second opinion when the 4B is doubted** (where it also shows fewer routing
under-escalations). It should not become the default for "difficult" tasks on the strength of
its parameter count.

**Suggested next step: Phase 4 (Embedding + Vector Retrieval).** Retrieval is the capability
most likely to change *what* these models can be asked, rather than how fast they answer, and
the embedding model is small enough not to disturb the co-residency picture. The 9B/4B routing
question should be revisited with a larger fixture set at that point.

**Phase 3 stops here.** No Phase 4 work has begun and no embedding model has been downloaded.
