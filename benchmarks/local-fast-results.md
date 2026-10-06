# local-fast — Phase 2 results

**Worker:** `local-fast` (Qwen3.5-4B) behind `llama-server` on `127.0.0.1:8111`
**Date:** 2026-09-29 · **Status:** measured on an **idle GPU** (no competing workload)

---

## 1. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop, Blackwell **sm_120**, 12227 MiB |
| Driver | **591.86** (CUDA 13.1 capability) |
| CUDA toolkit | not installed (`nvcc` absent); CUDA runtime DLLs supplied by `cudart` package |
| llama.cpp | **b11254**, commit `8019dc563`, `0.5.0-dev`, Clang 20.1.8, CUDA-13.4 backend |
| Host RAM | 31.5 GB (2 × 16 GB DDR5-5600) |
| Desktop VRAM baseline | ~2.3–2.6 GB (compositor + apps, no model loaded) |

## 2. Model

| Item | Value |
|---|---|
| Source | `Qwen/Qwen3.5-4B` rev `851bf6e8…` (Apache-2.0) → `unsloth/Qwen3.5-4B-GGUF` rev `e87f1764…` |
| Deployed quant | **Q4_K_M**, 2740.9 MB file, effective **5.193 bits/weight**, SHA-256 verified |
| Context | **8192** (`parallel: 1`) |
| Thinking mode | **disabled** via `chat_template_kwargs.enable_thinking=false` |

## 3. Runtime measurements (deployed configuration)

| Metric | Value |
|---|---|
| **VRAM, model resident** | **2981 MiB** GPU attribution (model 2603 + context 306 + compute 72) |
| VRAM, GPU total while loaded | 5484 MiB (≈ 2.5 GB desktop + 2981 MiB worker) |
| **VRAM headroom** | **6461 MiB free** — far above the ≥ 2048 MiB requirement |
| SysRAM | host attribution 515 MiB (model 497 + compute 18) |
| Prompt processing | **3974 t/s** (816-token prompt) |
| Generation | **128.6 t/s** |
| **TTFT** | **0.254 s** on an 816-token prompt; **0.13 s** mean on short structured prompts |
| Time to ready after start | ~2–3 s (warm page cache) |
| OOM | **none** |
| CUDA errors | **none** |

## 4. Quantisation comparison — Q4_K_M vs Q5_K_M vs Q6_K

`llama-bench`, `-ngl 99 -t 12 -p 512 -n 128 -r 5`, idle GPU. Raw: `results/quant_sweep.csv`.

| Quant | File size | Prompt t/s | **Gen t/s** | Gen vs Q4_K_M |
|---|---|---|---|---|
| **Q4_K_M** | 2614 MB | **4770** | **130.14** | — |
| Q5_K_M | 2998 MB | 4510 | 113.91 | **−12.5 %** |
| Q6_K | 3363 MB | 4089 | 102.57 | **−21.2 %** |

Both larger quants are **slower on both prompt and generation**, monotonically. Additional
structural facts (see `quant-analysis.md`):

* Q4_K_M is not really 4-bit — 30.8 % of its weight bytes are already **Q6_K**, so it already
  carries upgraded precision in quality-critical tensors.
* All three are **imatrix** quantisations (248 entries / 80 chunks).
* Q5_K_M costs +14.7 % size for a 12.5 % generation loss.

### Decision: **Q4_K_M**

The specification's rule permits Q5_K_M only if VRAM stays clearly under the safe limit, the
latency cost is acceptable, **and** quality is clearly better. Here:

* VRAM was never the deciding factor — both fit with >6 GB to spare;
* Q5_K_M is measurably **slower**, which directly harms the Fast Worker's stated latency goal;
* quality was **not** clearly better — Q4_K_M already scores **100 % JSON parse and schema
  adherence over 20/20 runs**, leaving no headroom for Q5_K_M to demonstrate.

Choosing a larger quant here would pay a real latency cost for a quality gain that the
measurements do not show. **Q4_K_M stands.**

## 5. Context sweep

`llama-bench` has no context flag, so each size was measured through the **deployed server**.
Raw: `results/ctx_probe_*.json`.

| Context | VRAM context | GPU `self` | Prompt t/s | Gen t/s | TTFT | Stable |
|---|---|---|---|---|---|---|
| 4096 | **178 MiB** | 2849 MiB | 4061 | 127.49 | 0.245 s | ✅ |
| **8192** | **306 MiB** | 2981 MiB | 3974 | 128.62 | 0.254 s | ✅ |
| 16384 | **562 MiB** | 3245 MiB | 3982 | 127.27 | 0.244 s | ✅ |

**Two results worth stating plainly:**

1. **The KV model predicted these exactly.** The CPU-derived fit
   (`context = 50 MiB + ctx × 32,768 B/token`) predicts 178 / 306 / 562 MiB — the GPU reported
   178 / 306 / 562. Agreement on all three, to the MiB.
2. **Throughput is flat across context** and TTFT stays ~0.25 s. 16K costs 256 MiB more than
   8K and does not slow generation. The 8192 default is therefore a policy choice, not a
   hardware limit — 16K is available for high-value tasks at negligible cost.

The earlier 457 MiB anomaly at 8192 is fully explained: it came from llama-server's default
**4-slot unified KV pool**, removed by `parallel: 1` (see `quant-analysis.md`).

## 6. Quality results

Full suite, Q4_K_M, ctx 8192, deterministic preset. Raw: `results/q4km_8k.json`.

| Test | Result |
|---|---|
| **A** 中文摘要 | 5/5 bullets, **7/10 facts**, **0 hallucinations**, 0 invented numbers |
| **B** JSON extraction | **parse 100.0 %, schema 100.0 % (20/20)** without grammar or JSON-schema constraints |
| **C** routing | **72.7 % (16/22)**, parse 100 % |
| **D** traceback | all four checks pass — correct exception, file, line, and a minimal fix |
| **E** code | `json_to_csv` **exec=True**, `sql_group_by` **exec=True**, shell + pytest structural 4/4 and 3/3 |
| **F** handoff compression | **9/9 sections**, 0.311 ratio, 6/6 concrete anchors |
| Thinking leaks | **0** across every call |

### Test C analysis — the weakest result, and why

All 6 misses are **systematic, not random**, and they cluster into three causes:

1. **Consistently one tier too low** (4 cases). RBAC design, a 3-year roadmap, a race-condition
   proof and a GC-behaviour analysis were all routed to `local-smart` where the key expects
   `claude` or `deepseek`. The model's stated reasoning is coherent — it is applying the
   supplied guidance more literally than my answer key assumes.
2. **No ASR route exists.** "把会议录音转成文字纪要" went to `local-ocr`; the valid-route list
   has no `local-asr` even though the specification defines an ASR worker. This is a **defect in
   my route taxonomy**, not a model error.
3. **One genuinely ambiguous label.** The production-safety case was answered `human`, which is
   defensible under "irreplaceable/dangerous → human"; my key says `claude`.

**Interpretation:** 72.7 % should be read as a *lower bound*. Two of the six "failures" are
arguably better answers than the key. Routing is nevertheless **not** production-ready as a
dispatcher and should be used only as advice, as the integration guide says.

**Follow-up (not done, belongs to Phase 11):** add `local-asr` to the taxonomy, and either
relax the key for tier-boundary cases or sharpen the routing guidance about when to escalate.

## 7. Repeatability (≥20×)

Test B was executed **20 times with 20 distinct inputs** (different task_id, project, priority,
GPU flag, worker and status each time — repeating one prompt would measure caching, not schema
adherence):

| Metric | Result |
|---|---|
| Valid JSON | **20/20 — 100 %** |
| Schema complete and correct | **20/20 — 100 %** |
| Field errors | **0** |
| Constraints used | none — no grammar, no `response_format` |

Reinforced by the stability run: **800 further calls, 100 % success**. For a machine-parsed task
this is the target outcome, and it means **GBNF grammar and JSON-schema constraints were not
needed** for Q4_K_M at temperature 0. They remain available for weaker configurations.

## 8. Sampling presets — measured, then fixed

A/B on the human-read tasks (Test A and Test F):

| Preset | Test A facts | Hallucinations | Test F |
|---|---|---|---|
| `deterministic` (temp 0.0, top_p 1.0, top_k 20, min_p 0.0) | 7/10 | 0 | 9/9 sections |
| **`general` (temp 0.3, top_p 0.9, top_k 40, min_p 0.05)** | **10/10** | 0 | 9/9 sections |

**The split is justified by measurement, not assumption:** temperature 0 costs real information
on summarisation (7/10 → 10/10) while `deterministic` is what delivers the 100 % structured
success rate. Values in `config/local-fast.yaml` are confirmed as deployed.

## 9. Long-run stability

10 minutes continuous cycling (summary + 4× JSON + routing + SQL + traceback per iteration),
VRAM sampled every second. Raw: `results/stability.json`, `results/vram_stability.csv`.

| Metric | Result |
|---|---|
| Iterations / total steps | **100 / 800** |
| Success rate | **100.0 %** |
| Errors | **0** |
| Crashes | **0** |
| Thinking leaks | **0** |
| Latency mean | 0.75 s |
| Latency drift (first → last quartile) | **+0.8 %** (0.746 → 0.751 s) |
| Generation throughput, mean | 124.81 t/s |
| **VRAM first → last** | **5696 → 5715 MiB (+19 MiB)** |
| VRAM peak | 5803 MiB |
| VRAM samples | 569 |

**No leak.** +19 MiB over 10 minutes and 800 calls is noise, not growth; peak exceeded the
final value, which a leak would not do. Latency is flat, so there is no degradation. The worker
is suitable for long-running use.

## 10. Deliverables

| File | Purpose |
|---|---|
| `config/local-fast.yaml` | single source of truth (model, runtime, presets) |
| `scripts/start-local-fast.ps1` / `stop-local-fast.ps1` / `check-local-fast.ps1` | lifecycle |
| `scripts/lf_config.py` | stdlib-only config loader (no PyYAML dependency) |
| `scripts/test_lifecycle_gates.ps1` | 10 refusal-path regression tests, all passing |
| `docs/models/local-fast.md` | model record and provenance |
| `docs/local-fast-api.md` | upstream integration contract |
| `benchmarks/local-fast/` | fixed task set, harness, fixtures, mock server |
| `benchmarks/local-fast/quant-analysis.md` | weight-level quantisation analysis |
| this file | measured results |

## 11. Phase 2 acceptance criteria (spec §17)

| # | Criterion | Status |
|---|---|---|
| 1 | Qwen3.5-4B correctly deployed | ✅ |
| 2 | GPU offload explicitly verified | ✅ 94 % util / 139 W under load, `compute capability 12.0` |
| 3 | OpenAI-compatible API callable | ✅ `/health`, `/v1/models`, `/v1/chat/completions` |
| 4 | `local-fast` alias usable | ✅ served via `--alias` |
| 5 | 8K context stable | ✅ plus 16K measured |
| 6 | No persistent CUDA/OOM problems | ✅ 0 errors across ~1000 calls |
| 7 | JSON extraction reliably parseable | ✅ 20/20, and 800/800 in stability |
| 8 | Chinese summary usable | ✅ 5/5 bullets, 7–10/10 facts, 0 hallucinations |
| 9 | Simple code usable | ✅ both executable cases pass |
| 10 | Context compression produces structured handoff | ✅ 9/9 sections |
| 11 | Final quant choice benchmark-backed | ✅ Q4_K_M |
| 12 | Reasonable VRAM headroom | ✅ 6461 MiB free |
| 13 | Start/stop/check scripts work | ✅ 10/10 gate tests, real start/stop validated |
| 14 | Model source and version fully recorded | ✅ |
| 15 | Phase 1 and Qwen-Image environments intact | ✅ untouched |

## 12. Known issues

1. **Routing accuracy 72.7 %** — systematic under-escalation plus a missing `local-asr` route.
   Use as advice only. Fix belongs to Phase 11.
2. **Test A recalls 7/10 facts at temperature 0** — use the `general` preset when completeness
   matters more than determinism.
3. **`docs/local-fast-api.md` was written before measurement** and still quotes 8192 as the
   assumption; the numbers there are consistent with these results but the measured tables here
   are authoritative.
4. **No autostart by design** — a cold start costs ~2–3 s with warm page cache.
5. **Qwen3.5-4B is natively a VLM**; only the text path is deployed, so images are not accepted.
