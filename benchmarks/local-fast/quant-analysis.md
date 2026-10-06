# local-fast — quantisation structural analysis

**Scope:** Phase 2, weight-level comparison of the three downloaded quantisations.
**Method:** direct GGUF container decode via [`inspect_gguf.py`](inspect_gguf.py) (stdlib only, no GPU).
**Raw data:** [`results/gguf_quants.json`](results/gguf_quants.json)
**Measured:** 2026-09-29

> This file records what is knowable **without a GPU**: file structure, where the bits
> actually went, and the resulting memory budget. Throughput, latency and answer quality
> require the GPU and are recorded in `../local-fast-results.md`.

---

## 1. Container facts

| Property | Q4_K_M | Q5_K_M | Q6_K |
|---|---|---|---|
| File size | **2740.9 MB** | 3143.7 MB | 3526.0 MB |
| Weight bytes | 2730.0 MB | 3132.7 MB | 3515.0 MB |
| Non-weight overhead | 11.0 MB | 11.0 MB | 11.0 MB |
| Overhead % | 0.40 % | 0.35 % | 0.31 % |
| Tensors | 426 | 426 | 426 |
| Parameters | 4,205,751,296 | same | same |
| **Effective bits/weight** | **5.193** | **5.959** | **6.686** |
| Still full precision | 3.8 MB (0.14 %) | 3.8 MB (0.12 %) | 3.8 MB (0.11 %) |

The 11 MB non-weight overhead is identical across all three, as expected — it is the
tokeniser (248,320-entry vocab), metadata and alignment, not weights.

## 2. Where the bits actually went

### Q4_K_M — 5.193 bpw, not "4-bit"
| Type | Bytes | Share | Tensors |
|---|---|---|---|
| Q4_K | 1361.0 MB | 49.85 % | 131 |
| **Q6_K** | **841.9 MB** | **30.84 %** | 22 |
| Q5_K | 519.0 MB | 19.01 % | 48 |
| Q8_0 | 4.2 MB | 0.15 % | 48 |
| F32 | 3.8 MB | 0.14 % | 177 |

A third of the weight bytes are already at **Q6_K** — the `_M` in `Q4_K_M` means
quality-critical tensors (attention V/output, feed-forward down projections) were
upgraded above the nominal 4-bit. The label understates the real precision.

### Q5_K_M — 5.959 bpw
| Type | Bytes | Share | Tensors |
|---|---|---|---|
| Q5_K | 1663.5 MB | 53.10 % | 131 |
| Q6_K | 1461.2 MB | 46.64 % | 70 |
| Q8_0 | 4.2 MB | 0.13 % | 48 |
| F32 | 3.8 MB | 0.12 % | 177 |

### Q6_K — 6.686 bpw
| Type | Bytes | Share | Tensors |
|---|---|---|---|
| Q6_K | 3239.6 MB | 92.16 % | 177 |
| Q8_0 | 271.6 MB | 7.73 % | 72 |
| F32 | 3.8 MB | 0.11 % | 177 |

**Step cost.** Q5_K_M is **+402.8 MB (+14.7 %)** over Q4_K_M; Q6_K is **+785.1 MB (+28.6 %)**.

## 3. Provenance is verifiable from the file itself

Every quantisation embeds its own audit trail:

```text
general.quantized_by      = Unsloth
general.license           = apache-2.0
general.base_model.0.repo = https://huggingface.co/Qwen/Qwen3.5-4B
quantize.imatrix.file     = Qwen3.5-4B-GGUF/imatrix_unsloth.gguf
quantize.imatrix.dataset  = unsloth_calibration_Qwen3.5-4B.txt
quantize.imatrix.entries_count = 248
quantize.imatrix.chunks_count  = 80
```

Two things worth noting:

1. The **license and upstream repo are declared inside the weights file**, not just on the
   web page — so provenance survives even if the file is copied elsewhere.
2. These are **imatrix (importance-matrix) quantisations**, calibrated on a real dataset
   (248 entries / 80 chunks). This matters: imatrix quants preserve quality noticeably
   better at low bit-widths than naive quantisation, which strengthens the case for
   accepting Q4_K_M rather than assuming a higher quant is needed.

## 4. Architecture discovered from the weights

| Key | Value |
|---|---|
| `general.architecture` | **`qwen35`** |
| Layers (`block_count`) | 32 |
| `full_attention_interval` | **4** → only **8 full-attention layers** |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | **4** |
| `attention.key_length` / `value_length` | 256 / 256 |
| `ssm.state_size` / `ssm.inner_size` | 128 / 4096 |
| `ssm.conv_kernel` / `ssm.group_count` | 4 / 16 |
| Native `context_length` | 262,144 |

**Correction to the model record:** the architecture key is **`qwen35`**, not `qwen3_5` as
the HuggingFace `qwen3_5` tag suggests. `docs/models/local-fast.md` has been corrected.

The `ssm.*` keys plus `full_attention_interval = 4` independently confirm the hybrid design
described in the model card: 24 of 32 layers are linear-attention (Gated DeltaNet, fixed-size
recurrent state), and only 8 are conventional full attention.

## 5. The memory budget — the decisive number

KV cache per token, from the real metadata (f16):

```text
8 full-attention layers × 4 KV heads × 256 width × 2 (K and V) × 2 bytes = 32,768 B/token = 32 KiB
```

Note on reading `key_length`/`value_length`: for this model the metadata reports **256**, which is
the **full KV projection width already** (`n_kv_heads × head_dim = 4 × 64 = 256`), not a per-head
dimension. The cache is therefore `2 tensors × 256 elements × 2 bytes = 1024 B` per layer per
token. Writing it as `2 × (key_length + value_length)` double-counts V — an earlier revision of
the inspector did exactly that, and the value only looked right because it was then multiplied by
the same factor again. `inspect_gguf.py` now emits an explicit `kv_formula` string, and
`test_grading.py` pins the arithmetic (32,768 B/token) so it cannot drift.

| Context | KV cache (f16) | Q4_K_M weights | Total weights+KV |
|---|---|---|---|
| 4K | **134.2 MB** | 2730 MB | 2864 MB |
| **8K (planned default)** | **268.4 MB** | 2730 MB | **2998 MB** |
| 16K | **536.9 MB** | 2730 MB | 3267 MB |
| 32K | 1073.7 MB | 2730 MB | 3804 MB |

**How much does the hybrid design actually save? 4×.** Holding KV geometry constant, a version
with all 32 layers as full attention would need `32/8 = 4×` more cache (128 KiB/token instead of
32 KiB). The entire saving comes from `full_attention_interval = 4`.

Do **not** quote a larger figure by comparing against a conventional 4B with 8 KV heads and
128-dim heads — that differs in KV geometry too, so it is not a like-for-like comparison.

**Consequences for the 12 GB card:**

* At the planned 8K default the model + KV is ~**3.0 GB**. Even allowing ~1 GB for CUDA
  context, compute buffers and graph, the worker should sit around **4–4.5 GB resident**.
* With ~2 GB consumed by the Windows desktop, that leaves roughly **5–6 GB genuinely free** —
  comfortably above the ≥2 GB headroom the specification requires.
* **16K is affordable** if a high-value task needs it (3.27 GB total): the context sweep is
  therefore a real measurement question rather than a memory-risk question.
* Q5_K_M at 16K would still be ~3.6 GB total, so a higher quant does not by itself endanger
  the VRAM budget. **If Q5_K_M is rejected, it should be on latency grounds, not memory.**

## 5b. Measured against llama.cpp's own accounting

llama.cpp emits a `fit_memory_breakdown` event on load. One was captured during a gate test
that accidentally launched the server (Q4_K_M, `-c 8192`):

```text
GPU  CUDA0 : model 2603 MiB | context 457 MiB | compute 72 MiB | self 3132 MiB
HOST Host  : self 515 MiB | model 497 MiB | compute 18 MiB
             n_slots = 4, n_ctx_slot = 8192, kv_unified = true
```

Two discrepancies from the analytical model above, both now resolved or bounded:

1. **Model weights match almost exactly** — 2603 MiB measured vs a 2740.9 MB file. This
   independently validates using ΔVRAM as a weight-accounting method (Phase 1 did the same).
2. **Context measured 457 MiB where the KV arithmetic predicts 256 MiB.** Resolved below:
   the *per-token* formula is exact; the difference is fixed backend overhead.

### Resolving the context question — SOLVED

The CPU backend reports a `host` row in the same breakdown event, so context scaling can be
measured without competing for the GPU. Sweeping `-c` with `-ngl 0`:

| ctx | context MiB | Δ | implied cost |
|---|---|---|---|
| 4096 | 178 | — | — |
| 8192 | 306 | +128 MiB | **32,768 B/token** |
| 16384 | 562 | +256 MiB | **32,768 B/token** |
| 32768 | 1074 | +512 MiB | **32,768 B/token** |

Least-squares fit: `context MiB = 50.0 + ctx × 0.03125`, i.e.
**`context = 50 MiB fixed + ctx × 32,768 B/token`**.

The slope is **exactly** the 32,768 B/token the metadata predicts, identical across all three
intervals. **The per-token KV model is confirmed to the byte.**

**And the GPU disagreement is now closed.** A second GPU measurement was taken with
`runtime.parallel = 1`:

| | 4 slots (default) | **1 slot** |
|---|---|---|
| `n_slots` / `kv_unified` | 4 / `true` | **1 / `false`** |
| context | 457 MiB | **306 MiB** |
| compute | 72 MiB | 72 MiB |
| model | 2603 MiB | 2603 MiB |
| **GPU `self` total** | **3132 MiB** | **2981 MiB** |

**306 MiB is exactly what the CPU-validated model predicts for 8192 tokens**
(50 + 8192 × 32768 B = 306 MiB). So:

* the KV model is right on **both** backends;
* the 457 MiB figure was caused by the default multi-slot unified KV pool, which
  overallocated context memory at roughly 51,200 B/token — about **1.56× the true KV cost**;
* `runtime.parallel = 1` removes that overallocation entirely.

The earlier retracted guess named the right culprit (`n_slots`) but with the wrong mechanism
and magnitude — the pool is not simply `4 × context` (which would be 1024 MiB), it overallocates
by a factor of about 1.56. The prediction, not the explanation, was wrong. Worth noting as a
reminder that a correct instinct is not the same as a verified claim.

**Consequence for the benchmark plan:** the GPU context sweep is no longer needed to explain a
discrepancy, but is still worth running at 8K and 16K to confirm the model holds with the GPU
backend across the range actually deployed.

**Action taken:** `runtime.parallel = 1`, on two grounds that do not depend on the unexplained
excess:
* with 4 slots each request gets `8192 / 4 = 2048` tokens of usable context, so a 4K-summary
  task would not fit; one slot gives the full 8192-token window, which is what a single-user
  worker wants;
* the measured 3132 MiB total is over the 2048 MB the specification reserves as headroom on a
  card where the desktop already holds ~2 GB.

The real effect of `parallel = 1` on the context figure must be **measured**, not assumed.

**Caveat on this measurement.** It was taken while an unrelated game held the GPU at ~100 %
utilisation, and it predates the `parallel = 1` change. The *shape* of the breakdown is
trustworthy (it is llama.cpp's own accounting); the totals need re-measuring on an idle GPU.
`runtime.expected_vram_mb` was set to 3200 from this figure, and the authoritative measured
numbers belong in `../local-fast-results.md` once the card is free.

**Revised expectation:** ~3.13 GB resident at 8K (measured, before the `parallel = 1` change).
The KV *term* in the tables above is now verified exactly; what remains unquantified is the
GPU-side extra allocation described below. The authoritative GPU numbers belong in
`../local-fast-results.md` once the card is free and a context sweep has been run.

## 6. What this does and does not settle

**Settled without a GPU:**
- provenance and license, from the weights themselves
- effective precision (5.193 / 5.959 / 6.686 bpw) rather than nominal quant names
- memory budget, and that KV cache is unusually cheap on this architecture
- that Q5_K_M's cost is +14.7 % size, which is affordable in VRAM terms

**Not settled, and must not be claimed yet:**
- tokens/s, prompt-processing speed, TTFT
- JSON/schema adherence, routing accuracy, summary faithfulness, code correctness
- whether Q4_K_M's 5.193 bpw is *enough* for the structured tasks — the deciding question

The specification's guidance is explicit: choose Q5_K_M only if VRAM stays clearly under
the safe limit, the latency cost is acceptable, **and** quality is clearly better; otherwise
prefer Q4_K_M to protect the low-latency goal. The memory half of that test now passes for
both. The latency and quality halves require the GPU.
