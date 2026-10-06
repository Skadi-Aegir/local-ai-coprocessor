# local-fast — Model Record (Phase 2)

**Logical alias:** `local-fast`
**Role:** Local Fast Worker — high-frequency, low-cost, verifiable tasks
**Phase:** 2
**Recorded:** 2026-09-29

> Upper layers must only ever reference the alias **`local-fast`**. GGUF filenames,
> repository names and quantisation names must never appear in agent code or prompts.

---

## 1. Source chain

| Step | Artefact | Value |
|---|---|---|
| Upstream model | `Qwen/Qwen3.5-4B` | official Qwen organisation |
| Upstream revision | commit | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Upstream last modified | | 2026-03-02 |
| Upstream downloads | | 7,229,569 · 977 likes |
| GGUF source | `unsloth/Qwen3.5-4B-GGUF` | official-derived quantisation by Unsloth |
| GGUF revision | commit | `e87f176479d0855a907a41277aca2f8ee7a09523` |
| GGUF `base_model` | declared in card | `Qwen/Qwen3.5-4B` ✅ matches upstream |
| License (both) | | **Apache-2.0** (no commercial restriction) |
| Gated | | No |

**Why this GGUF source.** The specification's priority order is (1) official Qwen repo,
(2) official or high-trust GGUF conversion. `Qwen/Qwen3.5-4B` publishes **safetensors only** —
there is no official Qwen GGUF for this model — so priority (2) applies. Unsloth was chosen
over the alternative (`bartowski/Qwen_Qwen3.5-4B-GGUF`, also Apache-2.0, verified) because:

* its card explicitly declares `base_model: Qwen/Qwen3.5-4B` (traceable provenance);
* it ships the full quant ladder including `UD-*` dynamic quants and `imatrix_unsloth.gguf_file`;
* its files are consistently smaller at equal nominal quant (Q4_K_M: 2.55 GB vs 2.81 GB),
  which suits a 12 GB card;
* higher community adoption (999 k vs 261 k downloads).

Both candidates were inspected; neither is an anonymous re-upload. No unknown source,
no network-drive mirror and no unversioned GGUF was used.

### Important architectural finding

`Qwen3.5-4B` is **not a text-only model**. The official card states
*"Type: Causal Language Model with Vision Encoder"* and `pipeline_tag: image-text-to-text`;
the repo ships `preprocessor_config.json` and `video_preprocessor_config.json`, and the GGUF
repo provides separate `mmproj-*.gguf` projector files.

**This deployment uses the text path only.** The `mmproj` projector was deliberately *not*
downloaded, and the Phase 1 llama.cpp build reports `modalities: text` for the loaded model.
Consequence: this worker cannot accept images. Visual work is reserved for
`local-vision` (Phase 8), so the §2 resource plan is unaffected. This is recorded here so a
later phase does not assume the VL capability already exists.

---

## 2. Parameters and architecture

| Property | Value |
|---|---|
| Parameters | **4B** (as planned — no substitution) |
| Architecture (GGUF key) | **`qwen35`** — note: the HF tag says `qwen3_5`; the weights say `qwen35` |
| Hidden dimension | 2560 |
| Layers | **32** |
| **Full-attention layers** | **8** (every 4th; `full_attention_interval = 4`) |
| Linear-attention layers | 24 (Gated DeltaNet) |
| Layer layout | `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))` |
| Gated DeltaNet | 32 linear-attention heads (V), 16 (QK), head dim 128, `ssm.state_size` 128, `ssm.inner_size` 4096, conv kernel 4, group count 16 |
| Gated Attention | 16 Q heads, **4 KV heads**, key/value length 256, RoPE dim 64 |
| FFN intermediate | 9216 |
| Vocabulary | 248,320 (padded, tied embeddings) |
| Native context | 262,144 (extensible to ~1,010,000) |
| Languages | 201 |
| MTP | trained with multi-steps |
| Parameters (counted from tensors) | 4,205,751,296 |

**KV-cache implication (matters for the 12 GB budget).** Only **8 of 32 layers** are
full attention, and those use just **4 KV heads**. The remaining 24 layers are Gated DeltaNet
linear attention with fixed-size recurrent state. Measured from the real metadata, the KV cache
is **32 KiB per token**:

```text
8 full-attention layers × 4 KV heads × 256 width × 2 (K and V) × 2 bytes (f16) = 32,768 B/token
```

Careful with `key_length`/`value_length` = 256 here: that is the **full KV projection width**
(`n_kv_heads × head_dim = 4 × 64`), not a per-head dimension. Writing the cache as
`2 × (key_length + value_length)` double-counts V. `inspect_gguf.py` emits an explicit
`kv_formula`, and `test_grading.py` pins the arithmetic.

| Context | KV cache | Q4_K_M weights + KV |
|---|---|---|
| 4K | 134 MB | 2864 MB |
| **8K (default)** | **268 MB** | **2998 MB** |
| 16K | 537 MB | 3267 MB |
| 32K | 1074 MB | 3804 MB |

**The hybrid design saves 4×**, not more: with identical KV geometry, 32 full-attention layers
would need `32/8` times the cache. Do not inflate this by comparing against a conventional 4B
with a different KV geometry.

This is why 16K stays affordable on this card, and why the quant decision is a
**latency/quality** question rather than a memory one. Full analysis:
`benchmarks/local-fast/quant-analysis.md`.

---

## 3. Deployed quantisations

All three were downloaded at the pinned GGUF revision and **SHA-256 verified**.

| Quant | File | Size | Effective bpw | SHA-256 | Status |
|---|---|---|---|---|---|
| **Q4_K_M** | `Qwen3.5-4B-Q4_K_M.gguf` | 2740.9 MB | **5.193** | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` | **deployed default** |
| Q5_K_M | `Qwen3.5-4B-Q5_K_M.gguf` | 3143.7 MB | 5.959 | `8814232b85594dcd46c50e5b8b29324a7efe9e746edbe8a3d1df3d3fce7aad39` | A/B candidate |
| Q6_K | `Qwen3.5-4B-Q6_K.gguf` | 3526.0 MB | 6.686 | `fdedd781c9ce676ab66b018ca247ff78e8a33c98098a822c1e2d5075e7718f66` | optional |

Note that the nominal labels understate real precision: **Q4_K_M is 5.193 bits/weight, not 4** —
a third of its weight bytes (841.9 MB) are already Q6_K, and critical tensors were upgraded by
the `_M` mixing rule. All three are **imatrix** quantisations
(`quantize.imatrix.dataset = unsloth_calibration_Qwen3.5-4B.txt`, 248 entries / 80 chunks),
which preserve quality better at low bit-width than naive quantisation.

Selection rationale, per-type breakdown and the memory budget:
[`quant-analysis.md`](../../benchmarks/local-fast/quant-analysis.md). The final choice is made
on measured latency, VRAM and quality — see `benchmarks/local-fast-results.md`. Q4_K_M is the
*provisional* default, chosen for the low-latency goal, not because it downloaded first.

Location: `models/llm/` (weights are git-ignored — spec §5).

---

## 4. Runtime contract

| Item | Value |
|---|---|
| Server | `llama-server` from Phase 1 build **b11254** (commit `8019dc563`) |
| Bind | **`127.0.0.1:8111`** — loopback only, no firewall rule, not exposed to LAN |
| Model alias | `local-fast` (via `--alias`) |
| Context | 8192 (default) |
| GPU offload | `-ngl 99` (all layers) |
| Threads | 12 |
| Thinking mode | **disabled** — Qwen3.5 defaults to `<think>`; sent as `chat_template_kwargs.enable_thinking=false` |

Config: [`config/local-fast.yaml`](../../config/local-fast.yaml) — single source of truth.
Lifecycle: [`scripts/start-local-fast.ps1`](../../scripts/start-local-fast.ps1),
[`scripts/stop-local-fast.ps1`](../../scripts/stop-local-fast.ps1),
[`scripts/check-local-fast.ps1`](../../scripts/check-local-fast.ps1).

### Thinking mode is not optional

The Qwen3.5 chat template emits `<think>` unless `enable_thinking` is explicitly `false`.
A reasoning trace before every answer would multiply token count and destroy the Fast Worker's
latency target for mechanical tasks. All deterministic call paths **must** pass
`enable_thinking: false`.

---

## 5. Intended and forbidden use

**Intended:** classification, routing assistance, JSON/structured extraction, log and file
summarisation, commit messages, simple translation, prompt/context compression, traceback
triage, simple Python/Shell/SQL, simple test generation, test-result interpretation, small
change suggestions, handoff assembly.

**Forbidden** (per spec §2.1): complex system architecture, security-critical changes,
complex mathematical reasoning, large-scale refactors, and decisions that cannot be verified.

---

## 6. Reproducing the download

```powershell
$base = 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523'
Invoke-WebRequest -Uri "$base/Qwen3.5-4B-Q4_K_M.gguf" -OutFile 'D:\MultiAgent\models\llm\Qwen3.5-4B-Q4_K_M.gguf'
(Get-FileHash 'D:\MultiAgent\models\llm\Qwen3.5-4B-Q4_K_M.gguf' -Algorithm SHA256).Hash
# must equal 00FE7986FF5F6B463E62455821146049DB6F9313603938A70800D1FB69EF11A4
```

Pin the revision — do not resolve `main`, which can move under a rolling release.
