# local-smart — Model Record (Phase 3)

**Logical alias:** `local-smart`
**Role:** Local Smart Worker — tasks needing real analysis rather than mechanical speed
**Phase:** 3
**Recorded:** 2026-09-30

> Upper layers must only ever reference the alias **`local-smart`**. GGUF filenames,
> repository names and quantisation names must never appear in agent code or prompts.

---

## 1. Source chain

| Step | Artefact | Value |
|---|---|---|
| Upstream model | `Qwen/Qwen3.5-9B` | official Qwen organisation |
| Upstream revision | commit | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Upstream last modified | | 2026-03-02 |
| Upstream downloads | | 9,586,772 · 2,064 likes |
| Model type | | **post-trained instruct/chat — NOT Base** |
| GGUF source | `unsloth/Qwen3.5-9B-GGUF` | same trust chain as the Phase 2 Fast Worker |
| GGUF revision | commit | `3885219b6810b007914f3a7950a8d1b469d598a5` |
| License (both) | | **Apache-2.0** |
| Gated | | No |

**Why this GGUF source.** As in Phase 2, `Qwen/Qwen3.5-9B` publishes **safetensors only** —
there is no official Qwen GGUF — so the specification's priority (2), a high-trust conversion,
applies. Unsloth was selected for the same verified reasons as Phase 2 (declared `base_model`,
full quant ladder, imatrix calibration, consistently smaller files at equal nominal quant). The
alternative `bartowski/Qwen_Qwen3.5-9B-GGUF` was inspected and is also Apache-2.0 and traceable;
it was not needed.

`Qwen/Qwen3.5-9B-Base` exists and was **deliberately not used** — the task requires an
instruct/chat model.

## 2. Architecture (read from the weights, not the model card)

| Property | 9B | 4B (for comparison) |
|---|---|---|
| Parameters (counted from tensors) | **8,953,803,264** | 4,205,751,296 |
| Architecture key | `qwen35` | `qwen35` |
| Hidden size | 4096 | 2560 |
| Layers | **32** | 32 |
| **Full-attention layers** | **8** (`full_attention_interval = 4`) | 8 |
| Linear-attention layers | 24 (Gated DeltaNet) | 24 |
| Attention heads (Q) | 16 | 16 |
| **KV heads** | **4** | 4 |
| key_length / value_length | 256 / 256 | 256 / 256 |
| FFN intermediate | 12288 | 9216 |
| Native context | 262,144 | 262,144 |
| **KV bytes per token (f16)** | **32,768** | 32,768 |

**The KV geometry is identical to the 4B.** Both have 8 full-attention layers × 4 KV heads ×
256 width. Consequently context costs exactly the same in both models, and the **weight delta is
the only material VRAM difference**. This was the key fact that let co-residency be predicted
before any download finished.

## 3. Deployed quantisations

All three downloaded at the pinned revision and **SHA-256 verified**.

| Quant | File | Size | Effective bpw | SHA-256 |
|---|---|---|---|---|
| **Q4_K_M** (deployed) | `Qwen3.5-9B-Q4_K_M.gguf` | 5417 MB | **5.066** | `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8` |
| Q5_K_M | `Qwen3.5-9B-Q5_K_M.gguf` | 6273 MB | — | `dc2a39aef291f91a9116ad214058da0d86eb648743a124bd8c333787c4b9c91c` |
| Q6_K | `Qwen3.5-9B-Q6_K.gguf` | 7113 MB | — | `91898433cf5ce0a8f45516a4cc3e9343b6e01d052d01f684309098c66a326c59` |

imatrix calibration: `unsloth_calibration_Qwen3.5-9B.txt`, 248 entries / 80 chunks.
Measured Q4_K_M at 5.066 bpw — like the 4B, the nominal label understates real precision.

## 4. Runtime contract

| Item | Value |
|---|---|
| Server | `llama-server` from the Phase 1 build **b11254** (commit `8019dc563`) |
| Bind | **`127.0.0.1:8112`** — loopback only |
| Model alias | `local-smart` |
| Context | **16384** (Phase 3 target; 8K and 32K also measured) |
| GPU offload | `-ngl 99` |
| Threads | 12 |
| Thinking mode | **disabled by default**; available only via the `reasoning` preset |

Config: [`config/local-smart.yaml`](../../config/local-smart.yaml).

### On thinking mode

Measured on this hardware (`benchmarks/local-smart/results/think_ab.json`):

| Task | non-think | think |
|---|---|---|
| Structured extraction | 0.57 s, 36 tokens, correct | 9.79 s, 710 tokens, correct |
| Hard root-cause analysis | 10.48 s, 742 tokens, **4/4** | 16.36 s, 1200 tokens (cap hit), **0/4** |

Thinking cost **17× the latency and 20× the tokens on a trivial extraction task for zero gain**,
and on the hard task it spent its whole budget reasoning and never produced a scored answer.
**Reasoning must never be auto-enabled.** It remains available as the opt-in `reasoning` preset.

## 5. Intended use

**Intended:** multi-file code understanding, traceback root-cause work, ordinary bug repair,
unit-test generation, moderate Python/Shell/SQL, project documentation summarisation,
multi-document integration, context compression, handoff generation, first-pass code review,
preparation of context for DeepSeek/Claude/Codex, and second-pass review of local results.

**Not for:** large architecture design, high-risk security decisions, unverifiable complex
engineering changes, large cross-module refactors, production deployment decisions.

> **Important measured caveat.** On this benchmark the 9B did **not** beat the 4B on any of the
> scored tasks, and lost on several. See
> [`benchmarks/local-smart-results.md`](../../benchmarks/local-smart-results.md) §"Fast vs Smart"
> and [`docs/escalation-policy.md`](../escalation-policy.md). The escalation policy, not this
> file, decides which worker a task should go to.
