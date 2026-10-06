# llama.cpp Baseline — Phase 1

**Project:** RTX 5070 Ti 12GB Local AI Coprocessor
**Host:** `<hostname>` · Windows 11 build 26200 · RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12227 MiB)
**Date:** 2026-09-29
**Status:** ✅ **Phase 1 ACCEPTED** — CUDA GPU offload verified with real inference, not just device enumeration

---

## 1. Deployment decision and why

The specification (§4) requires llama.cpp with CUDA, and gates on CUDA ≥ 12.8 for Blackwell.
Phase 0 established:

* Driver **591.86** exposes **CUDA 13.1** → the 12.8 floor is satisfied with margin.
* **No CUDA toolkit is installed** (`nvcc` absent) → a from-source build is not possible today.* VS 2022 with MSVC 14.44, CMake 3.31.6 and Ninja 1.12.1 **are** present (so only `nvcc` is missing).

Two viable paths were considered:

| Option | Action required | Verdict |
|---|---|---|
| **A. Official prebuilt CUDA binary** | download only | ✅ **CHOSEN** |
| B. Build from source | install CUDA Toolkit (~3–5 GB) + build | deferred to Phase 2 if needed |

Option A was chosen because it requires **no** driver change, **no** toolkit install and **no**
new system dependency, which keeps Phase 1 strictly within §17's "no irreversible system
operations" boundary. Build recipe taken from the official release assets of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

**Important finding for later phases:** the official Windows CUDA assets are published for
CUDA **12.4** and **13.4** — there is no 12.8 Windows asset — and the release binaries do **not**
ship the CUDA runtime. A separate `cudart-*.zip` must be downloaded alongside the binary or the
CUDA backend will fail to load. This was discovered and handled during installation.

---

## 2. What was actually installed

Everything lives under the isolated project tree `D:\MultiAgent`. **Nothing was installed
system-wide**, no PATH change, no registry change, no pre-existing environment touched.

| Component | Version / file | Size | SHA256 |
|---|---|---|---|
| llama.cpp build | **b11254**, commit **8019dc563** (`0.5.0-dev`) | — | — |
| Binary package | `llama-b11254-bin-win-cuda-13.4-x64.zip` | 146.4 MB | `86CEB1AFDF0C4E667BFCFB879D8710A6A7606E93F4D603280178DC9E786FCDFB` |
| CUDA runtime | `cudart-llama-b11254-bin-win-cuda-13.4-x64.zip` | 403.9 MB | `738F8C251AC22B70C3AE6F83A10CF222725DF0395246A2CF58F32BDB85FBE668` |
| Install dir | `D:\MultiAgent\tools\llama.cpp\b11254-cuda13.4` | | |
| Build compiler | Clang 20.1.8 for Windows x86_64 (official build) | | |

Runtime DLLs supplied by the `cudart` package (these are what make the CUDA backend loadable):

```text
cudart64_13.dll       0.53 MB
cublas64_13.dll      52.4  MB
cublasLt64_13.dll   469.93 MB
```

Backend loading, as reported by the binary itself:

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 12226 MiB):
  Device 0: NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0, VMM: yes, VRAM: 12226 MiB
load_backend: loaded CUDA backend from D:\...\ggml-cuda.dll
load_backend: loaded RPC backend from D:\...\ggml-rpc.dll
load_backend: loaded CPU backend from D:\...\ggml-cpu-alderlake.dll
```

`compute capability 12.0` is the direct confirmation that the sm_120 Blackwell path is active.

### Exact commands used

```powershell
# 1. official prebuilt CUDA binary
Invoke-WebRequest -Uri 'https://github.com/ggml-org/llama.cpp/releases/download/b11254/llama-b11254-bin-win-cuda-13.4-x64.zip' `
  -OutFile 'D:\MultiAgent\tools\downloads\llama-b11254-bin-win-cuda-13.4-x64.zip'
Expand-Archive -Path 'D:\MultiAgent\tools\downloads\llama-b11254-bin-win-cuda-13.4-x64.zip' `
  -DestinationPath 'D:\MultiAgent\tools\llama.cpp\b11254-cuda13.4'

# 2. CUDA runtime (required — the binary package does not include it)
Invoke-WebRequest -Uri 'https://github.com/ggml-org/llama.cpp/releases/download/b11254/cudart-llama-bin-win-cuda-13.4-x64.zip' `
  -OutFile 'D:\MultiAgent\tools\downloads\cudart-llama-b11254-bin-win-cuda-13.4-x64.zip'
Expand-Archive -Path 'D:\MultiAgent\tools\downloads\cudart-llama-b11254-bin-win-cuda-13.4-x64.zip' `
  -DestinationPath 'D:\MultiAgent\tools\llama.cpp\b11254-cuda13.4'

# 3. acceptance check
& 'D:\MultiAgent\tools\llama.cpp\b11254-cuda13.4\llama-cli.exe' --list-devices
```

---

## 3. Test model used for this baseline

Phase 2 is **not** started — no Qwen3.5-4B was downloaded. To prove GPU offload with *real*
inference rather than device enumeration alone, one deliberately small, correctly-licensed
model was used. Source, license and size were verified against the official repository API
before download, per the task requirement:

| Item | Value |
|---|---|
| Repo | [`Qwen/Qwen3-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) — **official Qwen org** |
| File | `Qwen3-0.6B-Q8_0.gguf` — the only quant published in that repo |
| License | **Apache-2.0** (verified via HF API `cardData.license`) |
| Gated | No |
| Size | 609.8 MB (633,495,552 bytes) |
| SHA256 | `9465E63A…43BB031` — **verified matching** the published LFS digest |
| Params | 596,049,920 · 28 layers · n_embd 1024 · vocab 151936 · n_ctx_train 40960 |

Q8_0 was selected precisely *because* it is the only official file: its 610 MB size makes it a
more meaningful VRAM/offload test than a 300 MB 4-bit file would be. This model is a
**Phase 1 harness fixture only** and does not alter the §2 resource plan.

---

## 4. Benchmark results

Harness: `llama-bench`, 512-token prompt + 128-token generation, 3 repetitions
(`-p 512 -n 128 -r 3`), batch 2048 / ubatch 512, f16 KV cache, flash-attn auto.
Raw data: `benchmarks/llama-bench_qwen3-0.6b-q8_0_*.csv`.
Driver `591.86` · idle VRAM baseline **~2011–2116 MiB** (desktop compositor + apps; not free VRAM).

### 4.1 Throughput — 12 threads

| Config | `-ngl` | Prompt processing (t/s) | Generation (t/s) | Speedup vs CPU (pp / tg) |
|---|---|---|---|---|
| **GPU full offload** | 99 | **28,061** | **490.5** | **5.1× / 5.3×** |
| GPU hybrid | 14 | 8,979 | 165.9 | 1.6× / 1.8× |
| CPU only (reference) | 0 | 5,485 | 91.9 | 1.0× / 1.0× |

### 4.2 Thread scaling

| Config | 8 threads | 12 threads | 24 threads |
|---|---|---|---|
| GPU full — pp (t/s) | 28,997.6 | 29,421.8 | 28,887.9 |
| GPU full — tg (t/s) | 500.7 | 502.8 | 498.8 |
| CPU only — pp (t/s) | 5,531.9 | 5,602.3 | 5,690.2 |
| CPU only — tg (t/s) | 75.6 | 91.7 | 78.0 |

Raw: `logs/thread_scaling.csv`

**Observations:**

* With full offload, throughput is **flat across thread counts (8→24)** — the work is genuinely
  GPU-bound, which is exactly the behaviour expected of a working offload. Only the CPU-side
  sampling/batch path consumes host threads.
* CPU-only generation peaks at **12 threads** and *degrades* at 24. The Core Ultra 9 275HX
  reports 24 logical processors with **no SMT**, and this tiny model is memory-bandwidth-bound,
  so extra threads only add contention.
* **Recommended default: `-t 12`** for both modes.

### 4.3 GPU utilisation and VRAM — the offload proof

Measured with `scripts/vram_watch.ps1` (`nvidia-smi` sampled every 300–500 ms) during sustained
inference. Raw: `benchmarks/vram_gpu_full_99.csv`, `benchmarks/vram_cpu_only.csv`.

| Metric | GPU full offload (`-ngl 99`) | CPU only (`-ngl 0`) |
|---|---|---|
| Peak VRAM used | **3,257 MiB** | 2,684 MiB |
| Mean GPU utilisation | **22.3 %** | 6.2 % |
| **Peak GPU utilisation** | **94 %** | 49 % (ambient desktop, not the model) |
| **Peak board power** | **139.19 W** (at the 140 W cap) | 70.47 W |
| Peak SM clock | 2,700 MHz | 2,700 MHz |
| Throughput (tg) | 490.5 t/s | 91.9 t/s |

Idle baseline for reference: ~2011–2116 MiB VRAM, 4–6 % utilisation, ~20 W.

**Why this proves real GPU compute (requirement #8):**

1. **GPU utilisation climbs to 94 %** during offloaded inference vs 6.2 % mean on CPU — the
   GPU is doing work, not idling as a display adapter.
2. **Board power hits the 140 W cap** (139.19 W) versus ~20 W idle — sustained compute-level draw.
3. **Throughput is 5.3× higher** than CPU and is **invariant to host thread count**, the
   signature of GPU-bound execution.
4. Backend diagnostics explicitly report `compute capability 12.0` and
   `loaded CUDA backend from …ggml-cuda.dll`.

**Honest caveat on the VRAM delta.** Peak VRAM rose to 3257 MiB against a ~2011–2116 MiB
baseline — an apparent delta of ~1150 MiB for a 610 MB model. The extra is the CUDA context,
compute buffers and KV cache (4096-ctx f16 KV for this architecture is ~450 MB), not the model
alone. The CPU-only run also showed ambient peaks (2684 MiB) caused by unrelated desktop
applications, since Windows gives no per-process VRAM attribution under WDDM. **Model-resident
VRAM was confirmed directly instead:** with `llama-server` loaded, VRAM read 3293 MiB, and
after shutdown it returned to **2023 MiB** — i.e. the model + runtime were genuinely released.

### 4.4 OpenAI-compatible server validation

`llama-server` was started on `127.0.0.1:8080` and exercised over HTTP:

| Endpoint | Result |
|---|---|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/models` | lists the GGUF; reports `n_ctx 4096`, `n_vocab 151936`, `n_params 596049920`, `ftype Q8_0` |
| `GET /props` | `n_ctx = 4096` confirmed |
| `POST /v1/chat/completions` | **worked** — returned `"Paris"`, `finish_reason: stop`, usage `prompt=34 completion=2` |

This matters beyond Phase 1: it confirms the **exact API surface the Gateway (§6) must proxy**
already exists and behaves. The Gateway can be built as a thin router in front of
llama-server rather than reimplementing inference.

Command:

```powershell
llama-server.exe -m D:\MultiAgent\models\llm\Qwen3-0.6B-Q8_0.gguf -ngl 99 -c 4096 `
  --host 127.0.0.1 --port 8080 -t 12
```

---

## 5. Phase 1 acceptance checklist

| Requirement | Status | Evidence |
|---|---|---|
| llama.cpp CUDA backend suitable for sm_120 deployed | ✅ | b11254 CUDA-13.4 official build; `compute capability 12.0` |
| CUDA environment verified sufficient before install | ✅ | driver 591.86 / CUDA 13.1 ≥ 12.8 required |
| No driver or CUDA major upgrade performed | ✅ | none needed; escalation not triggered |
| No existing environment modified | ✅ | all artefacts under `D:\MultiAgent` only |
| Independent project directory | ✅ | `D:\MultiAgent` with spec's logical layout |
| Install commands and versions recorded | ✅ | §2 and this file |
| **CUDA GPU offload works, not CPU inference** | ✅ | 94 % util · 139 W · 5.3× tg · thread-invariant |
| Benchmark recorded | ✅ | §4, CSVs in `benchmarks/` |
| Functional test passed | ✅ | `/v1/chat/completions` → `"Paris"` |
| No Qwen3.5-4B downloaded | ✅ | Phase 2 not started |

---

## 6. Known issues and risks

| # | Issue | Severity | Notes / mitigation |
|---|---|---|---|
| 1 | **Disk headroom** — D: started at ~31.5 GB free (5 %) | **Low** (was High) | D: recovered to **~72 GB free** during the session, independent of this project (which uses 1.83 GB total). Full model pool now fits; re-check before each download. C: stays tight at ~34 GB. |
| 2 | No CUDA toolkit (`nvcc`) | Medium | Blocks from-source builds, custom quants, `llama-imatrix`. Worked around with official binaries. |
| 3 | `cudart` must be fetched separately | Low | Already handled & documented; would silently break a naive install. |
| 4 | Prebuilt is a rolling `b*` tag (prerelease channel) | Low | Pinned to **b11254** for reproducibility; recorded commit `8019dc563`. |
| 5 | WDDM shared-memory spill can mask OOM as slowness | Medium | Always co-record VRAM with tokens/s (harness does). |
| 6 | Usable VRAM ~11.0 GB, not 12 GB | Expected | Budget accordingly in Phase 2/3 model selection. |
| 7 | Rolling-build API drift (e.g. `-no-cnv` removed in favour of `--single-turn`) | Low | Pin the build; re-verify CLI flags after any upgrade. |
| 8 | Windows binaries are built with **Clang**, not MSVC | Info | Useful later if reproducing the official build. |

### On the toolchain-drift instruction

Per the task's standing instruction not to assume a specific GGUF filename: no filename from the
specification was assumed anywhere. Model source, license, size and quantisation were each
verified against the official repository API before download, and the checksum was validated
after. The **§2 resource plan (4B Fast / 9B Smart / 0.6B Embedding / 0.6B Reranker / 4B VL) was
not altered** — the 0.6B Qwen3 fixture is a harness artefact, not a planning decision.

---

## 7. Recommendation for Phase 2

**Phase 1 is complete and the environment is safe to advance to Phase 2 (Qwen3.5-4B Fast Worker).**

Before or during Phase 2:

1. **Disk space is no longer blocking** — D: has ~72 GB free. Confirm before each download, and
   keep weights off C: (~34 GB free).
2. Download Qwen3.5-4B with the same discipline used here: verify the official repo, license,
   available quantisations, file size and Blackwell compatibility, then checksum.
3. Start with **Q4_K_M**, context **8192**, `-ngl 99`, `-t 12` as the baseline configuration.
4. Benchmark against this baseline using `scripts/llama_bench_suite.ps1`; record VRAM with
   throughput so spill is detectable.
5. Keep the Fast Worker behind `llama-server` so Phase 10's Gateway is a routing layer, not a
   rewrite.
6. **Phase 2 stop condition:** Phase 2 must not begin until the operator confirms; Phase 1 work
   has stopped here.

---

*Phase 1 complete. No pre-existing environment was modified.*
