# Environment Report — Phase 0

**Project:** RTX 5070 Ti 12GB Local AI Coprocessor
**Subject machine:** `<hostname>` (Windows 11)
**Detected:** 2026-09-29
**Phase:** 0 — hardware and software environment detection (**read-only; nothing was modified**)
**Authority:** 《RTX 5070 Ti Local AI Worker 部署实施规范》§15 Phase 0, §17

---

## 0. Executive summary

| Question | Answer |
|---|---|
| Is the GPU usable for CUDA compute? | **Yes.** RTX 5070 Ti Laptop GPU, Blackwell **sm_120**, driver **591.86**, CUDA **13.1** |
| Is the driver/CUDA new enough for llama.cpp? | **Yes, comfortably.** Requirement is CUDA ≥ 12.8; installed driver exposes 13.1 |
| Is a CUDA *toolkit* installed? | **No** — `nvcc` is absent entirely. Driver runtime only |
| Is a driver/CUDA upgrade required for Phase 1? | **No.** Therefore §17's mandatory escalation was **not** triggered |
| Does the existing Qwen-Image-2.1 environment exist and work? | **Yes** — ComfyUI deployment with torch 2.11.0+cu128, `sm_120` supported |
| Was the Qwen-Image environment touched? | **No.** Read-only inspection only |
| Biggest constraint found | **Disk space** was the tightest resource at detection (D: ~31.5 GB free, 5 %); it recovered to ~72 GB free during the session. VRAM is healthy |

**Upgrade decision: no NVIDIA driver or CUDA major-version change is needed, so no escalation was raised.**

---

## 1. GPU

| Property | Value |
|---|---|
| Product | NVIDIA GeForce RTX 5070 Ti **Laptop** GPU |
| Architecture | **Blackwell** |
| Compute capability | **sm_120** (12.0) — confirmed independently by torch |
| VRAM total | 12227 MiB (~11.94 GB) |
| VRAM free at detection | 10049 MiB |
| Driver model | WDDM |
| Bus ID | 00000000:02:00.0 |
| VBIOS | 98.05.31.00.59 |
| Power | current/requested limit 140.00 W (default 65 W, max 140 W) |
| Temperature (idle) | 49 °C |
| ECC | N/A (consumer) |
| MIG | N/A |

Independently confirmed by the existing Qwen-Image torch runtime:

```text
capability    : (12, 0)      -> sm_120
arch_list     : ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
device        : NVIDIA GeForce RTX 5070 Ti Laptop GPU
```

### Two GPU-specific caveats for later phases

1. **Usable VRAM is not 12 GB.** Windows/WDDM reserves memory for the desktop. Even with
   `nvidia-smi` showing `Compute Mode: Default` and nothing heavy running, only ~11.0 GB was
   allocatable. Budget **~10.5–11.0 GB usable**, not 12227 MiB.
2. **WDDM shared-memory spill is a real benchmarking hazard.** `nvidia-smi -q` reports a
   16384 MiB "shared" pool of which only 28 MiB was free. Under WDDM, CUDA can silently
   spill into system RAM instead of failing with OOM, which would make benchmark numbers look
   plausible while actually running far slower. **All later benchmarks must record VRAM
   alongside tokens/s** so a spill is detectable.
3. Per-process VRAM attribution is unavailable under WDDM ("Not available in WDDM driver
   model"), so VRAM accounting must be done as a global delta, not per process.

---

## 2. NVIDIA driver and CUDA

| Item | Value |
|---|---|
| Driver version | **591.86** (installed 2026-09 via NVIDIA App 11.0.6.379) |
| CUDA version reported by driver | **13.1** |
| `CUDA_PATH` environment variable | not set |
| `nvcc` on PATH | **absent** |
| CUDA toolkit installation | **none** — `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` does not exist |
| CUDA runtime DLLs in `System32` | none (`cudart64_*.dll` absent) — only `nvml.dll` |
| NVIDIA Container services | present (1.48) |

**Interpretation.** The machine has the *driver* (which carries the CUDA user-mode driver and
a 13.1 feature level) but no *toolkit*. Nothing can be compiled against CUDA locally, and no
CUDA runtime DLL is available to dynamically-linked CUDA binaries until one is supplied.

**Against the specification's §4 gate (CUDA ≥ 12.8 for Blackwell):** satisfied with margin.
Driver 591.86 ≫ the 12.8 floor, so **no driver upgrade is required and no CUDA major-version
change is required**. §17 escalation was therefore not invoked.

---

## 3. Toolchains

| Tool | Status | Version | Notes |
|---|---|---|---|
| Git | present | **2.50.0.windows.2** | `D:\Git\cmd\git.exe` |
| CMake | **present (bundled)** | **3.31.6-msvc6** | VS 2022 private copy — not on PATH, no install needed |
| Ninja | **present (bundled)** | **1.12.1** | VS 2022 private copy |
| MSVC toolset | present | **14.44.35207** | VS 2022 Community 17.14.37012.4 |
| Windows SDK | present | **10.0.26100.0** | |
| `vcvars64.bat` | present | — | usable for MSVC builds |
| Python | present | **3.13.5** (64-bit) | `D:\Python\python.exe`, pip 26.0.1 |
| MinGW gcc | present | 15.1.0 | **not usable for CUDA** — nvcc requires cl.exe on Windows |
| winget | present | 1.29.380 |
| Node.js | present | 24.21.0 |
| VS Code | present | 1.126.0 |
| WSL | **not installed** | — | `wsl --install` never run; no Linux distro |
| Docker | absent | — | |
| conda / uv | absent | — | |

**Consequence:** a from-source CUDA build of llama.cpp is *nearly* possible today — the
compiler, CMake and Ninja are all present. The **only** missing component is `nvcc`
(the CUDA toolkit). This is recorded as a Phase 1 decision point in
`benchmarks/llama-cpp-baseline.md`.

---

## 4. CPU, RAM, storage

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 9 275HX — 24 cores / 24 logical, 2700 MHz |
| RAM | **31.5 GB** total (2 × 16 GB Micron DDR5-5600) |
| RAM free at detection | ~16.5 GB |
| Page file | `C:\pagefile.sys`, 27648 MB allocated |
| OS | Windows 11 家庭版 中文版, build **26200** |

### Storage — the binding constraint

| Drive | Size | Free at detection | Free at end of Phase 1 | Free % (end) |
|---|---|---|---|---|
| C: | 300 GB | ~34 GB | 33.7 GB | 11.2 % |
| D: | 652.9 GB | **~31.5 GB** | **72.1 GB** | **11 %** |

Disk, not VRAM, was expected to be the limiting resource for this project. The planned model
pool (4B + 9B LLM, 0.6B embedding, 0.6B reranker, 4B VL, Whisper turbo, plus PaddleOCR) is on
the order of **15–25 GB** of weights alone, before caches and FAISS indices.

**Headroom note.** D: gained roughly **+40 GB of free space** during the Phase 1 session,
independent of any action taken here (this project consumes 1.83 GB total). At ~72 GB free the
full planned model pool now fits comfortably, so disk is **downgraded from blocker to watch
item**. C: remains tight at ~34 GB and should not be used for weights. Current largest
consumers of D: are game launchers (miHoYo 245 GB, Hypergryph 87 GB, Steam 78 GB).
**Re-check free space before each model download in Phases 2–8.** See §9.

---

## 5. Existing model / AI environments

### 5.1 Qwen-Image-2.1 — present and healthy

| Item | Value |
|---|---|
| Location | `D:\DSH\computer_science\qwen-image-deploy` |
| Runtime | ComfyUI + dedicated venv (Python 3.13.5) |
| torch | **2.11.0+cu128**, CUDA 12.8 build, `cuda_available = True` |
| Weights | `ComfyUI\models\diffusion_models\qwen_image_2.1_int8_convrot.safetensors` (6.76 GB) |
| | `ComfyUI\models\text_encoders\qwen3vl_8b_w4a8.safetensors` (5.88 GB) |
| | `ComfyUI\models\vae\qwen_image_2.1_vae_bf16.safetensors` (0.63 GB) |
| Workflows | t2i, image_edit, background_removal (`workflows\*.json`) |
| Entry point | `start_comfyui.bat` (port 8188 — **not currently running**) |
| License | ⚠️ Qwen Research License — **non-commercial use only** |

The deployment's own report records working end-to-end generation: 1024 px / 25 steps ≈ 24 s,
2048 px / 25 steps ≈ 129 s. Its torch build already supports `sm_120`, which is strong
independent evidence that this GPU's compute path is functional.

**This environment was inspected read-only and is explicitly out of scope for modification**
(§17, §2.9). Phase 9 will wrap it in an API *from the outside* rather than rebuild it.

### 5.2 Everything else

| Item | Finding |
|---|---|
| HuggingFace cache | `C:\Users\<user>\.cache\huggingface\hub` — only `paraphrase-multilingual-MiniLM-L12-v2` (~0.44 GB) |
| Ollama | **not installed** |
| LM Studio | **not installed** |
| ComfyUI (other than above) | none |
| Other GGUF / safetensors weights | none found on C: or D: |
| System Python packages | `torch`, `transformers`, `sentence_transformers` present; **`faiss`, `llama_cpp`, `diffusers`, `onnxruntime` absent** |

**Notable discrepancy with the specification:** §2.9 describes Qwen-Image-2.1 as an existing
asset to be preserved. It does exist — but under `D:\DSH\computer_science\`, **not** under the
project tree, and it is a **ComfyUI** deployment rather than a bare model directory. Phase 9
must therefore target that path. There is **no** pre-existing local LLM, embedding, reranker,
OCR, ASR or VL runtime to reuse — Phases 2–8 all start from zero.

---

## 6. Network

| Target | Result |
|---|---|
| DNS | `github.com` resolves to **198.18.0.23** (transparent local proxy / fake-IP range) |
| github.com over HTTPS | works via API and `git ls-remote` |
| huggingface.co over HTTPS | works (API + CDN) |
| Measured HuggingFace CDN throughput | **≈ 1.98 MB/s** (~320 s for a 640 MB file) |

Slow-ish egress matters for planning: a 9B Q4_K_M model (~5.5 GB) is roughly a
**45–50 minute** download at this rate. Downloads should be run as background jobs and
verified by checksum.

---

## 7. Safety statement for Phase 0

Phase 0 was executed under the specification's read-only mandate. Specifically:

* No NVIDIA driver or CUDA component was installed, removed or modified.
* No file inside `D:\DSH\computer_science\qwen-image-deploy` was created, modified or deleted.
* The system Python at `D:\Python` was **not** modified (probes used `-c` one-liners and
  `pip list`, which are read-only; no packages were installed).
* No BIOS, partition, or disk configuration was touched.
* New project material was written **only** under `D:\MultiAgent`.

One incidental incident is recorded for completeness: a single harness-side probe file
(`D:\MultiAgent\.dsh-write-probe.txt`) was created and immediately deleted inside the new
project tree. It never touched any pre-existing directory.

---

## 8. Phase 0 acceptance checklist

| Requirement (spec §15, user instruction) | Status |
|---|---|
| GPU detected | ✅ sm_120, RTX 5070 Ti Laptop, 12227 MiB |
| NVIDIA driver recorded | ✅ 591.86 |
| CUDA runtime/toolkit recorded | ✅ driver 13.1; **toolkit absent** |
| Python recorded | ✅ 3.13.5 |
| Git recorded | ✅ 2.50.0 |
| CMake recorded | ✅ 3.31.6 (VS-bundled) |
| Disk free recorded | ✅ C: 34 GB, D: 31.5 GB |
| RAM recorded | ✅ 31.5 GB |
| Existing model directories found | ✅ HF cache; no LLM weights |
| Existing Qwen-Image-2.1 environment found | ✅ `D:\DSH\computer_science\qwen-image-deploy` |
| No existing environment modified | ✅ |
| Independent project directory created | ✅ `D:\MultiAgent` |

---

## 9. Constraints and recommendations carried into Phase 1+

1. **Disk headroom recovered during the session** (D: 31.5 → ~72 GB free, external to this
   work). The full planned model pool now fits. Keep C: (~34 GB) clear of weights and re-check
   free space before each download in Phases 2–8.
2. **Budget ~10.5–11.0 GB usable VRAM**, not 12 GB, and always record VRAM with throughput to
   catch WDDM shared-memory spill.
3. **No `nvcc`.** Either add the CUDA toolkit (matching the existing 13.x driver) or stay on
   official prebuilt binaries.
4. **Qwen-Image-2.1 is under a non-commercial research license** — relevant to §19's KPI
   framing if any generated media is used commercially.
5. **No Linux/WSL present**, so the specification's `/opt/local-ai` layout is not applicable;
   this project uses `D:\MultiAgent` with the same logical structure.

---

*Phase 0 complete. No environment was modified.*
