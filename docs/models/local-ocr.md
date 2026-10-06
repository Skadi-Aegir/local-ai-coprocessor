# local-ocr — Model Record (Phase 6)

**Logical name:** `local-ocr` ｜ **Service:** `127.0.0.1:8116` ｜ **Device:** **CPU only**
**Status:** models present and verified running on 2026-10-05. Accuracy numbers live in
`benchmarks/ocr-results.md`, not here — this page records identity, provenance and the runtime
contract.

## 1. Source chain

* Upstream project: **PaddleOCR** (PaddlePaddle), models distributed as **PP-OCRv5** official models.
* Downloaded by `benchmarks/ocr/b1_models.py` through the paddlex official-model manager
  (`paddlex.inference.utils.official_models.official_models.get_model_path`), with
  `MODEL_SOURCE = huggingface`.
* Package versions actually installed in `services/ocr/venv`:
  **`paddlepaddle 3.3.1`** (CPU build) and **`paddleocr 3.7.0`**, pulling **`paddlex 3.7.2`**.
  Full list: `benchmarks/ocr/results/OCR-01-env/pip_freeze.txt`.
* **Only the CPU wheel is installed.** `paddle.device.is_compiled_with_cuda()` returns **False** and
  `paddle.device.get_device()` returns **`cpu`**. No `paddlepaddle-gpu`, no `nvidia-*`, no
  `torch`, no CUDA toolkit and no cuDNN were installed; nothing on the system (PATH, registry,
  global Python) was changed. The install is confined to `services/ocr/venv`, created with the
  existing `D:\Python\python.exe` (3.13.5) — no new interpreter.

## 2. Official specifications — the four models

Two tiers are installed deliberately, because B3 compares them. **Both tiers are present; neither has
been selected yet** — the selection rule is pre-registered in `benchmarks/ocr/questions.md` and the
outcome is recorded in `benchmarks/ocr-results.md`.

| model | files | size | `inference.pdiparams` sha256 (first 32) | `inference.json` sha256 (first 32) |
|---|---:|---:|---|---|
| `PP-OCRv5_mobile_det` | 15 | 4.72 MB | `afa1820cb16c1fd0dad589d0f8b38913` | `05feef1acb00aa4cd7362b15f7f501fc` |
| `PP-OCRv5_mobile_rec` | 15 | 16.40 MB | `2460da90875937c94db97eba74ae3d9e` | `24587345250c7332d0fc6f9a44e794d0` |
| `PP-OCRv5_server_det` | 15 | 84.27 MB | `183146fe9d9910352f68482f623bcbbb` | `af5876933d8806a1b50d895867e0781e` |
| `PP-OCRv5_server_rec` | 15 | 81.29 MB | `63853f062a5f4089befc16f565a68277` | `8e6e12e5d42531840310977fffb58165` |

Full file lists and complete SHA256 values: `benchmarks/ocr/results/OCR-01-env/models.json`.

**PP-OCRv6 also exists in this paddlex build** (`PP-OCRv6_tiny/small/medium_det`, `_rec`) and is
strictly newer. **PP-OCRv5 was chosen deliberately**: it is the version the task names, and its
mobile and server tiers are the intended comparison pair. Recording v6's existence here so the
choice is visible rather than accidental — a later round may evaluate it, with its own frozen set.

### Preprocessing deliberately DISABLED

`use_doc_orientation_classify=False`, `use_doc_unwarping=False`,
`use_textline_orientation=False`.

These three stages serve photographed or scanned pages: document rotation classification, page
dewarping, and text-line rotation classification. The Phase 6 target is **screenshots, terminal
screenshots and software UI**, which are already upright and undistorted, so the stages would add
latency and failure modes without adding accuracy. Disabling them also avoids downloading three
further models. Table recognition and PDF text-layer extraction are out of scope entirely.

## 3. Inference backend

**CPU.** One important measured caveat:

> **oneDNN must be disabled on this build.** With the default (oneDNN enabled) both tiers **loaded
> successfully and then failed at predict**:
>
> ```
> NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
> [pir::ArrayAttribute<pir::DoubleAttribute>]  (onednn_instruction.cc:118)
> ```
>
> This is a oneDNN/PIR kernel incompatibility in `paddlepaddle 3.3.1`, **not a bad model or a failed
> download** — the same files succeed unchanged once oneDNN is off. Two equivalent remedies were
> measured to work: `enable_mkldnn=False` on the `PaddleOCR` constructor, or the environment variable
> `FLAGS_use_mkldnn=0`. Setting only `FLAGS_enable_pir_api=0` does **not** fix it.
> The API flag is used in this project because it is visible in the call rather than hidden in the
> environment. First failing run preserved as
> `benchmarks/ocr/results/OCR-01-env/b1_smoke.onednn_default_failed.json`.

* **No GPU is used.** VRAM is identical before and after a run (measured; see
  `benchmarks/ocr-results.md`). The models are CPU-only and the Paddle build has no CUDA support at
  all, so there is no path by which this service could occupy VRAM.
* Thread count is set explicitly rather than left to Paddle's default; the value used is recorded
  with each benchmark run.

## 4. Environment (isolated)

| item | value |
|---|---|
| interpreter | `D:\Python\python.exe` 3.13.5 (the pre-existing one; no new interpreter) |
| venv | `services/ocr/venv` |
| install | `paddlepaddle==3.3.1`, `paddleocr==3.7.0` (pinned), 72 packages frozen |
| device | `cpu`, `is_compiled_with_cuda() == False` |
| oneDNN | **disabled** (see §3) |

**Model location.** `D:\MultiAgent\models\ocr\<model_name>\`, one directory per model, staged there
by `b1_models.py` from a cache that was then deleted.

**Nothing landed on C:.** paddlex defaults its cache to `%USERPROFILE%\.paddlex`; that path is
redirected by setting **`PADDLE_PDX_CACHE_HOME`** *before importing paddlex* (it is read at import
time in `paddlex/utils/cache.py`). Measured after the download: `%USERPROFILE%\.paddlex` contains
**0 files** (it existed and was empty before this round, and is unchanged). The build also refuses to
proceed if the effective cache root begins with `C:\`.

**Models are not in Git.** `models/ocr/**` is ignored, with only `models/ocr/.gitkeep` re-admitted.
This rule was required and is not redundant: the pre-existing weight patterns (`*.gguf`,
`*.safetensors`, `*.bin`, `*.onnx`, `*.pt`, …) **do not match** PaddleOCR's `inference.json`,
`inference.pdmodel`, `inference.pdiparams` or dictionary files, so before this round a downloaded OCR
model **would have been committable**. Verified with `git check-ignore -v` on six representative
paths before any download: all ignored by `models/ocr/**`, and `.gitkeep` still committable.

## 5. Measured runtime — smoke

CPU, both tiers, on a synthetic high-contrast English line ("local-ocr smoke test 8116").
Evidence: `benchmarks/ocr/results/OCR-01-env/b1_smoke.json`.

| tier | load | predict | transcript | confidence |
|---|---:|---:|---|---:|
| mobile (`PP-OCRv5_mobile_*`) | 0.89 s | **0.256 s** | `local-ocr smoke test 8116` | 0.982 |
| server (`PP-OCRv5_server_*`) | 0.48 s | **1.191 s** | `local-ocr smoke test 8116` | 0.992 |

Both transcribed the line **exactly**. Full accuracy figures (CER, line recall, per category, p50/p95
latency) are in `benchmarks/ocr-results.md`; the numbers above are a one-image wiring check and are
**not** a quality measurement.

## 6. Usage contract

* Call `POST /v1/ocr` on `127.0.0.1:8116` with `{"image": "<base64>"}`; optional `"request_id"`.
* Returns `{text, blocks, confidence, image, model, elapsed_s}`; `blocks` are in reading order
  (top-to-bottom, then left-to-right) and `confidence` is a character-weighted mean of block
  confidences, `null` when no text was found.
* **Loopback only.** The service binds `127.0.0.1` and refuses any other host.
* **Images only** (PNG / JPEG). PDF text-layer extraction and table recognition are **not**
  implemented, by scope.
* Not a general-purpose document parser: it returns recognised text and boxes, not structure.

## 7. Known limitations

1. **oneDNN must stay disabled** (§3). Re-enabling it reintroduces the predict failure. This is a
   workaround for an upstream incompatibility, not a preference.
2. **CPU-only by decision (AD-21 condition 2).** Latency is seconds, not milliseconds; the service
   is for screenshots, not for high-volume streams.
3. **Accuracy is unmeasured for Chinese-heavy, low-contrast and small-font material until B3 runs.**
   No claim is made here about how the tiers compare.
4. **Screenshots only.** No rotation handling, no dewarping (disabled by choice), no table structure,
   no PDF.
5. **No authentication, no TLS.** Loopback-only, and the task excludes public exposure.
6. **PP-OCRv6 is available and unevaluated** (§2).
7. **The four models occupy ~187 MB on D: and none of it is in Git** — a fresh clone has no models
   and must run `b1_models.py` before the service can start.
