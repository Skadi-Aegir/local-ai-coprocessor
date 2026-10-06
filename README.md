# Local AI Coprocessor

A set of small, local model services that run on one consumer GPU (built and measured on an
**RTX 5070 Ti 12 GB**, Windows 11) and do the cheap, frequent work for a cloud agent: fast
answers, embeddings, reranking, OCR, and a retrieval-backed Q&A path that cites its sources and
refuses when the evidence is not there.

[中文说明](README.zh-CN.md)

Every service binds to `127.0.0.1` only, has no authentication, and is meant to be called by
other programs on the same machine. Callers use **logical names** (`local-fast`,
`local-embedding`, …), never model paths or quantisation names.

| logical name | port | model | runs on | residency |
|---|---|---|---|---|
| `local-fast` | 8111 | Qwen3.5-4B, GGUF Q4_K_M, `llama-server` | GPU | resident |
| `local-smart` | 8112 | Qwen3.5-9B, GGUF Q4_K_M, `llama-server` | GPU | on demand |
| `local-embedding` | 8113 | Qwen3-Embedding-0.6B (1024-d) | GPU | resident |
| `local-reranker` | 8114 | Qwen3-Reranker-0.6B | GPU | on demand |
| QA server (`cli/ask_server.py`) | 8115 | dense retrieval + `local-fast` | — | on demand |
| `local-ocr` | 8116 | PP-OCRv5 (mobile tier) | CPU | on demand |

Why this split, and the 2048 MiB VRAM margin every resident set must keep, is in
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) (AD-21) and
[docs/gpu-scheduling.md](docs/gpu-scheduling.md).

## Repository layout

```
config/       one YAML per service: model, revision, SHA-256, runtime parameters, paths
scripts/      start/stop/check scripts per service (PowerShell), fetch_models.py, lf_config.py
services/     retrieval (embedding server, chunker, index), reranker, ocr, gpu_scheduler
cli/          ask.py (Q&A CLI) and ask_server.py (Q&A over HTTP)
docs/         design documents and one model record per service (docs/models/)
benchmarks/   every measurement behind a decision, with the scripts that produced it
models/       model files, NOT in git (see "Models")
vector/       built indexes, NOT in git (only the build manifests are)
```

The project is developed in recorded phases. [PROJECT_STATE.md](PROJECT_STATE.md) is the current
state, [OPEN_ISSUES.md](OPEN_ISSUES.md) what is still open, and [ERRATA.md](ERRATA.md) corrects
earlier reports instead of rewriting them.

## Requirements

* Windows 10/11 and Windows PowerShell 5.1 — the lifecycle scripts are PowerShell only.
* An NVIDIA GPU with 12 GB VRAM for the resident set as configured (`local-fast` +
  `local-embedding`); `local-smart` and `local-reranker` need additional headroom.
* Python 3.13.
* [llama.cpp](https://github.com/ggml-org/llama.cpp) build **b11254** (CUDA), unpacked into
  `tools/llama.cpp/b11254-cuda13.4/` (or change `runtime.llama_dir` / `runtime.server_exe` in
  `config/local-fast.yaml` and `config/local-smart.yaml`).
* About 12 GB of disk for the models.

## Setup

All paths in `config/*.yaml` are written as `${REPO}\...` and resolve to the checkout's root, so the
repository can live anywhere. Set `LOCAL_AI_ROOT` to point them somewhere else.

```powershell
# 1. Python environments, one per service (the services are isolated on purpose)
python -m venv services\llm\venv            # stdlib only; used by the lifecycle scripts
python -m venv services\retrieval\venv
services\retrieval\venv\Scripts\pip install -r services\retrieval\requirements.txt
python -m venv services\reranker\venv
services\reranker\venv\Scripts\pip install -r services\reranker\requirements.txt
python -m venv services\ocr\venv            # only if you want local-ocr
services\ocr\venv\Scripts\pip install -r services\ocr\requirements.txt

# 2. Embedding + reranker models, at pinned revisions, SHA-256 verified
python scripts\fetch_models.py
```

### Models

| model | how to get it |
|---|---|
| Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B | `python scripts\fetch_models.py` (set `HF_ENDPOINT` to use a mirror) |
| Qwen3.5-4B / 9B GGUF | download from `unsloth/Qwen3.5-4B-GGUF` / `unsloth/Qwen3.5-9B-GGUF` at the revision in `config/local-fast.yaml` / `config/local-smart.yaml` into `models\llm\`; the start script refuses a file whose SHA-256 differs from the config. See [docs/models/local-fast.md §6](docs/models/local-fast.md) |
| PP-OCRv5 | `services\ocr\venv\Scripts\python benchmarks\ocr\b1_models.py` downloads both tiers into `models\ocr\` |

All of them are Apache-2.0 works of their authors and are not redistributed in this repository.

## Running

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-local-fast.ps1
powershell -ExecutionPolicy Bypass -File scripts\start-local-embedding.ps1
powershell -ExecutionPolicy Bypass -File scripts\check-local-fast.ps1
powershell -ExecutionPolicy Bypass -File scripts\stop-local-fast.ps1
```

Each start script reads everything from its config file and refuses to start when the machine is
not safe: model missing or checksum mismatch, not enough free VRAM for the margin, port busy. The
`local-fast` HTTP API is OpenAI-compatible; see [docs/local-fast-api.md](docs/local-fast-api.md).

### Q&A over your files

With `local-fast` and `local-embedding` running, build the QA index and ask:

```powershell
services\retrieval\venv\Scripts\python benchmarks\phaseA\a_build_index.py
services\retrieval\venv\Scripts\python cli\ask.py "Which port does local-ocr listen on?"
```

The index is built from this repository's own `docs/`, `config/`, `scripts/`, `services/` and
`cli/` (`INCLUDE_DIRS` in `a_build_index.py`). Answers cite the
file and line range the model was actually shown; an answer without a valid citation is turned
into a refusal. The manifests in `vector/qa/` describe the index the recorded results were
measured on — a fresh build gets a new version.

## Tests

Offline, no GPU needed:

```powershell
services\retrieval\venv\Scripts\python cli\test_qa_path.py
services\retrieval\venv\Scripts\python services\retrieval\test_chunker.py
services\retrieval\venv\Scripts\python services\retrieval\test_index_store.py
services\retrieval\venv\Scripts\python benchmarks\phaseA\test_a_build_index.py
services\ocr\venv\Scripts\python services\ocr\test_ocr_server.py
python benchmarks\local-fast\test_grading.py
```

## A note on recorded results

Files under `benchmarks/*/results/` and the reports are kept exactly as measured. They contain the
original absolute paths of the development machine (`D:\MultiAgent\...`); only the user name and
host name were replaced with `<user>` and `<hostname>`.

## License

[MIT](LICENSE). The models this project downloads carry their own licenses (Apache-2.0).
