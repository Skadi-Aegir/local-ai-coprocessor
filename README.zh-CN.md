# 本地 AI 协处理器

一组跑在单张消费级显卡上的本地小模型服务（在 **RTX 5070 Ti 12 GB**、Windows 11 上开发和实测），
替云端 Agent 承担便宜、高频的工作：快速回答、向量化、重排、OCR，以及一条带引用的检索问答链路——
证据不足时明确拒答，而不是凭模型记忆作答。

[English](README.md)

所有服务只绑定 `127.0.0.1`，没有鉴权，供本机其他程序调用。调用方只使用**逻辑名**
（`local-fast`、`local-embedding` 等），不接触模型路径或量化名。

| 逻辑名 | 端口 | 模型 | 运行于 | 常驻 |
|---|---|---|---|---|
| `local-fast` | 8111 | Qwen3.5-4B GGUF Q4_K_M（`llama-server`） | GPU | 常驻 |
| `local-smart` | 8112 | Qwen3.5-9B GGUF Q4_K_M（`llama-server`） | GPU | 按需 |
| `local-embedding` | 8113 | Qwen3-Embedding-0.6B（1024 维） | GPU | 常驻 |
| `local-reranker` | 8114 | Qwen3-Reranker-0.6B | GPU | 按需 |
| 问答服务 `cli/ask_server.py` | 8115 | 稠密检索 + `local-fast` | — | 按需 |
| `local-ocr` | 8116 | PP-OCRv5（mobile 档） | CPU | 按需 |

为什么这样划分、以及常驻集合必须保留的 2048 MiB 显存余量，见
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md)（AD-21）和 [docs/gpu-scheduling.md](docs/gpu-scheduling.md)。

## 环境要求

* Windows 10/11 + Windows PowerShell 5.1（启停脚本只有 PowerShell 版本）
* 12 GB 显存的 NVIDIA 显卡（按当前配置跑常驻的 `local-fast` + `local-embedding`）
* Python 3.13
* [llama.cpp](https://github.com/ggml-org/llama.cpp) **b11254**（CUDA 版），解压到 `tools/llama.cpp/b11254-cuda13.4/`
* 约 12 GB 磁盘空间存放模型

## 安装与运行

`config/*.yaml` 中的路径都写成 `${REPO}\...`，自动解析为仓库根目录，所以仓库放在哪里都可以；
设置环境变量 `LOCAL_AI_ROOT` 可指向别处。具体命令（建 venv、装依赖、下载模型、启动服务、构建索引、
跑测试）见 [README.md](README.md) 的 Setup / Running / Tests 三节，命令本身中英文版相同。

模型文件不随仓库分发：
* 向量化和重排模型：`python scripts\fetch_models.py`（按固定 revision 下载并校验 SHA-256；国内可设 `HF_ENDPOINT=https://hf-mirror.com`）
* GGUF 模型：按 `config/local-fast.yaml`、`config/local-smart.yaml` 中记录的 revision 下载到 `models\llm\`
* OCR 模型：`benchmarks\ocr\b1_models.py`

## 关于记录的实验结果

`benchmarks/*/results/` 和各报告按实测原样保留，其中含开发机的原始绝对路径（`D:\MultiAgent\...`），
只把用户名和主机名替换成了 `<user>`、`<hostname>`。

## 许可证

[MIT](LICENSE)。项目下载的模型各自遵循其许可证（Apache-2.0）。
