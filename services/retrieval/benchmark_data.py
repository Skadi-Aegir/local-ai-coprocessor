"""
benchmark_data.py - the Phase 4 retrieval benchmark: corpus, queries, ground truth.

Ground truth is defined by HAND, by reading this repository, not derived from the
retriever's own output. Each query names the single file that a competent engineer
would expect to be retrieved; a hit counts when the returned chunk comes from that file.

Query types, as the specification requires - deliberately NOT all keyword lookups:
  exact        the wording overlaps the source
  semantic     asks for a concept, no shared keywords
  paraphrase   the source states it one way, the query another
  cross_lang   source is mostly English, query is Chinese (and vice versa)
  code_concept asks what the code does, never naming the file or symbol

Corpus: real files from this project. `docs`/`config` supply documentation chunks,
`scripts` supply code chunks, and the mixed set draws on both.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Directories copied into the benchmark corpus.
CORPUS_DIRS = ["docs", "config", "scripts"]
CORPUS_FILES = [
    "benchmarks/local-fast-results.md",
    "benchmarks/local-smart-results.md",
    "benchmarks/llama-cpp-baseline.md",
    "README.md",
]

# Sub-directory the corpus is materialised into, to keep project names stable.
PROJECT = "local-ai"


# --------------------------------------------------------------------------
# documentation queries (docs / config / reports)
# --------------------------------------------------------------------------

DOC_QUERIES: list[dict] = [
    # --- exact
    dict(q="What context size does local-fast use by default?",
         file="config/local-fast.yaml", type="exact"),
    dict(q="What port does local-smart listen on?",
         file="config/local-smart.yaml", type="exact"),
    dict(q="Which model and quantisation does local-smart deploy?",
         file="docs/models/local-smart.md", type="exact"),

    # --- semantic (little or no lexical overlap)
    dict(q="Why is it a bad idea to have two local models generate at the same time?",
         file="benchmarks/local-smart-results.md", type="semantic"),
    dict(q="What is the main risk if the desktop uses more graphics memory than expected?",
         file="benchmarks/local-smart-results.md", type="semantic"),
    dict(q="How much memory does the small worker leave free for other programs?",
         file="benchmarks/local-fast-results.md", type="semantic"),

    # --- paraphrase
    dict(q="Two models running together cut total output roughly in half",
         file="benchmarks/local-smart-results.md", type="paraphrase"),
    dict(q="Do I need to tell the model not to reason before answering?",
         file="docs/local-fast-api.md", type="paraphrase"),
    dict(q="Which files decide how long the prompt window is?",
         file="config/local-fast.yaml", type="paraphrase"),

    # --- cross language (source English, query Chinese)
    dict(q="本地快速工作进程默认使用多大的上下文窗口？",
         file="config/local-fast.yaml", type="cross_lang"),
    dict(q="为什么不应该让两个本地模型同时常驻显存？",
         file="benchmarks/local-smart-results.md", type="cross_lang"),
    dict(q="量化选择的依据是什么？",
         file="docs/models/local-smart.md", type="cross_lang"),
    dict(q="如何判断模型是否真的在用 GPU 而不是 CPU？",
         file="benchmarks/llama-cpp-baseline.md", type="cross_lang"),

    # --- code concept aimed at config/docs
    dict(q="Where would I change the sampling randomness for structured extraction?",
         file="config/local-smart.yaml", type="code_concept"),
    dict(q="What document explains how an upstream agent should call the fast worker?",
         file="docs/local-fast-api.md", type="semantic"),
]


# --------------------------------------------------------------------------
# code queries (scripts/)
# --------------------------------------------------------------------------

CODE_QUERIES: list[dict] = [
    # --- exact
    dict(q="vram_watch.ps1", file="scripts/vram_watch.ps1", type="exact"),
    dict(q="What does start-local-fast.ps1 refuse to run without?",
         file="scripts/start-local-fast.ps1", type="exact"),

    # --- code concept: never names the file or symbol
    dict(q="Where is GPU memory sampled repeatedly while a job is running?",
         file="scripts/vram_watch.ps1", type="code_concept"),
    dict(q="Which code parses the YAML configuration file into nested values?",
         file="scripts/lf_config.py", type="code_concept"),
    dict(q="What checks that the service is reachable and answering before declaring success?",
         file="scripts/check-local-fast.ps1", type="code_concept"),
    dict(q="Where are the refusal paths of the launcher tested without needing a GPU?",
         file="scripts/test_lifecycle_gates.ps1", type="code_concept"),
    dict(q="Which script measures how long a model takes to load and unload?",
         file="scripts/phase3_coresidency.ps1", type="code_concept"),
    dict(q="Where is the code that stops a running model and frees its port?",
         file="scripts/stop-local-fast.ps1", type="code_concept"),

    # --- semantic
    dict(q="How does the project decide whether free graphics memory is sufficient?",
         file="scripts/start-local-fast.ps1", type="semantic"),
    dict(q="Code that measures prompt processing and generation speed separately",
         file="scripts/llama_bench_suite.ps1", type="semantic"),

    # --- paraphrase
    dict(q="a helper that reads one value out of a config by dotted key path",
         file="scripts/lf_config.py", type="paraphrase"),
    dict(q="Kill the worker, wait for the socket to close, report what is left",
         file="scripts/stop-local-fast.ps1", type="paraphrase"),

    # --- cross language
    dict(q="哪个脚本负责在负载运行时持续采样显存？",
         file="scripts/vram_watch.ps1", type="cross_lang"),
    dict(q="配置文件的解析逻辑写在哪里？",
         file="scripts/lf_config.py", type="cross_lang"),
]


# --------------------------------------------------------------------------
# mixed: docs + code in one query set
# --------------------------------------------------------------------------

MIXED_QUERIES: list[dict] = [
    dict(q="Where is the fast worker's port configured, and where is it actually bound?",
         file="config/local-fast.yaml", type="mixed_config"),
    dict(q="Which document and which script both mention the VRAM safety margin?",
         file="docs/local-fast-api.md", type="mixed_doc"),
    dict(q="How is the JSON adherence of the fast worker measured?",
         file="benchmarks/local-fast-results.md", type="mixed_report"),
    dict(q="Where would I look to confirm the embedding model is loaded on the GPU rather than the CPU?",
         file="scripts/check-local-fast.ps1", type="mixed_code"),
    dict(q="What evidence shows the model really ran on the GPU and not on the CPU?",
         file="benchmarks/llama-cpp-baseline.md", type="mixed_report"),
    dict(q="Which file would a new agent read first to learn how to call the fast worker?",
         file="docs/local-fast-api.md", type="mixed_doc"),
    dict(q="Where is thinking mode switched off, in configuration and in code?",
         file="config/local-fast.yaml", type="mixed_config"),
    dict(q="Where are the retrieval service's own tests kept?",
         file="scripts/test_lifecycle_gates.ps1", type="mixed_code"),
    dict(q="Which document lists what the fast worker must never be asked to do?",
         file="docs/models/local-fast.md", type="mixed_doc"),
    dict(q="How long does it take to switch between the two local models?",
         file="benchmarks/local-smart-results.md", type="mixed_report"),
]


ALL_SETS = {
    "doc": DOC_QUERIES,
    "code": CODE_QUERIES,
    "mixed": MIXED_QUERIES,
}


# --------------------------------------------------------------------------
# Phase 5 ground-truth revision
# --------------------------------------------------------------------------
# Phase 4 scored each query against a SINGLE expected file. That was shown to be wrong
# for this corpus: the project deliberately documents the same facts in several places,
# so a factually correct answer retrieved from another file was scored as a miss, and
# 74.4% R@5 was understated as a result.
#
# Phase 5 therefore allows several VALID sources per query. Both metrics are reported:
#   strict    - hit only if the ORIGINAL single file is retrieved (Phase 4's number,
#               preserved unchanged so the historical result stays comparable)
#   multi     - hit if ANY valid source is retrieved
#
# The added sources below are not guesses. Each one was read and confirmed to contain
# information that answers the query; where a query genuinely has only one valid source,
# the set stays a singleton and the two metrics agree. Sources that merely mention the
# topic without answering were deliberately NOT added - widening ground truth until
# everything passes would be exactly the kind of standard-lowering the project forbids.
VALID_SOURCES: dict[str, list[str]] = {
    # --- doc queries
    "What context size does local-fast use by default?": [
        "config/local-fast.yaml", "docs/models/local-fast.md", "docs/local-fast-api.md"],
    "What port does local-smart listen on?": [
        "config/local-smart.yaml", "docs/models/local-smart.md"],
    "Which model and quantisation does local-smart deploy?": [
        "docs/models/local-smart.md", "config/local-smart.yaml"],
    "Why is it a bad idea to have two local models generate at the same time?": [
        "benchmarks/local-smart-results.md", "docs/escalation-policy.md",
        "docs/retrieval-architecture.md"],
    "What is the main risk if the desktop uses more graphics memory than expected?": [
        "benchmarks/local-smart-results.md", "docs/environment-report.md",
        "docs/models/local-fast.md"],
    "How much memory does the small worker leave free for other programs?": [
        "benchmarks/local-fast-results.md", "benchmarks/local-smart-results.md",
        "docs/models/local-fast.md"],
    "Two models running together cut total output roughly in half": [
        "benchmarks/local-smart-results.md", "benchmarks/embedding-results.md"],
    "Do I need to tell the model not to reason before answering?": [
        "docs/local-fast-api.md", "docs/models/local-fast.md"],
    "Which files decide how long the prompt window is?": [
        "config/local-fast.yaml", "config/local-smart.yaml", "docs/models/local-fast.md"],
    "本地快速工作进程默认使用多大的上下文窗口？": [
        "config/local-fast.yaml", "docs/models/local-fast.md", "docs/local-fast-api.md"],
    "为什么不应该让两个本地模型同时常驻显存？": [
        "benchmarks/local-smart-results.md", "docs/escalation-policy.md",
        "docs/retrieval-architecture.md"],
    "量化选择的依据是什么？": [
        "docs/models/local-smart.md", "benchmarks/local-smart-results.md",
        "docs/models/local-fast.md"],
    "如何判断模型是否真的在用 GPU 而不是 CPU？": [
        "benchmarks/llama-cpp-baseline.md", "docs/environment-report.md"],
    "Where would I change the sampling randomness for structured extraction?": [
        "config/local-smart.yaml", "config/local-fast.yaml", "docs/local-fast-api.md"],
    "What document explains how an upstream agent should call the fast worker?": [
        "docs/local-fast-api.md"],

    # --- code queries
    "vram_watch.ps1": ["scripts/vram_watch.ps1"],
    "What does start-local-fast.ps1 refuse to run without?": [
        "scripts/start-local-fast.ps1", "docs/local-fast-api.md"],
    "Where is GPU memory sampled repeatedly while a job is running?": [
        "scripts/vram_watch.ps1", "benchmarks/embedding-results.md"],
    "Which code parses the YAML configuration file into nested values?": [
        "scripts/lf_config.py"],
    "What checks that the service is reachable and answering before declaring success?": [
        "scripts/check-local-fast.ps1", "scripts/check-local-embedding.ps1",
        "scripts/check-local-reranker.ps1"],
    "Where are the refusal paths of the launcher tested without needing a GPU?": [
        "scripts/test_lifecycle_gates.ps1"],
    "Which script measures how long a model takes to load and unload?": [
        "scripts/phase3_coresidency.ps1", "scripts/coresidency.ps1",
        "benchmarks/local-smart-results.md"],
    "Where is the code that stops a running model and frees its port?": [
        "scripts/stop-local-fast.ps1", "scripts/stop-local-embedding.ps1",
        "scripts/stop-local-reranker.ps1"],
    "How does the project decide whether free graphics memory is sufficient?": [
        "scripts/start-local-fast.ps1", "config/local-fast.yaml",
        "docs/local-fast-api.md"],
    "Code that measures prompt processing and generation speed separately": [
        "scripts/llama_bench_suite.ps1", "benchmarks/llama-cpp-baseline.md",
        "benchmarks/local-fast-results.md"],
    "a helper that reads one value out of a config by dotted key path": [
        "scripts/lf_config.py"],
    "Kill the worker, wait for the socket to close, report what is left": [
        "scripts/stop-local-fast.ps1", "scripts/stop-local-embedding.ps1",
        "scripts/stop-local-reranker.ps1"],
    "哪个脚本负责在负载运行时持续采样显存？": [
        "scripts/vram_watch.ps1", "benchmarks/embedding-results.md"],
    "配置文件的解析逻辑写在哪里？": ["scripts/lf_config.py"],

    # --- mixed queries
    "Where is the fast worker's port configured, and where is it actually bound?": [
        "config/local-fast.yaml", "docs/local-fast-api.md",
        "scripts/start-local-fast.ps1"],
    "Which document and which script both mention the VRAM safety margin?": [
        "docs/local-fast-api.md", "scripts/start-local-fast.ps1",
        "config/local-fast.yaml", "docs/models/local-fast.md"],
    "How is the JSON adherence of the fast worker measured?": [
        "benchmarks/local-fast-results.md", "docs/models/local-fast.md"],
    "Where would I look to confirm the embedding model is loaded on the GPU rather than the CPU?": [
        "scripts/check-local-fast.ps1", "scripts/check-local-embedding.ps1",
        "docs/models/local-embedding.md", "benchmarks/embedding-results.md"],
    "What evidence shows the model really ran on the GPU and not on the CPU?": [
        "benchmarks/llama-cpp-baseline.md", "benchmarks/embedding-results.md",
        "docs/models/local-fast.md"],
    "Which file would a new agent read first to learn how to call the fast worker?": [
        "docs/local-fast-api.md", "docs/models/local-fast.md"],
    "Where is thinking mode switched off, in configuration and in code?": [
        "config/local-fast.yaml", "docs/local-fast-api.md",
        "docs/models/local-fast.md"],
    "Where are the retrieval service's own tests kept?": [
        "scripts/test_lifecycle_gates.ps1", "docs/retrieval-architecture.md",
        "services/retrieval/test_index_store.py", "services/retrieval/test_chunker.py"],
    "Which document lists what the fast worker must never be asked to do?": [
        "docs/models/local-fast.md", "docs/local-fast-api.md"],
    "How long does it take to switch between the two local models?": [
        "benchmarks/local-smart-results.md", "docs/escalation-policy.md",
        "docs/models/local-smart.md"],
}


def valid_sources(q: dict) -> list[str]:
    """
    Valid source files for a query.

    Falls back to the original single `file` when no revision is recorded, so every
    query always has at least one valid answer and the two metrics are computable.
    """
    return VALID_SOURCES.get(q["q"], [q["file"]])


def materialise_corpus(dest: Path) -> int:
    """Copy the benchmark corpus into `dest`. Returns the number of files copied."""
    import shutil
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in CORPUS_DIRS:
        src = REPO / d
        if src.is_dir():
            shutil.copytree(src, dest / d)
            n += sum(1 for _ in (dest / d).rglob("*") if _.is_file())
    for f in CORPUS_FILES:
        src = REPO / f
        if src.is_file():
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / f)
            n += 1
    return n


if __name__ == "__main__":
    for name, qs in ALL_SETS.items():
        types = {}
        for q in qs:
            types[q["type"]] = types.get(q["type"], 0) + 1
        print(f"{name:6}: {len(qs):2} queries  {types}")
    print(f"total : {sum(len(v) for v in ALL_SETS.values())} queries")
