"""
freeze_baseline.py - capture the Phase 5 baseline so Phase 5.5 results are comparable.

Writes an immutable record of what was deployed and measured at the end of Phase 5:
model revisions, weight hashes, service ports, environment package versions, the git
commit, and the historical result figures.

Frozen means frozen: the file is written ONCE and any later run refuses to overwrite it
unless --force is passed, and the historical numbers are copied from the reports rather
than recomputed. Phase 5.5 must be able to say "this is what we started from" without the
baseline quietly moving underneath it.

Usage:
  python freeze_baseline.py --out baseline.json
  python freeze_baseline.py --out baseline.json --force    # explicit re-freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RETRIEVAL_SRC = REPO / "services" / "retrieval"
sys.path.insert(0, str(RETRIEVAL_SRC))

MODELS = {
    "local-fast": {
        "repo": "Qwen/Qwen3.5-4B-GGUF",
        "quant": "Q4_K_M",
        "path": REPO / "models/llm/Qwen3.5-4B-Q4_K_M.gguf",
        "port": 8111,
        "impl": "llama.cpp b11254 cuda13.4",
    },
    "local-embedding": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "license": "apache-2.0",
        "path": REPO / "models/embedding/Qwen3-Embedding-0.6B/model.safetensors",
        "port": 8113,
        "impl": "sentence-transformers 6.1.0 / torch 2.11.0+cu128",
    },
    "local-reranker": {
        "repo": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
        "license": "apache-2.0",
        "path": REPO / "models/reranker/Qwen3-Reranker-0.6B/model.safetensors",
        "port": 8114,
        "impl": "transformers 5.17.0 / torch 2.11.0+cu128",
    },
}

# Historical Phase 4/5 figures, copied from the reports as published. NOT recomputed here:
# the point of a frozen baseline is to record what was claimed at the time.
HISTORICAL = {
    "phase4": {
        "retrieval_strict_R@5": 74.4,
        "retrieval_strict_R@1": 46.2,
        "rag_accuracy_pct": 91.7,
        "rag_questions": 12,
        "rag_index": "served 'local-ai' index, 1173 chunks",
        "rag_arm": "dense top-5",
        "token_reduction_pct": 99.02,
    },
    "phase5": {
        "candidate_recall_at_20_multi_pct": 92.3,
        "dense_strict_R@1": 35.9,
        "dense_strict_R@5": 66.7,
        "dense_strict_MRR": 0.4882,
        "rerank20_strict_R@5": 71.8,
        "rerank20_multi_R@5": 87.2,
        "rerank50_strict_MRR": 0.6239,
        "rag_accuracy_dense_pool10_pct": 66.7,
        "rag_accuracy_reranked_pool10_pct": 83.3,
        "rag_index": "benchmark index, 186 chunks",
        "hard_negative_strict_pct": 87.5,
        "duplicate_stale_current_wins": "1/5",
    },
}

BASELINE_RETRIEVAL = {
    "dimensions": 1024,
    "instruction": "ON (query side only)",
    "chunking": "structure-aware",
    "candidate_pool": 10,
    "final_top_k": 5,
    "max_length": 2048,
    "batch_size": 20,
    "max_batch_tokens": 8192,
}


def sha256(p: Path, chunk: int = 1 << 20) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def port_listening(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    if out.exists() and not args.force:
        print(f"REFUSING to overwrite the frozen baseline: {out}")
        print("A frozen baseline that can be rewritten is not a baseline. "
              "Pass --force only if you intend to replace it.")
        return 3

    models = {}
    for name, m in MODELS.items():
        entry = {k: str(v) for k, v in m.items() if k != "path"}
        entry["path"] = str(m["path"])
        entry["exists"] = m["path"].exists()
        entry["size_mb"] = round(m["path"].stat().st_size / 2**20, 2) if m["path"].exists() else None
        if m["path"].suffix == ".safetensors":
            entry["sha256"] = sha256(m["path"])
        entry["healthy"] = port_listening(m["port"])
        models[name] = entry

    # environment versions actually importable in the retrieval venv
    env = {}
    try:
        import torch
        import transformers
        import sentence_transformers as st
        import faiss
        import numpy as np
        env = {"python": sys.version.split()[0], "torch": torch.__version__,
               "transformers": transformers.__version__,
               "sentence_transformers": st.__version__, "faiss": faiss.__version__,
               "numpy": np.__version__,
               "cuda_available": torch.cuda.is_available(),
               "cuda_version": torch.version.cuda}
    except Exception as e:  # noqa: BLE001
        env = {"error": f"{type(e).__name__}: {e}"}

    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": "end of Phase 5 / start of Phase 5.5",
        "git_commit": git("rev-parse", "HEAD"),
        "git_commit_subject": git("log", "-1", "--pretty=%s"),
        "git_dirty": bool(git("status", "--porcelain")),
        "models": models,
        "environment": env,
        "retrieval_baseline": BASELINE_RETRIEVAL,
        "historical_results": HISTORICAL,
        "indexes": {},
    }

    # index inventory, so a later result can be tied to the exact index it used
    try:
        from index_store import IndexStore
        for label, p in (("served_local_ai", REPO / "vector/embedding/local-ai"),
                         ("bench_d1024_structured",
                          REPO / "cache/retrieval_bench/indexes/d1024_structured")):
            if p.exists():
                s = IndexStore.load(p, 1024)
                record["indexes"][label] = s.stats()
                s.close()
    except Exception as e:  # noqa: BLE001
        record["indexes_error"] = f"{type(e).__name__}: {e}"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"frozen baseline -> {out}")
    print(f"  commit      {record['git_commit'][:12]} ({record['git_commit_subject'][:50]})")
    print(f"  dirty       {record['git_dirty']}")
    for n, m in models.items():
        h = (m.get("sha256") or "")[:16]
        print(f"  {n:<16} port {m['port']}  healthy={m['healthy']}  "
              f"{m['size_mb']} MB  sha256={h or 'n/a'}")
    for k, v in record["indexes"].items():
        print(f"  index {k:<24} {v.get('chunks_in_metadata')} chunks")
    print(f"  env torch {env.get('torch')} transformers {env.get('transformers')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
