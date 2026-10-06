"""
build_index.py - CLI to build or refresh a project index.

Usage:
  python build_index.py --project local-ai --root <repo> --index-root <repo>\\vector\\embedding --dimension 1024
  python build_index.py --project local-ai --root ... --strategy fixed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder import Embedder, EmbedderConfig
from index_builder import IndexBuilder
from index_store import IndexStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--index-root", default=str(Path(__file__).resolve().parents[2] / r"vector\embedding"))
    ap.add_argument("--model-path",
                    default=str(Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B"))
    ap.add_argument("--dimension", type=int, default=1024)
    ap.add_argument("--strategy", default="structured", choices=["structured", "fixed"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--include", default=None,
                    help="comma-separated suffixes to limit the scan, e.g. .py,.md")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="discard the existing index and re-embed everything")
    args = ap.parse_args()

    emb = Embedder(EmbedderConfig(
        model_path=args.model_path, dimension=args.dimension,
        batch_size=args.batch_size, max_seq_length=args.max_seq_length))
    store = IndexStore(Path(args.index_root) / args.project, args.dimension)
    builder = IndexBuilder(emb, store, strategy=args.strategy)
    try:
        if args.reset:
            store.reset(verbose=True)
        include = [s.strip() for s in args.include.split(",")] if args.include else None
        rep = builder.build(args.project, args.root, include=include, verbose=args.verbose)
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=1))
        print(json.dumps(store.stats(), ensure_ascii=False))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

