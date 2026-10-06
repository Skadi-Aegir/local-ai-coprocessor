"""
label_helper.py - inspect the benchmark index so ground truth can be labelled against
REAL chunks instead of guessed ones.

Phase 4/5 labelled each query with a source FILE. That was already shown to be too coarse:
the same fact appears in several files, and a correct answer from another file scored as a
miss. Phase 5.5 needs CHUNK-level labels for the new queries, and a chunk id can only be
assigned by looking at what is actually in the index.

This tool does not label anything. It only reports, so a human can decide.

Usage:
  python label_helper.py --list-files
  python label_helper.py --grep "some phrase"
  python label_helper.py --show <source_path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from index_store import IndexStore  # noqa: E402

INDEX = REPO / "cache/retrieval_bench/indexes/d1024_structured"
DIM = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--list-files", action="store_true")
    ap.add_argument("--grep", default=None)
    ap.add_argument("--show", default=None)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    store = IndexStore.load(args.index, DIM)
    try:
        if args.list_files:
            rows = store.conn.execute(
                "SELECT source_path, COUNT(*) c, MIN(start_line), MAX(end_line) "
                "FROM chunks GROUP BY source_path ORDER BY source_path").fetchall()
            print(f"{'chunks':>6}  {'lines':>12}  path")
            for p, c, lo, hi in rows:
                print(f"{c:>6}  {f'{lo}-{hi}':>12}  {p}")
            print(f"\ntotal {sum(r[1] for r in rows)} chunks in "
                  f"{len(rows)} files")
            return 0

        if args.grep:
            q = f"%{args.grep}%"
            rows = store.conn.execute(
                "SELECT chunk_id, source_path, start_line, end_line, symbol, content "
                "FROM chunks WHERE content LIKE ? ORDER BY source_path, start_line LIMIT ?",
                (q, args.limit)).fetchall()
            print(f"{len(rows)} chunks match {args.grep!r}\n")
            for cid, p, s, e, sym, content in rows:
                print(f"--- {p}:{s}-{e}  [{cid}]  symbol={sym!r}")
                print("    " + content[:400].replace("\n", "\n    "))
                print()
            return 0

        if args.show:
            rows = store.conn.execute(
                "SELECT chunk_id, start_line, end_line, symbol, content FROM chunks "
                "WHERE source_path=? ORDER BY start_line", (args.show,)).fetchall()
            print(f"{len(rows)} chunks in {args.show}\n")
            for cid, s, e, sym, content in rows:
                print(f"--- {args.show}:{s}-{e}  [{cid}]  symbol={sym!r}")
                print("    " + content[:300].replace("\n", "\n    "))
                print()
            return 0

        ap.print_help()
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
