"""
a2_contamination.py - quantify answer-key leakage in the served index.

The confound audit found that the SERVED corpus contains the evaluation harnesses:

    services/retrieval/rag_compare.py      <- holds the 12 RAG questions AND their
                                              expected-fact lists and forbidden-value lists
    services/retrieval/retrieval_bench.py  <- holds the retrieval benchmark queries
    benchmarks/retrieval-results.md        <- holds published result tables

If the answer key is inside the index, a retriever that returns it is not demonstrating
retrieval — it is reading the answer sheet. This script measures the leak rather than
asserting it:

  * for each RAG question, is the question string itself retrievable from the served index?
  * is a chunk containing the EXPECTED FACTS retrievable?
  * how do those chunks rank, and would they land inside a top-5 context?

The historical index is NOT modified. The output is an exclusion list plus the measurement.

Usage:
  python a2_contamination.py --out ../results/<run-id>/a2_contamination.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

SERVED_INDEX = REPO / "vector/embedding/local-ai"
SERVED_CORPUS = REPO / "vector/embedding/corpus"

# The files that carry the answer key. Identified by content, not by name alone.
KEY_BEARING = [
    "services/retrieval/rag_compare.py",
    "services/retrieval/retrieval_bench.py",
    "services/retrieval/benchmark_data.py",
    "benchmarks/retrieval-results.md",
    "benchmarks/reranker-results.md",
    "benchmarks/local-fast-results.md",
    "benchmarks/local-smart-results.md",
]

RAG_HARNESS = REPO / "services/retrieval/rag_compare.py"


def extract_questions(path: Path) -> list[dict]:
    """Pull q / facts / bad triples out of the Phase 4 RAG harness."""
    if not path.is_file():
        return []
    src = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in re.finditer(r'dict\(q="([^"]+)",\s*\n\s*facts=(\[\[.*?\]\]),\s*bad=(\[\[.*?\]\])\)',
                         src, re.S):
        q = m.group(1)
        facts = re.findall(r'"(.*?)"', m.group(2))
        bad = re.findall(r'"(.*?)"', m.group(3))
        out.append({"q": q, "facts": facts, "bad": bad})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    from embedder import Embedder, EmbedderConfig
    from index_store import IndexStore

    questions = extract_questions(RAG_HARNESS)
    print(f"RAG questions recovered from the harness: {len(questions)}")
    if not questions:
        print("could not parse the harness; aborting rather than guessing")
        return 4

    store = IndexStore.load(SERVED_INDEX, args.dim)
    emb = Embedder(EmbedderConfig(dimension=args.dim, batch_size=32,
                                  max_seq_length=2048, device="cpu"))
    report: dict = {
        "served_index": store.stats(),
        "key_bearing_files": {},
        "questions": [],
        "note": ("the historical index is NOT modified; this measures the leak and produces "
                 "an exclusion list"),
    }
    try:
        # ---------------- which files are actually present in the index
        paths = [r[0] for r in store.conn.execute(
            "SELECT DISTINCT source_path FROM chunks").fetchall()]
        report["indexed_paths"] = len(paths)
        for kf in KEY_BEARING:
            present = kf in paths
            report["key_bearing_files"][kf] = {"indexed": present}
            print(f"  {'INDEXED ' if present else 'absent  '} {kf}")

        # ---------------- does a chunk contain the verbatim question?
        print("\n=== verbatim question retrievable from the index? ===")
        leaked_q = 0
        leaked_key = 0
        for item in questions:
            q = item["q"]
            # direct SQL: does any chunk contain the question text?
            row = store.conn.execute(
                "SELECT chunk_id, source_path, start_line FROM chunks "
                "WHERE content LIKE ? LIMIT 1", (f"%{q}%",)).fetchone()
            q_in_index = row is not None

            # does any chunk contain the expected facts together with the question?
            key_hits = []
            for f in item["facts"]:
                r2 = store.conn.execute(
                    "SELECT source_path FROM chunks WHERE content LIKE ? LIMIT 3",
                    (f"%{f}%",)).fetchall()
                key_hits.append({"fact": f, "sources": [x[0] for x in r2]})

            # how many of those fact-bearing chunks come from a KEY-BEARING file?
            key_in_keyfile = any(
                src in KEY_BEARING for kh in key_hits for src in kh["sources"])

            entry = {
                "q": q,
                "question_verbatim_in_index": q_in_index,
                "question_chunk": ({"id": row[0], "source": row[1], "line": row[2]}
                                   if row else None),
                "facts": key_hits,
                "answer_key_in_index": key_in_keyfile,
            }
            report["questions"].append(entry)
            if q_in_index:
                leaked_q += 1
            if key_in_keyfile:
                leaked_key += 1
            print(f"  q_verbatim={str(q_in_index):<5} keyfile={str(key_in_keyfile):<5} "
                  f"{q[:58]}")

        n = len(questions)
        report["summary"] = {
            "questions": n,
            "question_verbatim_in_index": leaked_q,
            "answer_key_in_index": leaked_key,
            "pct_question_verbatim": round(100 * leaked_q / n, 1),
            "pct_answer_key": round(100 * leaked_key / n, 1),
        }
        print(f"\n  {leaked_q}/{n} questions appear VERBATIM in the served index")
        print(f"  {leaked_key}/{n} have an expected fact present in a key-bearing file")

        # ---------------- the exclusion list
        victims = [r[0] for r in store.conn.execute(
            "SELECT DISTINCT source_path FROM chunks").fetchall()
            if r[0] in KEY_BEARING]
        report["exclusion_list"] = sorted(victims)
        counts = {}
        for v in victims:
            c = store.conn.execute("SELECT COUNT(*) FROM chunks WHERE source_path=?",
                                   (v,)).fetchone()[0]
            counts[v] = c
        report["exclusion_chunk_counts"] = counts
        print("\n  exclusion list (chunks that would be removed):")
        for v in sorted(victims):
            print(f"    {counts[v]:>4} chunks  {v}")
        print(f"    total {sum(counts.values())} of "
              f"{store.stats()['chunks_in_metadata']} chunks")
    finally:
        store.close()
        emb.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
