"""
p2_decon_build.py - P55R-02 item 3, step 1: build a DECONTAMINATED VIEW of the frozen index.

Does NOT modify the historical index. Produces a separate filtered copy plus a review record of
the exclusion decision.

EXCLUSION POLICY, and why it is not simply "drop the six files"
  The six contaminated files are not all the same kind of thing, and the phase requires saying how
  the mixed ones are handled:

  TIER 1 - STRUCTURAL LEAKAGE, excluded entirely
      services/retrieval/rag_compare.py      the 12 questions AND their expected-fact lists AND
                                             their forbidden-value lists, as Python literals
      services/retrieval/retrieval_bench.py  benchmark harness
      services/retrieval/benchmark_data.py   benchmark data
    These carry the scoring rules themselves. A retriever that surfaces them is being handed the
    answer key, so they have no legitimate place in a corpus used to measure retrieval.

  TIER 2 - MIXED: legitimate project facts AND answer values
      benchmarks/local-fast-results.md
      benchmarks/local-smart-results.md
      benchmarks/retrieval-results.md
    These are real project documentation - they are where the measured performance figures of this
    project legitimately live, and a count of the results files shows they are the project's own
    output. But the evaluation questions were WRITTEN FROM them, so some of their chunks carry the
    expected values verbatim.

    HANDLING: these files are excluded from the decontaminated view AS WELL, because there is no
    honest way to keep the value and remove the leak when the value IS the leak. Excluding them is
    the conservative choice, and its cost is reported rather than hidden: any question whose only
    supporting sources were these files becomes unanswerable in the decontaminated view, and those
    questions are FLAGGED (step 2) rather than silently dropped or have their answers edited.

Usage:
  python p2_decon_build.py --src <index dir> --out <new index dir> --report <json>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

TIER1 = ["services/retrieval/rag_compare.py",
         "services/retrieval/retrieval_bench.py",
         "services/retrieval/benchmark_data.py"]
TIER2 = ["benchmarks/local-fast-results.md",
         "benchmarks/local-smart-results.md",
         "benchmarks/retrieval-results.md"]
EXCLUDED = TIER1 + TIER2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    report_path = Path(args.report)
    if report_path.exists():
        print(f"REFUSING to overwrite {report_path}")
        return 3
    if out.exists():
        print(f"REFUSING to overwrite existing index dir {out}")
        return 3

    from index_store import IndexStore
    import numpy as np

    print(f"opening frozen index {src} (READ ONLY)")
    store = IndexStore.load(src, 1024)
    stats_before = store.stats()

    n = stats_before["vectors_in_index"]
    rows = store.conn.execute(
        "SELECT chunk_id, source_path, content, start_line, end_line, faiss_id "
        "FROM chunks ORDER BY faiss_id").fetchall()
    print(f"  {n} vectors, {len(rows)} metadata rows")

    vectors = np.vstack([store.index.reconstruct(i) for i in range(n)])
    print(f"  reconstructed matrix {vectors.shape}")

    exclude = set(EXCLUDED)
    keep_vecs = []
    keep_chunks = []
    dropped_by_file: dict[str, int] = {}
    kept_by_file: dict[str, int] = {}
    for cid, source, content, sline, eline, fpos in rows:
        if source in exclude:
            dropped_by_file[source] = dropped_by_file.get(source, 0) + 1
            continue
        keep_vecs.append(vectors[fpos])
        keep_chunks.append({"chunk_id": cid, "project": "local-ai",
                            "source_path": source, "content": content or "",
                            "start_line": sline, "end_line": eline})
        kept_by_file[source] = kept_by_file.get(source, 0) + 1

    keep_vecs = np.vstack(keep_vecs)
    print(f"  keeping {len(keep_chunks)} of {n} chunks "
          f"({n - len(keep_chunks)} excluded across {len(dropped_by_file)} files)")

    # write the filtered copy into a NEW directory; the original is never written to
    out.mkdir(parents=True, exist_ok=False)
    store2 = IndexStore(out, 1024)
    added = store2.add(keep_vecs, keep_chunks)
    # add() writes the SQLite metadata but NOT index.faiss, so without an explicit save() the
    # copy reports chunks_in_metadata=1061 with vectors_in_index=0 and consistent=False. Caught
    # by checking stats() after the build rather than assuming the write had happened.
    store2.save()
    stats_after = store2.stats()
    store2.close()

    store.close()

    report = {
        "step": "decontamination build",
        "source_index": str(src),
        "source_stats": stats_before,
        "decontaminated_index": str(out),
        "decontaminated_stats": stats_after,
        "vectors_added": added,
        "excluded_files": EXCLUDED,
        "tier1_structural_leakage": TIER1,
        "tier2_mixed_legitimate_facts_and_leakage": TIER2,
        "dropped_chunks_by_file": dropped_by_file,
        "kept_chunks_by_file": kept_by_file,
        "total_dropped": sum(dropped_by_file.values()),
        "total_kept": len(keep_chunks),
        "note": "the historical index was opened read-only and is unchanged; this is a separate copy",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out} and {report_path}")
    print(f"  dropped: {json.dumps(dropped_by_file, indent=4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

