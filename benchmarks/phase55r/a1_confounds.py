"""
a1_confounds.py - the remaining A1 confounds, read from source.

The aggregate audit showed the question sets and top_k are identical across all three
generations, so those are NOT confounds. Two candidates remain and both are read here rather
than assumed:

  1. PACK PATH - Phase 4 asked the HTTP service for a packed context; Phase 5 packed locally
     from SearchHit objects. If the two produce different text for the same hits, the arms
     are not comparable even with an identical index.

  2. TRUNCATION - the server's pack() had a per-chunk character cap that was raised during
     Phase 4. If the cap differs between the paths, the model was shown different evidence.

Also records the served corpus's file inventory by area, which is the material question for
contamination: does the served index contain the benchmark harnesses or their questions?
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

CHECKS = {
    "server_pack": REPO / "services/retrieval/server.py",
    "retriever_pack": REPO / "services/retrieval/retriever.py",
    "pipeline_pack": REPO / "services/retrieval/rerank_pipeline.py",
}


def read_default(path: Path, cls: str, field: str):
    """Pull a dataclass default out of source, e.g. max_chars_per_chunk."""
    if not path.is_file():
        return None
    src = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(rf"{field}\s*:\s*int\s*=\s*(\d+)", src)
    return int(m.group(1)) if m else None


def main() -> int:
    print("=== pack implementations and their truncation limits ===\n")
    out: dict = {"packs": {}, "truncation": {}}

    for name, p in CHECKS.items():
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        out["packs"][name] = {
            "exists": True,
            "has_pack_def": bool(re.search(r"def pack\(", src)),
            "max_chars_per_chunk": read_default(p, "RetrieveConfig", "max_chars_per_chunk"),
            "uses_truncated_marker": "truncated" in src,
            "header_includes_only_facts": "only project facts" in src,
        }
        cap = out["packs"][name]["max_chars_per_chunk"]
        print(f"  {name:<16} pack_def={out['packs'][name]['has_pack_def']} "
              f"cap={cap} truncates={out['packs'][name]['uses_truncated_marker']}")

    # the served index's corpus inventory: is the benchmark material inside it?
    corpus = REPO / "vector/embedding/corpus"
    print("\n=== served corpus inventory (contamination question) ===")
    if corpus.is_dir():
        files = [p.relative_to(corpus).as_posix() for p in corpus.rglob("*") if p.is_file()]
        areas = Counter(f.split("/")[0] for f in files)
        out["served_corpus_inventory"] = {"file_count": len(files), "areas": dict(areas)}
        for a, n in sorted(areas.items()):
            print(f"  {a:<20} {n} files")

        # does it contain the harnesses, the questions, or the results?
        suspicious = [f for f in files if re.search(
            r"(rag_compare|rerank_bench|retrieval_bench|eval_v2|benchmark_v2|"
            r"cost_eval|baseline_reconcile|retrieval-results|reranker-results|"
            r"retrieval-v2-results|environment_snapshot)", f)]
        out["served_corpus_inventory"]["harness_or_result_files"] = suspicious
        print(f"\n  files that ARE harnesses or result reports: {len(suspicious)}")
        for f in suspicious:
            print(f"    {f}")
    else:
        print("  served corpus not present")

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
