"""
p56_build_clean_corpus.py - build a corpus containing ONLY actual task material.

Excludes, in addition to the P55R-02 answer-key/test-harness exclusions:
  * anything whose filename begins with `test_` (test scripts)
  * anything under a `fixtures/` directory (test fixtures)
  * benchmark harness scripts (run_bench, ctx_sweep, ctx_probe, compare_models, repair_suite,
    run_phase3, summarise_run, stability_test, concurrency_test, token_savings,
    bench_throughput, probe_embedding, verify_pipeline, verify_fixtures, mock_server, ...)

Keeps: config/, docs/, and the production service and script sources - i.e. the material an agent
would legitimately consult when answering a question about this project.

Does NOT modify any historical index; writes a new one.

Usage: python p56_build_clean_corpus.py --out <index dir> --report <json>
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

# already excluded by the P55R-02 contamination audit
ANSWER_KEY = [
    "services/retrieval/rag_compare.py",
    "services/retrieval/retrieval_bench.py",
    "services/retrieval/benchmark_data.py",
    "benchmarks/local-fast-results.md",
    "benchmarks/local-smart-results.md",
    "benchmarks/retrieval-results.md",
]

# harness / test material: answer keys for OTHER benchmarks, and pure scaffolding
HARNESS_PATTERNS = [
    r"(^|/)test_",                       # test_*.py anywhere
    r"/fixtures/",                       # test fixtures
    r"(^|/)conftest\.py$",
    r"benchmarks/local-fast/(common|ctx_probe|ctx_sweep|inspect_gguf|mock_server|run_bench|"
    r"stability_test|summarize_run|tasks)\.py$",
    r"benchmarks/local-smart/(compare_models|compare_reviews|concurrency_test|repair_suite|"
    r"run_phase3|show_code_case|show_failures|tasks_c_review|think_ab|verify_fixtures)\.py$",
    r"services/retrieval/(bench_throughput|probe_embedding|token_savings|verify_pipeline)\.py$",
    r"services/retrieval/(benchmark_data|retrieval_bench|rag_compare)\.py$",
    r"scripts/(test_lifecycle_gates|llama_bench_suite|phase2_sweep|phase3_coresidency)\.ps1$",
]


def excluded(path: str) -> str | None:
    if path in ANSWER_KEY:
        return "answer_key"
    for pat in HARNESS_PATTERNS:
        if re.search(pat, path):
            return "harness_or_test"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "vector/embedding/local-ai"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    rep = Path(args.report)
    if rep.exists():
        print(f"REFUSING to overwrite {rep}")
        return 3
    if out.exists():
        print(f"REFUSING to overwrite existing index {out}")
        return 3

    from index_store import IndexStore
    import numpy as np

    st = IndexStore.load(src, 1024)
    stats_src = st.stats()
    rows = st.conn.execute(
        "SELECT chunk_id, source_path, content, start_line, end_line, faiss_id "
        "FROM chunks ORDER BY faiss_id").fetchall()
    vecs = np.vstack([st.index.reconstruct(i) for i in range(len(rows))])
    print(f"source: {stats_src['vectors_in_index']} chunks, {len(rows)} rows")

    keep_v, keep_c, drop = [], [], {}
    for (cid, sp, content, s0, s1, fid) in rows:
        why = excluded(sp)
        if why:
            drop.setdefault(why, {}).setdefault(sp, 0)
            drop[why][sp] += 1
            continue
        keep_v.append(vecs[fid])
        keep_c.append({"chunk_id": cid, "project": "local-ai", "source_path": sp,
                       "content": content or "", "start_line": s0, "end_line": s1})

    keep_v = np.vstack(keep_v)
    out.mkdir(parents=True, exist_ok=False)
    st2 = IndexStore(out, 1024)
    st2.add(keep_v, keep_c)
    st2.save()                      # add() writes metadata only; save() persists the FAISS file
    stats_out = st2.stats()
    st2.close()
    st.close()

    report = {
        "step": "clean task-material corpus",
        "source_index": str(src), "source_stats": stats_src,
        "clean_index": str(out), "clean_stats": stats_out,
        "kept_chunks": len(keep_c),
        "dropped_total": sum(sum(v.values()) for v in drop.values()),
        "dropped_by_reason": {k: {"files": len(v), "chunks": sum(v.values()),
                                  "detail": v} for k, v in drop.items()},
        "kept_paths": sorted({c["source_path"] for c in keep_c}),
        "note": "historical index opened read-only and unchanged; this is a separate index",
    }
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nclean corpus: {stats_out['vectors_in_index']} chunks, "
          f"{len(report['kept_paths'])} files")
    for k, v in report["dropped_by_reason"].items():
        print(f"  dropped {k}: {v['chunks']} chunks across {v['files']} files")
    print(f"wrote {out} and {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
