"""
P57-C0 step 2 - DIAGNOSE why the current top-5 fails to gather evidence that is split across chunks.

Method: for a set of probe questions whose required facts are known to sit in known files/lines,
retrieve at several candidate depths and record, for each depth:
    * how many of the required facts are covered by the retrieved chunks
    * WHERE the required chunks rank
    * the score gap between rank 5 and the best required chunk that rank 5 excludes

This is measurement, not tuning: nothing is changed, and the index is opened read-only.

OLD q05 IS DIAGNOSTIC ONLY. Its 8114 fact does not exist in the clean corpus, so q05 can never be
"answered correctly" and is NOT counted as a success sample anywhere here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
EVID = REPO / "benchmarks/p57c0/results"
# Use the CLI's OWN retriever wrapper, so what is measured is exactly what the user path does.
sys.path.insert(0, str(REPO / "cli"))

import ask as ASK  # noqa: E402

# Probes: each names the question and the FILE that must be retrieved to answer one of its facts.
# The 8114 probe is included ONLY to show that the corpus lacks the fact - it is not an answerable
# sample and is excluded from every count below.
PROBES = [
    {"id": "d01", "question": "Which components does a question pass through before reaching "
                             "local-fast, and which ports do they use?",
     "required_files": ["config/local-fast.yaml", "docs/retrieval-architecture.md"],
     "note": "two facts in two files: the fast port and the embedding port"},
    {"id": "d02", "question": "How do the embedding service memory figures and the scheduler "
                             "margin relate to each other?",
     "required_files": ["docs/models/local-embedding.md", "docs/gpu-scheduling.md"],
     "note": "memory meaning in one file, the margin policy in another"},
    {"id": "d03", "question": "What does the escalation policy require when local retrieval is "
                             "weak, and where is that threshold configured?",
     "required_files": ["docs/escalation-policy.md", "services/gpu_scheduler/gpu_scheduler.py"],
     "note": "policy in docs, the implemented threshold in code"},
    {"id": "d04", "question": "What are the two unload figures on the staged RAG path and what "
                             "latency does a full cycle take?",
     "required_files": ["docs/staged-lifecycle-design.md",
                        "docs/phase55r-evidence-consolidation.md"],
     "note": "the two documents split the numbers"},
    {"id": "d05", "question": "What per-chunk character cap does the pack apply and why was it "
                             "raised?",
     "required_files": ["services/retrieval/retriever.py"],
     "note": "single-file control: must stay answerable at top-5"},
    {"id": "d06", "question": "Which ports are reserved for the reranker and the OCR service?",
     "required_files": ["config/local-reranker.yaml", "config/local-ocr.yaml"],
     "note": "two config files"},
    {"id": "d07", "question": "How does the reranker turn model output into a score, and what "
                             "test pins that arithmetic?",
     "required_files": ["docs/reranking-architecture.md", "config/local-reranker.yaml"],
     "note": "explanation and configuration"},
    {"id": "d08_8114_DIAGNOSTIC_ONLY", "question": "Trace a question end to end: which components, "
                                                   "in what order, and what ports do they use?",
     "required_files": ["config/local-fast.yaml", "docs/retrieval-architecture.md",
                        "config/local-reranker.yaml"],
     "note": "DIAGNOSTIC ONLY - the 8114 fact is absent from the clean corpus, so this can never be "
             "answered correctly and is EXCLUDED from all counts"},
]

DEPTHS = [5, 8, 10]


def main() -> int:
    EVID.mkdir(parents=True, exist_ok=True)
    out = EVID / "diagnosis.json"
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    r = ASK.Retriever(IDX, embedder="service")
    rep: dict = {"index": str(IDX), "depths": DEPTHS, "probes": {}}
    try:
        for p in PROBES:
            print(f"\n=== {p['id']} ===")
            print(f"  Q: {p['question'][:88]}")
            deep = r.search(p["question"], top_k=max(DEPTHS))
            rows = []
            for i, h in enumerate(deep, 1):
                rows.append({"rank": i, "score": round(h.get("score", 0), 4),
                             "source_path": h.get("source_path"),
                             "start_line": h.get("start_line"), "end_line": h.get("end_line")})
            for i, row in enumerate(rows, 1):
                mark = "  <-- required" if row["source_path"] in p["required_files"] else ""
                print(f"   {i:>2}. {row['score']:.4f}  {row['source_path']}:{row['start_line']}"
                      f"{mark}")

            found = {row["source_path"] for row in rows}
            per_depth = {}
            for d in DEPTHS:
                top = rows[:d]
                got = {x["source_path"] for x in top}
                cov = [f for f in p["required_files"] if f in got]
                per_depth[str(d)] = {
                    "required_covered": cov,
                    "required_missing": [f for f in p["required_files"] if f not in got],
                    "complete": len(cov) == len(p["required_files"]),
                    "best_rank_of_missing": min(
                        [x["rank"] for x in rows
                         if x["source_path"] in p["required_files"] and x["rank"] > d],
                        default=None),
                }
            rep["probes"][p["id"]] = {
                "question": p["question"], "required_files": p["required_files"],
                "note": p["note"], "ranks": rows,
                "required_ever_retrieved_in_deep": [f for f in p["required_files"] if f in found],
                "per_depth": per_depth,
            }
            for d in DEPTHS:
                pd = per_depth[str(d)]
                flag = "COMPLETE" if pd["complete"] else f"MISSING {pd['required_missing']}"
                extra = (f"  (best missing at rank {pd['best_rank_of_missing']})"
                         if pd["best_rank_of_missing"] else "")
                print(f"   depth {d:>2}: {flag}{extra}")

        # summary, excluding the diagnostic-only probe
        answerable = [k for k, v in rep["probes"].items() if not k.endswith("DIAGNOSTIC_ONLY")]
        print("\n=== summary over ANSWERABLE probes (diagnostic probe excluded) ===")
        rep["summary"] = {"n_answerable": len(answerable), "by_depth": {}}
        for d in DEPTHS:
            comp = [k for k in answerable if rep["probes"][k]["per_depth"][str(d)]["complete"]]
            miss = [k for k in answerable if not rep["probes"][k]["per_depth"][str(d)]["complete"]]
            rep["summary"]["by_depth"][str(d)] = {
                "complete": comp, "missing": miss,
                "n_complete": len(comp), "n_missing": len(miss)}
            print(f"  depth {d:>2}: {len(comp)}/{len(answerable)} complete   "
                  f"incomplete={miss}")
        print(f"\n  NOTE: {len([k for k in rep['probes'] if k.endswith('DIAGNOSTIC_ONLY')])} "
              f"diagnostic-only probe excluded from these counts (8114 fact absent from corpus)")
    finally:
        r.close()

    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
