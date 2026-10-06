"""
P57-C0 step 2d - CORRECT the diagnosis: check at CHUNK level, not file level.

The first diagnosis asserted "required file X missing from top-5", which is invalid: a fact's carrier
file being absent says nothing if ANOTHER carrier was retrieved. This is the corrected measurement:
for each fact, was there a retrieved chunk at rank <= D whose CONTENT actually contains the fact?

This is the check that decides whether the pack can answer the question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402

# fact -> the literal strings, ANY of which satisfies it
FACTS = {
    "d01": {"fast_port": ["8111"], "embed_port": ["8113"]},
    "d02": {"torch_reserved_meaning": ["torch_reserved", "reserved_mb"],
            "margin_policy": ["2048"]},
    "d03": {"escalation_threshold": ["0.30", "evidence floor", "EVIDENCE_FLOOR"]},
    "d04": {"unload_figures": ["1368", "1392", "2591", "2595"], "cycle_latency": ["22", "24"]},
    "d05": {"char_cap": ["8000"], "why_raised": ["2000", "truncat"]},
    "d06": {"reranker_port": ["8114"], "ocr_port": ["8116"]},
    "d07": {"score_method": ["softmax", "logit"], "config": ["CrossEncoder", "LogitScore"]},
}
DEPTHS = [5, 8, 10]

store = IndexStore.load(IDX, 1024)
try:
    def chunk_has(chunk_id: str, needles: list[str]) -> bool:
        row = store.conn.execute("SELECT content FROM chunks WHERE chunk_id=?",
                                 (chunk_id,)).fetchone()
        if not row:
            return False
        body = (row[0] or "").lower()
        return any(n.lower() in body for n in needles)

    r = ASK.Retriever(IDX, embedder="service")
    rep: dict = {"index_version": json.load(open(IDX / "manifest.json",
                                                encoding="utf-8"))["index_version"],
                 "depths": DEPTHS, "probes": {}}
    try:
        for pid, facts in FACTS.items():
            q = next(p["question"] for p in json.load(
                open(REPO / "benchmarks/p57c0/results/diagnosis.json", encoding="utf-8")
            )["probes"].values() if False) if False else None
            # reuse the stored question text
            diag = json.load(open(REPO / "benchmarks/p57c0/results/diagnosis.json",
                                  encoding="utf-8"))
            key = next(k for k in diag["probes"] if k.startswith(pid + "_") or k == pid)
            q = diag["probes"][key]["question"]

            hits = r.search(q, top_k=max(DEPTHS))
            rows = [{"rank": i, "chunk_id": h.get("chunk_id"), "score": round(h.get("score", 0), 4),
                     "source_path": h.get("source_path"), "start_line": h.get("start_line")}
                    for i, h in enumerate(hits, 1)]

            print(f"\n=== {pid} === {q[:78]}")
            per_depth = {}
            for d in DEPTHS:
                got, missing = {}, {}
                for fname, needles in facts.items():
                    rank = next((x["rank"] for x in rows[:d]
                                 if chunk_has(x["chunk_id"], needles)), None)
                    if rank:
                        got[fname] = rank
                    else:
                        missing[fname] = next(
                            (x["rank"] for x in rows if chunk_has(x["chunk_id"], needles)), None)
                per_depth[str(d)] = {"gathered": got, "missing": missing,
                                     "complete": not missing}
                flag = "COMPLETE" if not missing else f"MISSING {list(missing)}"
                extra = ""
                if missing:
                    ranks = [v for v in missing.values() if v]
                    if ranks:
                        extra = f"  (nearest carrier at rank {min(ranks)})"
                    else:
                        extra = "  (NO carrier anywhere in the top 10)"
                print(f"   depth {d:>2}: {flag}{extra}   gathered={got}")
            rep["probes"][pid] = {"question": q, "facts": facts, "ranks": rows,
                                  "per_depth": per_depth}
    finally:
        r.close()
finally:
    store.close()

answerable = list(FACTS)
print("\n\n=== CORRECTED SUMMARY (chunk-level, answerable probes) ===")
rep["summary"] = {"n": len(answerable), "by_depth": {}}
for d in DEPTHS:
    comp = [k for k in answerable if rep["probes"][k]["per_depth"][str(d)]["complete"]]
    miss = [k for k in answerable if not rep["probes"][k]["per_depth"][str(d)]["complete"]]
    rep["summary"]["by_depth"][str(d)] = {"complete": comp, "missing": miss}
    print(f"  depth {d:>2}: {len(comp)}/{len(answerable)} complete   incomplete={miss}")

p = REPO / "benchmarks/p57c0/results/diagnosis_chunklevel.json"
p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {p}")
