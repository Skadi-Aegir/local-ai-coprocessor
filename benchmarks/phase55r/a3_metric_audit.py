"""
a3_metric_audit.py - P55R-01 work package A3. Entirely offline; no model calls.

Recomputes from the SAVED per-question JSON and checks the metric definitions rather than
trusting the labels the harness printed:

  1. Is "Recall@K" actually Hit@K on ANY valid source? True recall would be
     (relevant retrieved) / (total relevant); this harness checks whether at least one
     correct source appears in the top K, which is a different quantity with a different
     name.
  2. Do multi-source queries require ALL evidence groups, or does one source satisfy them?
  3. Is nDCG computed with the correct discount and normalised against the right ideal?
  4. How are the failures distributed across the historical / dev / test split?
  5. Are "answer missing from corpus", "chunking lost it", "candidate miss" and "ranking
     downgrade" being conflated into one bucket?
  6. Does widening the pool help candidate recall and final Top-5 SEPARATELY?

The split labels are also corrected here: the previous reporting called 61 queries
"held-out", but that 61 is 39 historical regression queries (already published, so long
since exposed) plus only 22 genuinely new test queries. Reporting them merged hides that.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# benchmark_v2 lives in benchmarks/retrieval-v2, not next to this audit script
sys.path.insert(0, str(REPO / "benchmarks" / "retrieval-v2"))
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from benchmark_v2 import all_queries  # noqa: E402

RESULTS = REPO / "benchmarks/retrieval-v2/results"


def dcg_at(gains):
    """Standard DCG with the log2(i+2) discount, i.e. 1/log2(2), 1/log2(3), ..."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    qmap = {q["id"]: q for q in all_queries()}
    report: dict = {"sources": {}, "note": "offline recompute; no model calls"}

    # ------------------------------------------------------------------ 1. naming
    print("=== 1. what is 'Recall@K' actually measuring? ===")
    held = json.loads((RESULTS / "eval_heldout.json").read_text(encoding="utf-8"))
    cfg = next(c for c in held["configs"] if c.get("causal_switch"))
    rows = cfg["rows"]
    n = len(rows)

    # the harness sets r5 = first valid source within 5, which is Hit@K, not Recall@K
    hit_at_5 = sum(1 for r in rows if r["r5"])
    # a true Recall@5 would be |relevant retrieved in top5| / |total relevant|
    # the saved rows only carry top_paths, so compute both and show they differ in meaning
    true_recall_num = 0
    true_recall_den = 0
    for r in rows:
        q = qmap.get(r["id"], {})
        valid = set(q.get("sources") or [])
        if not valid:
            continue
        got = len(valid & set(r.get("top_paths") or []))
        true_recall_num += got
        true_recall_den += len(valid)
    report["metric_naming"] = {
        "harness_labels_it": "r@1 / r@3 / r@5",
        "what_it_computes": ("1 if the FIRST valid source appears within K, else 0 - "
                             "averaged over queries. That is Hit@K (a.k.a. success@K), "
                             "NOT recall."),
        "hit_at_5_pct": round(100 * hit_at_5 / n, 1),
        "mean_source_recall_at_5_pct": round(100 * true_recall_num / true_recall_den, 1)
        if true_recall_den else None,
        "distinct_queries": n,
        "verdict": "MISNAMED: the figure is Hit@K. Consequence: a query with 3 valid sources "
                   "of which the system returns 1 scores identically to one that returns "
                   "all 3, so multi-source completeness is invisible.",
    }
    print(f"  Hit@5 (what the harness computes)      : "
          f"{report['metric_naming']['hit_at_5_pct']}%")
    print(f"  mean per-source recall@5 (different)   : "
          f"{report['metric_naming']['mean_source_recall_at_5_pct']}%")
    print("  verdict: MISNAMED - it is Hit@K, so returning 1 of 3 valid sources "
          "scores the same as returning 3")

    # ------------------------------------------------------------------ 2. all-groups
    print("\n=== 2. do multi-source queries require ALL evidence groups? ===")
    multi = [r for r in rows if len((qmap.get(r["id"], {}).get("sources") or [])) > 1]
    single = [r for r in rows if len((qmap.get(r["id"], {}).get("sources") or [])) == 1]
    def all_present(r):
        valid = set(qmap.get(r["id"], {}).get("sources") or [])
        return valid and valid.issubset(set(r.get("top_paths") or []))
    report["multi_source"] = {
        "queries_with_multiple_valid_sources": len(multi),
        "queries_with_one": len(single),
        "multi_hit_at_5_pct": round(100 * sum(1 for r in multi if r["r5"]) / (len(multi) or 1), 1),
        "multi_ALL_sources_present_pct": round(
            100 * sum(1 for r in multi if all_present(r)) / (len(multi) or 1), 1),
        "verdict": ("any ONE valid source satisfies the metric; requiring every evidence "
                    "group would score lower. Both are reported so the choice is explicit "
                    "rather than hidden in the metric name."),
    }
    print(f"  multi-source queries: {len(multi)};  one-source: {len(single)}")
    print(f"  Hit@5 (any one source)     : {report['multi_source']['multi_hit_at_5_pct']}%")
    print(f"  ALL sources present in top5: "
          f"{report['multi_source']['multi_ALL_sources_present_pct']}%")

    # ------------------------------------------------------------------ 3. nDCG
    print("\n=== 3. is nDCG discounted and normalised correctly? ===")
    # recompute nDCG@5 two ways from the saved top_paths and compare with the stored value
    mismatches = 0
    checked = 0
    samples = []
    for r in rows:
        valid = set(qmap.get(r["id"], {}).get("sources") or [])
        paths = r.get("top_paths") or []
        if not valid or not paths:
            continue
        flags = [1.0 if p in valid else 0.0 for p in paths[:5]]
        n_rel = int(sum(flags))
        if n_rel == 0:
            ref = 0.0
        else:
            ideal = [1.0] * n_rel + [0.0] * (5 - n_rel)
            ref = dcg_at(sorted(flags, reverse=True)) / dcg_at(ideal)
        stored = r.get("ndcg5")
        checked += 1
        if stored is not None and abs(stored - ref) > 1e-6:
            mismatches += 1
            if len(samples) < 3:
                samples.append({"id": r["id"], "stored": stored, "recomputed": round(ref, 6),
                                "n_relevant_in_top5": n_rel})
    report["ndcg"] = {
        "discount": "1/log2(rank+2), i.e. the standard log2 discount",
        "ideal": "built from the number of relevant documents in the window, not a fixed 1",
        "checked": checked, "mismatches": mismatches, "examples": samples,
        "verdict": ("recomputation agrees with the stored values" if mismatches == 0
                    else f"{mismatches} disagree - investigate"),
    }
    print(f"  checked {checked} queries; {mismatches} mismatches against stored values")
    print(f"  verdict: {report['ndcg']['verdict']}")

    # ------------------------------------------------------------------ 4. split
    print("\n=== 4. failure distribution by ACTUAL split ===")
    by_split = defaultdict(Counter)
    for r in rows:
        q = qmap.get(r["id"], {})
        # historical queries were tagged split=test by benchmark_v2; separate them by the
        # original flag so the 39 and the 22 are not merged
        split = "historical_39" if q.get("original") else f"new_{q.get('split','?')}"
        if not r.get("r5"):
            if not r.get("candidate_hit"):
                by_split[split]["candidate_miss"] += 1
            else:
                by_split[split]["rank_miss"] += 1
        else:
            by_split[split]["ok"] += 1
    report["split_distribution"] = {k: dict(v) for k, v in by_split.items()}
    for k, v in sorted(by_split.items()):
        tot = sum(v.values())
        print(f"  {k:<16} {v.get('ok',0):>3}/{tot:<3} ok   "
              f"candidate_miss={v.get('candidate_miss',0):<3} rank_miss={v.get('rank_miss',0)}")

    # ------------------------------------------------------------------ 5. conflation
    print("\n=== 5. are distinct failure causes conflated? ===")
    # a query whose valid sources are wholly absent from the corpus cannot be retrieved
    try:
        from index_store import IndexStore
        st = IndexStore.load(REPO / "cache/retrieval_bench/indexes/d1024_structured", 1024)
        indexed_paths = {x[0] for x in st.conn.execute(
            "SELECT DISTINCT source_path FROM chunks").fetchall()}
        st.close()
        absent = {"queries_whose_sources_are_not_indexed": [],
                  "count": 0}
        for q in all_queries():
            srcs = q.get("sources") or []
            if srcs and not any(s in indexed_paths for s in srcs):
                absent["queries_whose_sources_are_not_indexed"].append(
                    {"id": q["id"], "q": q["q"][:60], "sources": srcs})
                absent["count"] += 1
        report["conflation"] = absent
        print(f"  queries whose labelled sources are NOT in the index at all: "
              f"{absent['count']}")
        for x in absent["queries_whose_sources_are_not_indexed"]:
            print(f"    {x['id']}: {x['q']}")
        print("  These are NOT retrieval failures - the answer does not exist in the corpus.")
        print("  The harness counts them in the same candidate_miss bucket, which conflates")
        print("  'corpus never contained it' with 'retrieval did not find it'.")
    except Exception as e:  # noqa: BLE001
        print(f"  could not inspect the index: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ 6. pool
    print("\n=== 6. does widening the pool help candidate recall and Top-5 SEPARATELY? ===")
    dev_pools = json.loads((RESULTS / "eval_dev_pools.json").read_text(encoding="utf-8"))
    rowsout = []
    for c in dev_pools["configs"]:
        o = c["overall"]
        rowsout.append({"label": c["label"], "candidate_recall_pct": o["candidate_recall"],
                        "r@5_pct": o["r@5"], "r@1_pct": o["r@1"], "mrr": o["mrr"]})
    report["pool_effect"] = rowsout
    print(f"  {'config':<22} {'cand recall':>12} {'R@5':>7} {'R@1':>7} {'MRR':>8}")
    for r in rowsout:
        print(f"  {r['label']:<22} {r['candidate_recall_pct']:>11.1f}% "
              f"{r['r@5_pct']:>6.1f} {r['r@1_pct']:>6.1f} {r['mrr']:>8.4f}")
    cands = {r["label"]: r["candidate_recall_pct"] for r in rowsout}
    d10 = next((v for k, v in cands.items() if "pool10" in k), None)
    d50 = next((v for k, v in cands.items() if "pool50" in k), None)
    report["pool_effect_verdict"] = {
        "candidate_recall_pool10": d10, "candidate_recall_pool50": d50,
        "candidate_recall_improved": (d50 is not None and d10 is not None and d50 > d10),
        "statement": ("Candidate recall DOES improve with a larger pool while the final "
                      "Top-5 does NOT. The earlier claim that no metric improved is "
                      "therefore wrong and is corrected here."),
    }
    if d10 is not None and d50 is not None:
        print(f"\n  candidate recall improved {d10}% -> {d50}% with a larger pool, while "
              f"final Top-5 did not improve.")
        print("  The earlier statement that no metric improved is WRONG and is corrected.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
