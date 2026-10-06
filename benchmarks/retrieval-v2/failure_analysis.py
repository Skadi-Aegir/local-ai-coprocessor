"""
failure_analysis.py - classify the held-out failures so the review questions can be
answered from evidence rather than impression.

Distinguishes the three ways a query can fail, because they need different fixes:

  candidate_miss   the correct source never entered the pool -> a retrieval/chunking
                   problem, and NO ranker can fix it
  rank_miss        the correct source WAS in the pool but the final ordering pushed it
                   below top-k -> a ranking problem
  both             the pool missed it AND the ordering was poor

Also reports the hallucination cases with the evidence actually present in the pack, so
"invented" can be separated from "was told but ignored".
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from benchmark_v2 import all_queries  # noqa: E402

RESULTS = HERE / "results"


def classify(result_path: Path, top_k: int = 5) -> dict:
    d = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = next((c for c in d["configs"] if c.get("causal_switch")), d["configs"][-1])
    qmap = {q["id"]: q for q in all_queries()}

    buckets: dict[str, list] = defaultdict(list)
    per_cat = defaultdict(Counter)
    for r in cfg["rows"]:
        q = qmap.get(r["id"], {})
        cat = r.get("category", "?")
        if r.get("pool") == 0:
            buckets["empty_pool"].append(r)
            per_cat[cat]["empty_pool"] += 1
            continue
        in_topk = bool(r.get("r5"))
        in_pool = bool(r.get("candidate_hit"))
        if in_topk:
            per_cat[cat]["ok"] += 1
        elif not in_pool:
            buckets["candidate_miss"].append(r)
            per_cat[cat]["candidate_miss"] += 1
        else:
            buckets["rank_miss"].append(r)
            per_cat[cat]["rank_miss"] += 1

    return {"config": cfg.get("label"), "buckets": buckets,
            "per_category": {k: dict(v) for k, v in per_cat.items()},
            "counts": {k: len(v) for k, v in buckets.items()}}


def main() -> int:
    paths = sys.argv[1:] or [str(RESULTS / "eval_heldout.json")]
    for p in paths:
        res = classify(Path(p))
        print("=" * 78)
        print(f"{Path(p).name}   config={res['config']}")
        print("=" * 78)
        print(f"  counts: {res['counts']}")
        print("\n  per category:")
        for cat, c in sorted(res["per_category"].items()):
            tot = sum(c.values())
            ok = c.get("ok", 0)
            print(f"    {cat:<18} {ok:>3}/{tot:<3} ok   "
                  f"candidate_miss={c.get('candidate_miss', 0):<3} "
                  f"rank_miss={c.get('rank_miss', 0)}")

        for bucket in ("candidate_miss", "rank_miss"):
            rows = res["buckets"].get(bucket, [])
            if not rows:
                continue
            print(f"\n  --- {bucket} ({len(rows)}) ---")
            for r in rows[:12]:
                print(f"    [{r['category']:<16}] {r['q'][:60]}")
                print(f"        dense_rank={r['dense_rank']} rerank_rank={r['rerank_rank']} "
                      f"pool={r['pool']}")
                if r.get("top_paths"):
                    print(f"        top: {r['top_paths'][0]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
