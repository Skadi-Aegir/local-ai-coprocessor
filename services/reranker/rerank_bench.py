"""
rerank_bench.py - Phase 5 two-stage retrieval benchmark.

Answers the question the phase exists for: can the reranker pull the correct chunks that
Phase 4 left at rank 6 / rank 13 into a stable final Top-5?

Pipeline under test:

    query -> embedding -> Top-N candidates -> reranker -> final Top-K

Reports, for every configuration:
  * CANDIDATE RECALL at Top 5/10/20/50 - did the right chunk even reach the reranker?
    This is reported FIRST and separately because if candidate recall is poor the
    reranker cannot be blamed, and the specification is explicit about that.
  * DENSE vs RERANKED R@1/R@3/R@5/MRR/nDCG@5
  * RANK PROMOTION - per query, dense rank -> reranked rank, so the Phase 4 failures can
    be inspected individually rather than only in aggregate.
  * BOTH ground truths: strict (Phase 4's single file) and multi (any valid source).

The reranker is called over HTTP so the measurement describes the deployed service.
Requests are written by this script as UTF-8 JSON, never composed in a shell, because a
PowerShell-composed request body produced an unreproducible score earlier in Phase 5.

Usage:
  python rerank_bench.py --out ..\\..\\benchmarks\\reranker\\results\\sweep.json
  python rerank_bench.py --pools 10,20,50 --instructions default,code,none
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "retrieval"))

from benchmark_data import ALL_SETS, PROJECT, valid_sources  # noqa: E402
from embedder import Embedder, EmbedderConfig  # noqa: E402
from index_store import IndexStore  # noqa: E402
from retriever import Retriever, RetrieveConfig  # noqa: E402

RERANK_URL = "http://127.0.0.1:8114/v1/rerank"
INDEX_DIR = (Path(__file__).resolve().parents[2] / r"cache\retrieval_bench\indexes\d1024_structured")
DIM = 1024


# ---------------------------------------------------------------- metrics
def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(flags, k=5):
    """
    Binary-gain nDCG@k.

    The ideal ranking depends on HOW MANY relevant documents exist, not on the usual
    single-relevant assumption. An earlier version hardcoded `ideal = [1] + [0]*...`,
    which is correct only when one document can be relevant. Under multi-source ground
    truth several retrieved documents are legitimately relevant, so achieved DCG exceeded
    the single-document ideal and nDCG came out ABOVE 1.0 (1.4376) - a value that is
    mathematically impossible and would have silently inflated every reranked score.

    The ideal is therefore built from the number of relevant documents actually present
    in the truncated window, which is what nDCG normalises against.
    """
    window = flags[:k]
    if not window:
        return 0.0
    gains = [1.0 if f else 0.0 for f in window]
    n_rel = int(sum(gains))
    if n_rel == 0:
        return 0.0
    ideal = [1.0] * n_rel + [0.0] * (len(window) - n_rel)
    idcg = dcg(ideal)
    val = dcg(sorted(gains, reverse=True)) / idcg if idcg > 0 else 0.0
    # nDCG is normalised and cannot exceed 1. Assert rather than trust: an earlier
    # single-relevant ideal produced 1.4376 under multi-source ground truth, which would
    # have inflated every reranked figure without ever looking wrong in a table.
    assert val <= 1.0 + 1e-9, f"nDCG@k exceeded 1.0 ({val}); the ideal is miscomputed"
    return val


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def rerank(query, documents, instruction_type="default", top_n=None, instruction=None):
    payload = {"query": query, "documents": documents, "instruction_type": instruction_type}
    if instruction is not None:
        payload.pop("instruction_type", None)
        payload["instruction"] = instruction
    if top_n is not None:
        payload["top_n"] = top_n
    return post(RERANK_URL, payload)


# ---------------------------------------------------------------- evaluation
def evaluate(retr: Retriever, pool_n: int, top_k: int, instruction_type: str,
             queries: list[dict], use_rerank: bool = True,
             instruction_override: str | None = None) -> tuple[dict, list[dict]]:
    """
    Run one configuration over the query set.

    Returns (metrics, per-query rows). Dense retrieval happens once at pool_n; reranking
    reorders that same pool, so the two stages are directly comparable on identical
    candidates - the only difference is ordering.
    """
    rows = []
    t_embed = t_rank = 0.0

    for q in queries:
        vs = valid_sources(q)
        strict = {q["file"]}

        # ---- stage 1: dense candidates
        t0 = time.time()
        hits = retr.retrieve(q["q"], top_k=pool_n, instruction=None)
        t_embed += time.time() - t0
        paths = [h.source_path for h in hits]
        # note: instruction handling is done below via the query-side prompt choice
        if not hits:
            rows.append({"q": q["q"], "type": q["type"], "pool": 0})
            continue

        dense_rank_strict = next((i + 1 for i, p in enumerate(paths) if p in strict), None)
        dense_rank_multi = next((i + 1 for i, p in enumerate(paths) if p in vs), None)

        # ---- rank the pool: dense order is the baseline, reranker the treatment
        if use_rerank:
            t0 = time.time()
            docs = [h.content for h in hits]
            if instruction_override is not None:
                rr = rerank(q["q"], docs, instruction=instruction_override, top_n=None)
            else:
                it = instruction_type
                if it == "auto":
                    it = "code" if q["type"].startswith("code") or "code" in q["type"] else "default"
                rr = rerank(q["q"], docs, instruction_type=it, top_n=None)
            t_rank += time.time() - t0
            order = [x["index"] for x in rr["results"]]
            rerank_scores = {x["index"]: x["score"] for x in rr["results"]}
        else:
            order = list(range(len(hits)))
            rerank_scores = {}

        reranked_paths = [paths[i] for i in order]
        rank_strict = next((i + 1 for i, p in enumerate(reranked_paths) if p in strict), None)
        rank_multi = next((i + 1 for i, p in enumerate(reranked_paths) if p in vs), None)

        def flags_for(rank_of, sets):
            return [p in sets for p in reranked_paths[:top_k]]

        rows.append({
            "q": q["q"], "type": q["type"], "valid_sources": vs,
            "pool": len(hits),
            "dense_paths": paths[:top_k],
            "reranked_paths": reranked_paths[:top_k],
            # strict ground truth (Phase 4's single file)
            "dense_rank_strict": dense_rank_strict,
            "rerank_rank_strict": rank_strict,
            "strict_r1": bool(rank_strict == 1),
            "strict_r3": bool(rank_strict and rank_strict <= 3),
            "strict_r5": bool(rank_strict and rank_strict <= 5),
            "strict_dense_r5": bool(dense_rank_strict and dense_rank_strict <= 5),
            "strict_rr": (1.0 / rank_strict) if rank_strict else 0.0,
            "strict_ndcg5": ndcg_at_k([p in strict for p in reranked_paths], top_k),
            # multi ground truth (any valid source)
            "dense_rank_multi": dense_rank_multi,
            "rerank_rank_multi": rank_multi,
            "multi_r1": bool(rank_multi == 1),
            "multi_r3": bool(rank_multi and rank_multi <= 3),
            "multi_r5": bool(rank_multi and rank_multi <= 5),
            "multi_dense_r5": bool(dense_rank_multi and dense_rank_multi <= 5),
            "multi_rr": (1.0 / rank_multi) if rank_multi else 0.0,
            "multi_ndcg5": ndcg_at_k([p in vs for p in reranked_paths], top_k),
            "rerank_top_score": max(rerank_scores.values()) if rerank_scores else None,
        })

    ok = [r for r in rows if r.get("pool")]
    n = len(ok) or 1

    def agg(prefix):
        base = "dense" if not use_rerank else "rerank"
        return {
            "r@1": round(100 * sum(1 for r in ok if r[f"{prefix}_r1"]) / n, 1),
            "r@3": round(100 * sum(1 for r in ok if r[f"{prefix}_r3"]) / n, 1),
            "r@5": round(100 * sum(1 for r in ok if r[f"{prefix}_r5"]) / n, 1),
            "mrr": round(sum(r[f"{prefix}_rr"] for r in ok) / n, 4),
            "ndcg@5": round(sum(r[f"{prefix}_ndcg5"] for r in ok) / n, 4),
        }

    metrics = {
        "pool_n": pool_n, "top_k": top_k, "instruction": instruction_type,
        "reranked": use_rerank, "queries": len(ok),
        "strict": agg("strict"), "multi": agg("multi"),
        "mean_embed_s": round(t_embed / n, 4),
        "mean_rerank_s": round(t_rank / n, 4) if use_rerank else 0.0,
    }
    return metrics, rows


def candidate_recall(retr: Retriever, queries: list[dict], pools=(5, 10, 20, 50)) -> dict:
    """Did the correct chunk reach the reranker's pool at all? Reported first, on purpose."""
    maxp = max(pools)
    out = {p: {"strict": 0, "multi": 0} for p in pools}
    detail = []
    for q in queries:
        vs = valid_sources(q)
        hits = retr.retrieve(q["q"], top_k=maxp, instruction=None)
        paths = [h.source_path for h in hits]
        row = {"q": q["q"], "type": q["type"], "pool": len(hits)}
        for p in pools:
            sub = paths[:p]
            s = any(x in {q["file"]} for x in sub)
            m = any(x in vs for x in sub)
            row[f"strict@{p}"] = s
            row[f"multi@{p}"] = m
            if s:
                out[p]["strict"] += 1
            if m:
                out[p]["multi"] += 1
        detail.append(row)
    n = len(queries)
    return {
        "pools": {str(p): {"strict": round(100 * out[p]["strict"] / n, 1),
                           "multi": round(100 * out[p]["multi"] / n, 1)}
                  for p in pools},
        "queries": n,
        "rows": detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=str(INDEX_DIR))
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--pools", default="10,20,50")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--instructions", default="default")
    ap.add_argument("--sets", default="doc,code,mixed")
    args = ap.parse_args()

    pools = [int(x) for x in args.pools.split(",") if x.strip()]
    instrs = [x.strip() for x in args.instructions.split(",") if x.strip()]
    sets = [x.strip() for x in args.sets.split(",") if x.strip()]
    queries = [q for s in sets for q in ALL_SETS[s]]

    emb = Embedder(EmbedderConfig(dimension=args.dim, batch_size=32, max_seq_length=2048))
    store = IndexStore.load(args.index, args.dim)
    retr = Retriever(emb, store, RetrieveConfig(top_k=5, instruction=None))
    report: dict = {"queries": len(queries), "sets": sets, "dim": args.dim}
    try:
        print(f"index: {args.index}  chunks={store.stats()['chunks_in_metadata']}")
        print(f"queries: {len(queries)}\n")

        # ---------------- candidate recall FIRST
        print("=== candidate recall (did the right chunk reach the pool?) ===")
        cr = candidate_recall(retr, queries, pools=[5, 10, 20, 50])
        report["candidate_recall"] = cr
        print(f"{'pool':>6} {'strict':>9} {'multi':>9}")
        for p in ("5", "10", "20", "50"):
            v = cr["pools"][p]
            print(f"{p:>6} {v['strict']:>8.1f}% {v['multi']:>8.1f}%")

        # ---------------- dense baseline
        print("\n=== dense baseline (no rerank) ===")
        dense_m, dense_rows = evaluate(retr, max(pools), args.top_k, "n/a", queries,
                                       use_rerank=False)
        report["dense"] = dense_m
        print(f"  strict R@1={dense_m['strict']['r@1']} R@3={dense_m['strict']['r@3']} "
              f"R@5={dense_m['strict']['r@5']} MRR={dense_m['strict']['mrr']} "
              f"nDCG@5={dense_m['strict']['ndcg@5']}")
        print(f"  multi  R@1={dense_m['multi']['r@1']} R@3={dense_m['multi']['r@3']} "
              f"R@5={dense_m['multi']['r@5']} MRR={dense_m['multi']['mrr']} "
              f"nDCG@5={dense_m['multi']['ndcg@5']}")

        # ---------------- reranked configurations
        configs = []
        for instr in instrs:
            for pool in pools:
                print(f"\n=== rerank-{pool} (instruction={instr}) ===", flush=True)
                m, rows = evaluate(retr, pool, args.top_k, instr, queries, use_rerank=True)
                m["instruction"] = instr
                configs.append({"metrics": m, "rows": rows})
                print(f"  strict R@1={m['strict']['r@1']} R@3={m['strict']['r@3']} "
                      f"R@5={m['strict']['r@5']} MRR={m['strict']['mrr']} "
                      f"nDCG@5={m['strict']['ndcg@5']}")
                print(f"  multi  R@1={m['multi']['r@1']} R@3={m['multi']['r@3']} "
                      f"R@5={m['multi']['r@5']} MRR={m['multi']['mrr']} "
                      f"nDCG@5={m['multi']['ndcg@5']}")
                print(f"  embed {m['mean_embed_s']}s + rerank {m['mean_rerank_s']}s "
                      f"= {round(m['mean_embed_s']+m['mean_rerank_s'],4)}s per query")
        report["configs"] = configs
        report["dense_rows"] = dense_rows

        # ---------------- rank promotion
        print("\n=== rank promotion: dense vs reranked (multi ground truth) ===")
        for cfg in configs:
            m = cfg["metrics"]
            proms = []
            for r in cfg["rows"]:
                dr, rr = r.get("dense_rank_multi"), r.get("rerank_rank_multi")
                if dr is not None and rr is not None and dr != rr:
                    proms.append((r["q"], dr, rr, dr - rr))
            cfg["promotions"] = [
                {"q": q, "dense_rank": d, "rerank_rank": a, "delta": delta}
                for q, d, a, delta in sorted(proms, key=lambda x: -x[3])
            ]
            better = [p for p in proms if p[3] > 0]
            worse = [p for p in proms if p[3] < 0]
            print(f"  pool={m['pool_n']} instr={m['instruction']}: "
                  f"{len(better)} promoted, {len(worse)} demoted")
            for q, d, a, delta in sorted(better, key=lambda x: -x[3])[:5]:
                print(f"      +{delta:2}  {d:2} -> {a:2}  {q[:60]}")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

        # ---------------- summary table
        print("\n" + "=" * 104)
        print(f"{'config':<26} {'ground':<7} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>7} {'nDCG@5':>7}")
        print("=" * 104)
        for label, mm in [("dense (no rerank)", dense_m)] + \
                         [(f"rerank-{c['metrics']['pool_n']}/{c['metrics']['instruction']}",
                           c["metrics"]) for c in configs]:
            for g in ("strict", "multi"):
                v = mm[g]
                print(f"{label if g=='strict' else '':<26} {g:<7} {v['r@1']:>6.1f} "
                      f"{v['r@3']:>6.1f} {v['r@5']:>6.1f} {v['mrr']:>7.4f} {v['ndcg@5']:>7.4f}")
        return 0
    finally:
        store.close()
        emb.close()


if __name__ == "__main__":
    raise SystemExit(main())
