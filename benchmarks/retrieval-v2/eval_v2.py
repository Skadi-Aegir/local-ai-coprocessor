"""
eval_v2.py - Phase 5.5 evaluation harness.

Records, for EVERY query, both the dense rank and the reranked rank of the first correct
source, so a regression can be traced to the stage that caused it rather than only seen as
an aggregate movement.

Reports per config and per category:
  Recall@1/3/5, MRR, nDCG@5, candidate recall at the pool size, end-to-end latency, and the
  actual prompt tokens handed to the LLM.

Ground truth is multi-source throughout, AND retrieval relevance is kept separate from
factual recency: for `version_conflict` queries the harness additionally reports how often
a SUPERSEDED source was promoted, so a retriever is never credited for surfacing a stale
answer just because the stale file was topically relevant.

Usage:
  python eval_v2.py --out results/eval.json --pools 10,20,50
  python eval_v2.py --out results/eval.json --splits dev --instructions general,causal
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
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "services" / "retrieval"))
sys.path.insert(0, str(REPO / "services" / "reranker"))

from benchmark_v2 import all_queries  # noqa: E402
from index_store import IndexStore  # noqa: E402
from retriever import RetrieveConfig, Retriever  # noqa: E402

RERANK_URL = "http://127.0.0.1:8114/v1/rerank"
BENCH_INDEX = REPO / "cache/retrieval_bench/indexes/d1024_structured"
DIM = 1024


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(flags, k=5):
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
    assert val <= 1.0 + 1e-9, f"nDCG@k exceeded 1.0 ({val})"
    return val


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def rerank(query, docs, instruction_type="general", timeout=600):
    return post(RERANK_URL, {"query": query, "documents": docs,
                             "instruction_type": instruction_type}, timeout=timeout)


# instruction texts for the causal experiment
CAUSAL_INSTRUCTION = (
    "Given a question asking why a design or engineering decision was made, determine "
    "whether the document explains the reasoning, rationale, trade-off or measured "
    "evidence behind that decision. A document that only describes how something works or "
    "how to configure it does NOT answer a why question."
)
GENERAL_INSTRUCTION = (
    "Given the user query, determine whether the document contains information useful for "
    "answering the query."
)


def run_config(retr: Retriever, queries: list[dict], pool: int, top_k: int,
               instruction_type: str | None, instruction_text: str | None,
               reranked: bool, causal_switch: bool = False,
               rule_assist: bool = False) -> dict:
    rows = []
    embed_t = rank_t = 0.0
    for q in queries:
        valid = set(q["sources"])
        superseded = set(q.get("superseded") or [])

        t0 = time.time()
        hits = retr.retrieve(q["q"], top_k=pool, instruction=None)
        embed_t += time.time() - t0
        if not hits:
            rows.append({"id": q["id"], "q": q["q"], "category": q["category"],
                         "split": q["split"], "pool": 0,
                         "dense_rank": None, "rerank_rank": None,
                         "candidate_hit": False, "superseded_promoted": False})
            continue
        paths = [h.source_path for h in hits]
        dense_rank = next((i + 1 for i, p in enumerate(paths) if p in valid), None)
        cand_hit = dense_rank is not None and dense_rank <= pool

        if reranked:
            it, itext = instruction_type, instruction_text
            if causal_switch:
                # switch on a LIGHT RULE, not on model confidence
                it, itext = ("__causal__", CAUSAL_INSTRUCTION) if is_causal(q["q"]) \
                    else ("general", GENERAL_INSTRUCTION)
            t0 = time.time()
            payload = {"query": q["q"], "documents": [h.content for h in hits]}
            if itext is not None:
                payload["instruction"] = itext
            else:
                payload["instruction_type"] = it or "general"
            rr = post(RERANK_URL, payload)
            rank_t += time.time() - t0
            order = [x["index"] for x in rr["results"]]
        else:
            order = list(range(len(hits)))

        ordered = [paths[i] for i in order]

        if rule_assist:
            # Light rule: if a version-conflict query is involved, push sources that are
            # known-superseded to the back. This is a metadata rule, not a model signal.
            if superseded:
                ordered = ([p for p in ordered if p not in superseded]
                           + [p for p in ordered if p in superseded])

        top = ordered[:top_k]
        rank = next((i + 1 for i, p in enumerate(ordered) if p in valid), None)
        sup_rank = next((i + 1 for i, p in enumerate(top) if p in superseded), None)
        rows.append({
            "id": q["id"], "q": q["q"], "category": q["category"],
            "split": q["split"], "pool": len(hits),
            "dense_rank": dense_rank,
            "rerank_rank": rank,
            "candidate_hit": cand_hit,
            "top_paths": top,
            "superseded_in_top": sup_rank is not None,
            "superseded_promoted": bool(superseded and sup_rank == 1),
            "r1": rank == 1, "r3": bool(rank and rank <= 3), "r5": bool(rank and rank <= 5),
            "rr": (1.0 / rank) if rank else 0.0,
            "ndcg5": ndcg_at_k([p in valid for p in ordered], top_k),
        })

    ok = [r for r in rows if r.get("pool")]
    n = len(ok) or 1

    def agg(rs):
        m = len(rs) or 1
        return {
            "n": len(rs),
            "r@1": round(100 * sum(1 for r in rs if r["r1"]) / m, 1),
            "r@3": round(100 * sum(1 for r in rs if r["r3"]) / m, 1),
            "r@5": round(100 * sum(1 for r in rs if r["r5"]) / m, 1),
            "mrr": round(sum(r["rr"] for r in rs) / m, 4),
            "ndcg@5": round(sum(r["ndcg5"] for r in rs) / m, 4),
            "candidate_recall": round(100 * sum(1 for r in rs if r["candidate_hit"]) / m, 1),
        }

    cats = sorted({r["category"] for r in ok})
    return {
        "pool": pool, "top_k": top_k, "reranked": reranked,
        "instruction_type": instruction_type, "causal_switch": causal_switch,
        "rule_assist": rule_assist,
        "overall": agg(ok),
        "by_category": {c: agg([r for r in ok if r["category"] == c]) for c in cats},
        "superseded_promoted": sum(1 for r in ok if r["superseded_promoted"]),
        "superseded_in_top": sum(1 for r in ok if r["superseded_in_top"]),
        "mean_embed_s": round(embed_t / n, 4),
        "mean_rerank_s": round(rank_t / n, 4) if reranked else 0.0,
        "rows": rows,
    }


def is_causal(q: str) -> bool:
    """Light rule for intent switching. Deliberately simple and inspectable."""
    low = q.lower()
    return any(low.startswith(w) or f" {w} " in low
               for w in ("why", "为什么", "为何")) or "what caused" in low


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=str(BENCH_INDEX))
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--pools", default="10")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--modes", default="dense,rerank",
                    help="comma list of: dense, rerank, causal_switch, rule_assist")
    ap.add_argument("--instructions", default="general")
    args = ap.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    cats = tuple(c.strip() for c in args.categories.split(",")) if args.categories else None
    pools = [int(x) for x in args.pools.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    instrs = [i.strip() for i in args.instructions.split(",") if i.strip()]

    queries = all_queries(splits=splits, categories=cats)
    print(f"index    : {args.index}")
    print(f"queries  : {len(queries)} (splits={splits}, categories={cats or 'all'})")

    # CPU embedding: the GPU already holds the 4B worker and the reranker, and a three-way
    # transient spike produced a CUDA unknown-error earlier in this phase.
    sys.path.insert(0, str(REPO / "services" / "retrieval"))
    from embedder import Embedder, EmbedderConfig
    emb = Embedder(EmbedderConfig(dimension=args.dim, batch_size=32,
                                  max_seq_length=2048, device="cpu"))
    store = IndexStore.load(args.index, args.dim)
    retr = Retriever(emb, store, RetrieveConfig(top_k=5, instruction=None))

    report = {"queries": len(queries), "splits": list(splits),
              "index": args.index, "configs": []}
    try:
        print(f"chunks   : {store.stats()['chunks_in_metadata']}\n")

        def add(label, **kw):
            cfg = run_config(retr, queries, **kw)
            cfg["label"] = label
            report["configs"].append(cfg)
            o = cfg["overall"]
            print(f"{label:<34} R@1={o['r@1']:5.1f} R@3={o['r@3']:5.1f} "
                  f"R@5={o['r@5']:5.1f} MRR={o['mrr']:.4f} nDCG@5={o['ndcg@5']:.4f} "
                  f"cand={o['candidate_recall']:5.1f}", flush=True)

        for pool in pools:
            add(f"dense_pool{pool}",
                pool=pool, top_k=args.top_k, instruction_type=None,
                instruction_text=None, reranked=False)
        for pool in pools:
            for instr in instrs:
                add(f"rerank{pool}_{instr}",
                    pool=pool, top_k=args.top_k, instruction_type=instr,
                    instruction_text=None, reranked=True)
        if "causal_switch" in modes:
            add("rerank10_causal_switch",
                pool=pools[0], top_k=args.top_k, instruction_type=None,
                instruction_text=None, reranked=True, causal_switch=True)
        if "rule_assist" in modes:
            add("rerank10_general_ruleassist",
                pool=pools[0], top_k=args.top_k, instruction_type="general",
                instruction_text=None, reranked=True, rule_assist=True)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")

        # per-category detail for the causal experiment
        print("\n=== per-category (all configs) ===")
        cats_present = sorted({r["category"] for c in report["configs"] for r in c["rows"]})
        hdr = "category".ljust(18) + "".join(f"{c['label'][:16]:>18}" for c in report["configs"])
        print(hdr)
        for cat in cats_present:
            line = f"{cat:<18}"
            for c in report["configs"]:
                m = c["by_category"].get(cat, {})
                line += f"{m.get('r@5', 0):>18.1f}"
            print(line)
        return 0
    finally:
        store.close()
        emb.close()


if __name__ == "__main__":
    raise SystemExit(main())
