"""
baseline_reconcile.py - why did Phase 4 report 91.7% RAG and Phase 5 66.7%?

Both used the same 12 questions and the same mechanical scorer, so the difference must come
from the RETRIEVAL ARM, not the scoring. The two RAG harnesses retrieved from different
indexes:

  Phase 4 rag_compare.py       -> HTTP /v1/retrieve on the SERVED index  (1173 chunks)
  Phase 5 rag_compare_rerank.py -> direct IndexStore on the BENCH index   (186 chunks)

This script holds everything else constant - same questions, same top_k=5, same local-fast,
same temperature, same grader - and varies ONLY the index, so the comparison is like for
like. It reports three arms:

  A  served index, top-5
  B  bench index, top-5
  C  served index, top-5, reranked from a top-20 pool

If A reproduces ~91.7% and B reproduces ~66.7%, the discrepancy is explained by corpus
composition and the Phase 4 figure was never comparable to the Phase 5 baseline.

Usage:
  python baseline_reconcile.py --out ../../benchmarks/retrieval-v2/results/baseline_reconcile.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# this harness lives in benchmarks/retrieval-v2/, so the service modules it reuses must be
# added explicitly rather than assumed to be siblings
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))
sys.path.insert(0, str(REPO / "services" / "reranker"))

from embedder import Embedder, EmbedderConfig  # noqa: E402
from index_store import IndexStore  # noqa: E402
from rerank_pipeline import PipelineConfig, two_stage_retrieve  # noqa: E402
from retriever import RetrieveConfig, Retriever  # noqa: E402

SERVED = (Path(__file__).resolve().parents[2] / r"vector\embedding\local-ai")
BENCH = (Path(__file__).resolve().parents[2] / r"cache\retrieval_bench\indexes\d1024_structured")
DIM = 1024
FAST = "http://127.0.0.1:8111"
EMBED_HTTP = "http://127.0.0.1:8113"

# identical to both Phase 4 rag_compare.py and Phase 5 rag_compare_rerank.py
TASKS = [
    dict(q="What context size does the local-fast worker use by default?",
         facts=[["8192", "8k"]], bad=[["4096"], ["16384"]]),
    dict(q="How much VRAM does local-fast occupy when resident?",
         facts=[["2981"]], bad=[["1500"], ["8000"]]),
    dict(q="What generation speed did local-fast reach in its benchmark?",
         facts=[["130"]], bad=[["500 tok"], ["50 tok"]]),
    dict(q="Why is concurrent inference of two local models not recommended?",
         facts=[["47", "throughput"], ["half", "47"]], bad=[[]]),
    dict(q="How long does it take to switch between the two local models?",
         facts=[["7", "second"], ["3", "4"]], bad=[["minute"]]),
    dict(q="What is the JSON schema adherence rate reported for local-fast?",
         facts=[["100"]], bad=[["50%"], ["80%"]]),
    dict(q="Which pooling method does the Qwen3 embedding model use?",
         facts=[["last"]], bad=[["mean pooling"]]),
    dict(q="What is the maximum embedding dimension of the embedding model?",
         facts=[["1024"]], bad=[["256"], ["4096"]]),
    dict(q="How much did retrieval reduce the token count?",
         facts=[["96", "99", "reduction"]], bad=[["10%"], ["20%"]]),
    dict(q="What port does the embedding service listen on?",
         facts=[["8113"]], bad=[["8111"], ["8112"]]),
    dict(q="What is the recommended batch size for embedding, and why?",
         facts=[["32"]], bad=[["128"], ["256"]]),
    dict(q="Which chunking strategy performed better, fixed or structure aware?",
         facts=[["structure"]], bad=[["fixed performed better", "fixed was better"]]),
]


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask(question, context, max_tokens=400):
    if context:
        system = ("Answer the question using ONLY the provided sources. If the sources do "
                  "not contain the answer, reply exactly: NOT IN SOURCES. Quote the "
                  "specific numbers and file names you rely on.")
        user = f"SOURCES:\n{context}\n\nQUESTION: {question}"
    else:
        system = ("Answer the question from your own knowledge. If you do not know, say "
                  "'I do not know' rather than guessing.")
        user = question
    r = post(f"{FAST}/v1/chat/completions", {
        "model": "local-fast",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"]["content"]


def score(answer, facts, bad):
    low = answer.lower()
    hits = [all(k.lower() in low for k in g) for g in facts if g]
    halluc = [all(k.lower() in low for k in g) for g in bad if g]
    refused = ("not in sources" in low or "i do not know" in low or "不知道" in low
               or "not provided" in low or "no information" in low)
    return {"fact_hits": sum(1 for h in hits if h), "fact_total": len(hits),
            "all_facts": bool(hits) and all(hits),
            "hallucination_flags": sum(1 for h in halluc if h), "refused": refused}


def http_retrieve_pack(query, top_k=5):
    """Exactly the Phase 4 code path: the SERVED index over HTTP."""
    r = post(f"{EMBED_HTTP}/v1/retrieve",
             {"query": query, "project": "local-ai", "top_k": top_k, "pack": True})
    return r.get("context", ""), r.get("results", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=str(BENCH))
    ap.add_argument("--served-index", default=str(SERVED))
    args = ap.parse_args()

    # ONE embedder, ONE device, ONE packing function for every arm. The only variable
    # between arms is which index is searched, which is the whole point of this script.
    # CPU is used because the 4B worker and the reranker already occupy the GPU and a
    # three-way transient spike produced "CUDA error: unknown error" on the first attempt.
    emb = Embedder(EmbedderConfig(dimension=DIM, batch_size=32, max_seq_length=2048,
                                  device="cpu"))
    bench_store = IndexStore.load(args.index, DIM)
    served_store = IndexStore.load(args.served_index, DIM)
    bench_retr = Retriever(emb, bench_store, RetrieveConfig(top_k=5, instruction=None))
    served_retr = Retriever(emb, served_store, RetrieveConfig(top_k=5, instruction=None))

    report = {"questions": len(TASKS), "arms": {}}
    arms = {}

    try:
        bench = bench_store.stats()
        served = served_store.stats()
        report["served_index"] = served
        report["bench_index"] = bench
        print(f"served index : {served['chunks_in_metadata']} chunks")
        print(f"bench  index : {bench['chunks_in_metadata']} chunks\n")

        for arm, retriever in (("A_served", served_retr),
                               ("B_bench", bench_retr),
                               ("C_served_reranked", served_retr),
                               ("D_bench_reranked", bench_retr)):
            print(f"=== arm {arm} ===", flush=True)
            rows = []
            for i, t in enumerate(TASKS, 1):
                if arm.endswith("reranked"):
                    pr = two_stage_retrieve(retriever, t["q"],
                                            PipelineConfig(candidate_top_n=20,
                                                           final_top_k=5,
                                                           instruction_type="general"))
                    ctx = pr.pack(retriever)
                    paths = [h.source_path for h in pr.hits]
                else:
                    hits = retriever.retrieve(t["q"], top_k=5, instruction=None)
                    ctx = retriever.pack(hits)
                    paths = [h.source_path for h in hits]
                ans = ask(t["q"], ctx)
                s = score(ans, t["facts"], t["bad"])
                rows.append({"q": t["q"], "paths": paths, **s})
                mark = "OK " if s["all_facts"] else "   "
                print(f"  {mark}[{i:2}/{len(TASKS)}] {s['fact_hits']}/{s['fact_total']} "
                      f"{t['q'][:52]}", flush=True)

            n = len(rows)
            arms[arm] = {
                "all_correct": sum(1 for r in rows if r["all_facts"]),
                "accuracy_pct": round(100 * sum(1 for r in rows if r["all_facts"]) / n, 1),
                "fact_hits": sum(r["fact_hits"] for r in rows),
                "fact_total": sum(r["fact_total"] for r in rows),
                "hallucinations": sum(1 for r in rows if r["hallucination_flags"]),
                "refusals": sum(1 for r in rows if r["refused"]),
                "rows": rows,
            }
            arms[arm]["fact_recall_pct"] = round(
                100 * arms[arm]["fact_hits"] / arms[arm]["fact_total"], 1)
            print(f"  -> {arms[arm]['all_correct']}/{n} = {arms[arm]['accuracy_pct']}% "
                  f"fact recall {arms[arm]['fact_recall_pct']}%\n", flush=True)

        report["arms"] = arms
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("=" * 74)
        print(f"{'arm':<22} {'correct':>9} {'acc%':>7} {'fact recall':>12} {'refus':>6}")
        print("=" * 74)
        for k, v in arms.items():
            print(f"{k:<22} {v['all_correct']:>4}/{len(TASKS):<4} {v['accuracy_pct']:>7.1f} "
                  f"{v['fact_recall_pct']:>11.1f}% {v['refusals']:>6}")
        print(f"\nwrote {out}")
        return 0
    finally:
        bench_store.close()
        served_store.close()
        emb.close()


if __name__ == "__main__":
    raise SystemExit(main())
