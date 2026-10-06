"""
rag_compare_rerank.py - Phase 5 end-to-end RAG: no-RAG vs dense-RAG vs reranked-RAG.

Reuses the Phase 4 RAG questions (12 questions whose answers exist only in this project's
own files, so the no-RAG arm must fail unless the model guesses) and adds a THIRD arm:

  A  no RAG        local-fast alone
  B  dense RAG     embedding top-5 -> local-fast
  C  reranked RAG  embedding top-20 -> reranker -> top-5 -> local-fast

The point of Phase 5 is not to improve a retrieval metric in isolation but to improve the
final ANSWER, so this is the measurement that decides whether the reranker earns its
place. Scoring stays mechanical (expected fact keywords must appear, forbidden values must
not) so it is repeatable.

Usage:
  python rag_compare_rerank.py --out ../../benchmarks/reranker/results/rag_end_to_end.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reranker"))

from embedder import Embedder, EmbedderConfig  # noqa: E402
from index_store import IndexStore  # noqa: E402
from rerank_pipeline import PipelineConfig, two_stage_retrieve  # noqa: E402
from retriever import RetrieveConfig, Retriever  # noqa: E402

INDEX = (Path(__file__).resolve().parents[2] / r"cache\retrieval_bench\indexes\d1024_structured")
DIM = 1024
FAST = "http://127.0.0.1:8111"

# Same questions as Phase 4's rag_compare.py, so the arms are directly comparable.
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
        return json.loads(r.read().decode("utf-8"))


def ask(question: str, context: str | None, max_tokens: int = 400) -> dict:
    if context:
        system = ("Answer the question using ONLY the provided sources. If the sources do "
                  "not contain the answer, reply exactly: NOT IN SOURCES. Quote the "
                  "specific numbers and file names you rely on.")
        user = f"SOURCES:\n{context}\n\nQUESTION: {question}"
    else:
        system = ("Answer the question from your own knowledge. If you do not know, say "
                  "'I do not know' rather than guessing.")
        user = question
    t0 = time.time()
    r = post(f"{FAST}/v1/chat/completions", {
        "model": "local-fast",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    txt = r["choices"][0]["message"]["content"]
    return {"answer": txt, "seconds": round(time.time() - t0, 2),
            "prompt_tokens": r.get("timings", {}).get("prompt_n"),
            "gen_tps": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}


def score(answer: str, facts, bad) -> dict:
    low = answer.lower()
    hits = [all(k.lower() in low for k in grp) for grp in facts if grp]
    halluc = [all(k.lower() in low for k in grp) for grp in bad if grp]
    refused = ("not in sources" in low or "i do not know" in low or "不知道" in low
               or "not provided" in low or "no information" in low)
    return {"fact_hits": sum(1 for h in hits if h), "fact_total": len(hits),
            "all_facts": bool(hits) and all(hits),
            "hallucination_flags": sum(1 for h in halluc if h),
            "refused": refused}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--candidate-top-n", type=int, default=20)
    ap.add_argument("--final-top-k", type=int, default=5)
    ap.add_argument("--instruction-type", default="general")
    args = ap.parse_args()

    emb = Embedder(EmbedderConfig(dimension=args.dim, batch_size=32, max_seq_length=2048))
    store = IndexStore.load(args.index, args.dim)
    retr = Retriever(emb, store, RetrieveConfig(top_k=5, instruction=None))
    pipe_cfg = PipelineConfig(candidate_top_n=args.candidate_top_n,
                              final_top_k=args.final_top_k,
                              instruction_type=args.instruction_type)

    rows = []
    try:
        for i, t in enumerate(TASKS, 1):
            # ---- arm A: no RAG
            a = ask(t["q"], None)
            sa = score(a["answer"], t["facts"], t["bad"])

            # ---- arm B: dense top-5
            dense_hits = retr.retrieve(t["q"], top_k=args.final_top_k, instruction=None)
            dense_ctx = retr.pack(dense_hits)
            b = ask(t["q"], dense_ctx)
            sb = score(b["answer"], t["facts"], t["bad"])

            # ---- arm C: reranked
            pr = two_stage_retrieve(retr, t["q"], pipe_cfg)
            rerank_ctx = pr.pack(retr)
            c = ask(t["q"], rerank_ctx)
            sc = score(c["answer"], t["facts"], t["bad"])

            rows.append({
                "q": t["q"],
                "no_rag": {**a, **sa},
                "dense_rag": {**b, **sb,
                              "sources": [h.source_path for h in dense_hits],
                              "context_tokens": len(dense_ctx) // 4},
                "reranked_rag": {**c, **sc,
                                 "sources": [h.source_path for h in pr.hits],
                                 "dense_sources": [h.source_path for h in
                                                   retr.retrieve(t["q"], top_k=args.final_top_k,
                                                                 instruction=None)],
                                 "candidate_count": pr.candidate_count,
                                 "reranker_used": pr.reranker_used,
                                 "fallback_reason": pr.fallback_reason,
                                 "embed_s": round(pr.embed_s, 4),
                                 "rerank_s": round(pr.rerank_s, 4),
                                 "context_tokens": len(rerank_ctx) // 4},
            })
            print(f"  [{i:2}/{len(TASKS)}] noRAG {sa['fact_hits']}/{sa['fact_total']} | "
                  f"dense {sb['fact_hits']}/{sb['fact_total']} | "
                  f"reranked {sc['fact_hits']}/{sc['fact_total']}  "
                  f"(cand={pr.candidate_count}, rerank={pr.rerank_s}s)", flush=True)

        n = len(rows)

        def agg(key):
            return {
                "all_correct": sum(1 for r in rows if r[key]["all_facts"]),
                "fact_hits": sum(r[key]["fact_hits"] for r in rows),
                "fact_total": sum(r[key]["fact_total"] for r in rows),
                "hallucinations": sum(1 for r in rows if r[key]["hallucination_flags"]),
                "refusals": sum(1 for r in rows if r[key]["refused"]),
                "median_prompt_tokens": sorted(r[key]["prompt_tokens"] or 0
                                               for r in rows)[n // 2],
                "median_latency_s": sorted(r[key]["seconds"] for r in rows)[n // 2],
            }

        summary = {"questions": n, "candidate_top_n": args.candidate_top_n,
                   "final_top_k": args.final_top_k,
                   "instruction_type": args.instruction_type,
                   "arms": {k: agg(k) for k in ("no_rag", "dense_rag", "reranked_rag")},
                   "rows": rows}
        for k, v in summary["arms"].items():
            v["accuracy_pct"] = round(100 * v["all_correct"] / n, 1)
            v["fact_recall_pct"] = round(100 * v["fact_hits"] / v["fact_total"], 1)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n" + "=" * 78)
        print("END-TO-END RAG: no-RAG vs dense-RAG vs reranked-RAG")
        print("=" * 78)
        print(f"{'arm':<14} {'correct':>9} {'acc%':>7} {'fact recall':>12} "
              f"{'halluc':>7} {'refus':>6} {'prompt tok':>11}")
        for k in ("no_rag", "dense_rag", "reranked_rag"):
            v = summary["arms"][k]
            print(f"{k:<14} {v['all_correct']:>4}/{n:<4} {v['accuracy_pct']:>7.1f} "
                  f"{v['fact_recall_pct']:>11.1f}% {v['hallucinations']:>7} "
                  f"{v['refusals']:>6} {v['median_prompt_tokens']:>11}")
        print(f"\nwrote {out}")
        return 0
    finally:
        store.close()
        emb.close()


if __name__ == "__main__":
    raise SystemExit(main())
