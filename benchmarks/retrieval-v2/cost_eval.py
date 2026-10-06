"""
cost_eval.py - Phase 5.5 cost evaluation across the four retrieval arms.

Arms:
  no_rag        local-fast alone
  dense_rag     embedding top-5 -> local-fast
  reranked_rag  embedding top-10 -> reranker -> top-5 -> local-fast
  adaptive      the same, but SKIPPING the reranker for queries whose dense result is
                unambiguous. "Adaptive" is defined by a LIGHT RULE, not by model
                confidence - see below.

THE TOKEN ACCOUNTING RULE THIS SCRIPT EXISTS TO ENFORCE
  Internal reranker tokens are NOT cloud LLM input tokens and the two are never added
  together. The reranker reads up to 10 candidate documents and emits one score each; that
  work never leaves the machine and costs no money. Only the packed context handed to the
  answer model is a "prompt token" in any cloud sense. Both are reported, in separate
  columns, labelled.

CLOUD NECESSITY
  Reported as a count over THIS fixed question set with the routing rule stated inline.
  It is explicitly NOT extrapolated to all workloads: 12 questions cannot estimate the
  escalation rate of a real workload, and the phase instructions forbid presenting it as a
  cost-saving ratio.

Usage:
  python cost_eval.py --out results/cost_eval.json
"""

from __future__ import annotations

import argparse
import json
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

from index_store import IndexStore  # noqa: E402
from retriever import RetrieveConfig, Retriever  # noqa: E402

BENCH_INDEX = REPO / "cache/retrieval_bench/indexes/d1024_structured"
DIM = 1024
FAST = "http://127.0.0.1:8111"
RERANK = "http://127.0.0.1:8114/v1/rerank"
EMBED_HTTP = "http://127.0.0.1:8113"

# Same 12 questions as Phase 4/5 so the arms stay comparable.
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

# Light adaptive rule: skip the reranker when the dense top-1 is clearly separated from
# top-2 by a MARGIN, which is a property of the retrieval scores themselves rather than a
# claim about model confidence. A margin is a decision boundary, not a probability.
MARGIN_THRESHOLD = 0.06


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
    return r["choices"][0]["message"]["content"], r.get("timings", {})


def score(answer, facts, bad):
    low = answer.lower()
    hits = [all(k.lower() in low for k in g) for g in facts if g]
    halluc = [all(k.lower() in low for k in g) for g in bad if g]
    refused = ("not in sources" in low or "i do not know" in low or "\u4e0d\u77e5\u9053" in low
               or "not provided" in low or "no information" in low)
    return {"fact_hits": sum(1 for h in hits if h), "fact_total": len(hits),
            "all_facts": bool(hits) and all(hits),
            "hallucination_flags": sum(1 for h in halluc if h), "refused": refused}


def pack_sources(hits, limit=8000):
    parts = []
    for i, h in enumerate(hits, 1):
        body = h.content[:limit]
        parts.append(f"SOURCE {i}\npath: {h.source_path}\nlines: {h.start_line}-{h.end_line}\n"
                     f"score: {h.score:.4f}\nchunk_id: {h.chunk_id}\ncontent:\n{body}")
    header = ("The following sources were retrieved from the project. They are the only "
              "project facts available to you. If the answer is not present, say so "
              "instead of guessing.\n")
    return header + "\n\n" + "\n\n".join(parts)


def est_tokens(text: str) -> int:
    """CJK-aware estimator; a flat chars/4 mis-sizes this mixed-language content."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(cjk * 0.6 + (len(text) - cjk) * 0.25)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--final-top-k", type=int, default=5)
    ap.add_argument("--candidate-top-n", type=int, default=10)
    args = ap.parse_args()

    from embedder import Embedder, EmbedderConfig
    emb = Embedder(EmbedderConfig(dimension=DIM, batch_size=32, max_seq_length=2048,
                                  device="cpu"))
    store = IndexStore.load(BENCH_INDEX, DIM)
    retr = Retriever(emb, store, RetrieveConfig(top_k=5, instruction=None))

    arms = ["no_rag", "dense_rag", "reranked_rag", "adaptive"]
    rows: dict[str, list] = {a: [] for a in arms}
    # token bookkeeping, kept in SEPARATE columns on purpose
    internal_pairs = {"no_rag": 0, "dense_rag": 0, "reranked_rag": 0, "adaptive": 0}
    rerank_calls = {"reranked_rag": 0, "adaptive": 0}

    try:
        for i, t in enumerate(TASKS, 1):
            # ---------------- candidate pool for this question
            cand = retr.retrieve(t["q"], top_k=args.candidate_top_n, instruction=None)
            dense5 = cand[: args.final_top_k]
            margin = (cand[0].score - cand[1].score) if len(cand) > 1 else 1.0
            # adaptive: skip reranking when the dense ordering is already clearly separated
            skip_rerank = margin >= MARGIN_THRESHOLD
            itype = "causal" if any(w in t["q"].lower() for w in ("why", "\u4e3a\u4ec0\u4e48")) else "general"
            INSTR = {"causal": ("Given a question asking why a design or engineering decision "
                                "was made, determine whether the document explains the "
                                "reasoning, rationale, trade-off or measured evidence behind "
                                "that decision."),
                     "general": ("Given the user query, determine whether the document "
                                 "contains information useful for answering the query.")}

            def do_rerank(hits):
                rr = post(RERANK, {"query": t["q"],
                                   "documents": [h.content for h in hits],
                                   "instruction": INSTR[itype]})
                internal_pairs["reranked_rag"] += len(hits)
                return [hits[x["index"]] for x in rr["results"]]

            for arm in arms:
                t0 = time.time()
                if arm == "no_rag":
                    ctx, hits_for_answer = "", []
                elif arm == "dense_rag":
                    hits_for_answer = dense5
                    ctx = pack_sources(dense5)
                elif arm == "reranked_rag":
                    ordered = do_rerank(cand)
                    rerank_calls["reranked_rag"] += 1
                    hits_for_answer = ordered[: args.final_top_k]
                    ctx = pack_sources(hits_for_answer)
                else:  # adaptive
                    if skip_rerank:
                        hits_for_answer = dense5
                    else:
                        ordered = do_rerank(cand)
                        rerank_calls["adaptive"] += 1
                        hits_for_answer = ordered[: args.final_top_k]
                    ctx = pack_sources(hits_for_answer)
                retrieval_s = time.time() - t0

                ans, tim = ask(t["q"], ctx if ctx else None)
                s = score(ans, t["facts"], t["bad"])
                # `prompt_n` from llama.cpp is KV-CACHE-AWARE: it counts tokens actually
                # evaluated, not the size of the prompt. Running four arms over the same
                # question re-sends an identical prompt, so later arms report a handful of
                # tokens and the number collapses to 4 - which says nothing about cost. The
                # cache-independent measure of what the model was asked to read is the
                # context itself, so BOTH are recorded and the estimated figure is the one
                # used for comparison.
                rows[arm].append({
                    "q": t["q"], "retrieval_s": round(retrieval_s, 4),
                    "llm_s": round(tim.get("predicted_ms", 0) / 1000, 3) if tim else None,
                    "prompt_tokens_cache_aware": tim.get("prompt_n") if tim else None,
                    "gen_tokens": tim.get("predicted_n") if tim else None,
                    "context_chars": len(ctx), "context_tokens_est": est_tokens(ctx),
                    "sources": len(hits_for_answer),
                    "reranker_used": arm in ("reranked_rag",) or (arm == "adaptive"
                                                                 and not skip_rerank),
                    "margin": round(margin, 4),
                    **s,
                })
            print(f"  [{i:2}/{len(TASKS)}] adaptive_skip={skip_rerank} "
                  f"margin={margin:.4f}  " +
                  " ".join(f"{a}={rows[a][-1]['fact_hits']}/{rows[a][-1]['fact_total']}"
                           for a in arms), flush=True)

        n = len(TASKS)
        out_arms = {}
        for arm in arms:
            rs = rows[arm]
            pt = [r["context_tokens_est"] for r in rs if r["context_tokens_est"]]
            out_arms[arm] = {
                "all_correct": sum(1 for r in rs if r["all_facts"]),
                "accuracy_pct": round(100 * sum(1 for r in rs if r["all_facts"]) / n, 1),
                "fact_hits": sum(r["fact_hits"] for r in rs),
                "fact_total": sum(r["fact_total"] for r in rs),
                "refusals": sum(1 for r in rs if r["refused"]),
                "hallucinations": sum(1 for r in rs if r["hallucination_flags"]),
                # SEPARATE token columns - never summed
                # cache-INDEPENDENT: the size of the context the model was asked to read.
                # The cache-aware prompt_n is recorded per row but is NOT used for
                # comparison, because re-sending an identical prompt makes it collapse.
                "median_llm_prompt_tokens_est": int(statistics.median(pt)) if pt else None,
                "median_llm_prompt_tokens_cache_aware": int(statistics.median(
                    [r["prompt_tokens_cache_aware"] for r in rs
                     if r["prompt_tokens_cache_aware"]] or [0])),
                "internal_reranker_pairs": internal_pairs[arm],
                "reranker_invocations": rerank_calls.get(arm, 0),
                "mean_retrieval_s": round(statistics.mean(r["retrieval_s"] for r in rs), 4),
                "mean_llm_s": round(statistics.mean([r["llm_s"] for r in rs
                                                     if r["llm_s"]] or [0]), 3),
                "rows": rs,
            }
            out_arms[arm]["fact_recall_pct"] = round(
                100 * out_arms[arm]["fact_hits"] / out_arms[arm]["fact_total"], 1)

        raw_corpus_tokens = est_tokens("\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in (REPO / "cache/retrieval_bench/corpus").rglob("*") if p.is_file()))

        report = {
            "questions": n, "final_top_k": args.final_top_k,
            "candidate_top_n": args.candidate_top_n,
            "adaptive_rule": {
                "type": "dense top-1 minus top-2 score MARGIN",
                "threshold": MARGIN_THRESHOLD,
                "why_not_confidence": ("a score margin is a decision boundary on the "
                                       "retrieval scores themselves, NOT a claim about "
                                       "model confidence, which this project has no "
                                       "calibration data for"),
                "reranks_skipped": sum(1 for r in rows["adaptive"] if not r["reranker_used"]),
            },
            "arms": out_arms,
            "raw_corpus_tokens_est": raw_corpus_tokens,
            "token_accounting_note": (
                "internal_reranker_pairs counts query-document pairs the LOCAL reranker "
                "scored. It is NOT a prompt-token count and is never added to "
                "median_llm_prompt_tokens."),
            "cloud_necessity_note": (
                "Cloud escalation is reported as a raw count over this fixed 12-question "
                "set ONLY. It is NOT extrapolated to real workloads: 12 questions cannot "
                "estimate an escalation rate."),
        }
        # cloud necessity by the documented routing rule, counted not extrapolated
        low_conf = 0
        for r in rows["reranked_rag"]:
            if r["fact_hits"] == 0:
                low_conf += 1
        report["cloud_necessity"] = {
            "rule": "a question whose reranked context produced no fact hit is escalated",
            "escalated": low_conf, "of": n,
            "note": "count on this fixed set only; not a workload escalation rate",
        }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n" + "=" * 96)
        print(f"{'arm':<14} {'acc%':>6} {'fact rec':>9} {'refus':>6} {'halluc':>7} "
              f"{'llm tok':>8} {'rr pairs':>9} {'rr calls':>9} {'retr s':>8}")
        print("=" * 96)
        for arm in arms:
            v = out_arms[arm]
            print(f"{arm:<14} {v['accuracy_pct']:>6.1f} {v['fact_recall_pct']:>8.1f}% "
                  f"{v['refusals']:>6} {v['hallucinations']:>7} "
                  f"{str(v['median_llm_prompt_tokens_est']):>8} "
                  f"{v['internal_reranker_pairs']:>9} {v['reranker_invocations']:>9} "
                  f"{v['mean_retrieval_s']:>8.3f}")
        print(f"\nraw corpus (est tokens): {raw_corpus_tokens}")
        print(f"adaptive skipped reranking on "
              f"{report['adaptive_rule']['reranks_skipped']}/{n} questions")
        print(f"cloud necessity (this fixed set only): {low_conf}/{n}")
        print(f"\nwrote {out}")
        return 0
    finally:
        store.close()
        emb.close()


if __name__ == "__main__":
    raise SystemExit(main())

