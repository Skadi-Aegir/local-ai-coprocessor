"""
rag_compare.py - Phase 4: does retrieval actually improve local-fast's answers?

For each question, runs local-fast twice:
  WITHOUT RAG : the bare question (the model must rely on its own weights)
  WITH RAG    : the same question plus the retrieved, pack()-formatted sources

Scoring is mechanical, not judged by the model:
  * each question carries a list of expected fact strings
  * an answer scores a fact when the fact's keywords appear in it
  * hallucination markers are strings that must NOT appear (invented values)
This is deliberately crude but repeatable, and it separates the two failure modes
that matter: MISSING facts (retrieval did not help) and INVENTED facts (retrieval made
the model confidently wrong).

The questions are about THIS project, whose facts live in the corpus and cannot be in
the model's training data, so the no-RAG arm should fail by construction unless the
model happens to guess. That is the point: it isolates retrieval as the variable.

Usage:
  python rag_compare.py --llm-url http://127.0.0.1:8111 --model local-fast --out ...json
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

EMBED_URL = "http://127.0.0.1:8113"

# Questions whose answers exist only in this project's own files.
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
         facts=[["last"], ["lasttoken", "last_token", "last-token"]], bad=[["mean pooling"]]),
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


def http_json(url, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def retrieve_pack(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    r = http_json(f"{EMBED_URL}/v1/retrieve",
                  {"query": query, "project": "local-ai", "top_k": top_k, "pack": True})
    return r.get("context", ""), r.get("results", [])


def ask(llm_url: str, model: str, question: str, context: str | None,
        max_tokens: int = 400) -> dict:
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
    r = http_json(f"{llm_url}/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    txt = r["choices"][0]["message"]["content"]
    return {
        "answer": txt,
        "seconds": round(time.time() - t0, 2),
        "prompt_tokens": r.get("timings", {}).get("prompt_n"),
        "gen_tps": round(r.get("timings", {}).get("predicted_per_second", 0), 1),
    }


def score(answer: str, facts: list[list[str]], bad: list[list[str]]) -> dict:
    low = answer.lower()
    hits = [all(k.lower() in low for k in grp) for grp in facts if grp]
    halluc = [all(k.lower() in low for k in grp) for grp in bad if grp]
    refused = "not in sources" in low or "i do not know" in low or "不知道" in low
    return {
        "fact_hits": sum(1 for h in hits if h),
        "fact_total": len(hits),
        "all_facts": bool(hits) and all(hits),
        "hallucination_flags": sum(1 for h in halluc if h),
        "refused": refused,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-url", default="http://127.0.0.1:8111")
    ap.add_argument("--model", default="local-fast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    print(f"model={args.model}  questions={len(TASKS)}")
    rows = []
    for i, t in enumerate(TASKS, 1):
        ctx, sources = retrieve_pack(t["q"], args.top_k)
        no_rag = ask(args.llm_url, args.model, t["q"], None)
        rag = ask(args.llm_url, args.model, t["q"], ctx)
        s_no = score(no_rag["answer"], t["facts"], t["bad"])
        s_rag = score(rag["answer"], t["facts"], t["bad"])
        rows.append({
            "q": t["q"], "sources": [s["source_path"] for s in sources],
            "no_rag": {**no_rag, **s_no},
            "rag": {**rag, **s_rag},
        })
        print(f"  [{i:2}/{len(TASKS)}] noRAG facts={s_no['fact_hits']}/{s_no['fact_total']} "
              f"hall={s_no['hallucination_flags']}  |  RAG facts={s_rag['fact_hits']}/{s_rag['fact_total']} "
              f"hall={s_rag['hallucination_flags']}", flush=True)

    n = len(rows)
    def agg(key, field):
        return sum(1 for r in rows if r[key][field])

    summary = {
        "model": args.model,
        "questions": n,
        "no_rag_all_facts_correct": agg("no_rag", "all_facts"),
        "rag_all_facts_correct": agg("rag", "all_facts"),
        "no_rag_fact_hits": sum(r["no_rag"]["fact_hits"] for r in rows),
        "rag_fact_hits": sum(r["rag"]["fact_hits"] for r in rows),
        "fact_total": sum(r["no_rag"]["fact_total"] for r in rows),
        "no_rag_hallucinations": agg("no_rag", "hallucination_flags"),
        "rag_hallucinations": agg("rag", "hallucination_flags"),
        "no_rag_refusals": agg("no_rag", "refused"),
        "rag_refusals": agg("rag", "refused"),
        "no_rag_median_prompt_tokens": sorted(r["no_rag"]["prompt_tokens"] or 0 for r in rows)[n // 2],
        "rag_median_prompt_tokens": sorted(r["rag"]["prompt_tokens"] or 0 for r in rows)[n // 2],
        "rows": rows,
    }
    summary["accuracy_no_rag_pct"] = round(100 * summary["no_rag_all_facts_correct"] / n, 1)
    summary["accuracy_rag_pct"] = round(100 * summary["rag_all_facts_correct"] / n, 1)
    summary["fact_recall_no_rag_pct"] = round(100 * summary["no_rag_fact_hits"] / summary["fact_total"], 1)
    summary["fact_recall_rag_pct"] = round(100 * summary["rag_fact_hits"] / summary["fact_total"], 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("RAG IMPACT (local-fast on project-specific facts)")
    print("=" * 66)
    print(f"  fully correct answers : noRAG {summary['no_rag_all_facts_correct']}/{n} "
          f"({summary['accuracy_no_rag_pct']}%)  ->  RAG {summary['rag_all_facts_correct']}/{n} "
          f"({summary['accuracy_rag_pct']}%)")
    print(f"  fact recall           : noRAG {summary['fact_recall_no_rag_pct']}%  ->  "
          f"RAG {summary['fact_recall_rag_pct']}%")
    print(f"  hallucinations        : noRAG {summary['no_rag_hallucinations']}  ->  "
          f"RAG {summary['rag_hallucinations']}")
    print(f"  refusals              : noRAG {summary['no_rag_refusals']}  ->  RAG {summary['rag_refusals']}")
    print(f"  median prompt tokens  : noRAG {summary['no_rag_median_prompt_tokens']}  ->  "
          f"RAG {summary['rag_median_prompt_tokens']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
