"""
cloud_context_prep.py - Phase 4: prepare escalation bundles WITHOUT calling the cloud.

The escalation policy says hard work goes to DeepSeek / Claude / Codex. Those calls cost
money and are not made here. What this module does is build the EXACT payload that would
be sent: the retrieved sources, packed and labelled, plus the routing metadata and token
accounting that decide whether retrieval is even worth doing.

Three escalation shapes are modelled, matching docs/escalation-policy.md:
  local_only   the answer is already in the local context; no escalation needed
  local_review local-fast answers and local-smart reviews (under-escalation check)
  cloud        a bundle prepared for an upstream model

Nothing is transmitted. Every bundle is written to disk so a human can inspect exactly
what a cloud model would have received, which is the point: the experiment measures the
SIZE and QUALITY of the context, not the cloud model's answer.

Usage:
  python cloud_context_prep.py --out ..\\..\\benchmarks\\retrieval\\results\\cloud_context_prep.json
  python cloud_context_prep.py --top-k 5 --budget-tokens 6000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))

from benchmark_data import ALL_SETS  # noqa: E402

EMBED_URL = "http://127.0.0.1:8113"

# Prompts a real escalation would use. They are part of the bundle so the payload is
# complete and reviewable.
CLOUD_SYSTEM = (
    "You are a senior engineer reviewing a task that a local model could not complete "
    "confidently. You are given the only project sources available. Cite file paths and "
    "line ranges for every claim. State explicitly when the sources do not answer the "
    "question."
)


class Tokenizer:
    def __init__(self):
        self.tok = None
        try:
            from tokenizers import Tokenizer as HFTok
            p = (Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B\tokenizer.json")
            if p.exists():
                self.tok = HFTok.from_file(str(p))
        except Exception:
            self.tok = None

    def count(self, text: str) -> int:
        if self.tok is not None:
            try:
                return len(self.tok.encode(text, add_special_tokens=False).ids)
            except Exception:
                pass
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return int(cjk * 0.6 + (len(text) - cjk) * 0.25)


def http_json(url, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def retrieve(query: str, top_k: int) -> dict:
    return http_json(f"{EMBED_URL}/v1/retrieve",
                     {"query": query, "project": "local-ai", "top_k": top_k, "pack": True})


def classify_escalation(results: list[dict], expected_file: str | None) -> str:
    """
    Decide the escalation shape from retrieval evidence alone.

    The rule mirrors the project's policy: if a source from the file a human would name
    is retrieved with a high score, local handling is plausible; if the best score is
    weak or the labelled source is absent, the task carries uncertainty and should be
    escalated.
    """
    if not results:
        return "cloud"
    top = results[0]["score"]
    has_expected = any(r["source_path"] == expected_file for r in results) if expected_file else False
    if top >= 0.60 and has_expected:
        return "local_only"
    if top >= 0.45 or has_expected:
        return "local_review"
    return "cloud"


def build_bundle(question: str, query_type: str, context: str, results: list[dict],
                 tk: Tokenizer, budget_tokens: int) -> dict:
    """Assemble the payload a cloud call would send, and measure it."""
    ctx_tokens = tk.count(context)
    user_msg = f"SOURCES:\n{context}\n\nTASK: {question}"
    prompt_tokens = tk.count(CLOUD_SYSTEM) + tk.count(user_msg)

    bundle = {
        "model_target": "deepseek-chat | claude-sonnet | codex",   # placeholder, not called
        "messages": [
            {"role": "system", "content": CLOUD_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "meta": {
            "question": question,
            "query_type": query_type,
            "sources": len(results),
            "source_paths": [r["source_path"] for r in results],
            "top_score": results[0]["score"] if results else None,
            "context_tokens": ctx_tokens,
            "prompt_tokens": prompt_tokens,
            "within_budget": prompt_tokens <= budget_tokens,
        },
    }
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--budget-tokens", type=int, default=6000)
    ap.add_argument("--dump-dir", default=None,
                    help="write each bundle as a .json file for inspection")
    args = ap.parse_args()

    tk = Tokenizer()
    print(f"tokenizer: {'real' if tk.tok else 'heuristic'}")
    print(f"budget   : {args.budget_tokens} prompt tokens per escalation\n")

    rows, bundles = [], []
    counts = {"local_only": 0, "local_review": 0, "cloud": 0}

    for setname, queries in ALL_SETS.items():
        for q in queries:
            try:
                r = retrieve(q["q"], args.top_k)
            except Exception as e:  # noqa: BLE001
                rows.append({"set": setname, "query": q["q"], "error": str(e)})
                continue
            context = r.get("context", "")
            results = r.get("results", [])
            route = classify_escalation(results, q["file"])
            counts[route] += 1
            bundle = build_bundle(q["q"], q["type"], context, results, tk, args.budget_tokens)
            rows.append({
                "set": setname, "type": q["type"], "query": q["q"],
                "expected": q["file"], "route": route, **bundle["meta"],
            })
            bundles.append(bundle)

    ok = [r for r in rows if "error" not in r]
    n = len(ok) or 1
    summary = {
        "questions": len(ok),
        "top_k": args.top_k,
        "budget_tokens": args.budget_tokens,
        "routing": counts,
        "routing_pct": {k: round(100 * v / n, 1) for k, v in counts.items()},
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in ok) / n, 1),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in ok), default=0),
        "within_budget": sum(1 for r in ok if r["within_budget"]),
        "mean_context_tokens": round(sum(r["context_tokens"] for r in ok) / n, 1),
        "mean_sources": round(sum(r["sources"] for r in ok) / n, 2),
        "rows": rows,
    }

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # one worked example, fully inspectable, so the payload shape is verifiable
    if bundles:
        example = max(bundles, key=lambda b: b["meta"]["prompt_tokens"])
        summary["largest_bundle_example"] = {
            "question": example["meta"]["question"],
            "source_paths": example["meta"]["source_paths"],
            "prompt_tokens": example["meta"]["prompt_tokens"],
            "user_message_head": example["messages"][1]["content"][:1200],
        }

    if args.dump_dir:
        dd = Path(args.dump_dir)
        dd.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(bundles, 1):
            (dd / f"bundle_{i:02d}.json").write_text(
                json.dumps(b, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["dumped_bundles"] = len(bundles)

    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print("=" * 68)
    print("CLOUD CONTEXT PREPARATION (no cloud call was made)")
    print("=" * 68)
    print(f"  questions          : {summary['questions']}")
    print(f"  routing            : {summary['routing']}")
    print(f"  routing %          : {summary['routing_pct']}")
    print(f"  mean prompt tokens : {summary['mean_prompt_tokens']}")
    print(f"  max prompt tokens  : {summary['max_prompt_tokens']} "
          f"(budget {args.budget_tokens})")
    print(f"  within budget      : {summary['within_budget']}/{summary['questions']}")
    print(f"  mean sources       : {summary['mean_sources']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
