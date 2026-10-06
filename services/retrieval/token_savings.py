"""
token_savings.py - Phase 4: how much context does retrieval actually save?

Compares, for a set of questions answerable from this project:

  baseline   the whole relevant corpus pasted in (what an agent does without retrieval)
  retrieved  query -> top-k chunks -> packed context block

and reports token counts at each stage plus the reduction. If a compression step is
configured, a third stage is reported: retrieved -> compressed.

Token counting uses the embedding model's own tokenizer when available, because a
chars/4 rule mis-sizes this project's mixed Chinese/English content badly.

The point is NOT maximum compression. It is substantially less context without losing
the answer, which is why the harness reports the retrieved-token count alongside the
reduction rather than only the ratio.

Usage:
  python token_savings.py --out ..\\..\\benchmarks\\retrieval\\results\\token_savings.json
  python token_savings.py --compress --llm-url http://127.0.0.1:8111 --llm-model local-fast
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
CORPUS = (Path(__file__).resolve().parents[2] / r"vector\embedding\corpus")


class Tokenizer:
    """Counts tokens with the embedding tokenizer if we can load it cheaply."""

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
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f")
        return int(cjk * 0.6 + (len(text) - cjk) * 0.25)


def http_json(url: str, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def whole_corpus_text() -> str:
    """Baseline: every file a manual reader would consult."""
    parts = []
    for p in sorted(CORPUS.rglob("*")):
        if p.is_file():
            try:
                parts.append(f"===== {p.relative_to(CORPUS).as_posix()} =====\n"
                             + p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n\n".join(parts)


def retrieve(query: str, top_k: int = 5, pack: bool = True) -> dict:
    return http_json(f"{EMBED_URL}/v1/retrieve",
                     {"query": query, "project": "local-ai", "top_k": top_k, "pack": pack})


def compress(text: str, llm_url: str, model: str, max_tokens: int = 700) -> tuple[str, dict]:
    r = http_json(f"{llm_url}/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content":
             "Compress the given sources into a compact brief for another engineer. "
             "Keep every fact, number, file path and identifier. Remove only redundancy "
             "and prose. Do not add anything."},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    return r["choices"][0]["message"]["content"], r.get("timings", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8111")
    ap.add_argument("--llm-model", default="local-fast")
    args = ap.parse_args()

    tk = Tokenizer()
    print(f"tokenizer: {'real (tokenizer.json)' if tk.tok else 'heuristic fallback'}")

    corpus_text = whole_corpus_text()
    corpus_files, corpus_tokens = 0, tk.count(corpus_text)
    for p in CORPUS.rglob("*"):
        if p.is_file():
            corpus_files += 1
    print(f"baseline corpus: {corpus_files} files, {corpus_tokens} tokens")

    rows = []
    for setname, queries in ALL_SETS.items():
        for q in queries:
            try:
                r = retrieve(q["q"], top_k=args.top_k, pack=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"set": setname, "query": q["q"], "error": str(e)})
                continue
            packed = r.get("context", "")
            ret_tokens = tk.count(packed)
            # raw chunk content without the packaging labels
            raw_tokens = tk.count("\n\n".join(x["content"] for x in r["results"]))
            hit = any(x["source_path"] == q["file"] for x in r["results"])
            row = {
                "set": setname, "type": q["type"], "query": q["q"],
                "expected": q["file"],
                "baseline_tokens": corpus_tokens,
                "retrieved_raw_tokens": raw_tokens,
                "retrieved_packed_tokens": ret_tokens,
                "results": len(r["results"]),
                "correct_file_retrieved": hit,
                "reduction_pct": round(100 * (1 - ret_tokens / corpus_tokens), 2) if corpus_tokens else None,
            }
            if args.compress:
                try:
                    comp, tim = compress(packed, args.llm_url, args.llm_model)
                    row["compressed_tokens"] = tk.count(comp)
                    row["compression_ratio"] = round(tk.count(comp) / ret_tokens, 3) if ret_tokens else None
                    row["compressed_total_reduction_pct"] = round(
                        100 * (1 - tk.count(comp) / corpus_tokens), 2) if corpus_tokens else None
                    row["compress_gen_tps"] = round(tim.get("predicted_per_second", 0), 1)
                except Exception as e:  # noqa: BLE001
                    row["compress_error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            print(f"  [{setname}/{q['type']:13}] {ret_tokens:6} tok  "
                  f"{row['reduction_pct']:6.2f}% saved  hit={hit}", flush=True)

    ok = [r for r in rows if "error" not in r]
    import statistics
    out = {
        "baseline_corpus_files": corpus_files,
        "baseline_corpus_tokens": corpus_tokens,
        "top_k": args.top_k,
        "queries": len(ok),
        "mean_retrieved_tokens": round(statistics.mean(r["retrieved_packed_tokens"] for r in ok), 1) if ok else None,
        "median_retrieved_tokens": round(statistics.median(r["retrieved_packed_tokens"] for r in ok), 1) if ok else None,
        "mean_reduction_pct": round(statistics.mean(r["reduction_pct"] for r in ok), 2) if ok else None,
        "retrieval_hit_rate": round(100 * sum(1 for r in ok if r["correct_file_retrieved"]) / len(ok), 1) if ok else None,
        "rows": rows,
    }
    if args.compress:
        comp = [r for r in ok if "compressed_tokens" in r]
        if comp:
            out["mean_compressed_tokens"] = round(statistics.mean(r["compressed_tokens"] for r in comp), 1)
            out["mean_compression_ratio"] = round(statistics.mean(r["compression_ratio"] for r in comp), 3)
            out["mean_total_reduction_pct"] = round(
                statistics.mean(r["compressed_total_reduction_pct"] for r in comp), 2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("TOKEN SAVINGS")
    print("=" * 62)
    print(f"  baseline (whole corpus)  : {out['baseline_corpus_tokens']} tokens / {out['baseline_corpus_files']} files")
    print(f"  retrieved (top {args.top_k}, packed): {out['mean_retrieved_tokens']} tokens mean")
    if "mean_compressed_tokens" in out:
        print(f"  compressed               : {out['mean_compressed_tokens']} tokens mean")
        print(f"  original -> retrieved -> final: {out['baseline_corpus_tokens']} -> "
              f"{out['mean_retrieved_tokens']} -> {out['mean_compressed_tokens']}")
        print(f"  total reduction          : {out['mean_total_reduction_pct']}%")
    print(f"  reduction (retrieved)    : {out['mean_reduction_pct']}%")
    print(f"  correct file retrieved   : {out['retrieval_hit_rate']}%")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
