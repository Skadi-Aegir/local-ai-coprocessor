"""
think_ab.py - Phase 3: thinking vs non-thinking mode on local-smart.

Runs the same hard tasks with enable_thinking true and false and compares:
  quality proxy (task-specific checks), latency, token usage, and reasoning leakage.

The question this answers: should reasoning be a per-task option, or the default?
The specification says reasoning must NOT be auto-enabled for simple tasks, so this
measures the actual cost and benefit rather than assuming either way.

Usage:
  python think_ab.py --base-url http://127.0.0.1:8112 --out results/think_ab.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))
from common import Client, extract_json, thinking_leak  # noqa: E402

# One hard task that genuinely benefits from deliberation, and one that does not.
HARD = (
    "A paginated API takes a 1-based `page` parameter. `page_bounds(page, per_page, total)` "
    "computes `start = page * per_page`. For page=1, per_page=10, total=25 the endpoint "
    "returns items 11-20 instead of 1-10. Explain precisely why, state the minimal fix, and "
    "name one other function that could be affected by the same mistake."
)
EASY = (
    "Extract the fields as JSON with keys id, status, priority: "
    "'task T-77 is pending with priority 40'."
)
EASY_GOLD = {"id": "T-77", "status": "pending", "priority": 40}


def score_hard(text: str) -> dict:
    low = (text or "").lower()
    return {
        "explains_1based": bool(re.search(r"1[- ]?based|one[- ]based|从 1 开始|1 起始", low)),
        "mentions_offbyone": bool(re.search(r"off[- ]?by[- ]?one|差一|偏移", low)),
        "gives_fix": bool(re.search(r"page - 1|page-1|\(page - 1\)|减去 1|减 1", low)),
        "names_other_function": "total_pages" in low,
    }


def run(c: Client, prompt: str, thinking: bool, max_tokens: int) -> dict:
    t0 = time.time()
    r = c.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens,
               temperature=0.6 if thinking else 0.0, enable_thinking=thinking)
    return {
        "thinking": thinking,
        "ok": r.ok,
        "error": r.error,
        "wall_s": round(time.time() - t0, 2),
        "gen_tps": round(r.gen_tps, 2),
        "completion_tokens": r.completion_tokens,
        "ttft_s": round(r.ttft_s, 3) if r.ttft_s else None,
        "leak": thinking_leak(r.text),
        "reasoning_chars": len(r.text or ""),
        "text": r.text or "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8112")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    c = Client(args.base_url, timeout=1800)
    c.health()

    out: dict = {"base_url": args.base_url, "tasks": {}}

    # ---------------- HARD task
    print("=== HARD task: root-cause explanation ===", flush=True)
    hard = {}
    for think in (False, True):
        res = run(c, HARD, think, 1200)
        res["score"] = score_hard(res["text"])
        res["score_total"] = sum(1 for v in res["score"].values() if v)
        res.pop("text", None)
        hard["think" if think else "nothink"] = res
        print(f"  thinking={think}: {res['wall_s']}s  {res['completion_tokens']} tok  "
              f"score {res['score_total']}/4  leak={res['leak']}", flush=True)
    out["tasks"]["hard"] = hard

    # ---------------- EASY task
    print("\n=== EASY task: structured extraction ===", flush=True)
    easy = {}
    for think in (False, True):
        res = run(c, EASY, think, 1200)
        parsed = extract_json(res["text"])
        res["json_ok"] = parsed is not None
        res["schema_ok"] = bool(isinstance(parsed, dict) and all(
            str(parsed.get(k, "")).lower() == str(v).lower() for k, v in EASY_GOLD.items()))
        res.pop("text", None)
        easy["think" if think else "nothink"] = res
        print(f"  thinking={think}: {res['wall_s']}s  {res['completion_tokens']} tok  "
              f"json={res['json_ok']} schema={res['schema_ok']}  leak={res['leak']}", flush=True)
    out["tasks"]["easy"] = easy

    # ---------------- derived
    h0, h1 = hard["nothink"], hard["think"]
    e0, e1 = easy["nothink"], easy["think"]
    out["summary"] = {
        "hard_latency_ratio": round(h1["wall_s"] / h0["wall_s"], 2) if h0["wall_s"] else None,
        "hard_token_ratio": round(h1["completion_tokens"] / h0["completion_tokens"], 2)
                            if h0["completion_tokens"] else None,
        "hard_score_gain": h1["score_total"] - h0["score_total"],
        "easy_latency_ratio": round(e1["wall_s"] / e0["wall_s"], 2) if e0["wall_s"] else None,
        "easy_token_ratio": round(e1["completion_tokens"] / e0["completion_tokens"], 2)
                            if e0["completion_tokens"] else None,
        "easy_schema_nothink": e0["schema_ok"],
        "easy_schema_think": e1["schema_ok"],
        "any_leak": bool(h0["leak"] or h1["leak"] or e0["leak"] or e1["leak"]),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("THINK / NON-THINK SUMMARY (local-smart)")
    print("=" * 62)
    for k, v in out["summary"].items():
        print(f"  {k:26} {v}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
