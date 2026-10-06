"""
concurrency_test.py - Phase 3: quantify GPU contention between local-fast and local-smart.

Compares, with BOTH models loaded (so VRAM is identical in every condition):
  1. baseline A : 4B generating, 9B idle
  2. baseline B : 9B generating, 4B idle
  3. concurrent : both generating at the same time

Reports each condition's generation tok/s and the derived contention penalty.

Why long generations: at ~60 tokens the wall time is dominated by fixed per-request
overhead (prompt processing, graph setup), which made an earlier short-prompt
comparison suggest a 35% penalty that was mostly measurement artefact. 800-token
generations make decode throughput the dominant term.

VRAM is sampled throughout by scripts/vram_watch.ps1, written separately.

Usage:
  python concurrency_test.py --base-fast http://127.0.0.1:8111 --base-smart http://127.0.0.1:8112
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))
from common import Client  # noqa: E402

PROMPT_SHORT = "Explain database indexing in detail, covering B-trees, hash indexes, and when each is appropriate."
PROMPT_LONG = ("Write a comprehensive technical guide to designing a rate limiter for a "
               "distributed API. Cover token bucket, leaky bucket, fixed and sliding windows, "
               "the tradeoffs of each, how to handle clock skew across nodes, what to store "
               "in Redis versus locally, and how to degrade gracefully under overload. "
               "Be thorough and use concrete numbers where useful.")


def run_one(client: Client, label: str, max_tokens: int, out: dict, key: str) -> None:
    r = client.chat([{"role": "user", "content": PROMPT_LONG}], max_tokens=max_tokens,
                    temperature=0.0, enable_thinking=False)
    out[key] = {
        "label": label, "ok": r.ok, "error": r.error,
        "wall_s": round(r.total_s, 2),
        "gen_tps": round(r.gen_tps, 2),
        "prompt_tps": round(r.prompt_tps, 1),
        "completion_tokens": r.completion_tokens,
        "ttft_s": round(r.ttft_s, 3) if r.ttft_s else None,
    }
    print(f"    {label}: {out[key]['gen_tps']} t/s, {out[key]['completion_tokens']} tok, "
          f"{out[key]['wall_s']}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-fast", default="http://127.0.0.1:8111")
    ap.add_argument("--base-smart", default="http://127.0.0.1:8112")
    ap.add_argument("--tokens", type=int, default=800)
    ap.add_argument("--out", default=str(HERE / "results" / "concurrency.json"))
    args = ap.parse_args()

    cf = Client(args.base_fast, timeout=1200)
    cs = Client(args.base_smart, timeout=1200)
    res: dict = {"tokens_requested": args.tokens, "conditions": {}}

    # ---- condition 1: 4B alone (9B resident but idle)
    print("\n[1] fast ACTIVE, smart IDLE (both models resident)", flush=True)
    run_one(cf, "fourb_alone", args.tokens, res["conditions"], "c1_fast_alone")

    # ---- condition 2: 9B alone (4B resident but idle)
    print("\n[2] smart ACTIVE, fast IDLE (both models resident)", flush=True)
    run_one(cs, "nineb_alone", args.tokens, res["conditions"], "c2_smart_alone")

    # ---- condition 3: both at once
    print("\n[3] BOTH generating concurrently", flush=True)
    t0 = time.time()
    th1 = threading.Thread(target=run_one, args=(cf, "fourb_concurrent", args.tokens,
                                                 res["conditions"], "c3_fast_concurrent"))
    th2 = threading.Thread(target=run_one, args=(cs, "nineb_concurrent", args.tokens,
                                                 res["conditions"], "c3_smart_concurrent"))
    th1.start(); th2.start()
    th1.join(); th2.join()
    res["concurrent_wall_s"] = round(time.time() - t0, 2)

    # ---- derived contention
    def pct(base, conc):
        if not base or not conc:
            return None
        return round(100 * (conc - base) / base, 1)

    a = res["conditions"].get("c1_fast_alone", {}).get("gen_tps")
    b = res["conditions"].get("c3_fast_concurrent", {}).get("gen_tps")
    c = res["conditions"].get("c2_smart_alone", {}).get("gen_tps")
    d = res["conditions"].get("c3_smart_concurrent", {}).get("gen_tps")
    res["fast_penalty_pct"] = pct(a, b)
    res["smart_penalty_pct"] = pct(c, d)
    res["fast_retained_pct"] = round(100 * b / a, 1) if a and b else None
    res["smart_retained_pct"] = round(100 * d / c, 1) if c and d else None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("CONCURRENCY RESULT (both models resident throughout)")
    print("=" * 66)
    print(f"  fast  alone      : {a} t/s")
    print(f"  fast  concurrent : {b} t/s   -> {res['fast_penalty_pct']}% "
          f"(retains {res['fast_retained_pct']}%)")
    print(f"  smart alone      : {c} t/s")
    print(f"  smart concurrent : {d} t/s   -> {res['smart_penalty_pct']}% "
          f"(retains {res['smart_retained_pct']}%)")
    print(f"  concurrent wall  : {res['concurrent_wall_s']}s for both")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
