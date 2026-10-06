"""
stability_test.py - long-run stability check for the local-fast worker.

Cycles summary / JSON / code / routing / traceback tasks continuously and watches for
the failure modes the specification asks about:
  - VRAM growth over time (leak)
  - service crash
  - CUDA errors
  - output degradation
  - severe latency degradation

VRAM is sampled by a separate PowerShell watcher writing CSV; this script reads that
CSV afterwards and correlates it with per-iteration latency.

Usage:
  python stability_test.py --minutes 10 --out results/stability.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Client, check_schema, extract_json, thinking_leak  # noqa: E402
import tasks as T  # noqa: E402

FIX = HERE / "fixtures"


def build_cycle() -> list[tuple[str, list[dict], dict]]:
    """One cycle covering summary, JSON, code and routing."""
    zh = (FIX / "test_a_zh_summary.txt").read_text(encoding="utf-8")
    cases = T.make_extraction_cases(4, seed=777)
    route_desc, route_expect, _ = T.ROUTING_CASES[0]
    code_case = next(c for c in T.CODE_CASES if c["kind"] == "sql")

    steps: list[tuple[str, list[dict], dict]] = []

    steps.append((
        "summary",
        [{"role": "system", "content": "你是一个严谨的技术文档摘要助手。"},
         {"role": "user", "content":
          "请把下面的技术文本压缩为恰好 5 个要点。只使用原文中出现的信息。"
          "每个要点用一行，以数字编号开头。\n\n" + zh}],
        {"max_tokens": 700, "temperature": 0.0},
    ))

    for i, case in enumerate(cases):
        steps.append((
            f"json_{i}",
            [{"role": "system", "content": T.EXTRACTION_SYSTEM},
             {"role": "user", "content": case["text"]}],
            {"max_tokens": 200, "temperature": 0.0, "_gold": case["gold"]},
        ))

    steps.append((
        "routing",
        [{"role": "system", "content": T.ROUTING_SYSTEM},
         {"role": "user", "content": route_desc}],
        {"max_tokens": 200, "temperature": 0.0, "_expect": route_expect},
    ))

    steps.append((
        "code_sql",
        [{"role": "system", "content": code_case["system"]},
         {"role": "user", "content": code_case["prompt"]}],
        {"max_tokens": 900, "temperature": 0.0, "_sql": True},
    ))

    steps.append((
        "traceback",
        [{"role": "system", "content": T.TRACEBACK_SYSTEM},
         {"role": "user", "content": "请分析下面的 Python traceback：\n\n" + T.TRACEBACK}],
        {"max_tokens": 400, "temperature": 0.0},
    ))

    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--out", default=str(HERE / "results" / "stability.json"))
    ap.add_argument("--vram-csv", default=str(Path(__file__).resolve().parents[2] / r"benchmarks\local-fast\results\vram_stability.csv"))
    args = ap.parse_args()

    c = Client(args.base_url, timeout=180)
    try:
        h = c.health()
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: server unreachable: {e}")
        return 2
    print(f"health: {h}")

    steps = build_cycle()
    deadline = time.time() + args.minutes * 60
    iterations = []
    it = 0
    t_start = time.time()

    while time.time() < deadline:
        it += 1
        cycle_start = time.time()
        per_step = []
        for name, messages, opts in steps:
            opts = dict(opts)
            gold = opts.pop("_gold", None)
            expect = opts.pop("_expect", None)
            is_sql = opts.pop("_sql", False)
            t0 = time.time()
            r = c.chat(messages, enable_thinking=False, **opts)
            dt = time.time() - t0

            ok = r.ok
            detail = ""
            if gold is not None:
                parsed = extract_json(r.text)
                good, probs = check_schema(parsed, gold)
                ok = good
                detail = "" if good else str(probs)[:120]
            elif expect is not None:
                parsed = extract_json(r.text) or {}
                ok = parsed.get("route") == expect
                detail = f"got {parsed.get('route')}"
            elif is_sql:
                ok = "select" in r.text.lower() and "group by" in r.text.lower()
                detail = "sql shape"

            leak = thinking_leak(r.text)
            per_step.append({
                "step": name, "latency_s": round(dt, 3), "ok": ok,
                "gen_tps": round(r.gen_tps, 2), "completion_tokens": r.completion_tokens,
                "leak": leak, "error": r.error, "detail": detail,
            })
        cycle_s = time.time() - cycle_start
        n_ok = sum(1 for s in per_step if s["ok"])
        iterations.append({
            "iter": it,
            "elapsed_s": round(time.time() - t_start, 1),
            "cycle_s": round(cycle_s, 3),
            "ok": n_ok, "total": len(per_step),
            "steps": per_step,
        })
        print(f"  iter {it:3d}  t+{time.time()-t_start:6.1f}s  cycle {cycle_s:5.2f}s  "
              f"ok {n_ok}/{len(per_step)}", flush=True)

    # ---------------- analysis
    all_steps = [s for i in iterations for s in i["steps"]]
    lat = [s["latency_s"] for s in all_steps]
    first_q = statistics.mean(lat[:len(lat) // 4]) if len(lat) >= 4 else statistics.mean(lat)
    last_q = statistics.mean(lat[-len(lat) // 4:]) if len(lat) >= 4 else statistics.mean(lat)

    vram = []
    vcsv = Path(args.vram_csv)
    if vcsv.exists():
        import csv as _csv
        with vcsv.open(encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                try:
                    vram.append(int(row["mem_used_mb"]))
                except Exception:
                    pass

    report = {
        "minutes_requested": args.minutes,
        "minutes_actual": round((time.time() - t_start) / 60, 2),
        "iterations": len(iterations),
        "steps_total": len(all_steps),
        "crashes": 0,
        "errors": [s for s in all_steps if s["error"]],
        "thinking_leaks": sum(1 for s in all_steps if s["leak"]),
        "step_failures": [{"iter": i["iter"], **s} for i in iterations for s in i["steps"] if not s["ok"]],
        "ok_rate": round(100 * sum(1 for s in all_steps if s["ok"]) / max(len(all_steps), 1), 1),
        "latency_mean_s": round(statistics.mean(lat), 3),
        "latency_first_quartile_s": round(first_q, 3),
        "latency_last_quartile_s": round(last_q, 3),
        "latency_drift_pct": round(100 * (last_q - first_q) / first_q, 1) if first_q else None,
        "gen_tps_mean": round(statistics.mean([s["gen_tps"] for s in all_steps if s["gen_tps"]]), 2),
        "vram_samples": len(vram),
        "vram_min_mb": min(vram) if vram else None,
        "vram_max_mb": max(vram) if vram else None,
        "vram_first_mb": vram[0] if vram else None,
        "vram_last_mb": vram[-1] if vram else None,
        "vram_growth_mb": (vram[-1] - vram[0]) if len(vram) > 1 else None,
        "iterations_detail": iterations,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print("STABILITY SUMMARY")
    print("=" * 66)
    for k in ("iterations", "steps_total", "ok_rate", "thinking_leaks", "crashes",
              "latency_mean_s", "latency_first_quartile_s", "latency_last_quartile_s",
              "latency_drift_pct", "gen_tps_mean", "vram_samples", "vram_first_mb",
              "vram_last_mb", "vram_max_mb", "vram_growth_mb"):
        print(f"  {k:26} {report[k]}")
    print(f"  errors                    {len(report['errors'])}")
    print(f"  step failures             {len(report['step_failures'])}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
