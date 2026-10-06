"""
p3_candidate_budget.py - P55R-03: test a CANDIDATE reranker transient budget, isolated.

DEFAULT POLICY IS NOT MODIFIED. The candidate lives in an isolated registry copy created for
this experiment only; `gpu_scheduler.REGISTRY` and the production default stay at 1800 MiB.

HOW THE CANDIDATE IS DERIVED - stated explicitly, as required, and NOT simply copied from the
observed peak:

    observed peak (this run, unique documents, 12 cycles)      881 MiB
    x sampling allowance 1.4
        the sampler ran at 0.02 s, so a short allocation spike can fall entirely between two
        samples; 1.4x is an explicit allowance for that blind window, not a measurement
    + fixed measurement margin                                 128 MiB
        covers nvidia-smi resolution and the per-call allocator growth seen across cycles
    = CANDIDATE                                                 1360 MiB

  which leaves 440 MiB of headroom under the declared 1800 MiB. The candidate is THEREFORE NOT
  an increase in risk relative to the declared value - it is still 440 MiB above what was ever
  observed plus the allowances. The declared 1800 MiB is retained as the default this round.

MARGIN GATE: 2048 MiB, unchanged. A round passes only if all five items complete, there is no
OOM, and the minimum free VRAM stays at or above the margin. One round first; two more only if
the first passes.

Usage:
  python p3_candidate_budget.py --out results/<run>/candidate_budget.json [--rounds N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))
sys.path.insert(0, str(HERE))

import gpu_scheduler as G                                   # noqa: E402
from gpu_scheduler import GpuScheduler, REGISTRY            # noqa: E402
import p2_residency as R                                    # noqa: E402

MARGIN = 2048
OBSERVED_PEAK = 881
SAMPLING_ALLOWANCE = 1.4
FIXED_MARGIN = 128
CANDIDATE = int(round(OBSERVED_PEAK * SAMPLING_ALLOWANCE)) + FIXED_MARGIN


def candidate_registry() -> dict:
    """An ISOLATED registry: a copy with only the reranker transient changed."""
    reg = {}
    for n, s in REGISTRY.items():
        if n == "local-reranker":
            reg[n] = G.ServiceSpec(s.name, s.port, s.vram_mb, CANDIDATE, s.essential,
                                   s.start_script, s.stop_script, s.load_s)
        else:
            reg[n] = G.ServiceSpec(s.name, s.port, s.vram_mb, s.peak_extra_mb, s.essential,
                                   s.start_script, s.stop_script, s.load_s)
    return reg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)

    reg = candidate_registry()
    report: dict = {
        "margin_mb": MARGIN,
        "default_policy_unchanged": {
            "registry_peak_extra_mb": REGISTRY["local-reranker"].peak_extra_mb,
            "note": "gpu_scheduler.REGISTRY is NOT modified; the candidate is an isolated copy",
        },
        "candidate_derivation": {
            "observed_peak_mb": OBSERVED_PEAK,
            "sampling_allowance_factor": SAMPLING_ALLOWANCE,
            "sampling_allowance_reason": ("sampler interval was 0.02 s, so a short allocation "
                                          "spike can fall between two samples"),
            "fixed_measurement_margin_mb": FIXED_MARGIN,
            "candidate_mb": CANDIDATE,
            "headroom_under_declared_mb": REGISTRY["local-reranker"].peak_extra_mb - CANDIDATE,
            "observed_plus_allowance_mb": int(round(OBSERVED_PEAK * SAMPLING_ALLOWANCE)),
        },
        "reflects_prior_rejection": ("the 262 MiB hot-sample figure is NOT used anywhere in this "
                                     "derivation; it was rejected as too thin a basis"),
        "rounds": [],
    }
    print(f"candidate = {OBSERVED_PEAK} x {SAMPLING_ALLOWANCE} + {FIXED_MARGIN} = {CANDIDATE} MiB")
    print(f"declared stays {REGISTRY['local-reranker'].peak_extra_mb} MiB")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    globals()["_scheduler_registry"] = reg

    for rnd in range(args.rounds):
        print(f"\n=== round {rnd+1} ===", flush=True)
        s = GpuScheduler(strategy="conditional", verbose=False, dry_run=False,
                         registry=reg, max_background_units=2)
        # the workload functions use the module-level REGISTRY for health/loading, so point the
        # whole process at the isolated copy for the duration of this experiment
        saved = G.REGISTRY
        G.REGISTRY = reg
        R.REGISTRY = reg
        try:
            results = []
            with R.Sampler(interval=0.1) as smp:
                t_round = time.time()
                results.append(R.run_through_scheduler(
                    s, "interactive", f"r{rnd}-gen", ["local-fast"], R.w_interactive))
                results.append(R.run_through_scheduler(
                    s, "background", f"r{rnd}-embed", ["local-embedding"], R.w_embed_batch))
                results.append(R.run_through_scheduler(
                    s, "background", f"r{rnd}-rag",
                    ["local-embedding", "local-reranker", "local-fast"], R.w_full_rag))
                bg, it = [], []
                tg = threading.Thread(target=lambda: bg.append(R.run_through_scheduler(
                    s, "background", f"r{rnd}-bg2", ["local-embedding"], R.w_embed_batch)))
                tg.start()
                time.sleep(0.05)
                ti = threading.Thread(target=lambda: it.append(R.run_through_scheduler(
                    s, "interactive", f"r{rnd}-int2", ["local-fast"], R.w_interactive)))
                ti.start()
                tg.join(timeout=R.OUTER_TIMEOUT)
                ti.join(timeout=R.OUTER_TIMEOUT)
                results.extend(bg or [{"request_id": f"r{rnd}-bg2", "status": "hung"}])
                results.extend(it or [{"request_id": f"r{rnd}-int2", "status": "hung"}])
                t_round_end = time.time()
            win = smp.between(t_round, t_round_end)
        finally:
            G.REGISTRY = saved
            R.REGISTRY = saved

        done = sum(1 for r in results if r["status"] == "completed")
        oom = any("out of memory" in str(r.get("error", "")).lower() for r in results)
        min_free = win.get("free_min")
        round_rec = {
            "round": rnd + 1,
            "requests": len(results), "completed": done,
            "refused": sum(1 for r in results if r["status"] == "refused"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "oom": oom,
            "vram_free_min_mb": min_free,
            "margin_mb": MARGIN,
            "margin_held": (min_free is not None and min_free >= MARGIN),
            "vram_window": win,
            "results": results,
            "round_s": round(t_round_end - t_round, 2),
        }
        round_rec["passed"] = (done == len(results)) and not oom and round_rec["margin_held"]
        report["rounds"].append(round_rec)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"  completed={done}/{len(results)} refused={round_rec['refused']} "
              f"failed={round_rec['failed']} oom={oom} free_min={min_free} "
              f"margin_held={round_rec['margin_held']}", flush=True)
        for r in results:
            print(f"    {r['request_id']:<12} {r['status']:<10} "
                  f"queue={r.get('queue_s')} exec={r.get('exec_s')} "
                  f"{(r.get('reason') or r.get('error') or '')[:90]}", flush=True)

        if rnd == 0 and not round_rec["passed"]:
            report["stopped"] = {"after_round": 1,
                                 "reason": "first round did not pass; not repeating",
                                 "completed": done, "of": len(results),
                                 "oom": oom, "margin_held": round_rec["margin_held"]}
            print("  ROUND FAILED - stopping, not repeating", flush=True)
            break

    passed = sum(1 for r in report["rounds"] if r.get("passed"))
    report["summary"] = {"rounds_run": len(report["rounds"]), "rounds_passed": passed,
                         "verdict": ("PASS" if passed == len(report["rounds"]) and passed > 0
                                     else "FAIL")}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {report['summary']['verdict']} "
          f"({passed}/{len(report['rounds'])} rounds passed)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
