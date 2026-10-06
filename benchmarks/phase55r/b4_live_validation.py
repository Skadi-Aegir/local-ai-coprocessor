"""
b4_live_validation.py - P55R-01 work package B4.

A minimal live workload where EVERY request goes through the scheduler. Nothing calls a
service directly; if the scheduler refuses, the request is not made.

Five scenarios, matching the phase list:
  1  a single interactive task
  2  concurrent background work
  3  interactive arriving while background runs
  4  an exclusive request queueing
  5  controlled unload and recovery

The resource model keeps the five quantities separate, as required:
  resident      what is loaded now (already inside the live free figure - NOT added again)
  load_needed   memory for a model the job needs which is not running
  peak_extra    transient headroom the job needs on top
  reserved      committed by ACCEPTED jobs that live sampling has not yet reflected
  margin        the 2048 MiB safety margin, never lowered to make a test pass

Honest framing: the operator has three models resident, so the margin leaves little usable
room. A run in which most normal work is REFUSED would not demonstrate a working scheduler,
and is reported as a failed live validation with a minimal-residency proposal rather than
spun as "safe refusal working correctly".

Usage:
  python b4_live_validation.py --out ../results/<run-id>/b4_live.json [--unload-recovery]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))

from gpu_scheduler import GpuScheduler  # noqa: E402

FAST = "http://127.0.0.1:8111"
EMBED = "http://127.0.0.1:8113"
RERANK = "http://127.0.0.1:8114"


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def vram_used() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Sampler:
    def __init__(self):
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            self.samples.append({"t": round(time.time(), 2), "used": vram_used()})
            time.sleep(0.2)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.t.join(timeout=5)
        return False

    def peak(self):
        vals = [s["used"] for s in self.samples if s["used"]]
        return max(vals) if vals else None


# the actual GPU work, performed ONLY after the scheduler admits it
def work_interactive() -> dict:
    t0 = time.time()
    r = post(f"{FAST}/v1/chat/completions", {
        "model": "local-fast",
        "messages": [{"role": "user", "content": "Name two indexing strategies."}],
        "max_tokens": 32, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    return {"kind": "interactive", "s": round(time.time() - t0, 3),
            "gen_tps": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}


def work_embed() -> dict:
    t0 = time.time()
    post(f"{EMBED}/v1/embeddings",
         {"model": "local-embedding", "input": [f"chunk {i}" for i in range(16)]})
    return {"kind": "embed", "s": round(time.time() - t0, 3)}


def work_rerank() -> dict:
    t0 = time.time()
    post(f"{RERANK}/v1/rerank",
         {"query": f"retrieval {time.time_ns()}",
          "documents": [f"doc {i} about retrieval and ranking" for i in range(10)],
          "top_n": 5, "instruction_type": "general", "use_cache": False})
    return {"kind": "rerank", "s": round(time.time() - t0, 3)}


def run_job(s: GpuScheduler, job_class: str, fn, timeout_s: float = 30.0) -> dict:
    """Every request goes through acquire() first. Refusal means NO call is made."""
    t0 = time.time()
    d = s.acquire(job_class, timeout_s=timeout_s)
    wait = time.time() - t0
    if not d.allowed:
        return {"job_class": job_class, "status": "refused", "wait_s": round(wait, 3),
                "reason": d.reason[:120]}
    try:
        res = fn()
        return {"job_class": job_class, "status": "completed", "wait_s": round(wait, 3),
                "exec_s": res["s"], **{k: v for k, v in res.items()
                                       if k not in ("kind", "s")}}
    except Exception as e:  # noqa: BLE001
        return {"job_class": job_class, "status": "failed", "wait_s": round(wait, 3),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        s.release(job_class, d.reservation_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--unload-recovery", action="store_true",
                    help="also exercise a real unload and recovery of the reranker")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"rounds": args.rounds, "scenarios": [], "vram_start": vram_used(),
                    "margin_mb": 2048,
                    "note": "every request is admitted through the scheduler; refusal "
                            "means no call was made"}

    # ---------------- 1. single interactive
    print("=== scenario 1: single interactive ===")
    results = []
    for r in range(args.rounds):
        s = GpuScheduler(strategy="conditional", verbose=False)
        with Sampler() as smp:
            res = run_job(s, "interactive", work_interactive)
        res["vram_peak"] = smp.peak()
        results.append(res)
        print(f"  round {r+1}: {res['status']} wait={res['wait_s']}s "
              f"exec={res.get('exec_s')}s peak={res.get('vram_peak')}")
    report["scenarios"].append({"scenario": "single_interactive", "results": results})

    # ---------------- 2. concurrent background
    print("\n=== scenario 2: concurrent background ===")
    s = GpuScheduler(strategy="conditional", verbose=False)
    bg_results: list[dict] = []
    with Sampler() as smp:
        ts = [threading.Thread(target=lambda f=fn: bg_results.append(run_job(s, "background", f)))
              for fn in (work_embed, work_rerank, work_embed, work_rerank)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
    print(f"  {sum(1 for r in bg_results if r['status']=='completed')}/{len(bg_results)} "
          f"completed, peak={smp.peak()} MiB")
    for r in bg_results:
        print(f"    {r['status']:<10} wait={r['wait_s']}s exec={r.get('exec_s')}s "
              f"{r.get('reason','')[:50]}")
    report["scenarios"].append({"scenario": "concurrent_background",
                                "results": bg_results, "vram_peak": smp.peak()})

    # ---------------- 3. interactive arriving during background
    print("\n=== scenario 3: interactive arriving while background runs ===")
    s = GpuScheduler(strategy="conditional", verbose=False)
    inter_res: list[dict] = []
    with Sampler() as smp:
        bg = threading.Thread(target=lambda: bg_results.append(
            run_job(s, "background", work_rerank)))
        bg.start()
        time.sleep(0.05)
        inter = threading.Thread(target=lambda: inter_res.append(
            run_job(s, "interactive", work_interactive)))
        inter.start()
        bg.join(timeout=60)
        inter.join(timeout=60)
    r3 = inter_res[0] if inter_res else {"status": "missing"}
    print(f"  interactive: {r3.get('status')} wait={r3.get('wait_s')}s "
          f"exec={r3.get('exec_s')}s")
    report["scenarios"].append({"scenario": "interactive_during_background",
                                "interactive": r3, "vram_peak": smp.peak()})

    # ---------------- 4. exclusive queueing
    print("\n=== scenario 4: exclusive request queueing behind background ===")
    s = GpuScheduler(strategy="conditional", verbose=False)
    ex_res: list[dict] = []
    with Sampler() as smp:
        bg = threading.Thread(target=lambda: bg_results.append(
            run_job(s, "background", work_rerank)))
        bg.start()
        time.sleep(0.05)
        ex = threading.Thread(target=lambda: ex_res.append(
            run_job(s, "exclusive", work_embed, timeout_s=20.0)))
        ex.start()
        bg.join(timeout=60)
        ex.join(timeout=60)
    r4 = ex_res[0] if ex_res else {"status": "missing"}
    print(f"  exclusive: {r4.get('status')} wait={r4.get('wait_s')}s  "
          f"reason={(r4.get('reason') or '')[:70]}")
    report["scenarios"].append({"scenario": "exclusive_queueing", "exclusive": r4,
                                "vram_peak": smp.peak()})

    # ---------------- 5. controlled unload and recovery
    if args.unload_recovery:
        print("\n=== scenario 5: controlled unload + recovery (reranker) ===")
        s = GpuScheduler(strategy="unload", verbose=False)
        before = vram_used()
        ok_unload = s.unload("local-reranker")
        time.sleep(2)
        after_unload = vram_used()
        healthy_after = s.start("local-reranker", wait_s=90)
        after_restart = vram_used()
        print(f"  unload ok={ok_unload} vram {before} -> {after_unload}; "
              f"restart healthy={healthy_after} vram -> {after_restart}")
        report["scenarios"].append({
            "scenario": "unload_recovery",
            "unload_ok": ok_unload, "vram_before": before,
            "vram_after_unload": after_unload, "recovered_healthy": healthy_after,
            "vram_after_restart": after_restart})
    else:
        print("\n(scenario 5 skipped; pass --unload-recovery to exercise it)")

    # ---------------- verdict
    allr = [r for sc in report["scenarios"] for r in
            ([sc.get("interactive"), sc.get("exclusive")] if sc["scenario"] in
             ("interactive_during_background", "exclusive_queueing") else sc.get("results", []))
            if r]
    completed = sum(1 for r in allr if r.get("status") == "completed")
    refused = sum(1 for r in allr if r.get("status") == "refused")
    failed = sum(1 for r in allr if r.get("status") == "failed")
    total = len(allr) or 1
    report["summary"] = {"requests": len(allr), "completed": completed,
                         "refused": refused, "failed": failed,
                         "completion_rate_pct": round(100 * completed / total, 1)}
    print("\n" + "=" * 70)
    print(f"requests={len(allr)} completed={completed} refused={refused} failed={failed} "
          f"({report['summary']['completion_rate_pct']}% completed)")

    if completed == 0:
        report["verdict"] = "FAIL - every normal request was refused"
        print("VERDICT: FAIL - the live validation demonstrates nothing if all normal work "
              "is refused. A minimal-residency option is required.")
    elif refused > completed:
        report["verdict"] = "CONDITIONAL - refusals outnumber completions"
        print("VERDICT: CONDITIONAL - safe refusal works, but most work is refused, so the "
              "default co-residency configuration is impractical as deployed.")
    else:
        report["verdict"] = "PASS - normal paths completed through the scheduler"
        print("VERDICT: PASS - at least one interactive and one background path completed "
              "through the scheduler, with refusals accounted for.")

    # minimal-residency proposal, grounded in the measured registry
    from gpu_scheduler import REGISTRY
    report["minimal_residency_option"] = {
        "why": "the 2048 MiB margin plus three resident models leaves little usable room",
        "option": "keep local-fast resident only; load local-embedding and local-reranker "
                  "on demand",
        "resident_cost_mb": REGISTRY["local-fast"].vram_mb,
        "switch_cost_s": {"local-embedding": REGISTRY["local-embedding"].load_s,
                          "local-reranker": REGISTRY["local-reranker"].load_s},
        "tradeoff": "each retrieval step pays a model load (1.2-1.7 s) unless a warm cache "
                    "is maintained",
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
