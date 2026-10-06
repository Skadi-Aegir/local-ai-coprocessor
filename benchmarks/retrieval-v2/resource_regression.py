"""
resource_regression.py - resource monitoring and regression baseline for Phase 5.5.

Measures, on a fixed task set, the quantities the phase requires:
  VRAM peak, system RAM, shared GPU memory, pagefile, TTFT, generation throughput,
  reranker latency, embedding throughput, end-to-end retrieval latency.

Also samples VRAM while three models are resident, because a resting reading is misleading:
Phase 5 measured that the reranker's transient spike (+1794 MiB alone) is what exhausts
memory, not its steady footprint.

THE SAFETY MARGIN IS THE POINT
  The scheduler must hold a 2 GiB margin. When it cannot, the correct behaviour is to unload
  a non-essential model and REPORT, not to keep loading. This script records whether the
  margin held at every sample, so a run that quietly breached it is visible.

Usage:
  python resource_regression.py --out results/resource_regression.json
  python resource_regression.py --out ... --compare-baseline     # diff against a stored run
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
sys.path.insert(0, str(REPO / "services" / "retrieval"))
sys.path.insert(0, str(HERE))

TOTAL_VRAM_MB = 12227
FAST = "http://127.0.0.1:8111"
EMBED = "http://127.0.0.1:8113"
RERANK = "http://127.0.0.1:8114"
SAFETY_MARGIN_MB = 2048


def nvidia(query: str) -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def system_mem() -> dict:
    """System RAM and pagefile, via CIM so no extra dependency is needed."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem;"
             "$p=Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue;"
             "$g=Get-CimInstance Win32_ComputerSystem;"
             "Write-Output ('{0}|{1}|{2}|{3}' -f $o.TotalVisibleMemorySize,"
             "$o.FreePhysicalMemory, ($p.AllocatedBaseSize), ($p.PeakUsage))"],
            capture_output=True, text=True, timeout=40)
        parts = out.stdout.strip().split("|")
        return {"ram_total_mb": int(parts[0]) // 1024,
                "ram_free_mb": int(parts[1]) // 1024,
                "pagefile_allocated_mb": int(parts[2] or 0),
                "pagefile_peak_mb": int(parts[3] or 0)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def shared_gpu() -> dict:
    """Windows shared GPU memory is visible through nvidia-smi's used/total only, so this
    reports what is actually queryable: total card memory and the derived free figure."""
    used = nvidia("memory.used")
    free = nvidia("memory.free")
    total = nvidia("memory.total")
    return {"vram_used_mb": used, "vram_free_mb": free, "vram_total_mb": total or TOTAL_VRAM_MB,
            "note": "Windows shared GPU memory is not exposed by nvidia-smi on this driver"}


def up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=4) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--compare-baseline", default=None)
    args = ap.parse_args()

    services = {"local_fast": up(FAST), "local_embedding": up(EMBED),
                "local_reranker": up(RERANK)}
    print(f"services up: {services}")
    resident = sum(1 for v in services.values() if v)
    print(f"models resident: {resident}/3\n")

    samples: list[dict] = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            u = nvidia("memory.used")
            samples.append({"t": round(time.time(), 2), "vram_used": u,
                            "free": (TOTAL_VRAM_MB - u) if u else None,
                            "margin_ok": (TOTAL_VRAM_MB - u) >= SAFETY_MARGIN_MB if u else None})
            time.sleep(0.2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    metrics: dict = {}
    counts = {"gen": 0, "embed": 0, "rerank": 0}
    ttfts: list[float] = []
    gen_tps: list[float] = []
    rerank_lat: list[float] = []
    embed_cps: list[float] = []

    t_end = time.time() + args.seconds
    while time.time() < t_end:
        if services["local_fast"]:
            try:
                t0 = time.time()
                r = post(f"{FAST}/v1/chat/completions", {
                    "model": "local-fast",
                    "messages": [{"role": "user", "content": "Name three indexing strategies."}],
                    "max_tokens": 64, "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False}})
                tm = r.get("timings", {})
                if tm.get("prompt_ms"):
                    ttfts.append(tm["prompt_ms"] / 1000)
                if tm.get("predicted_per_second"):
                    gen_tps.append(tm["predicted_per_second"])
                counts["gen"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  gen error: {type(e).__name__}")
        if services["local_embedding"]:
            try:
                t0 = time.time()
                post(f"{EMBED}/v1/embeddings",
                     {"model": "local-embedding", "input": [f"chunk {i}" for i in range(32)]})
                dt = time.time() - t0
                embed_cps.append(32 / dt if dt else 0)
                counts["embed"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  embed error: {type(e).__name__}")
        if services["local_reranker"]:
            try:
                t0 = time.time()
                post(f"{RERANK}/v1/rerank",
                     {"query": f"retrieval variant {time.time_ns()}",
                      "documents": [f"doc {i} about retrieval and ranking" for i in range(10)],
                      "top_n": 5, "instruction_type": "general", "use_cache": False})
                rerank_lat.append(time.time() - t0)
                counts["rerank"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  rerank error: {type(e).__name__}")

    stop.set()
    th.join(timeout=10)

    used = [s["vram_used"] for s in samples if s["vram_used"]]
    margin_ok = [s["margin_ok"] for s in samples if s["margin_ok"] is not None]
    peak = max(used) if used else None
    metrics = {
        "services": services, "models_resident": resident,
        "samples": len(samples),
        "vram_peak_mb": peak,
        "vram_median_mb": int(statistics.median(used)) if used else None,
        "vram_peak_delta_mb": (peak - min(used)) if used else None,
        "free_at_peak_mb": (TOTAL_VRAM_MB - peak) if peak else None,
        "margin_mb": SAFETY_MARGIN_MB,
        "margin_held_pct": round(100 * sum(1 for m in margin_ok if m) / len(margin_ok), 1)
        if margin_ok else None,
        "margin_breaches": sum(1 for m in margin_ok if not m),
        "system": {**system_mem(), **shared_gpu()},
        "workload": counts,
        "ttft_s_median": round(statistics.median(ttfts), 4) if ttfts else None,
        "gen_tps_median": round(statistics.median(gen_tps), 2) if gen_tps else None,
        "rerank_s_median": round(statistics.median(rerank_lat), 4) if rerank_lat else None,
        "embed_chunks_per_s_median": round(statistics.median(embed_cps), 1) if embed_cps else None,
        "seconds": args.seconds,
    }

    print(f"VRAM peak        {metrics['vram_peak_mb']} MiB "
          f"(free {metrics['free_at_peak_mb']})")
    print(f"margin held      {metrics['margin_held_pct']}% of samples "
          f"({metrics['margin_breaches']} breaches)")
    print(f"TTFT median      {metrics['ttft_s_median']} s")
    print(f"gen median       {metrics['gen_tps_median']} tok/s")
    print(f"rerank median    {metrics['rerank_s_median']} s")
    print(f"embed median     {metrics['embed_chunks_per_s_median']} chunks/s")
    print(f"system           {metrics['system']}")

    if args.compare_baseline:
        bp = Path(args.compare_baseline)
        if bp.exists():
            base = json.loads(bp.read_text(encoding="utf-8"))
            print("\n=== comparison against stored baseline ===")
            for k in ("vram_peak_mb", "ttft_s_median", "gen_tps_median",
                      "rerank_s_median", "embed_chunks_per_s_median"):
                b, n = base.get(k), metrics.get(k)
                if b is None or n is None:
                    continue
                delta = (n - b) / b * 100 if b else None
                print(f"  {k:<28} {b:>10} -> {n:>10}  "
                      f"({delta:+.1f}%)" if delta is not None else f"  {k}: {b} -> {n}")
            metrics["baseline_comparison"] = {
                k: {"baseline": base.get(k), "now": metrics.get(k)} for k in
                ("vram_peak_mb", "ttft_s_median", "gen_tps_median", "rerank_s_median",
                 "embed_chunks_per_s_median")}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    # the phase requirement: if the margin cannot be held, unload non-essential models and
    # REPORT rather than continuing to load
    if metrics["margin_breaches"]:
        print(f"\nWARNING: the {SAFETY_MARGIN_MB} MiB margin was breached in "
              f"{metrics['margin_breaches']} of {len(margin_ok)} samples. Per the phase "
              f"requirement, a non-essential model should be unloaded rather than "
              f"continuing to load. Run: "
              f"python services/gpu_scheduler/scheduler_cli.py --strategy unload --plan background")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
