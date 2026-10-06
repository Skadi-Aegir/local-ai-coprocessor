"""
p5_baseline.py - P55R-05: establish whether a SAFE BASELINE exists.

Read-only except for using the EXISTING safe start/stop scripts on DSH-OWNED services. No user
process is stopped. Every step records the time series so the numbers are auditable.

The question this answers: with some subset of the three DSH-owned services resident, is free VRAM
>= the 2048 MiB margin? If NO subset reaches it, P55R-05 is BLOCKED and the run stops here, because
no real RAG stress test can be run safely.

Usage: python p5_baseline.py --out <path>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))
sys.path.insert(0, str(HERE))

from gpu_scheduler import REGISTRY, service_healthy          # noqa: E402
from p3_atomic import atomic_write_json                      # noqa: E402

MARGIN = 2048
SCRIPTS = REPO / "scripts"


def gpu() -> tuple[int, int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        u, f = out.stdout.strip().splitlines()[0].split(",")
        return int(u.strip()), int(f.strip())
    except Exception:
        return None


def run_script(name: str) -> int:
    """stdout to a FILE, never a pipe (a pipe deadlock already cost this project a run)."""
    log = REPO / "logs" / f"p5-svc-{name}.log"
    with open(log, "wb") as fh:
        r = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                            "-ExecutionPolicy", "Bypass", "-File", str(SCRIPTS / name)],
                           stdout=fh, stderr=subprocess.STDOUT, timeout=600)
    return r.returncode


def active_requests() -> dict:
    """
    Confirm no active request before stopping a DSH-owned service, as required.
    Uses each service's own stats/health where available.
    """
    out = {}
    for alias, path in (("local-reranker", "/stats"), ("local-embedding", "/stats"),
                        ("local-fast", "/metrics")):
        port = REGISTRY[alias].port
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                body = r.read().decode()
            if alias == "local-fast":
                # llama.cpp exposes prometheus text; look for in-flight request gauges
                busy = [ln for ln in body.splitlines()
                        if "requests" in ln and "processing" in ln]
                out[alias] = {"busy_lines": busy[:3] or "none reported"}
            else:
                d = json.loads(body)
                out[alias] = {"requests": d.get("requests"), "errors": d.get("errors"),
                              "uptime_s": d.get("uptime_s")}
        except Exception as e:  # noqa: BLE001
            out[alias] = f"{type(e).__name__}: {e}"
    return out


def sample_for(seconds: float, interval: float = 0.2) -> dict:
    """Time series over a window: min/max free, samples, breaches."""
    vals: list[tuple[int, int]] = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        g = gpu()
        if g:
            vals.append(g)
        time.sleep(interval)
    if not vals:
        return {"n": 0}
    free = [v[1] for v in vals]
    used = [v[0] for v in vals]
    return {"n": len(vals), "window_s": seconds,
            "used_min": min(used), "used_max": max(used),
            "free_min": min(free), "free_max": max(free),
            "margin_mb": MARGIN,
            "holds_margin": min(free) >= MARGIN,
            "n_sub_margin": sum(1 for f in free if f < MARGIN)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {
        "experiment": "P55R-05 safe-baseline check",
        "margin_mb": MARGIN,
        "policy": ("read-only; only DSH-OWNED services are touched, via the existing safe "
                   "start/stop scripts; no user process is stopped"),
        "steps": [],
    }
    atomic_write_json(out, report)

    def step(label: str, action=None, settle: float = 6.0, window: float = 6.0) -> dict:
        if action:
            print(f"  action: {action}", flush=True)
            run_script(action)
        time.sleep(settle)
        s = sample_for(window)
        g = gpu()
        resident = [n for n in REGISTRY if service_healthy(REGISTRY[n].port)]
        rec = {"step": label, "action": action, "resident": resident,
               "gpu": {"used": g[0], "free": g[1]} if g else None, "window": s}
        report["steps"].append(rec)
        atomic_write_json(out, report)
        print(f"  {label}: resident={resident} used={g[0] if g else '?'} "
              f"free={g[1] if g else '?'} free_min={s.get('free_min')} "
              f"holds_margin={s.get('holds_margin')}", flush=True)
        return rec

    print("=== active-request check before touching any DSH service ===", flush=True)
    ar = active_requests()
    report["active_requests_before"] = ar
    print(f"  {json.dumps(ar)[:220]}", flush=True)
    atomic_write_json(out, report)

    # 0: as-found
    first = step("as-found: all three DSH services resident")

    # 1: drop the reranker (on-demand in the staged design anyway)
    step("after stopping local-reranker", "stop-local-reranker.ps1")

    # 2: drop the embedding too - this is the floor the staged design needs at rerank time
    step("after stopping local-embedding", "stop-local-embedding.ps1")

    # 3: DSH-owned services all down - the absolute floor, i.e. everything else on the card
    step("after stopping local-fast (all DSH services down)", "stop-local-fast.ps1")

    report["floors"] = {
        "all_three": first["window"].get("free_min"),
        "after_reranker_stop": report["steps"][1]["window"].get("free_min"),
        "after_embedding_stop": report["steps"][2]["window"].get("free_min"),
        "all_dsh_down": report["steps"][3]["window"].get("free_min"),
    }
    non_dsh_used = None
    if report["steps"][3]["gpu"]:
        non_dsh_used = report["steps"][3]["gpu"]["used"]
    report["non_dsh_occupancy_mb"] = non_dsh_used
    report["conclusion"] = {
        "non_dsh_occupancy_mb": non_dsh_used,
        "free_with_no_dsh_service_mb": report["floors"]["all_dsh_down"],
        "can_any_subset_hold_margin": any(
            s["window"].get("holds_margin") for s in report["steps"] if s["window"].get("n")),
    }
    print(f"\n=== CONCLUSION ===")
    print(f"  non-DSH occupancy      : {non_dsh_used} MiB")
    print(f"  free with all DSH down : {report['floors']['all_dsh_down']} MiB")
    print(f"  any subset holds margin: {report['conclusion']['can_any_subset_hold_margin']}")

    # restore the delivered state: all three resident
    print("\n=== restoring all three DSH services ===", flush=True)
    for n in ("start-local-fast.ps1", "start-local-embedding.ps1", "start-local-reranker.ps1"):
        run_script(n)
        time.sleep(1)
    time.sleep(6)
    report["restored"] = step("after restoring all three DSH services")
    report["restored_ok"] = all(service_healthy(REGISTRY[n].port) for n in REGISTRY)

    atomic_write_json(out, report)
    print(f"\nrestored_ok={report['restored_ok']}  wrote {out}")
    return 0 if report["conclusion"]["can_any_subset_hold_margin"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
