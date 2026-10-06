"""
b4b_switch_cost.py - the minimal-residency alternative, measured.

The B4 live validation showed the default configuration refuses ALL normal work: with
local-fast, local-embedding and local-reranker resident, only ~152 MiB is usable after the
2048 MiB margin, which is below even a background job's 600 MiB need.

This measures whether the MINIMAL-RESIDENCY alternative actually works, and what it costs:
  keep local-fast resident
  unload local-embedding and local-reranker
  then admit the same work that was refused before
  then measure the cost of bringing them back

Nothing here lowers the margin. Both the refusal and the recovery are measured as they are.

Usage:
  python b4b_switch_cost.py --out ../results/<run-id>/b4b_switch_cost.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))

from gpu_scheduler import REGISTRY, GpuScheduler, service_healthy  # noqa: E402


def vram() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        used, free = out.stdout.strip().splitlines()[0].split(",")
        return int(used.strip()), int(free.strip())
    except Exception:
        return None


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"margin_mb": 2048, "note": "margin never lowered; recovery measured"}
    s = GpuScheduler(strategy="unload", verbose=True)

    print("=== BEFORE: three models resident ===")
    resident = s.running()
    u0, f0 = vram()
    plan_before = s.plan("background")
    print(f"  resident={resident}  vram used={u0} free={f0}")
    print(f"  background admission: {plan_before.allowed}  ({plan_before.reason[:80]})")
    report["before"] = {"resident": resident, "vram_used": u0, "vram_free": f0,
                        "background_allowed": plan_before.allowed,
                        "reason": plan_before.reason,
                        "usable_mb": f0 - 2048}

    print("\n=== UNLOAD the two non-essential models ===")
    unloads = {}
    for name in ("local-reranker", "local-embedding"):
        t0 = time.time()
        ok = s.unload(name)
        dt = time.time() - t0
        unloads[name] = {"ok": ok, "seconds": round(dt, 2)}
        print(f"  {name}: ok={ok} in {dt:.2f}s")
    time.sleep(2)
    u1, f1 = vram()
    print(f"  vram used={u1} free={f1}  (reclaimed {f1 - f0} MiB)")
    report["unload"] = {"per_service": unloads, "vram_used": u1, "vram_free": f1,
                        "reclaimed_mb": f1 - f0}

    print("\n=== AFTER: can the previously refused work now be admitted? ===")
    admitted = {}
    for cls in ("background", "interactive", "exclusive"):
        d = s.plan(cls)
        admitted[cls] = {"allowed": d.allowed, "required_mb": d.required_mb,
                         "usable_mb": d.free_mb - d.margin_mb, "reason": d.reason[:90]}
        print(f"  {cls:<12} allowed={d.allowed}  required={d.required_mb} "
              f"usable={d.free_mb - d.margin_mb}")

    # ------------------------------------------------------------------
    # CORRECT ORDERING (this is the P55R-02 fix)
    #
    # The previous version unloaded the reranker, then sent a request to 8114, and only
    # RECOVERED the service afterwards - so the request was aimed at a service it had just
    # stopped, and the "on-demand load" path was never actually exercised. plan().allowed was
    # also treated as if it proved the workload could run.
    #
    # The order that actually validates on-demand loading is:
    #   reserve -> unload if the budget requires it -> load the dependency ->
    #   health check -> REAL request -> release
    # ------------------------------------------------------------------
    print("\n=== ON-DEMAND LOAD PATH: reserve -> unload -> load -> health -> request -> release ===")
    ondemand: dict = {}

    # 1. a request declares what it depends on
    d_req = s.acquire("background", timeout_s=30, requires=["local-reranker"],
                      request_id="ondemand-rerank-1")
    ondemand["reserved"] = {"allowed": d_req.allowed, "required_mb": d_req.required_mb,
                            "reason": d_req.reason[:100],
                            "reservation_id": d_req.reservation_id}
    print(f"  1. reserve        : allowed={d_req.allowed} required={d_req.required_mb} "
          f"id={d_req.reservation_id}")

    if d_req.allowed:
        try:
            # 2. load the DECLARED dependency and wait for health
            t_load = time.time()
            healthy = s.start("local-reranker", wait_s=180)
            load_s = time.time() - t_load
            health_ok = service_healthy(REGISTRY["local-reranker"].port)
            ondemand["load"] = {"healthy": healthy, "health_checked": health_ok,
                                "load_seconds": round(load_s, 2)}
            print(f"  2. load + health  : healthy={healthy} direct_probe={health_ok} "
                  f"in {load_s:.2f}s")

            # 3. the REAL request, and its success is judged by a response, not by a plan
            t_req = time.time()
            try:
                body = post("http://127.0.0.1:8114/v1/rerank",
                            {"query": "what context size does local-fast use",
                             "documents": ["context: 8192 in config/local-fast.yaml",
                                           "unrelated text about pooling"],
                             "top_n": 2, "instruction_type": "general", "use_cache": False})
                first_inference_s = time.time() - t_req
                results = body.get("results") or []
                ondemand["request"] = {
                    "completed": bool(results), "n_results": len(results),
                    "first_inference_seconds": round(first_inference_s, 3),
                    "top_score": results[0].get("relevance_score") if results else None,
                }
                print(f"  3. REAL request   : completed={bool(results)} "
                      f"results={len(results)} first_inference={first_inference_s:.3f}s")
            except Exception as e:  # noqa: BLE001
                ondemand["request"] = {"completed": False,
                                       "error": f"{type(e).__name__}: {e}"}
                print(f"  3. REAL request   : FAILED {type(e).__name__}: {e}")

            # 4. health-passed time and first-inference time are reported SEPARATELY: a
            #    service that answers /health is not necessarily ready to serve inference
            ondemand["health_vs_inference"] = {
                "health_seconds": ondemand["load"]["load_seconds"],
                "first_inference_seconds": ondemand["request"].get(
                    "first_inference_seconds"),
                "note": "health passing is NOT the same as first successful inference; both "
                        "are reported and neither is used as a proxy for the other",
            }
        finally:
            s.release("background", d_req.reservation_id)
            print(f"  4. release        : reserved now {s.reserved_mb()} MiB")

    report["on_demand_path"] = ondemand

    print("\n=== RECOVERY: bring both services back, health-checked ===")
    recover = {}
    for name in ("local-embedding", "local-reranker"):
        t0 = time.time()
        ok = s.start(name, wait_s=120)
        dt = time.time() - t0
        recover[name] = {"healthy": ok, "seconds": round(dt, 2),
                         "health_checked": service_healthy(REGISTRY[name].port)}
        print(f"  {name}: healthy={ok} in {dt:.2f}s "
              f"(direct health check {recover[name]['health_checked']})")
    time.sleep(2)
    u2, f2 = vram()
    print(f"  vram used={u2} free={f2}")
    report["recovery"] = {"per_service": recover, "vram_used": u2, "vram_free": f2,
                          "resident_after": s.running()}

    report["switch_cost"] = {
        "unload_seconds": {k: v["seconds"] for k, v in unloads.items()},
        "reload_seconds": {k: v["seconds"] for k, v in recover.items()},
        "total_round_trip_s": round(sum(v["seconds"] for v in unloads.values())
                                    + sum(v["seconds"] for v in recover.values()), 2),
    }
    print(f"\n  switch cost: unload {report['switch_cost']['unload_seconds']} "
          f"reload {report['switch_cost']['reload_seconds']} "
          f"round trip {report['switch_cost']['total_round_trip_s']}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
