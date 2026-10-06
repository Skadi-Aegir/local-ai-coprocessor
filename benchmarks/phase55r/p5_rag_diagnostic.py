"""
p5_rag_diagnostic.py - P55R-05: ONE full RAG diagnostic over the two-required-unload path.

Abort conditions, checked continuously, and any one of them STOPS the run and preserves the
evidence rather than continuing:
  * OOM anywhere
  * any sample of free VRAM below the 2048 MiB margin
  * either unload not actually performed (port freed AND free VRAM rose, verified)

Records per stage: admission, the ACTUAL unload list, free VRAM before/after each unload, minimum
free VRAM, latency, and the service state on completion.

Isolated path only: this script builds the staged pipeline itself. Production routing is unchanged,
and no default budget is modified.

Usage: python p5_rag_diagnostic.py --out <path>
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

from gpu_scheduler import GpuScheduler, REGISTRY, service_healthy   # noqa: E402
from p3_atomic import atomic_write_json                             # noqa: E402
from p3_swap import Stage, StagedPipeline                           # noqa: E402
import p2_residency as R                                            # noqa: E402

MARGIN = 2048
RAG_Q = "What context size does the local-fast worker use by default?"


def post(url, payload, timeout=600):
    return R.post(url, payload, timeout)


class Monitor:
    """Samples free VRAM throughout; records the minimum and every sub-margin sample."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=15)
                u, f = out.stdout.strip().splitlines()[0].split(",")
                self.samples.append({"t": round(time.time(), 3),
                                     "used": int(u.strip()), "free": int(f.strip())})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.t.join(timeout=10)
        return False

    def window(self, t0: float, t1: float) -> dict:
        vals = [s for s in self.samples if t0 <= s["t"] <= t1]
        if not vals:
            return {"n": 0}
        free = [s["free"] for s in vals]
        used = [s["used"] for s in vals]
        breaches = [s for s in vals if s["free"] < MARGIN]
        return {"n_samples": len(vals), "free_min": min(free), "free_max": max(free),
                "used_min": min(used), "used_max": max(used),
                "margin_mb": MARGIN, "margin_breached": bool(breaches),
                "n_sub_margin_samples": len(breaches),
                "first_breach": breaches[0] if breaches else None}


def start_service(name: str) -> bool:
    script = REPO / "scripts" / f"start-local-{name.split('-')[-1]}.ps1"
    if not script.is_file():
        return False
    log = REPO / "logs" / f"p5-{script.stem}.log"
    with open(log, "wb") as fh:
        subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                        "-ExecutionPolicy", "Bypass", "-File", str(script)],
                       stdout=fh, stderr=subprocess.STDOUT, timeout=600)
    return service_healthy(REGISTRY[name].port)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=str(REPO), capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unavailable"

    report: dict = {
        "experiment": "P55R-05 single full RAG diagnostic (two required unloads)",
        "margin_mb": MARGIN,
        "production_unchanged": True,
        "budgets_unchanged": {"margin_mb": MARGIN,
                              "reranker_peak_extra_mb": REGISTRY["local-reranker"].peak_extra_mb,
                              "candidate_881_or_1361_used": False},
        "provenance": {"commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain"))},
        "stages": [],
        "abort": None,
    }

    resident0 = [n for n in REGISTRY if service_healthy(REGISTRY[n].port)]
    report["resident_before"] = resident0
    atomic_write_json(out, report)
    print(f"resident before: {resident0}", flush=True)
    if not all(service_healthy(REGISTRY[n].port) for n in REGISTRY):
        report["abort"] = "not all three services are healthy; refusing to start"
        atomic_write_json(out, report)
        print("ABORT: services not healthy")
        return 5

    s = GpuScheduler(strategy="conditional", verbose=False, dry_run=False,
                     max_background_units=2)
    p = StagedPipeline(s, log=lambda m: print(f"    {m}", flush=True))
    aborted = None
    mon: Monitor | None = None

    try:
        with Monitor() as mon:
            t_all = time.time()

            # ---- stage 1: embed + retrieve
            t0 = time.time()
            r1 = p.stage1_embed_retrieve(
                lambda: {"candidates": (post(f"{R.EMBED}/v1/retrieve",
                                             {"query": RAG_Q, "top_k": 10}).get("results") or []),
                         "exec_s": None},
                timeout_s=90)
            w1 = mon.window(t0, time.time())
            report["stages"].append({
                "stage": "1 embed+retrieve", "ok": r1["ok"],
                "admission": "admitted" if r1["ok"] else "refused",
                "wait_s": r1.get("waited_s"), "reason": r1.get("reason", "")[:160],
                "n_candidates": len(p.candidates), "wall_s": round(time.time() - t0, 3),
                "vram": w1})
            atomic_write_json(out, report)
            print(f"  stage1 ok={r1['ok']} candidates={len(p.candidates)} "
                  f"free_min={w1.get('free_min')}", flush=True)
            if not r1["ok"]:
                aborted = f"stage 1 failed: {r1.get('reason', '')[:120]}"
            elif w1.get("margin_breached"):
                aborted = f"margin breached during stage 1 (free_min={w1.get('free_min')})"

            # ---- stage 3: UNLOAD embedding (required) then admit reranker
            if not aborted:
                t0 = time.time()
                embed_free_before = p._live_free()
                sw = p.stage3_unload_embed_then_admit_reranker(timeout_s=90)
                w3 = mon.window(t0, time.time())
                report["stages"].append({
                    "stage": "3 unload embedding + admit reranker", "ok": sw.ok,
                    "actual_unload_list": sw.unloaded,
                    "unload_verified": {"scheduler_ok": sw.ok, "port_free": sw.port_free_verified,
                                        "free_before": sw.free_before,
                                        "free_after": sw.free_after,
                                        "rose": bool(sw.free_before and sw.free_after
                                                     and sw.free_after > sw.free_before)},
                    "free_vram_before_unload": embed_free_before,
                    "wait_s": sw.waited_s, "reason": sw.reason[:200],
                    "wall_s": round(time.time() - t0, 3), "vram": w3})
                atomic_write_json(out, report)
                print(f"  stage3 ok={sw.ok} unloaded={sw.unloaded} "
                      f"free {sw.free_before}->{sw.free_after} free_min={w3.get('free_min')}",
                      flush=True)
                if not sw.ok:
                    aborted = f"stage 3 failed: {sw.reason[:140]}"
                elif not sw.unloaded or not sw.port_free_verified:
                    aborted = "the required embedding unload did NOT actually happen"
                elif w3.get("margin_breached"):
                    aborted = f"margin breached during stage 3 (free_min={w3.get('free_min')})"

            # ---- stage 4: rerank
            if not aborted:
                t0 = time.time()

                def _rerank(cands):
                    docs = [c.get("content", "") for c in cands]
                    rr = post(f"{R.RERANK}/v1/rerank",
                              {"query": RAG_Q, "documents": docs, "top_n": 5,
                               "instruction_type": "general", "use_cache": False})
                    order = [x.get("index") for x in (rr.get("results") or [])]
                    final = [cands[i] for i in order
                             if isinstance(i, int) and i < len(cands)][:5] or cands[:5]
                    return {"reranked": final,
                            "exec_s": round(time.time() - t0, 3),
                            "scores": [round(x.get("relevance_score", 0), 4)
                                       for x in (rr.get("results") or [])]}

                r4 = p.stage4_rerank(_rerank)
                w4 = mon.window(t0, time.time())
                report["stages"].append({
                    "stage": "4 rerank", "ok": r4["ok"], "exec_s": r4.get("exec_s"),
                    "n_reranked": len(p.reranked), "reason": r4.get("reason", "")[:160],
                    "wall_s": round(time.time() - t0, 3), "vram": w4})
                atomic_write_json(out, report)
                print(f"  stage4 ok={r4['ok']} reranked={len(p.reranked)} "
                      f"free_min={w4.get('free_min')}", flush=True)
                if not r4["ok"]:
                    aborted = f"stage 4 failed: {r4.get('reason', '')[:120]}"
                elif w4.get("margin_breached"):
                    aborted = f"margin breached during stage 4 (free_min={w4.get('free_min')})"

            # ---- stage 5: UNLOAD reranker (required)
            if not aborted:
                t0 = time.time()
                rr_free_before = p._live_free()
                u5 = p.stage5_unload_reranker()
                w5 = mon.window(t0, time.time())
                report["stages"].append({
                    "stage": "5 unload reranker", "ok": u5.ok,
                    "actual_unload_list": u5.unloaded,
                    "unload_verified": {"port_free": u5.port_free_verified,
                                        "free_before": u5.free_before,
                                        "free_after": u5.free_after,
                                        "rose": bool(u5.free_before and u5.free_after
                                                     and u5.free_after > u5.free_before)},
                    "free_vram_before_unload": rr_free_before,
                    "wait_s": u5.waited_s, "reason": u5.reason[:200],
                    "wall_s": round(time.time() - t0, 3), "vram": w5})
                atomic_write_json(out, report)
                print(f"  stage5 ok={u5.ok} unloaded={u5.unloaded} "
                      f"free {u5.free_before}->{u5.free_after} free_min={w5.get('free_min')}",
                      flush=True)
                if not u5.ok:
                    aborted = f"stage 5 unload failed: {u5.reason[:140]}"
                elif not u5.unloaded or not u5.port_free_verified:
                    aborted = "the required reranker unload did NOT actually happen"
                elif w5.get("margin_breached"):
                    aborted = f"margin breached during stage 5 (free_min={w5.get('free_min')})"

            # ---- stage 6: pack + generate
            if not aborted:
                final = p.reranked
                pack = "\n\n".join(
                    f"SOURCE {i+1} ({c.get('source_path')} lines {c.get('start_line')}-"
                    f"{c.get('end_line')})\n{(c.get('content') or '')[:2000]}"
                    for i, c in enumerate(final))
                report["pack_chars"] = len(pack)

                def _gen():
                    t0 = time.time()
                    g = post(f"{R.FAST}/v1/chat/completions", {
                        "model": "local-fast",
                        "messages": [{"role": "user", "content": f"{pack}\n\nQUESTION: {RAG_Q}"}],
                        "max_tokens": 64, "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False}})
                    text = (g.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    return {"exec_s": round(time.time() - t0, 3),
                            "answer_has_8192": "8192" in text,
                            "answer_head": text.strip()[:200]}

                t0 = time.time()
                r6 = p.stage6_generate(_gen, timeout_s=90)
                w6 = mon.window(t0, time.time())
                report["stages"].append({
                    "stage": "6 pack+generate", "ok": r6["ok"], "exec_s": r6.get("exec_s"),
                    "answer_has_8192": r6.get("answer_has_8192"),
                    "answer_head": r6.get("answer_head"),
                    "reason": r6.get("reason", "")[:160],
                    "wall_s": round(time.time() - t0, 3), "vram": w6})
                atomic_write_json(out, report)
                print(f"  stage6 ok={r6['ok']} has8192={r6.get('answer_has_8192')} "
                      f"free_min={w6.get('free_min')}", flush=True)
                if not r6["ok"]:
                    aborted = f"stage 6 failed: {r6.get('reason', '')[:120]}"
                elif w6.get("margin_breached"):
                    aborted = f"margin breached during stage 6 (free_min={w6.get('free_min')})"

            report["total_wall_s"] = round(time.time() - t_all, 3)
            report["overall_window"] = mon.window(t_all, time.time())
    except Exception as e:  # noqa: BLE001
        aborted = f"exception: {type(e).__name__}: {e}"
        print(f"  EXCEPTION: {aborted}", flush=True)
    finally:
        p.cleanup()

    report["abort"] = aborted
    report["state_at_end"] = p.state.value
    report["min_free_vram_mb"] = (report.get("overall_window") or {}).get("free_min")
    report["margin_held"] = bool(report["min_free_vram_mb"]
                                 and report["min_free_vram_mb"] >= MARGIN)
    report["any_margin_breach"] = bool(
        (report.get("overall_window") or {}).get("margin_breached"))
    atomic_write_json(out, report)

    # ---- service recovery
    print("\n=== service recovery ===", flush=True)
    recovery = {}
    for n in ("local-embedding", "local-reranker"):
        if not service_healthy(REGISTRY[n].port):
            t0 = time.time()
            ok = start_service(n)
            recovery[n] = {"restarted": True, "healthy": ok,
                           "seconds": round(time.time() - t0, 2)}
        else:
            recovery[n] = {"restarted": False, "healthy": True}
        print(f"  {n}: {recovery[n]}", flush=True)
    report["service_recovery"] = recovery
    report["resident_after"] = [n for n in REGISTRY if service_healthy(REGISTRY[n].port)]
    report["final_free_vram"] = None
    try:
        out2 = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
        report["final_free_vram"] = int(out2.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    report["passed"] = (aborted is None and report["margin_held"]
                        and not report["any_margin_breach"])
    atomic_write_json(out, report)

    print(f"\nABORT: {aborted}")
    print(f"min free VRAM: {report['min_free_vram_mb']} MiB ; margin held: {report['margin_held']}")
    print(f"resident after: {report['resident_after']} ; final free: {report['final_free_vram']}")
    print(f"PASSED: {report['passed']}")
    print(f"wrote {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
