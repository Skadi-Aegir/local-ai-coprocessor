"""
p3_live_staged.py - P55R-04: the five-item workload over the STAGED path, ISOLATED.

Production routing is NOT changed. This script constructs the staged pipeline itself and drives it;
nothing in services/ or the default scheduler policy is modified. The 881 / 1361 MiB figures are NOT
written into any default budget - the registry used here is the DEFAULT one.

Margin gate, all three must hold for a round to pass:
  * 5 of 5 items complete
  * no OOM
  * minimum free VRAM across the whole round >= 2048 MiB

Per item: admission, completion, wait and execution latency, and the round's VRAM window.
Per round: the full record, written atomically so a crash preserves the last complete JSON.

Usage:
  python p3_live_staged.py --out <path> [--rounds N]
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

from gpu_scheduler import GpuScheduler, REGISTRY, service_healthy  # noqa: E402
from p3_atomic import atomic_write_json                           # noqa: E402
from p3_swap import Stage, StagedPipeline                         # noqa: E402
import p2_residency as R                                          # noqa: E402

MARGIN = 2048
OUTER = 900.0


def post(url, payload, timeout=600):
    return R.post(url, payload, timeout)


def vram():
    return R.vram()


class WideSampler:
    """Continuous sampler; reports the whole round's window and every sub-margin sample."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            v = vram()
            if v:
                self.samples.append({"t": round(time.time(), 3), "used": v[0], "free": v[1]})
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
        used = [s["used"] for s in vals]
        free = [s["free"] for s in vals]
        gaps = [round(vals[i + 1]["t"] - vals[i]["t"], 3) for i in range(len(vals) - 1)]
        breaches = [s for s in vals if s["free"] < MARGIN]
        return {
            "n_samples": len(vals),
            "interval_mean_s": round(sum(gaps) / len(gaps), 3) if gaps else None,
            "used_min": min(used), "used_max": max(used),
            "free_min": min(free), "free_max": max(free),
            "margin_mb": MARGIN,
            "margin_breached": bool(breaches),
            "n_sub_margin_samples": len(breaches),
        }


# ---------------------------------------------------------------- the five items
def item1_interactive(s: GpuScheduler, rid: str) -> dict:
    t0 = time.time()
    d = s.acquire("interactive", timeout_s=60, requires=["local-fast"], request_id=rid)
    wait = time.time() - t0
    if not d.allowed:
        return {"request_id": rid, "item": "1 interactive generation", "status": "refused",
                "wait_s": round(wait, 3), "reason": d.reason}
    try:
        out = R.w_interactive()
        return {"request_id": rid, "item": "1 interactive generation",
                "status": "completed" if out.get("ok") else "failed",
                "wait_s": round(wait, 3), "exec_s": out.get("s"), "gen_tps": out.get("gen_tps")}
    except Exception as e:  # noqa: BLE001
        return {"request_id": rid, "item": "1 interactive generation", "status": "failed",
                "wait_s": round(wait, 3), "error": f"{type(e).__name__}: {e}"}
    finally:
        s.release("interactive", d.reservation_id)


def item2_embedding(s: GpuScheduler, rid: str) -> dict:
    t0 = time.time()
    d = s.acquire("background", timeout_s=60, requires=["local-embedding"], request_id=rid)
    wait = time.time() - t0
    if not d.allowed:
        return {"request_id": rid, "item": "2 embedding background task", "status": "refused",
                "wait_s": round(wait, 3), "reason": d.reason}
    try:
        out = R.w_embed_batch()
        return {"request_id": rid, "item": "2 embedding background task",
                "status": "completed" if out.get("ok") else "failed",
                "wait_s": round(wait, 3), "exec_s": out.get("s"), "vectors": out.get("vectors")}
    except Exception as e:  # noqa: BLE001
        return {"request_id": rid, "item": "2 embedding background task", "status": "failed",
                "wait_s": round(wait, 3), "error": f"{type(e).__name__}: {e}"}
    finally:
        s.release("background", d.reservation_id)


def stage1_real():
    """
    Real embed + retrieve. Returns the candidate pool for the CPU handoff.

    This is the retrieval half of full RAG, run under the staging model rather than as one request
    that declares all three services at once.
    """
    t0 = time.time()
    r = post(f"{R.EMBED}/v1/retrieve", {"query": R.RAG_Q, "top_k": 10})
    cands = r.get("results") or []
    return {"candidates": cands, "exec_s": round(time.time() - t0, 3)}


def stage4_real(cands: list[dict]) -> dict:
    """Rerank the CPU-resident candidates using the reranker service."""
    t0 = time.time()
    docs = [c.get("content", "") for c in cands]
    rr = post(f"{R.RERANK}/v1/rerank",
              {"query": R.RAG_Q, "documents": docs, "top_n": 5,
               "instruction_type": "general", "use_cache": False})
    order = [x.get("index") for x in (rr.get("results") or [])]
    final = [cands[i] for i in order if isinstance(i, int) and i < len(cands)][:5] or cands[:5]
    return {"reranked": final, "exec_s": round(time.time() - t0, 3),
            "scores": [round(x.get("relevance_score", 0), 4) for x in (rr.get("results") or [])]}


def make_stage5(pack_holder: dict):
    def stage5():
        t0 = time.time()
        pack = pack_holder.get("pack", "")
        g = post(f"{R.FAST}/v1/chat/completions", {
            "model": "local-fast",
            "messages": [{"role": "user", "content": f"{pack}\n\nQUESTION: {R.RAG_Q}"}],
            "max_tokens": 64, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}})
        text = (g.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"exec_s": round(time.time() - t0, 3), "answer_has_8192": "8192" in text,
                "answer_head": text.strip()[:150]}
    return stage5


def item3_full_rag(s: GpuScheduler, rid: str) -> dict:
    """Full RAG over the STAGED path."""
    rec: dict = {"request_id": rid, "item": "3 full RAG (staged)", "stages": []}
    p = StagedPipeline(s)
    t_all = time.time()
    try:
        t0 = time.time()
        r1 = p.stage1_embed_retrieve(stage1_real, timeout_s=60)
        rec["stages"].append({"stage": "1 embed+retrieve", "ok": r1["ok"],
                              "wait_s": r1.get("waited_s"),
                              "reason": r1.get("reason", "")[:140],
                              "wall_s": round(time.time() - t0, 3)})
        if not r1["ok"]:
            rec.update({"status": "refused", "reason": r1.get("reason", ""),
                        "wait_s": r1.get("waited_s")})
            return rec

        t0 = time.time()
        sw = p.stage3_swap_to_reranker(timeout_s=60)
        rec["stages"].append({"stage": "3 swap (unload embedding, admit reranker)", "ok": sw.ok,
                              "wait_s": sw.waited_s, "unloaded": sw.unloaded,
                              "reason": sw.reason[:140],
                              "wall_s": round(time.time() - t0, 3)})
        if not sw.ok:
            rec.update({"status": "refused", "reason": sw.reason, "wait_s": sw.waited_s})
            return rec

        t0 = time.time()
        r4 = p.stage4_rerank(stage4_real)
        rec["stages"].append({"stage": "4 rerank", "ok": r4["ok"], "exec_s": r4.get("exec_s"),
                              "n_reranked": len(r4.get("reranked") or []),
                              "reason": r4.get("reason", "")[:140],
                              "wall_s": round(time.time() - t0, 3)})
        if not r4["ok"]:
            rec.update({"status": "failed", "error": r4.get("reason", "")})
            return rec

        final = r4["reranked"]
        pack = "\n\n".join(
            f"SOURCE {i+1} ({c.get('source_path')} lines {c.get('start_line')}-"
            f"{c.get('end_line')})\n{(c.get('content') or '')[:2000]}"
            for i, c in enumerate(final))
        holder = {"pack": pack}

        t0 = time.time()
        r5 = p.stage5_generate(make_stage5(holder), timeout_s=60)
        rec["stages"].append({"stage": "5 generate", "ok": r5["ok"], "exec_s": r5.get("exec_s"),
                              "reason": r5.get("reason", "")[:140],
                              "wall_s": round(time.time() - t0, 3)})
        rec["n_sources"] = len(final)
        rec["pack_chars"] = len(pack)
        rec["answer_has_8192"] = r5.get("answer_has_8192")
        rec["total_wall_s"] = round(time.time() - t_all, 3)
        rec["status"] = "completed" if r5["ok"] else "failed"
        if not r5["ok"]:
            rec["error"] = r5.get("reason", "")
        return rec
    finally:
        p.cleanup()


def item4_interactive_during_background(s: GpuScheduler, rid_bg: str, rid_it: str) -> list[dict]:
    """An interactive request arriving while a background task runs - both through the scheduler."""
    out: list[dict] = []

    def bg():
        out.append(item2_embedding(s, rid_bg))

    def it():
        time.sleep(0.05)
        out.append(item1_interactive(s, rid_it))

    tb = threading.Thread(target=bg, daemon=True)
    ti = threading.Thread(target=it, daemon=True)
    tb.start(); ti.start()
    tb.join(timeout=OUTER); ti.join(timeout=OUTER)
    while len(out) < 2:
        out.append({"request_id": "MISSING", "item": "4 interactive during background",
                    "status": "hung"})
    for r in out:
        r["item"] = "4 interactive during background"
    return out


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

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=str(REPO), capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unavailable"

    report: dict = {
        "experiment": "P55R-04 staged five-item workload (ISOLATED PATH)",
        "not_production": ("the production routing is NOT changed; this script builds the staged "
                           "pipeline itself"),
        "budgets_unchanged": {"margin_mb": MARGIN,
                              "reranker_peak_extra_mb": REGISTRY["local-reranker"].peak_extra_mb,
                              "candidate_1361_written_to_default": False,
                              "candidate_881_written_to_default": False},
        "provenance": {"commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain")),
                       "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                       "registry": {n: {"port": s.port, "vram_mb": s.vram_mb,
                                        "peak_extra_mb": s.peak_extra_mb,
                                        "essential": s.essential}
                                    for n, s in REGISTRY.items()}},
        "rounds": [],
    }
    atomic_write_json(out, report)          # provenance on disk before any work

    for rnd in range(args.rounds):
        print(f"\n=== round {rnd+1}/{args.rounds} ===", flush=True)
        s = GpuScheduler(strategy="conditional", verbose=False, dry_run=False,
                         max_background_units=2)
        n = rnd + 1
        with WideSampler() as smp:
            t0 = time.time()
            results: list[dict] = []
            results.append(item1_interactive(s, f"r{n}-1-gen"))
            results.append(item2_embedding(s, f"r{n}-2-embed"))
            results.append(item3_full_rag(s, f"r{n}-3-rag"))
            results.extend(item4_interactive_during_background(
                s, f"r{n}-4a-bg", f"r{n}-4b-int"))
            t1 = time.time()
        win = smp.window(t0, t1)

        completed = sum(1 for r in results if r.get("status") == "completed")
        oom = any("out of memory" in json.dumps(r).lower() for r in results)
        min_free = win.get("free_min")
        rec = {
            "round": n,
            "requests": len(results), "completed": completed,
            "refused": sum(1 for r in results if r.get("status") == "refused"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "oom": oom,
            "vram_window": win,
            "vram_free_min_mb": min_free,
            "margin_mb": MARGIN,
            "margin_held": (min_free is not None and min_free >= MARGIN),
            "wall_s": round(t1 - t0, 2),
            "results": results,
        }
        rec["passed"] = (completed == 5) and not oom and bool(rec["margin_held"])
        report["rounds"].append(rec)
        atomic_write_json(out, report)       # after EVERY round

        print(f"  completed={completed}/{len(results)} refused={rec['refused']} "
              f"failed={rec['failed']} oom={oom} free_min={min_free} "
              f"margin_held={rec['margin_held']} wall={rec['wall_s']}s", flush=True)
        for r in results:
            print(f"    {r.get('request_id'):<14} {r.get('status'):<10} "
                  f"wait={r.get('wait_s')} exec={r.get('exec_s')} "
                  f"{(r.get('reason') or r.get('error') or '')[:80]}", flush=True)
            for st in r.get("stages", []):
                print(f"        {st['stage']:<42} ok={st['ok']} "
                      f"wait={st.get('wait_s')} wall={st.get('wall_s')} "
                      f"{(st.get('reason') or '')[:60]}", flush=True)

        if rnd == 0 and not rec["passed"]:
            report["stopped"] = {"after_round": 1,
                                 "reason": "first round did not meet the gate; not repeating",
                                 "completed": completed, "oom": oom,
                                 "margin_held": rec["margin_held"], "free_min": min_free}
            print("  ROUND FAILED - stopping, not repeating", flush=True)
            break

    passed = sum(1 for r in report["rounds"] if r.get("passed"))
    report["summary"] = {"rounds_run": len(report["rounds"]), "rounds_passed": passed,
                         "verdict": ("PASS" if passed == len(report["rounds"]) and passed > 0
                                     else "FAIL")}
    atomic_write_json(out, report)
    print(f"\nVERDICT: {report['summary']['verdict']} "
          f"({passed}/{len(report['rounds'])} rounds passed the gate)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
