"""
p5_five_items.py - P55R-05: the fixed five-item workload over the two-required-unload path.

Gate for a round to pass, all three:
  * 5 of 5 items complete
  * no OOM
  * minimum free VRAM across the whole round >= 2048 MiB

One round first. Two more ONLY if the first passes. The staged RAG item uses the same
required-unload pipeline that the single diagnostic validated.

Isolated path only: production routing unchanged, no default budget modified.

Usage: python p5_five_items.py --out <path> --rounds N
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
from p3_swap import StagedPipeline                                  # noqa: E402
import p2_residency as R                                            # noqa: E402
from p5_rag_diagnostic import Monitor, start_service, MARGIN, RAG_Q  # noqa: E402


def post(url, payload, timeout=600):
    return R.post(url, payload, timeout)


def i1_interactive(s, rid) -> dict:
    t0 = time.time()
    d = s.acquire("interactive", timeout_s=90, requires=["local-fast"], request_id=rid)
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


def i2_embedding(s, rid) -> dict:
    t0 = time.time()
    d = s.acquire("background", timeout_s=90, requires=["local-embedding"], request_id=rid)
    wait = time.time() - t0
    if not d.allowed:
        return {"request_id": rid, "item": "2 embedding background", "status": "refused",
                "wait_s": round(wait, 3), "reason": d.reason}
    try:
        out = R.w_embed_batch()
        return {"request_id": rid, "item": "2 embedding background",
                "status": "completed" if out.get("ok") else "failed",
                "wait_s": round(wait, 3), "exec_s": out.get("s"), "vectors": out.get("vectors")}
    except Exception as e:  # noqa: BLE001
        return {"request_id": rid, "item": "2 embedding background", "status": "failed",
                "wait_s": round(wait, 3), "error": f"{type(e).__name__}: {e}"}
    finally:
        s.release("background", d.reservation_id)


def i3_full_rag(s, rid, mon: Monitor) -> dict:
    """Full RAG over the staged path with BOTH required unloads."""
    rec: dict = {"request_id": rid, "item": "3 full RAG (staged, 2 required unloads)",
                 "stages": []}
    p = StagedPipeline(s)
    t_all = time.time()
    try:
        t0 = time.time()
        r1 = p.stage1_embed_retrieve(
            lambda: {"candidates": (post(f"{R.EMBED}/v1/retrieve",
                                         {"query": RAG_Q, "top_k": 10}).get("results") or [])},
            timeout_s=90)
        rec["stages"].append({"stage": "1 embed+retrieve", "ok": r1["ok"],
                              "wait_s": r1.get("waited_s"), "n": len(p.candidates),
                              "reason": r1.get("reason", "")[:120],
                              "wall_s": round(time.time() - t0, 3)})
        if not r1["ok"]:
            rec.update({"status": "refused", "reason": r1.get("reason", ""),
                        "wait_s": r1.get("waited_s")})
            return rec

        t0 = time.time()
        sw = p.stage3_unload_embed_then_admit_reranker(timeout_s=90)
        rec["stages"].append({"stage": "3 unload embedding + admit reranker", "ok": sw.ok,
                              "actual_unload_list": sw.unloaded,
                              "free_before": sw.free_before, "free_after": sw.free_after,
                              "port_free_verified": sw.port_free_verified,
                              "wait_s": sw.waited_s, "reason": sw.reason[:160],
                              "wall_s": round(time.time() - t0, 3)})
        if not sw.ok or not sw.unloaded:
            rec.update({"status": "failed",
                        "reason": f"embedding unload not performed: {sw.reason[:120]}"})
            return rec

        t0 = time.time()

        def _rerank(cands):
            docs = [c.get("content", "") for c in cands]
            rr = post(f"{R.RERANK}/v1/rerank",
                      {"query": RAG_Q, "documents": docs, "top_n": 5,
                       "instruction_type": "general", "use_cache": False})
            order = [x.get("index") for x in (rr.get("results") or [])]
            final = [cands[i] for i in order
                     if isinstance(i, int) and i < len(cands)][:5] or cands[:5]
            return {"reranked": final, "exec_s": round(time.time() - t0, 3)}

        r4 = p.stage4_rerank(_rerank)
        rec["stages"].append({"stage": "4 rerank", "ok": r4["ok"], "exec_s": r4.get("exec_s"),
                              "n": len(p.reranked), "reason": r4.get("reason", "")[:120],
                              "wall_s": round(time.time() - t0, 3)})
        if not r4["ok"]:
            rec.update({"status": "failed", "error": r4.get("reason", "")})
            return rec

        t0 = time.time()
        u5 = p.stage5_unload_reranker()
        rec["stages"].append({"stage": "5 unload reranker", "ok": u5.ok,
                              "actual_unload_list": u5.unloaded,
                              "free_before": u5.free_before, "free_after": u5.free_after,
                              "port_free_verified": u5.port_free_verified,
                              "reason": u5.reason[:160], "wall_s": round(time.time() - t0, 3)})
        if not u5.ok or not u5.unloaded:
            rec.update({"status": "failed",
                        "reason": f"reranker unload not performed: {u5.reason[:120]}"})
            return rec

        pack = "\n\n".join(
            f"SOURCE {i+1} ({c.get('source_path')} lines {c.get('start_line')}-"
            f"{c.get('end_line')})\n{(c.get('content') or '')[:2000]}"
            for i, c in enumerate(p.reranked))

        def _gen():
            t0 = time.time()
            g = post(f"{R.FAST}/v1/chat/completions", {
                "model": "local-fast",
                "messages": [{"role": "user", "content": f"{pack}\n\nQUESTION: {RAG_Q}"}],
                "max_tokens": 64, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}})
            text = (g.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return {"exec_s": round(time.time() - t0, 3), "answer_has_8192": "8192" in text,
                    "answer_head": text.strip()[:150]}

        t0 = time.time()
        r6 = p.stage6_generate(_gen, timeout_s=90)
        rec["stages"].append({"stage": "6 pack+generate", "ok": r6["ok"],
                              "exec_s": r6.get("exec_s"),
                              "answer_has_8192": r6.get("answer_has_8192"),
                              "reason": r6.get("reason", "")[:120],
                              "wall_s": round(time.time() - t0, 3)})
        rec["n_sources"] = len(p.reranked)
        rec["pack_chars"] = len(pack)
        rec["answer_has_8192"] = r6.get("answer_has_8192")
        rec["total_wall_s"] = round(time.time() - t_all, 3)
        rec["status"] = "completed" if r6["ok"] else "failed"
        if not r6["ok"]:
            rec["error"] = r6.get("reason", "")
        return rec
    finally:
        p.cleanup()


def i4_interactive_during_bg(s, rid_bg, rid_it) -> list[dict]:
    out: list[dict] = []

    def bg():
        out.append(i2_embedding(s, rid_bg))

    def it():
        time.sleep(0.05)
        out.append(i1_interactive(s, rid_it))

    tb = threading.Thread(target=bg, daemon=True)
    ti = threading.Thread(target=it, daemon=True)
    tb.start(); ti.start()
    tb.join(timeout=600); ti.join(timeout=600)
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
        "experiment": "P55R-05 fixed five-item workload (staged, 2 required unloads)",
        "production_unchanged": True,
        "budgets_unchanged": {"margin_mb": MARGIN,
                              "reranker_peak_extra_mb": REGISTRY["local-reranker"].peak_extra_mb,
                              "candidate_881_or_1361_used": False},
        "provenance": {"commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain"))},
        "rounds": [],
    }
    atomic_write_json(out, report)

    for rnd in range(args.rounds):
        n = rnd + 1
        print(f"\n=== round {n}/{args.rounds} ===", flush=True)
        # restore residency: every round starts from all three healthy
        for svc in ("local-embedding", "local-reranker"):
            if not service_healthy(REGISTRY[svc].port):
                start_service(svc)
        time.sleep(2)
        report["resident_at_round_start"] = [x for x in REGISTRY
                                             if service_healthy(REGISTRY[x].port)]

        s = GpuScheduler(strategy="conditional", verbose=False, dry_run=False,
                         max_background_units=2)
        with Monitor() as mon:
            t0 = time.time()
            results: list[dict] = []
            results.append(i1_interactive(s, f"r{n}-1-gen"))
            results.append(i2_embedding(s, f"r{n}-2-embed"))

            # Item 3 stage 1 is an EMBEDDING call, so it needs local-embedding healthy - but that
            # is a precondition the item does not itself establish. It worked only because item 2
            # ran immediately before it and left embedding up; on the earlier ordering it failed
            # with a refused connection. The precondition is now made EXPLICIT and its cost is
            # reported separately rather than silently leaking into item 3's latency.
            t_pre3 = time.time()
            pre3 = {}
            for svc in ("local-embedding", "local-reranker", "local-fast"):
                if not service_healthy(REGISTRY[svc].port):
                    t0s = time.time()
                    ok = start_service(svc)
                    pre3[svc] = {"was_down": True, "restored": ok,
                                 "seconds": round(time.time() - t0s, 2)}
                else:
                    pre3[svc] = {"was_down": False, "restored": True, "seconds": 0.0}
            report["precondition_before_item3"] = {
                "services": pre3, "wall_s": round(time.time() - t_pre3, 2),
                "note": ("item 3 requires the embedding service; the requirement is declared and "
                         "satisfied here, and its latency is reported separately"),
            }
            atomic_write_json(out, report)

            results.append(i3_full_rag(s, f"r{n}-3-rag", mon))

            # Item 4 must start from a HEALTHY residency. Item 3 deliberately unloads the
            # embedding and reranker models to fit the margin, so it leaves the machine with them
            # down - and item 4a is an embedding call, which then fails on a refused connection.
            # That is a harness ordering defect, not a pipeline failure: the unloads are supposed
            # to happen. This restoration is recorded explicitly with its own timing, and it is
            # NOT counted as part of item 4's latency.
            t_restore = time.time()
            restored = {}
            for svc in ("local-embedding", "local-reranker"):
                if not service_healthy(REGISTRY[svc].port):
                    t0s = time.time()
                    ok = start_service(svc)
                    restored[svc] = {"was_down": True, "restored": ok,
                                     "seconds": round(time.time() - t0s, 2)}
                else:
                    restored[svc] = {"was_down": False, "restored": True, "seconds": 0.0}
            report["restore_before_item4"] = {
                "services": restored, "wall_s": round(time.time() - t_restore, 2),
                "note": ("item 3 intentionally unloads both models; item 4 needs them, so the "
                         "restoration is explicit and its latency is reported separately rather "
                         "than folded into item 4"),
            }
            atomic_write_json(out, report)
            print(f"  restored before item 4: {restored} "
                  f"({report['restore_before_item4']['wall_s']}s)", flush=True)

            results.extend(i4_interactive_during_bg(s, f"r{n}-4a-bg", f"r{n}-4b-int"))
            t1 = time.time()
        win = mon.window(t0, t1)

        completed = sum(1 for r in results if r.get("status") == "completed")
        oom = any("out of memory" in json.dumps(r).lower() for r in results)
        unloads_both = False
        rag = next((r for r in results if r.get("item", "").startswith("3 ")), None)
        if rag:
            st = {x["stage"]: x for x in rag.get("stages", [])}
            e = st.get("3 unload embedding + admit reranker", {})
            rr = st.get("5 unload reranker", {})
            unloads_both = bool(e.get("actual_unload_list") and rr.get("actual_unload_list"))
        rec = {"round": n, "requests": len(results), "completed": completed,
               "refused": sum(1 for r in results if r.get("status") == "refused"),
               "failed": sum(1 for r in results if r.get("status") == "failed"),
               "oom": oom, "both_unloads_performed": unloads_both,
               "vram_window": win, "vram_free_min_mb": win.get("free_min"),
               "margin_mb": MARGIN,
               "margin_held": bool(win.get("free_min") and win["free_min"] >= MARGIN),
               "wall_s": round(t1 - t0, 2), "results": results}
        rec["passed"] = (completed == 5 and not oom and rec["margin_held"]
                         and unloads_both)
        report["rounds"].append(rec)
        atomic_write_json(out, report)

        print(f"  completed={completed}/{len(results)} refused={rec['refused']} "
              f"failed={rec['failed']} oom={oom} both_unloads={unloads_both} "
              f"free_min={win.get('free_min')} margin_held={rec['margin_held']}", flush=True)
        for r in results:
            print(f"    {(r.get('request_id') or '?'):<14} {(r.get('status') or '?'):<10} "
                  f"wait={r.get('wait_s')} exec={r.get('exec_s')} "
                  f"{(r.get('reason') or r.get('error') or '')[:70]}", flush=True)
            for stg in r.get("stages", []):
                print(f"        {stg['stage']:<42} ok={stg['ok']} "
                      f"unloaded={stg.get('actual_unload_list')} "
                      f"free {stg.get('free_before')}->{stg.get('free_after')}", flush=True)

        if rnd == 0 and not rec["passed"]:
            report["stopped"] = {"after_round": 1,
                                 "reason": "first round did not meet the gate; not repeating",
                                 "completed": completed, "oom": oom,
                                 "margin_held": rec["margin_held"],
                                 "both_unloads_performed": unloads_both,
                                 "free_min": win.get("free_min")}
            print("  ROUND FAILED - stopping, not repeating", flush=True)
            break

    # restore residency at the end
    for svc in ("local-embedding", "local-reranker"):
        if not service_healthy(REGISTRY[svc].port):
            start_service(svc)
    report["restored"] = [x for x in REGISTRY if service_healthy(REGISTRY[x].port)]
    passed = sum(1 for r in report["rounds"] if r.get("passed"))
    report["summary"] = {"rounds_run": len(report["rounds"]), "rounds_passed": passed,
                         "verdict": ("PASS" if passed == len(report["rounds"]) and passed > 0
                                     else "FAIL")}
    atomic_write_json(out, report)
    print(f"\nVERDICT: {report['summary']['verdict']} ({passed}/{len(report['rounds'])})")
    print(f"restored: {report['restored']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
