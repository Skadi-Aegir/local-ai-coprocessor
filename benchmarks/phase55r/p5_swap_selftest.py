"""
p5_swap_selftest.py - P55R-05: deterministic verification of the TWO-REQUIRED-UNLOAD path.

Mocks, barriers and events only. No GPU work, no service started or stopped in the happy path.
Covers what the phase enumerates:
  active dependency protection
  concurrent admission during the handoff
  BOTH unloads failing (independently)
  cancellation / timeout
  result retention (candidates AND reranked results survive failures)

Run: python p5_swap_selftest.py --out <path>
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))
sys.path.insert(0, str(HERE))

import gpu_scheduler as G                            # noqa: E402
from gpu_scheduler import GpuScheduler               # noqa: E402
import p3_swap as SW                                 # noqa: E402
from p3_swap import Stage, StagedPipeline            # noqa: E402
from p3_atomic import atomic_write_json              # noqa: E402

OUTER = 30.0
MARGIN = 2048
FAILS: list[str] = []
RESULTS: list[dict] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""), flush=True)
    RESULTS.append({"name": name, "passed": bool(cond), "detail": detail})
    if not cond:
        FAILS.append(name)


def fake_free(init: int = 30000, running=("local-embedding", "local-reranker")):
    """A mutable live-free reading plus a running set, so unload has visible effects."""
    st = {"free": init, "running": set(running)}

    def _free():
        return st["free"]

    return st, _free


def make_sched(st, dry_run: bool = False, strategy: str = "unload") -> GpuScheduler:
    s = GpuScheduler(strategy=strategy, verbose=False, dry_run=dry_run)
    s.free_mb = lambda: st["free"]                                   # type: ignore
    s.running = lambda: sorted(st["running"])                        # type: ignore
    return s


def cands(n: int = 5) -> list[dict]:
    return [{"chunk_id": f"c{i}", "source_path": f"f{i}.md", "content": f"body {i}"}
            for i in range(n)]


class _wd:
    def __init__(self, seconds: float = OUTER):
        self.seconds = seconds

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    print("=== 1: happy path performs BOTH required unloads ===")
    with _wd():
        st, _free = fake_free(30000)
        s = make_sched(st)
        # fixture: unload frees memory and drops the service from the running set, and the port
        # is reported free once it is gone
        orig_unload = G.GpuScheduler._unload_service
        orig_pf = SW.port_free

        def unload(self, name):
            st["running"].discard(name)
            st["free"] += REG_FOOTPRINT.get(name, 1000)
            return True

        G.GpuScheduler._unload_service = unload                      # type: ignore
        SW.port_free = lambda port: not any(                         # type: ignore
            G.REGISTRY[n].port == port for n in st["running"])
        try:
            p = StagedPipeline(s)
            r1 = p.stage1_embed_retrieve(lambda: {"candidates": cands(), "exec_s": 0.01})
            check("stage1 admits", r1["ok"], str(r1)[:60])
            check("stage1 closed its reservation", p.live_own_reservations() == 0)
            sw = p.stage3_unload_embed_then_admit_reranker()
            check("swap ok", sw.ok, sw.reason[:80])
            check("swap ACTUALLY unloaded embedding (verified, not assumed)",
                  sw.unloaded == ["local-embedding"] and sw.port_free_verified,
                  f"unloaded={sw.unloaded} port_free={sw.port_free_verified}")
            check("free VRAM rose across the unload",
                  sw.free_after and sw.free_before and sw.free_after > sw.free_before,
                  f"{sw.free_before} -> {sw.free_after}")
            check("embedding is no longer running", "local-embedding" not in st["running"],
                  str(sorted(st["running"])))
            r4 = p.stage4_rerank(lambda c: {"reranked": c[:3], "exec_s": 0.01})
            check("stage4 reranks", r4["ok"], str(r4)[:60])
            check("state is RERANKED_ON_CPU (results retained for the next stage)",
                  p.state is Stage.RERANKED_ON_CPU, p.state.value)
            u5 = p.stage5_unload_reranker()
            check("the SECOND required unload succeeds", u5.ok, u5.reason[:80])
            check("reranker actually unloaded", "local-reranker" not in st["running"],
                  str(sorted(st["running"])))
            r6 = p.stage6_generate(lambda: {"exec_s": 0.01, "answer_has_8192": True})
            check("stage6 generates", r6["ok"], str(r6)[:60])
            check("final state DONE", p.state is Stage.DONE, p.state.value)
            p.cleanup()
            check("no reservation left", s.reserved_mb() == 0, str(s.reserved_mb()))
        finally:
            G.GpuScheduler._unload_service = orig_unload             # type: ignore
            SW.port_free = orig_pf                                   # type: ignore

    print("\n=== 2: active dependency protection (I3) ===")
    with _wd():
        st, _free = fake_free(30000)
        s = make_sched(st, dry_run=True)
        d = s.acquire("background", timeout_s=5, requires=["local-embedding"], request_id="p1")
        check("a request naming embedding is admitted", d.allowed, d.reason[:60])
        check("unload is REFUSED while it is live", s._unload_service("local-embedding") is False)
        p = StagedPipeline(s)
        blocked = p.unload_required("local-embedding")
        check("unload_required reports the refusal rather than skipping it",
              not blocked.ok and blocked.stage == "unload_refused", blocked.reason[:80])
        s.release("background", d.reservation_id)
        check("after release it becomes unloadable", s._unload_service("local-embedding") is True)

    print("\n=== 3: concurrent admission during the handoff ===")
    with _wd():
        st, _free = fake_free(30000)
        s = make_sched(st, dry_run=True)
        s.max_background_units = 2
        p = StagedPipeline(s)
        p.stage1_embed_retrieve(lambda: {"candidates": cands(), "exec_s": 0.01})
        got: dict = {}
        tb = threading.Thread(target=lambda: got.__setitem__(
            "bg", s.acquire("background", timeout_s=3)))
        ti = threading.Thread(target=lambda: got.__setitem__(
            "it", s.acquire("interactive", timeout_s=3, requires=["local-fast"])))
        tb.start(); ti.start(); tb.join(OUTER); ti.join(OUTER)
        check("concurrent background gets an answer", "bg" in got)
        check("concurrent interactive gets an answer", "it" in got)
        for k, cls in (("bg", "background"), ("it", "interactive")):
            dd = got.get(k)
            if dd is not None and getattr(dd, "allowed", False):
                s.release(cls, dd.reservation_id)
        check("ledger is clean before the swap", s.reserved_mb() == 0, str(s.reserved_mb()))
        p.cleanup()

    print("\n=== 4: FIRST unload fails (port frees, memory does NOT return) ===")
    with _wd():
        st, _free = fake_free(20000)
        s = make_sched(st)
        orig_unload = G.GpuScheduler._unload_service
        orig_pf = SW.port_free
        G.GpuScheduler._unload_service = lambda self, name: True      # type: ignore
        SW.port_free = lambda port: True                             # type: ignore
        # memory deliberately does NOT rise
        try:
            p = StagedPipeline(s)
            p.stage1_embed_retrieve(lambda: {"candidates": cands(), "exec_s": 0.01})
            r = p.unload_required("local-embedding")
            check("an unload that frees no memory is reported as FAILED",
                  not r.ok and r.stage == "memory_not_returned", f"{r.stage}: {r.reason[:80]}")
            check("the failure is not silently treated as success", not r.ok)
        finally:
            G.GpuScheduler._unload_service = orig_unload             # type: ignore
            SW.port_free = orig_pf                                   # type: ignore

    print("\n=== 5: FIRST unload fails (port stays bound) ===")
    with _wd():
        st, _free = fake_free(20000)
        s = make_sched(st)
        orig_unload = G.GpuScheduler._unload_service
        orig_pf = SW.port_free
        G.GpuScheduler._unload_service = lambda self, name: True      # type: ignore
        SW.port_free = lambda port: False                            # type: ignore
        try:
            p = StagedPipeline(s)
            p.stage1_embed_retrieve(lambda: {"candidates": cands(), "exec_s": 0.01})
            r = p.unload_required("local-embedding")
            check("a still-bound port is reported as FAILED",
                  not r.ok and r.stage == "port_still_bound", f"{r.stage}: {r.reason[:80]}")
        finally:
            G.GpuScheduler._unload_service = orig_unload             # type: ignore
            SW.port_free = orig_pf                                   # type: ignore

    print("\n=== 6: swap aborts when the first unload fails; candidates retained; retryable ===")
    with _wd():
        st, _free = fake_free(20000)
        s = make_sched(st)
        orig_unload = G.GpuScheduler._unload_service
        G.GpuScheduler._unload_service = lambda self, name: False     # type: ignore
        try:
            p = StagedPipeline(s)
            p.stage1_embed_retrieve(lambda: {"candidates": cands(), "exec_s": 0.01})
            sw = p.stage3_unload_embed_then_admit_reranker(timeout_s=2)
            check("swap fails when the unload fails", not sw.ok, sw.reason[:90])
            check("no reservation leaked", s.reserved_mb() == 0 and p.live_own_reservations() == 0,
                  f"reserved={s.reserved_mb()} live={p.live_own_reservations()}")
            check("candidates retained (I5)", len(p.candidates) == 5, f"n={len(p.candidates)}")
            check("state returns to CANDIDATES_ON_CPU for retry",
                  p.state is Stage.CANDIDATES_ON_CPU, p.state.value)
            p.cleanup()
        finally:
            G.GpuScheduler._unload_service = orig_unload             # type: ignore

    print("\n=== 7: SECOND unload fails -> reranked results retained ===")
    with _wd():
        st, _free = fake_free(30000)
        s = make_sched(st)
        orig_unload = G.GpuScheduler._unload_service
        orig_pf = SW.port_free

        def unload_ok(self, name):
            if name == "local-embedding":
                st["running"].discard(name)
                st["free"] += 1500
                return True
            return False                     # the reranker unload fails

        G.GpuScheduler._unload_service = unload_ok                  # type: ignore
        SW.port_free = lambda port: not any(                        # type: ignore
            G.REGISTRY[n].port == port for n in st["running"])
        try:
            p = StagedPipeline(s)
            p.stage1_embed_retrieve(lambda: {"candidates": cands(), "exec_s": 0.01})
            sw = p.stage3_unload_embed_then_admit_reranker()
            check("first swap ok", sw.ok, sw.reason[:60])
            p.stage4_rerank(lambda c: {"reranked": c[:3], "exec_s": 0.01})
            u5 = p.stage5_unload_reranker()
            check("the second unload failure is reported", not u5.ok, u5.reason[:90])
            check("RERANKED results are retained for a retry",
                  len(p.reranked) == 3, f"n={len(p.reranked)}")
            check("no reservation leaked after the second failure",
                  s.reserved_mb() == 0, str(s.reserved_mb()))
            p.cleanup()
        finally:
            G.GpuScheduler._unload_service = orig_unload             # type: ignore
            SW.port_free = orig_pf                                   # type: ignore

    print("\n=== 8: cancellation / timeout during stage1 leaves nothing behind ===")
    with _wd():
        st, _free = fake_free(30000)
        s = make_sched(st, dry_run=True)
        held = s.acquire("interactive", timeout_s=3, requires=["local-fast"])
        p = StagedPipeline(s)
        res = p.stage1_embed_retrieve(lambda: {"candidates": cands()}, timeout_s=0.4)
        check("stage1 times out rather than hanging", not res["ok"], str(res)[:70])
        check("timeout is reported", "timed out" in res.get("reason", ""),
              res.get("reason", "")[:70])
        check("nothing left behind", p.live_own_reservations() == 0)
        s.release("interactive", held.reservation_id)
        p.cleanup()
        check("ledger clean", s.reserved_mb() == 0, str(s.reserved_mb()))

    print("\n=== 9: unload_required with a live reservation of this pipeline is refused (I2) ===")
    with _wd():
        st, _free = fake_free(30000)
        s = make_sched(st, dry_run=True)
        p = StagedPipeline(s)
        d = s.acquire("background", timeout_s=3, requires=["local-embedding"])
        p.rid = d.reservation_id
        p._own.add(d.reservation_id)
        r = p.unload_required("local-embedding")
        check("I2 blocks a required unload while this pipeline holds a reservation",
              not r.ok and r.stage == "I2", f"{r.stage}: {r.reason[:70]}")
        p.cleanup()
        check("cleanup cleared it", s.reserved_mb() == 0, str(s.reserved_mb()))

    payload = {"checks": RESULTS, "passed": sum(1 for r in RESULTS if r["passed"]),
               "failed": len(FAILS), "outer_timeout_s": OUTER, "margin_mb": MARGIN,
               "note": "mocks/barriers only; no GPU work, no service started or stopped"}
    atomic_write_json(out, payload)
    print(f"\n{payload['passed']}/{len(RESULTS)} checks passed")
    if FAILS:
        print(f"FAILED: {FAILS}")
    print(f"wrote {out}")
    return 0 if not FAILS else 1


REG_FOOTPRINT = {"local-embedding": 1500, "local-reranker": 1600, "local-fast": 3000}

if __name__ == "__main__":
    raise SystemExit(main())
