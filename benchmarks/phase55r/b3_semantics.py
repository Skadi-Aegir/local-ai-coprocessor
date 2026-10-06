"""
b3_semantics.py - P55R-01 work package B3.

Tests the unified concurrency semantics and the scenarios the phase enumerates: both arrival
orders, queueing, timeout, abnormal exit, double release, and exclusive starvation. It also
MEASURES the interactive wait bound rather than asserting one exists.

Everything runs against stubbed memory and stubbed process execution: no GPU work, no service
started or stopped.

The bound being measured, stated honestly: without preemption an interactive request cannot
avoid waiting for background work already in flight. The scheduler's only lever is to stop
admitting NEW background units once an interactive request is waiting, and to cap how many
background units may run at once. The wait is therefore bounded by
`max_background_units x background_unit_duration`, and the measured figure is reported rather
than a promise that no wait occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))

from gpu_scheduler import GpuScheduler  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def stub(s: GpuScheduler, free: int = 30000) -> GpuScheduler:
    s.free_mb = lambda: free            # type: ignore[method-assign]
    s.running = lambda: []              # type: ignore[method-assign]
    return s


def replay_peak(s: GpuScheduler) -> int:
    active: dict[str, int] = {}
    peak = 0
    for h in s.history:
        if h["event"] == "acquire":
            active[h["job_class"]] = active.get(h["job_class"], 0) + 1
            peak = max(peak, sum(active.values()))
        elif h["event"] == "release":
            active[h["job_class"]] = max(0, active.get(h["job_class"], 0) - 1)
    return peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"note": "stubbed memory and process execution; no GPU work", "cases": []}

    # ---------------- 1. arrival order A: background first, interactive second
    print("=== 1: ARRIVAL ORDER A - background running, interactive arrives ===")
    s = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True,
                          max_background_units=2), 30000)
    bg = s.acquire("background", timeout_s=3)
    check("background is admitted first", bg.allowed, bg.reason[:60])
    s._active["background"] = max(s._active.get("background", 1), 3)  # simulate 3 in flight
    d_int = s.acquire("interactive", timeout_s=1.0)
    print(f"  interactive while 3 background units active: allowed={d_int.allowed}")
    check("interactive does NOT overlap running background", not d_int.allowed,
          d_int.reason[:70])
    # and while it is waiting, no NEW background may start.
    #
    # This drives a REAL interactive waiter rather than setting the counter directly. Setting
    # `_waiting_interactive = 1` by hand creates a state the scheduler cannot resolve: with
    # FIFO ticketing the background request takes a LATER ticket than the phantom waiter, so
    # it correctly waits its turn and times out as "queued behind an earlier request" - a
    # correct behaviour that the assertion then misread as a policy failure. A real waiter
    # holds the earlier ticket and genuinely blocks new background work, which is the rule
    # under test.
    s2 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 30000)
    held = s2.acquire("background", timeout_s=3)
    s2._active["background"] = 1
    waiter_result: dict = {}

    def interactive_waiter():
        waiter_result["d"] = s2.acquire("interactive", timeout_s=3.0)

    wt = threading.Thread(target=interactive_waiter, daemon=True)
    wt.start()
    time.sleep(0.3)                    # let it register as waiting
    print(f"  waiting interactive count: {s2._waiting_interactive}")
    d_new_bg = s2.acquire("background", timeout_s=0.5)
    print(f"  new background while an interactive waits: allowed={d_new_bg.allowed}")
    check("no new background unit is admitted while an interactive waits",
          not d_new_bg.allowed, d_new_bg.reason[:90])
    # The refusal must IDENTIFY the blocker. With FIFO the interactive waiter legitimately
    # holds an EARLIER ticket, so the background request is refused as queued behind it -
    # which is the same policy outcome expressed through the queue rather than through the
    # waiting counter. Accepting either wording and requiring the word "interactive" is what
    # makes the assertion test the intent (background blocked by interactive) instead of one
    # particular code path.
    check("the refusal identifies an interactive request as the blocker",
          "interactive" in d_new_bg.reason, d_new_bg.reason[:110])
    s2._active.clear()
    s2.release("background", held.reservation_id)
    wt.join(timeout=8)
    check("the waiting interactive eventually gets its answer (no hang)",
          "d" in waiter_result,
          f"result={'present' if 'd' in waiter_result else 'thread never returned'}")
    wd = waiter_result.get("d")
    if wd is not None and getattr(wd, "allowed", False):
        s2.release("interactive", wd.reservation_id)
    if bg.allowed:
        s.release("background", bg.reservation_id)

    # ---------------- 2. arrival order B: interactive first, background second
    print("\n=== 2: ARRIVAL ORDER B - interactive running, background arrives ===")
    s3 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 30000)
    i1 = s3.acquire("interactive", timeout_s=3)
    check("interactive is admitted first", i1.allowed, i1.reason[:60])
    d_bg = s3.acquire("background", timeout_s=0.4)
    print(f"  background while interactive runs: allowed={d_bg.allowed}")
    check("background does NOT join an active interactive job", not d_bg.allowed,
          d_bg.reason[:70])
    i2 = s3.acquire("interactive", timeout_s=0.4)
    check("a second interactive does NOT share with the first", not i2.allowed,
          i2.reason[:70])
    s3.release("interactive", i1.reservation_id)

    # ---------------- 3. queueing and the MEASURED interactive wait bound
    print("\n=== 3: interactive wait bound, MEASURED ===")
    # Each worker acquires, holds, RELEASES, then acquires again. It must NOT hold one slot
    # while requesting another: an earlier version did exactly that and deadlocked - the
    # workers held slots which the interactive was waiting for, while the interactive's
    # presence blocked the workers' NEXT acquire, so they timed out still holding. The
    # scheduler behaved correctly (its history shows acquire, acquire, timeout, timeout) and
    # the hang was produced by the test holding resources while asking for more. Real jobs
    # do not do that, and the shape is changed rather than the scheduler being weakened.
    s4 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True,
                           max_background_units=2), 30000)
    unit = 0.30
    holds = []
    stop_holders = threading.Event()

    def hold_bg() -> None:
        while not stop_holders.is_set():
            d = s4.acquire("background", timeout_s=5)
            if not d.allowed:
                return
            holds.append(d)
            time.sleep(unit)
            s4.release("background", d.reservation_id)   # release BEFORE the next request

    ts = [threading.Thread(target=hold_bg, daemon=True) for _ in range(2)]
    for t in ts:
        t.start()
    time.sleep(0.1)
    print(f"  [diag] before interactive: active={dict(s4._active)} "
          f"reserved={s4.reserved_mb()} waiting={s4._waiting_interactive} "
          f"holds={len(holds)}", flush=True)
    t0 = time.time()
    d_wait = s4.acquire("interactive", timeout_s=10)
    waited = time.time() - t0
    print(f"  [diag] after interactive: active={dict(s4._active)} "
          f"holds={len(holds)} alive={[t.is_alive() for t in ts]}", flush=True)
    print(f"  [diag] background history: "
          f"{[h['event'] for h in s4.history if h['job_class'] == 'background'][:14]}",
          flush=True)
    stop_holders.set()
    for t in ts:
        t.join(timeout=10)
    print(f"  interactive waited {waited:.3f}s while {len(holds)} background units ran "
          f"(unit={unit}s, max_bg={s4.max_background_units})")
    theoretical = s4.max_background_units * unit
    report["wait_bound"] = {"unit_s": unit, "max_background_units": s4.max_background_units,
                            "theoretical_bound_s": theoretical,
                            "measured_wait_s": round(waited, 3),
                            "background_units_observed": len(holds),
                            "admitted": d_wait.allowed}
    check("the interactive wait is finite and within the theoretical bound",
          waited <= theoretical + 1.0, f"waited={waited:.3f}s bound={theoretical:.3f}s")
    check("the interactive request IS eventually admitted (no starvation)",
          d_wait.allowed, d_wait.reason[:70])
    if d_wait.allowed:
        s4.release("interactive", d_wait.reservation_id)

    # ---------------- 4. timeout
    print("\n=== 4: timeout returns rather than hanging ===")
    s5 = stub(GpuScheduler(strategy="serial", verbose=False, dry_run=True), 30000)
    held5 = s5.acquire("interactive", timeout_s=3)
    t0 = time.time()
    d_to = s5.acquire("background", timeout_s=0.5)
    dt = time.time() - t0
    print(f"  waited {dt:.2f}s, allowed={d_to.allowed}, reason={d_to.reason[:60]}")
    check("a blocked request times out instead of hanging", 0.4 < dt < 3.0, f"{dt:.2f}s")
    check("the timeout is stated in the reason", "timed out" in d_to.reason, d_to.reason[:70])
    s5.release("interactive", held5.reservation_id)
    check("no reservation leaked after timeout", s5.reserved_mb() == 0,
          f"{s5.reserved_mb()}")

    # ---------------- 5. abnormal exit: release in a finally, then again
    print("\n=== 5: abnormal exit and release accounting ===")
    s6 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 30000)
    d6 = s6.acquire("background", timeout_s=3)
    try:
        raise RuntimeError("simulated job failure")
    except RuntimeError:
        pass
    finally:
        s6.release("background", d6.reservation_id)
    check("a job that dies still releases its reservation", s6.reserved_mb() == 0,
          f"{s6.reserved_mb()}")
    check("active count returns to empty", s6._active == {}, str(s6._active))
    # the budget must be available again
    d6b = s6.acquire("background", timeout_s=1)
    check("the budget is reusable after an abnormal exit", d6b.allowed, d6b.reason[:60])
    s6.release("background", d6b.reservation_id)

    # ---------------- 6. double release
    print("\n=== 6: double release ===")
    s7 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 30000)
    d7 = s7.acquire("background", timeout_s=3)
    s7.release("background", d7.reservation_id)
    before = s7.reserved_mb()
    s7.release("background", d7.reservation_id)
    s7.release("background")
    check("double release cannot drive the reservation negative",
          s7.reserved_mb() == before == 0, f"{s7.reserved_mb()}")
    check("double release is recorded",
          sum(1 for h in s7.history if h["event"] == "double_release_ignored") >= 1,
          str([h["event"] for h in s7.history]))

    # ---------------- 7. exclusive starvation
    print("\n=== 7: exclusive starvation under a background stream ===")
    s8 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 30000)
    stop = threading.Event()
    done = {"bg": 0}

    def bg_stream():
        while not stop.is_set():
            d = s8.acquire("background", timeout_s=0.2)
            if d.allowed:
                done["bg"] += 1
                time.sleep(0.05)
                s8.release("background", d.reservation_id)

    th = threading.Thread(target=bg_stream, daemon=True)
    th.start()
    time.sleep(0.2)
    t0 = time.time()
    d_ex = s8.acquire("exclusive", timeout_s=2.0)
    ex_wait = time.time() - t0
    stop.set()
    th.join(timeout=5)
    print(f"  exclusive: allowed={d_ex.allowed} waited={ex_wait:.2f}s "
          f"background completions={done['bg']}")
    report["exclusive_starvation"] = {"allowed": d_ex.allowed,
                                      "waited_s": round(ex_wait, 2),
                                      "background_completions": done["bg"]}
    check("exclusive is refused or served within its timeout, never hung forever",
          ex_wait < 6, f"{ex_wait:.2f}s")
    check("the exclusive decision is explained", bool(d_ex.reason), d_ex.reason[:70])
    if d_ex.allowed:
        s8.release("exclusive", d_ex.reservation_id)

    # ---------------- 8. the process boundary is stated
    print("\n=== 8: process boundary ===")
    st = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True)).status()
    check("status() states the in-process lock boundary",
          "in-process" in (st.get("process_boundary") or ""), str(st.get("process_boundary")))
    check("status() reports reserved memory",
          "reserved_mb" in st, str(sorted(st.keys())[:8]))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("B3 passed: arrival orders, queueing, timeout, abnormal exit, double release and "
          "exclusive starvation all behave as specified")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
