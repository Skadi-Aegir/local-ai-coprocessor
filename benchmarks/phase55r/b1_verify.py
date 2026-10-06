"""
b1_verify.py - prove the B1 fixes bind, using the SAME forced interleaving that reproduced
the four defects against the previous revision.

The earlier harness forced the interleaving by calling plan() and reserving in separate
steps, which is what the old code did internally. The FIXED code cannot be made to behave
that way from outside, because planning and reserving are one atomic operation. So this
harness instead drives the real public API concurrently and asserts the invariants hold.

Both directions are checked: the invariants must hold under contention AND normal use must
still be admitted, so the fix cannot pass by refusing everything.

Usage:
  python b1_verify.py --out ../results/<run-id>/b1_verify.json
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


def stub(s: GpuScheduler, free: int) -> GpuScheduler:
    s.free_mb = lambda: free          # type: ignore[method-assign]
    s.running = lambda: []            # type: ignore[method-assign]
    return s


def contend(s: GpuScheduler, classes: list[str], hold: float = 0.15,
            barrier: threading.Barrier | None = None) -> list[dict]:
    """Drive the REAL acquire() from several threads at once."""
    out: list[dict] = [{} for _ in classes]

    def worker(i: int, cls: str) -> None:
        if barrier:
            barrier.wait(timeout=10)
        d = s.acquire(cls, timeout_s=2.0)
        out[i] = {"cls": cls, "allowed": d.allowed, "reason": d.reason,
                  "rid": getattr(d, "reservation_id", None)}
        if d.allowed:
            time.sleep(hold)          # hold the slot so overlap would be observable
            s.release(cls, getattr(d, "reservation_id", None))

    ts = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(classes)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    return out


def max_concurrent(s: GpuScheduler) -> int:
    """Replay history to find the peak simultaneous active jobs."""
    active: dict[str, int] = {}
    peak = 0
    for h in s.history:
        if h["event"] == "acquire":
            k = h["job_class"]
            active[k] = active.get(k, 0) + 1
            peak = max(peak, sum(active.values()))
        elif h["event"] == "release":
            k = h["job_class"]
            active[k] = max(0, active.get(k, 0) - 1)
    return peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"note": "drives the real acquire()/release() API; no GPU work",
                    "cases": []}

    # ---------------- 1. serial must not double-admit
    print("=== 1: serial, two interactive jobs race ===")
    s = stub(GpuScheduler(strategy="serial", verbose=False, dry_run=True), 20000)
    res = contend(s, ["interactive", "interactive"], barrier=threading.Barrier(2))
    peak = max_concurrent(s)
    admitted = sum(1 for r in res if r["allowed"])
    print(f"  admitted={admitted}  peak_concurrent={peak}  active={dict(s._active)}")
    for r in res:
        print(f"    {'ADMITTED' if r['allowed'] else 'refused '} {r['reason'][:66]}")
    report["cases"].append({"case": "serial_double_admit", "admitted": admitted,
                            "peak_concurrent": peak, "active": dict(s._active)})
    check("serial never has more than one job active", peak <= 1, f"peak={peak}")
    check("serial admitted at least one job (fix must not refuse everything)",
          admitted >= 1, f"admitted={admitted}")
    check("no reservation leaked", s.reserved_mb() == 0, f"reserved={s.reserved_mb()} MiB")

    # ---------------- 2. exclusive must never co-admit
    print("\n=== 2: exclusive racing a background job ===")
    s2 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 20000)
    res2 = contend(s2, ["exclusive", "background"], barrier=threading.Barrier(2))
    peak2 = max_concurrent(s2)
    excl_admitted = [r for r in res2 if r["cls"] == "exclusive" and r["allowed"]]
    print(f"  peak_concurrent={peak2}  exclusive_admitted={len(excl_admitted)}")
    for r in res2:
        print(f"    {'ADMITTED' if r['allowed'] else 'refused '} {r['cls']:<10} "
              f"{r['reason'][:58]}")
    report["cases"].append({"case": "exclusive_co_admit", "peak_concurrent": peak2,
                            "exclusive_admitted": len(excl_admitted)})
    check("exclusive never runs concurrently with anything", peak2 <= 1, f"peak={peak2}")
    check("no reservation leaked", s2.reserved_mb() == 0, f"reserved={s2.reserved_mb()} MiB")

    # ---------------- 3. interactive must not overlap background
    print("\n=== 3: interactive racing a background job ===")
    s3 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 20000)
    res3 = contend(s3, ["background", "interactive"], barrier=threading.Barrier(2))
    peak3 = max_concurrent(s3)
    print(f"  peak_concurrent={peak3}")
    for r in res3:
        print(f"    {'ADMITTED' if r['allowed'] else 'refused '} {r['cls']:<10} "
              f"{r['reason'][:58]}")
    report["cases"].append({"case": "interactive_overlap_background",
                            "peak_concurrent": peak3})
    # both may be admitted, but never simultaneously: replay counts simultaneous
    check("interactive and background never overlap", peak3 <= 1, f"peak={peak3}")

    # ---------------- 4. budget must not be double-spent
    print("\n=== 4: three background jobs against a budget that fits one ===")
    s4 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 2048 + 700)
    res4 = contend(s4, ["background"] * 3, barrier=threading.Barrier(3))
    admitted4 = sum(1 for r in res4 if r["allowed"])
    peak4 = max_concurrent(s4)
    print(f"  usable=700 MiB, each needs 600 MiB -> at most 1 may hold the budget")
    print(f"  admitted={admitted4}  peak_concurrent={peak4}  reserved_now={s4.reserved_mb()}")
    report["cases"].append({"case": "budget_double_spend", "admitted": admitted4,
                            "peak_concurrent": peak4, "usable_mb": 700,
                            "each_mb": 600})
    check("the budget is not spent more than once", peak4 <= 1, f"peak={peak4}")
    check("no reservation leaked after release", s4.reserved_mb() == 0,
          f"reserved={s4.reserved_mb()} MiB")

    # ---------------- 5. normal path still works (fix must not refuse everything)
    print("\n=== 5: normal path is still admitted ===")
    s5 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 20000)
    d = s5.acquire("interactive", timeout_s=5)
    check("a lone interactive job is admitted", d.allowed, d.reason[:70])
    check("an admitted job holds a reservation", s5.reserved_mb() > 0,
          f"reserved={s5.reserved_mb()} MiB")
    check("an id is returned for the reservation", bool(getattr(d, "reservation_id", None)),
          str(getattr(d, "reservation_id", None)))
    s5.release("interactive", d.reservation_id)
    check("release clears the reservation", s5.reserved_mb() == 0,
          f"reserved={s5.reserved_mb()} MiB")

    # background may still overlap background within budget
    s6 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True,
                           max_background_units=3), 20000)
    res6 = contend(s6, ["background"] * 3, hold=0.4, barrier=threading.Barrier(3))
    peak6 = max_concurrent(s6)
    print(f"\n  background/background peak_concurrent={peak6} "
          f"(3 units, ample budget)")
    check("background jobs may still overlap each other", peak6 >= 2, f"peak={peak6}")

    # ---------------- 6. double release and starvation
    print("\n=== 6: double release and exclusive starvation ===")
    s7 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 20000)
    d = s7.acquire("background", timeout_s=5)
    s7.release("background", d.reservation_id)
    s7.release("background")                    # second release: must be a no-op
    check("double release does not create memory", s7.reserved_mb() == 0,
          f"reserved={s7.reserved_mb()} MiB")
    check("double release is recorded, not silent",
          any(h["event"] == "double_release_ignored" for h in s7.history),
          str([h["event"] for h in s7.history][-4:]))
    check("active count did not go negative",
          all(v > 0 for v in s7._active.values()) and s7._active == {},
          str(s7._active))

    # an exclusive request must be REFUSED, not starved forever, against a held background
    s8 = stub(GpuScheduler(strategy="conditional", verbose=False, dry_run=True), 20000)
    held = s8.acquire("background", timeout_s=5)
    t0 = time.time()
    d8 = s8.acquire("exclusive", timeout_s=1.0)
    waited = time.time() - t0
    check("exclusive against a running background is refused or times out, not hung",
          not d8.allowed and waited < 5, f"allowed={d8.allowed} waited={waited:.2f}s")
    check("the refusal is explained", bool(d8.reason), d8.reason[:70])
    s8.release("background", held.reservation_id)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("B1 verification passed: all four defects are fixed and the normal path works")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
