"""
b1_atomicity.py - P55R-01 work package B1: admission atomicity.

The suspicion: `acquire()` calls `plan()` (which takes and RELEASES the lock) and only then,
under a *second* lock acquisition, increments `_active`. Two threads can therefore both pass
the planning stage before either registers, so the policy is evaluated against a stale view
of the world.

This is proven deterministically with a thread barrier rather than by hoping a race occurs.
The barrier forces both threads to finish planning before either reserves, which is the exact
interleaving that a real race would produce only occasionally.

No GPU work is performed and no service is started or stopped.

Usage:
  python b1_atomicity.py --out ../results/<run-id>/b1_atomicity.json
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


def make(stub_free: int = 20000) -> GpuScheduler:
    s = GpuScheduler(strategy="serial", safety_margin_mb=2048, verbose=False, dry_run=True)
    s.free_mb = lambda: stub_free        # type: ignore[method-assign]
    s.running = lambda: []               # type: ignore[method-assign]
    return s


def race(s: GpuScheduler, classes: list[str], barrier: threading.Barrier) -> list:
    """
    Force the interleaving: every thread PLANS first, waits at the barrier, then RESERVES.
    That is precisely what the current code does internally, made deterministic.
    """
    results: list = [None] * len(classes)

    def worker(i: int, cls: str) -> None:
        # step 1: plan (as acquire() does), with no reservation yet
        d = s.plan(cls)
        barrier.wait(timeout=10)          # every thread has now planned
        # step 2: reserve (as acquire() does, under a second lock acquisition)
        if d.allowed:
            with s._lock:
                s._active[cls] = s._active.get(cls, 0) + 1
        results[i] = d

    threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(classes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"harness": "thread-barrier forced TOCTOU",
                    "note": "no GPU work; no service started or stopped", "cases": []}

    # ---------------- case 1: serial double admission
    print("=== case 1: serial, two jobs race after planning ===")
    s = make()
    b = threading.Barrier(2)
    res = race(s, ["interactive", "interactive"], b)
    admitted = sum(1 for d in res if d.allowed)
    active = dict(s._active)
    print(f"  allowed by plan: {admitted}/2   _active after: {active}")
    report["cases"].append({"case": "serial_double_admit", "planned_allowed": admitted,
                            "active_after": active})
    check("serial must NOT admit two overlapping jobs", admitted <= 1,
          f"both planned as allowed: {[d.reason for d in res]}")

    # ---------------- case 2: exclusive co-admission with another class
    print("\n=== case 2: exclusive racing a background job ===")
    s2 = GpuScheduler(strategy="conditional", safety_margin_mb=2048, verbose=False,
                      dry_run=True)
    s2.free_mb = lambda: 20000           # type: ignore[method-assign]
    s2.running = lambda: []              # type: ignore[method-assign]
    b2 = threading.Barrier(2)
    res2 = race(s2, ["exclusive", "background"], b2)
    admitted2 = sum(1 for d in res2 if d.allowed)
    active2 = dict(s2._active)
    print(f"  allowed by plan: {admitted2}/2   _active after: {active2}")
    report["cases"].append({"case": "exclusive_co_admit", "planned_allowed": admitted2,
                            "active_after": active2})
    check("exclusive must NEVER be co-admitted", not ("exclusive" in active2
                                                      and len(active2) > 1),
          f"active={active2}")

    # ---------------- case 3: interactive joining background (doc contradiction)
    print("\n=== case 3: interactive racing a background job ===")
    s3 = GpuScheduler(strategy="conditional", safety_margin_mb=2048, verbose=False,
                      dry_run=True)
    s3.free_mb = lambda: 20000           # type: ignore[method-assign]
    s3.running = lambda: []              # type: ignore[method-assign]
    b3 = threading.Barrier(2)
    res3 = race(s3, ["background", "interactive"], b3)
    active3 = dict(s3._active)
    print(f"  allowed by plan: {sum(1 for d in res3 if d.allowed)}/2   "
          f"_active after: {active3}")
    report["cases"].append({"case": "interactive_join_background",
                            "planned_allowed": sum(1 for d in res3 if d.allowed),
                            "active_after": active3})
    overlap = "interactive" in active3 and "background" in active3
    check("interactive must NOT overlap background (conservative policy)", not overlap,
          f"active={active3}")

    # ---------------- case 4: memory budget double-spend
    print("\n=== case 4: two jobs sharing one memory budget ===")
    s4 = GpuScheduler(strategy="conditional", safety_margin_mb=2048, verbose=False,
                      dry_run=True)
    # exactly enough for ONE background job (600 headroom + margin) and not two
    s4.free_mb = lambda: 2048 + 700      # type: ignore[method-assign]
    s4.running = lambda: []              # type: ignore[method-assign]
    b4 = threading.Barrier(3)
    res4 = race(s4, ["background", "background", "background"], b4)
    admitted4 = sum(1 for d in res4 if d.allowed)
    print(f"  allowed by plan: {admitted4}/3   _active after: {dict(s4._active)}")
    print(f"  each needs 600 MiB headroom; usable was 700 MiB -> at most ONE fits")
    report["cases"].append({"case": "budget_double_spend", "planned_allowed": admitted4,
                            "active_after": dict(s4._active), "usable_mb": 700,
                            "each_mb": 600})
    check("one memory budget must not be spent more than once", admitted4 <= 1,
          f"{admitted4} jobs admitted against a budget that fits 1")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    if FAILS:
        print(f"REPRODUCED ({len(FAILS)} failures): {FAILS}")
        print("These are the risks B1 describes, confirmed deterministically.")
        return 1
    print("B1: no atomicity defect reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
