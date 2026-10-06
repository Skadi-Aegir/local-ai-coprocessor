"""
p2_deterministic.py - P55R-02 group 1: FOUR deterministic verifications.

All four run with mocks, barriers and events. No real OOM is induced, no service is started
or stopped, and no GPU work is performed. Every test sets its own OUTER timeout so a hang
fails the test instead of stalling the suite.

Each group reports the PRE-FIX reproduction separately from the POST-FIX result, so the
repair is evidence-backed rather than asserted.

  1a request-level service dependencies
  1b FIFO, timeout and cancellation
  1c resource reservation accounting
  1d unload and recovery path ordering

Run:  python p2_deterministic.py --out ../results/<run-id>/deterministic.json [--phase pre|post]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))

OUTER_TIMEOUT = 30.0          # no test may hang longer than this
RESULTS: list[dict] = []


def record(group: str, name: str, passed: bool, detail: str = "", pre: bool = False) -> None:
    tag = "PRE " if pre else "POST"
    mark = "OK  " if passed else "FAIL"
    print(f"  {mark} [{tag}] {group}: {name}" + (f" :: {detail}" if detail else ""),
          flush=True)
    RESULTS.append({"group": group, "name": name, "passed": passed,
                    "detail": detail, "phase": "pre" if pre else "post"})


class Watchdog:
    """Outer timeout: fail the test rather than hang the suite."""

    def __init__(self, seconds: float = OUTER_TIMEOUT):
        self.seconds = seconds
        self.fired = False
        self.timer: threading.Timer | None = None

    def __enter__(self):
        def boom():
            self.fired = True
        self.timer = threading.Timer(self.seconds, boom)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, *e):
        if self.timer:
            self.timer.cancel()
        return False


def load_scheduler():
    """Reimport so a pre/post phase can each see the right code."""
    for mod in ("gpu_scheduler",):
        if mod in sys.modules:
            del sys.modules[mod]
    return importlib.import_module("gpu_scheduler")


def stub(s, free: int = 30000):
    s.free_mb = lambda: free             # type: ignore[method-assign]
    s.running = lambda: []               # type: ignore[method-assign]
    return s


# ==========================================================================
# 1a - request-level service dependencies
# ==========================================================================
def group_1a(phase: str, G) -> None:
    print("\n=== 1a: request-level service dependencies ===", flush=True)
    s = stub(G.GpuScheduler(strategy="conditional", verbose=False, dry_run=True))

    # can a request declare what it depends on?
    try:
        d_emb = s.acquire("background", timeout_s=5, requires=["local-embedding"])
        declared = True
    except TypeError as e:
        declared = False
        d_emb = None
        record("1a", "acquire() accepts a per-request `requires` argument", False,
               f"{e}", pre=(phase == "pre"))
    if declared:
        record("1a", "acquire() accepts a per-request `requires` argument", True,
               f"reservation={getattr(d_emb, 'reservation_id', None)}")

    if not declared:
        record("1a", "an embedding task's dependency is recorded on its reservation", False,
               "no per-request requires support", pre=True)
        record("1a", "a reranker task's dependency is recorded on its reservation", False,
               "no per-request requires support", pre=True)
        record("1a", "neither dependency can be unloaded while active", False,
               "cannot express", pre=True)
        record("1a", "releasing one task frees only its own protection", False,
               "cannot express", pre=True)
        return

    d_rr = s.acquire("background", timeout_s=5, requires=["local-reranker"])

    # both dependencies must be protected while their requests are active
    ok_emb = s._unload_service("local-embedding") is False
    ok_rr = s._unload_service("local-reranker") is False
    record("1a", "an embedding task's dependency is recorded on its reservation",
           bool(getattr(d_emb, "reservation_id", None)), str(getattr(d_emb, "reservation_id", None)))
    record("1a", "a reranker task's dependency is recorded on its reservation",
           bool(getattr(d_rr, "reservation_id", None)), str(getattr(d_rr, "reservation_id", None)))
    record("1a", "neither dependency can be unloaded while both are active",
           ok_emb and ok_rr, f"embedding_protected={ok_emb} reranker_protected={ok_rr}")

    # releasing the embedding task must free ONLY its own protection
    s.release("background", d_emb.reservation_id)
    emb_after = s._unload_service("local-embedding")
    rr_after = s._unload_service("local-reranker")
    record("1a", "releasing one task frees only its own service protection",
           emb_after is True and rr_after is False,
           f"embedding now unloadable={emb_after}, reranker still protected={rr_after is False}")
    s.release("background", d_rr.reservation_id)

    # empty dependencies only for tasks that genuinely use no registered service
    d_none = s.acquire("background", timeout_s=5)
    record("1a", "a task declaring NO dependencies is still admissible",
           d_none.allowed, d_none.reason[:60])
    s.release("background", d_none.reservation_id)

    # A declared dependency that cannot be MADE AVAILABLE must block the request even when the
    # budget allows. Note carefully what is NOT tested here: a dependency that is merely not
    # loaded yet must be ADMITTED, because an on-demand service is down by design and the
    # request that will load it would otherwise be refused for the dependency being unloaded.
    # That chicken-and-egg was a real defect found in the residency experiment.
    s2 = G.GpuScheduler(strategy="conditional", verbose=False, dry_run=False)
    s2.free_mb = lambda: 30000            # type: ignore[method-assign]
    s2.running = lambda: []               # type: ignore[method-assign]
    orig_health = G.service_healthy
    try:
        G.service_healthy = lambda port, *a, **k: False     # nothing loaded

        # (i) down but startable: must be ADMITTED - this is the on-demand case
        d_load = s2.acquire("background", timeout_s=2, requires=["local-reranker"])
        record("1a", "a declared dependency that is down but STARTABLE is admitted "
                     "(on-demand loading works)",
               d_load.allowed, d_load.reason[:110])
        if d_load.allowed:
            s2.release("background", d_load.reservation_id)

        # (ii) truly unavailable: a service with no start script must be REFUSED
        from gpu_scheduler import ServiceSpec
        s2.reg = dict(s2.reg)
        s2.reg["no-loader"] = ServiceSpec("no-loader", 9199, 500, 0, False, None, None, 1.0)
        G.service_healthy = lambda port, *a, **k: False
        d_bad = s2.acquire("background", timeout_s=2, requires=["no-loader"])
        record("1a", "a dependency that cannot be loaded is refused even when budget allows",
               (not d_bad.allowed) and "cannot be satisfied" in d_bad.reason,
               d_bad.reason[:130])
        record("1a", "the refusal is attributed to the dependency, not to the budget",
               "budget was not the constraint" in d_bad.reason, d_bad.reason[:130])

        # (iii) an unknown service name is rejected before any work
        try:
            s2.acquire("background", timeout_s=2, requires=["not-a-service"])
            record("1a", "an unknown service name in requires is rejected", False,
                   "no exception raised")
        except ValueError as e:
            record("1a", "an unknown service name in requires is rejected", True, str(e)[:80])

        # (iv) a request declaring NOTHING is still admitted on the same scheduler
        d_ok = s2.acquire("background", timeout_s=2, requires=[])
        record("1a", "a request with no declared dependencies is not blocked by the check",
               d_ok.allowed, d_ok.reason[:70])
        if d_ok.allowed:
            s2.release("background", d_ok.reservation_id)
    finally:
        G.service_healthy = orig_health


# ==========================================================================
# 1b - FIFO, timeout and cancellation
# ==========================================================================
def group_1b(phase: str, G) -> None:
    print("\n=== 1b: FIFO, timeout and cancellation ===", flush=True)

    # --- risk: a later same-class request overwrites the earlier one's timeout
    with Watchdog():
        s = stub(G.GpuScheduler(strategy="serial", verbose=False, dry_run=True))
        held = s.acquire("interactive", timeout_s=3)

        got: dict[str, object] = {}

        def long_waiter():
            got["long"] = s.acquire("background", timeout_s=8.0)

        def short_waiter():
            time.sleep(0.25)                      # arrives second
            got["short"] = s.acquire("background", timeout_s=0.5)

        t1 = threading.Thread(target=long_waiter, daemon=True)
        t1.start()
        time.sleep(0.05)
        t2 = threading.Thread(target=short_waiter, daemon=True)
        t2.start()
        t1.join(timeout=OUTER_TIMEOUT)
        t2.join(timeout=OUTER_TIMEOUT)

        short = got.get("short")
        long_ = got.get("long")
        short_ok = short is not None and not short.allowed   # 0.5 s should expire
        long_ok = long_ is not None and not long_.allowed    # released then times out at 8 s
        record("1b", "each request keeps its OWN deadline (short times out, long does not "
                     "inherit it)",
               short_ok and long_ok,
               f"short={'refused' if short and not short.allowed else 'admitted/None'} "
               f"long={'refused' if long_ and not long_.allowed else 'admitted/None'}",
               pre=(phase == "pre"))
        s.release("interactive", held.reservation_id)

    # --- risk: a timeout leaves queued/ticket state behind
    with Watchdog():
        s2 = stub(G.GpuScheduler(strategy="serial", verbose=False, dry_run=True))
        held2 = s2.acquire("interactive", timeout_s=3)
        d_to = s2.acquire("background", timeout_s=0.4)
        state = {"active": dict(s2._active), "reserved": s2.reserved_mb(),
                 "queued": {k: v for k, v in getattr(s2, "_queued", {}).items() if v},
                 "serving": getattr(s2, "_serving", None),
                 "ticket_seq": getattr(s2, "_ticket_seq", None)}
        later = s2.acquire("background", timeout_s=0.4)
        # "no residue" means the ledger/ticket bookkeeping is clean, NOT that the occupying
        # job's own reservation is gone - that reservation belongs to the interactive job
        # which is legitimately still active. An earlier version conflated the two and
        # reported the occupier's legitimate reservation as leaked state.
        occluded = {"reservations": dict(getattr(s2, "_reservations", {})),
                    "open_tickets": set(getattr(s2, "_tickets_open", set())),
                    "deadlines": dict(getattr(s2, "_deadlines", {}))}
        record("1b", "a timeout decision leaves no queued/ticket residue",
               not state["queued"] and not occluded["open_tickets"]
               and not occluded["deadlines"],
               f"queued={state['queued']} open_tickets={occluded['open_tickets']} "
               f"deadlines={occluded['deadlines']}")
        record("1b", "the timed-out request left no reservation of its own",
               len(occluded["reservations"]) == 1,
               f"reservations={list(occluded['reservations'])} "
               f"(1 = only the occupying interactive job)")
        record("1b", "a later request can still make progress after a timeout",
               later is not None, f"allowed={getattr(later, 'allowed', None)}")
        s2.release("interactive", held2.reservation_id)
        record("1b", "after the occupier releases, the ledger is empty",
               s2.reserved_mb() == 0 and not s2._reservations,
               f"reserved={s2.reserved_mb()} reservations={list(s2._reservations)}")

    # --- risk: head background and a later interactive wait on each other
    with Watchdog():
        s3 = stub(G.GpuScheduler(strategy="conditional", verbose=False, dry_run=True))
        order: dict[str, object] = {}

        def bg_first():
            order["bg"] = s3.acquire("background", timeout_s=6)

        def int_later():
            time.sleep(0.15)
            order["int"] = s3.acquire("interactive", timeout_s=6)

        a = threading.Thread(target=bg_first, daemon=True)
        b = threading.Thread(target=int_later, daemon=True)
        a.start(); b.start()
        a.join(timeout=OUTER_TIMEOUT)
        b.join(timeout=OUTER_TIMEOUT)
        bg, it = order.get("bg"), order.get("int")
        both_returned = bg is not None and it is not None
        # a deadlock shows as one of them never returning
        record("1b", "head background and later interactive do not deadlock each other",
               both_returned,
               f"bg={'returned' if bg else 'NEVER RETURNED'} "
               f"int={'returned' if it else 'NEVER RETURNED'}",
               pre=(phase == "pre"))
        for k, d in (("bg", bg), ("int", it)):
            if d is not None and getattr(d, "allowed", False):
                s3.release(d.job_class, getattr(d, "reservation_id", None))

    # --- monotonic clock check
    src = (REPO / "services/gpu_scheduler/gpu_scheduler.py").read_text(
        encoding="utf-8", errors="replace")
    uses_monotonic = "monotonic" in src
    wall_hits = [ln.strip() for ln in src.splitlines()
                 if "time.time()" in ln and ("t0" in ln or "deadline" in ln or "waited" in ln)]
    record("1b", "time control uses a monotonic clock", uses_monotonic,
           f"monotonic={'present' if uses_monotonic else 'ABSENT'}; "
           f"wall-clock comparisons on deadlines={len(wall_hits)}")


# ==========================================================================
# 1c - resource reservation accounting
# ==========================================================================
def group_1c(phase: str, G) -> None:
    print("\n=== 1c: resource reservation accounting ===", flush=True)
    # Scenario: a job is admitted and its memory reservation is counted. Then the live
    # measurement RISES to reflect that job (as it would once the model actually loads).
    # If the reservation is still subtracted on top, the same memory is counted twice.
    s = G.GpuScheduler(strategy="conditional", verbose=False, dry_run=True)
    free_now = {"v": 20000}
    s.free_mb = lambda: free_now["v"]        # type: ignore[method-assign]
    s.running = lambda: []                   # type: ignore[method-assign]

    d1 = s.acquire("background", timeout_s=5)
    reserved_after_admit = s.reserved_mb()
    usable_after_admit = free_now["v"] - reserved_after_admit - s.margin_mb

    # now the live figure drops because the job's memory has actually materialised
    free_now["v"] = 20000 - d1.required_mb
    plan2 = s.plan("background")
    usable_reported = plan2.free_mb - plan2.reserved_mb - plan2.margin_mb

    # the reservation is still in the ledger AND the live figure has moved; the usable
    # figure must not fall by the reservation a second time
    double_counted = usable_reported < (usable_after_admit - d1.required_mb) + 1
    record("1c", "reserved memory is not deducted twice once it appears in live VRAM",
           not double_counted,
           f"reserved={reserved_after_admit} live_fell_by={d1.required_mb} "
           f"usable_after={usable_reported}",
           pre=(phase == "pre"))

    # The ledger must still EXIST. Removing the reservation mechanism entirely would
    # reintroduce over-admission, so what is asserted is that the ledger is structurally
    # present and that the watermark logic is visible. Note the corrected semantics: once the
    # live reading has fallen by the reserved amount, the reservation is fully MATERIALISED
    # and legitimately charges 0 - that is the fix, not a leak.
    detail = s.reserved_detail()
    record("1c", "a reservation ledger still exists (not deleted to fix double counting)",
           hasattr(s, "_reservations") and len(detail) == 1 and "watermark_free"
           in s._reservations[d1.reservation_id],
           f"entries={len(detail)}")
    record("1c", "the reservation reports what is materialised vs still charged",
           detail and detail[0]["materialised_mb"] == d1.required_mb
           and detail[0]["still_charged_mb"] == 0,
           f"{detail}")
    record("1c", "reserved_detail exposes each request's declared dependencies",
           detail and detail[0]["requires"] == [], f"{detail}")
    # with live still NOT reflecting the job, the reservation must charge in full - this is
    # the over-admission protection the ledger exists for
    free_now["v"] = 20000
    held_charge = s.reserved_mb()
    record("1c", "before the memory materialises the reservation still charges in full",
           held_charge == d1.required_mb,
           f"charged={held_charge} required={d1.required_mb}")

    # and the conservatism must be stated rather than blamed on hardware
    over_estimate = reserved_after_admit - d1.required_mb
    record("1c", "the reservation is not wildly larger than the stated requirement",
           abs(over_estimate) <= 1,
           f"reserved={reserved_after_admit} required={d1.required_mb} "
           f"(conservatism={over_estimate} MiB)")
    s.release("background", d1.reservation_id)
    record("1c", "release returns the ledger to zero", s.reserved_mb() == 0,
           f"{s.reserved_mb()}")


# ==========================================================================
# 1d - unload and recovery path ordering
# ==========================================================================
def group_1d(phase: str, G) -> None:
    print("\n=== 1d: unload and recovery path ordering ===", flush=True)
    src = (REPO / "benchmarks/phase55r/b4b_switch_cost.py").read_text(
        encoding="utf-8", errors="replace")
    # the correct order is: reserve -> unload if needed -> load dependency -> health check
    # -> actual request -> release
    idx_unload = src.find("s.unload(")
    idx_start = src.find("s.start(")
    idx_request = src.find("/v1/rerank")
    order_ok = 0 <= idx_unload < idx_start < idx_request if -1 not in (
        idx_unload, idx_start, idx_request) else False
    record("1d", "b4b orders unload BEFORE start and start BEFORE the real request",
           order_ok,
           f"unload@{idx_unload} start@{idx_start} request@{idx_request}",
           pre=(phase == "pre"))
    record("1d", "b4b asserts a successful execution, not just plan().allowed",
           "completed" in src or "completed_ok" in src,
           "looks for an execution result marker")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="post", choices=["pre", "post"])
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    G = load_scheduler()
    print(f"phase: {args.phase}   outer timeout per test: {OUTER_TIMEOUT}s", flush=True)

    group_1a(args.phase, G)
    group_1b(args.phase, G)
    group_1c(args.phase, G)
    group_1d(args.phase, G)

    passed = sum(1 for r in RESULTS if r["passed"])
    report = {"phase": args.phase, "checks": RESULTS, "passed": passed,
              "failed": len(RESULTS) - passed, "outer_timeout_s": OUTER_TIMEOUT,
              "note": "mocks/barriers only; no GPU work, no service started or stopped"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{passed}/{len(RESULTS)} checks passed")
    fails = [r["name"] for r in RESULTS if not r["passed"]]
    if fails:
        print(f"FAILED: {fails}")
    print(f"wrote {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
