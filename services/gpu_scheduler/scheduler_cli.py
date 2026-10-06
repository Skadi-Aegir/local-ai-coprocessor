"""
scheduler_cli.py - command line surface for the GPU scheduler prototype.

Separated from gpu_scheduler.py so the policy module stays importable and testable without
a CLI, and so `--selftest` can exercise the POLICY (pure logic, no services touched)
independently of the LIVE mode, which genuinely starts and stops services.

  python scheduler_cli.py --status
  python scheduler_cli.py --plan interactive
  python scheduler_cli.py --selftest          # policy only, no services touched
  python scheduler_cli.py --live-demo         # starts/stops services for real
  python scheduler_cli.py --compare           # fixed task set across strategies
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gpu_scheduler import (JOB_PROFILE, REGISTRY, GpuScheduler, ServiceSpec,  # noqa: E402
                           gpu_used_mb, service_healthy)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------------------
# policy self-test: a fake registry with synthetic memory so the LOGIC is tested
# deterministically, with no services started and no GPU state involved.
# --------------------------------------------------------------------------
def selftest() -> int:
    print("=== GPU scheduler policy self-test (no services touched) ===\n")

    # ---- 1. serial refuses overlap
    #
    # Memory is stubbed for EVERY policy check in this file. A policy self-test must be
    # deterministic: run with three models resident the machine has ~2 GB free, the margin
    # eats it, and "a lone job is allowed" flips to a memory refusal. That is the GPU being
    # busy, not the policy being wrong, and a regression suite that changes its answer
    # depending on unrelated load is worse than no suite. Memory behaviour has its own
    # dedicated checks further down, which deliberately use real-looking shortfalls.
    print("--- strategy: serial ---")
    s = GpuScheduler(strategy="serial", safety_margin_mb=2048, verbose=False, dry_run=True)
    s.free_mb = lambda: 20000          # type: ignore[method-assign]
    s.running = lambda: []             # type: ignore[method-assign]
    s._active = {"interactive": 1}
    d = s.plan("interactive")
    check("serial: a second job may not run alongside an active one", not d.allowed,
          d.reason[:70])
    s._active = {}
    check("serial: a lone job is allowed", s.plan("background").allowed)

    # ---- 2. conditional lets background overlap background
    #
    # These assert the OVERLAP POLICY, so free memory is stubbed to a large value. Testing
    # them against real free memory conflated two different questions - "is this overlap
    # permitted?" and "does it fit?" - and on a machine with ~3.5 GiB free every case was
    # refused for memory, which looked like a policy bug but was a saturated GPU.
    print("\n--- strategy: conditional (overlap policy, memory stubbed) ---")
    c = GpuScheduler(strategy="conditional", safety_margin_mb=2048, verbose=False,
                     dry_run=True)
    c.free_mb = lambda: 20000          # type: ignore[method-assign]
    c.running = lambda: []             # type: ignore[method-assign]
    c._active = {"background": 1}
    check("conditional: background may join background", c.plan("background").allowed)
    c._active = {"interactive": 1}
    check("conditional: background may NOT join an active interactive job "
          "(B3 unification: the old policy allowed exactly this)",
          not c.plan("background").allowed)
    c._active = {"background": 1}
    d = c.plan("interactive")
    check("conditional: interactive may NOT join a background job", not d.allowed,
          d.reason[:70])
    c._active = {"interactive": 1}
    check("conditional: interactive may NOT join another interactive job "
          "(it runs alone)", not c.plan("interactive").allowed)
    # the OLD policy allowed background to join an interactive job and allowed two
    # interactives to share. Both are now refused, which is the B3 unification: the old
    # documentation claimed interactive does not share with background while permitting
    # exactly that in the other direction.
    check("conditional: background may NOT join an active interactive job",
          not c.plan("background").allowed)
    c._active = {"background": 2}
    check("conditional: background may still join background",
          c.plan("background").allowed)

    # and the reverse: serial must refuse EVERY overlap, memory permitting
    print("\n--- strategy: serial (overlap policy, memory stubbed) ---")
    s2 = GpuScheduler(strategy="serial", safety_margin_mb=2048, verbose=False, dry_run=True)
    s2.free_mb = lambda: 20000         # type: ignore[method-assign]
    s2.running = lambda: []            # type: ignore[method-assign]
    for active in ({"background": 1}, {"interactive": 1}):
        s2._active = dict(active)
        check(f"serial refuses overlap with {sorted(active)}",
              not s2.plan("background").allowed)
    s2._active = {}
    check("serial allows a lone job", s2.plan("background").allowed)

    # ---- 3. exclusive never shares
    print("\n--- exclusive jobs (memory stubbed) ---")
    e = GpuScheduler(strategy="conditional", verbose=False, dry_run=True)
    e.free_mb = lambda: 20000          # type: ignore[method-assign]
    e.running = lambda: []             # type: ignore[method-assign]
    for active in ({"interactive": 1}, {"background": 1}, {"exclusive": 1}):
        e._active = dict(active)
        d = e.plan("exclusive")
        check(f"exclusive refuses to share with {sorted(active)}", not d.allowed,
              d.reason[:60])
    e._active = {}
    check("exclusive is allowed when the GPU is idle", e.plan("exclusive").allowed)

    # ---- 4. essential services are never unloaded
    print("\n--- unload safety ---")
    u = GpuScheduler(strategy="unload", verbose=False, dry_run=True)
    check("local-fast is registered as essential", REGISTRY["local-fast"].essential)
    check("embedding and reranker are unloadable",
          not REGISTRY["local-embedding"].essential
          and not REGISTRY["local-reranker"].essential)
    check("Qwen-Image is NOT in the registry (cannot be unloaded by this prototype)",
          not any("qwen" in n.lower() and "image" in n.lower() for n in REGISTRY))
    check("an essential service is refused by _unload_service",
          u._unload_service("local-fast") is False)

    # ---- 5. the safety margin is enforced even when memory looks available
    print("\n--- safety margin enforcement ---")
    fake_free = 3000            # 3000 free, margin 2048 -> 952 usable
    m = GpuScheduler(strategy="conditional", safety_margin_mb=2048, verbose=False,
                     dry_run=True)
    m.free_mb = lambda: fake_free          # type: ignore[method-assign]
    m.running = lambda: []                 # type: ignore[method-assign]
    d = m.plan("interactive")              # needs 2981 + 900 = 3881
    check("a request larger than the usable margin is refused", not d.allowed,
          d.reason[:80])
    check("the refusal names the margin", "margin" in d.reason.lower())
    m.free_mb = lambda: 9000               # type: ignore[method-assign]
    check("the same request is allowed with more free memory",
          m.plan("interactive").allowed)

    # ---- 6. unload plan picks non-essential services, cheapest first
    print("\n--- unload planning ---")
    p = GpuScheduler(strategy="unload", safety_margin_mb=2048, verbose=False,
                     dry_run=True)
    p.free_mb = lambda: 2200               # type: ignore[method-assign]
    p.running = lambda: ["local-embedding", "local-reranker"]   # type: ignore
    d = p.plan("background")               # needs 600 headroom
    check("unload is proposed when memory is short", d.allowed or "cannot satisfy"
          in d.reason, d.reason[:70])
    if d.unload:
        check("unload candidates are non-essential only",
              all(not REGISTRY[n].essential for n in d.unload), str(d.unload))
        loads = [REGISTRY[n].load_s for n in d.unload]
        check("cheapest-to-reload service is chosen first", loads == sorted(loads),
              str(d.unload))

    # ---- 7. refusal is explicit, not a silent continue
    print("\n--- refusal behaviour ---")
    r = GpuScheduler(strategy="unload", safety_margin_mb=2048, verbose=False, dry_run=True)
    r.free_mb = lambda: 2100               # nothing can help: margin eats it all
    r.running = lambda: []                 # type: ignore[method-assign]
    d = r.plan("exclusive")                # needs 2000
    check("an unsatisfiable request is REFUSED", not d.allowed, d.reason[:70])
    check("the refusal explains it will not keep loading",
          "REFUSING" in d.reason or "cannot" in d.reason.lower(), d.reason[:70])

    # ---- 8. priority ordering under contention
    print("\n--- priority: interactive before background ---")
    order = []
    pr = GpuScheduler(strategy="serial", safety_margin_mb=2048, verbose=False, dry_run=True)
    pr.free_mb = lambda: 12000             # type: ignore[method-assign]

    def worker(cls, delay):
        time.sleep(delay)
        d = pr.acquire(cls, timeout_s=5)
        if d.allowed:
            order.append(cls)
            time.sleep(0.3)
            pr.release(cls)

    # a background job grabs the GPU first, then an interactive request arrives while a
    # second background job is queued behind it
    ts = [threading.Thread(target=worker, args=("background", 0.0)),
          threading.Thread(target=worker, args=("background", 0.05)),
          threading.Thread(target=worker, args=("interactive", 0.10))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    check("serial strategy ran jobs one at a time", len(order) <= 3, str(order))

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print(f"scheduler self-test passed ({len(FAILS)} failures)")
    return 0


# --------------------------------------------------------------------------
# live comparison on a fixed task set
# --------------------------------------------------------------------------
TASK_SET = [
    ("interactive", "answer one question with the fast worker"),
    ("background", "embed a batch of chunks"),
    ("background", "rerank a candidate pool"),
    ("interactive", "answer a second question"),
]


def compare(strategies=("serial", "conditional", "unload")) -> int:
    """
    Run the SAME fixed task set under each strategy and report wall time and refusals.

    Each task is a real GPU operation, so this measures the scheduling policy rather than
    a simulation of it.
    """
    import urllib.request

    def run_task(cls: str, desc: str) -> None:
        if cls == "interactive":
            payload = {"model": "local-fast",
                       "messages": [{"role": "user", "content": "Name two index types."}],
                       "max_tokens": 32, "temperature": 0.0,
                       "chat_template_kwargs": {"enable_thinking": False}}
            url = "http://127.0.0.1:8111/v1/chat/completions"
        elif "embed" in desc:
            payload = {"model": "local-embedding",
                       "input": [f"chunk {i}" for i in range(16)]}
            url = "http://127.0.0.1:8113/v1/embeddings"
        else:
            payload = {"query": "how does retrieval work",
                       "documents": [f"doc {i} about retrieval and ranking" for i in range(10)],
                       "top_n": 5, "instruction_type": "general"}
            url = "http://127.0.0.1:8114/v1/rerank"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()

    results = {}
    for strat in strategies:
        s = GpuScheduler(strategy=strat, safety_margin_mb=2048, verbose=False)
        print(f"\n=== strategy: {strat} ===")
        t0 = time.time()
        allowed = refused = 0

        def worker(cls, desc, out):
            d = s.acquire(cls, timeout_s=60)
            if not d.allowed:
                out.append(("refused", cls, desc, d.reason))
                return
            try:
                t = time.time()
                run_task(cls, desc)
                out.append(("ok", cls, desc, round(time.time() - t, 3)))
            finally:
                s.release(cls)

        out: list = []
        threads = [threading.Thread(target=worker, args=(c, d, out)) for c, d in TASK_SET]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=180)
        wall = time.time() - t0

        for status, cls, desc, detail in out:
            if status == "refused":
                refused += 1
                print(f"  REFUSED {cls:<12} {desc}: {str(detail)[:60]}")
            else:
                allowed += 1
                print(f"  ok      {cls:<12} {desc} ({detail}s)")
        results[strat] = {"wall_s": round(wall, 3), "completed": allowed,
                          "refused": refused,
                          "history_events": len(s.history)}
        print(f"  -> wall {wall:.2f}s, {allowed} completed, {refused} refused")

    print("\n" + "=" * 66)
    print(f"{'strategy':<14} {'wall s':>9} {'completed':>10} {'refused':>9}")
    print("=" * 66)
    for k, v in results.items():
        print(f"{k:<14} {v['wall_s']:>9.2f} {v['completed']:>10} {v['refused']:>9}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--plan", default=None, choices=sorted(JOB_PROFILE))
    ap.add_argument("--strategy", default="conditional",
                    choices=["serial", "conditional", "unload"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.compare:
        return compare()

    s = GpuScheduler(strategy=args.strategy, safety_margin_mb=2048, verbose=False)
    if args.plan:
        d = s.plan(args.plan)
        obj = d.to_dict()
        if args.json:
            print(json.dumps(obj, indent=1))
        else:
            print(f"job_class : {obj['job_class']}")
            print(f"allowed   : {obj['allowed']}")
            print(f"reason    : {obj['reason']}")
            print(f"free      : {obj['free_mb']} MiB   margin {obj['margin_mb']} MiB   "
                  f"required {obj['required_mb']} MiB")
            if obj["unload"]:
                print(f"unload    : {obj['unload']}")
        return 0 if d.allowed else 4

    st = s.status()
    print(json.dumps(st, indent=1) if args.json else "\n".join(
        f"{k:<24} {v}" for k, v in st.items() if k != "registry"))
    if not args.json:
        print("\nregistry:")
        for n, v in st["registry"].items():
            print(f"  {n:<17} port {v['port']}  {v['vram_mb']:>5} MiB "
                  f"(+{v['peak_extra_mb']} peak)  essential={v['essential']}  "
                  f"running={v['running']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
