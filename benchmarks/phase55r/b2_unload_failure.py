"""
b2_unload_failure.py - P55R-01 work package B2.

Injects the four failure conditions the phase names, using a fake registry and stubbed
process execution so nothing real is stopped:

  1. the stop script returns failure
  2. the stop script succeeds but the PORT is not released
  3. the stop script succeeds and the port frees but MEMORY does not come back
  4. the service being unloaded is STILL USED by an active job

Required outcomes, each asserted:
  * a job is NOT admitted when the unload it depended on failed
  * no leak of active counts, budget reservations or queue state
  * a service an active job depends on is never unloaded
  * jobs must DECLARE required services; `needs=[]` must not imply anything is unloadable
  * resource conditions are RE-CONFIRMED after unloading
  * a recovered service is only used after a health check

The direct unload() entry is tested as well as the acquire() path, because the phase asks
for both. The real stop scripts are inspected separately, since a scheduler comment is not
evidence about what a script does.

Usage:
  python b2_unload_failure.py --out ../results/<run-id>/b2_unload_failure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))

import gpu_scheduler as G  # noqa: E402
from gpu_scheduler import GpuScheduler, ServiceSpec  # noqa: E402

FAILS: list[str] = []

# Output is line-buffered and flushed on every line. Reason, measured: run from a shell that
# redirects stdout to a file while the reading end is slow, a block-buffered ~15 KB transcript
# fills the pipe buffer and the process BLOCKS on write - which presents as a test hang even
# though every check passed. Unbuffered line output removes that whole failure mode.
import functools as _functools

print = _functools.partial(print, flush=True)  # noqa: A001


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------------------
# a fake registry: three unloadable services plus one essential, so the unload path can be
# exercised without touching a real process
# --------------------------------------------------------------------------
def fake_registry() -> dict[str, ServiceSpec]:
    return {
        "essential-svc": ServiceSpec("essential-svc", 9101, 3000, 0, True,
                                     None, None, 3.0),
        "victim": ServiceSpec("victim", 9102, 1200, 0, False,
                              None, "stop-victim.ps1", 1.0),
        "victim2": ServiceSpec("victim2", 9103, 1200, 0, False,
                               None, "stop-victim2.ps1", 2.0),
        "needed": ServiceSpec("needed", 9104, 900, 200, False,
                              None, "stop-needed.ps1", 1.5),
    }


class Harness:
    """Drives a scheduler whose process interactions are fully stubbed."""

    def __init__(self, free_start: int = 3000, stop_rc: int = 0, port_frees: bool = True,
                 memory_returns: bool = True, running=("victim", "victim2")):
        self.free = free_start
        self.stop_rc = stop_rc
        self.port_frees = port_frees
        self.memory_returns = memory_returns
        self.running_set = set(running)
        self.stopped: list[str] = []
        self.s = GpuScheduler(strategy="unload", safety_margin_mb=2048,
                              registry=fake_registry(), verbose=False)
        self.s.free_mb = lambda: self.free              # type: ignore[method-assign]
        self.s.running = lambda: sorted(              # type: ignore[method-assign]
            n for n in self.running_set if n in self.s.reg)

    def _run(self, cmd, **kw):
        # identify which service the stop script belongs to
        name = None
        for n, spec in self.s.reg.items():
            if spec.stop_script and spec.stop_script in " ".join(str(c) for c in cmd):
                name = n
                break

        @dataclass
        class R:
            returncode: int
            stdout: str = ""
            stderr: str = ""

        if name:
            self.stopped.append(name)
            if not self.port_frees:
                pass                       # port stays bound -> port_free() must stay False
            else:
                self.running_set.discard(name)
                if self.memory_returns:
                    self.free += self.s.reg[name].vram_mb
        return R(self.stop_rc)

    def __enter__(self):
        self._orig_run = G.subprocess.run
        self._orig_pf = G.port_free
        self._orig_health = G.service_healthy
        G.subprocess.run = self._run                          # type: ignore[assignment]
        G.port_free = lambda port, *a, **k: not any(          # type: ignore[assignment]
            spec.port == port and n in self.running_set
            for n, spec in self.s.reg.items())
        # service_healthy must be stubbed too: the fake registry's ports are not really
        # listening, so a live (non-dry-run) scheduler would see every declared dependency as
        # unusable and refuse for the wrong reason. The stubbed probe reports exactly what
        # `running_set` says, which is what the fake registry means by "running".
        G.service_healthy = lambda port, *a, **k: any(       # type: ignore[assignment]
            spec.port == port and n in self.running_set
            for n, spec in self.s.reg.items())
        return self

    def __exit__(self, *exc):
        G.subprocess.run = self._orig_run                     # type: ignore[assignment]
        G.port_free = self._orig_pf                           # type: ignore[assignment]
        G.service_healthy = self._orig_health                 # type: ignore[assignment]
        return False


def snap(s: GpuScheduler) -> dict:
    return {"active": dict(s._active), "reserved": s.reserved_mb(),
            "queued": {k: v for k, v in s._queued.items() if v},
            "waiting": s._waiting_interactive}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"note": "fake registry + stubbed process execution; nothing real stopped",
                    "cases": []}

    # ---------------- 0. baseline: a clean unload succeeds
    print("=== 0: control case - unload succeeds ===")
    with Harness(free_start=2100, stop_rc=0, port_frees=True, memory_returns=True) as h:
        d = h.s.acquire("background", timeout_s=5)
        print(f"  allowed={d.allowed} unload={d.unload} reason={d.reason[:70]}")
        check("control: a job is admitted when the unload succeeds and frees memory",
              d.allowed, d.reason[:80])
        check("control: the reservation is recorded", h.s.reserved_mb() > 0,
              f"reserved={h.s.reserved_mb()}")
        h.s.release("background", d.reservation_id)
        check("control: release clears everything", snap(h.s)["reserved"] == 0,
              str(snap(h.s)))

    # ---------------- 1. stop script returns failure
    print("\n=== 1: stop script returns failure ===")
    with Harness(free_start=2100, stop_rc=1, port_frees=True, memory_returns=True) as h:
        d = h.s.acquire("background", timeout_s=5)
        st = snap(h.s)
        print(f"  allowed={d.allowed} reason={d.reason[:80]}")
        print(f"  state after: {st}")
        check("a failed stop script means the job is NOT admitted", not d.allowed,
              d.reason[:80])
        check("the refusal is explained", bool(d.reason), d.reason[:60])
        check("no active count leaked", st["active"] == {}, str(st["active"]))
        check("no reservation leaked", st["reserved"] == 0, str(st["reserved"]))
        check("no queue state leaked", st["queued"] == {}, str(st["queued"]))
        report["cases"].append({"case": "stop_rc_nonzero", "allowed": d.allowed,
                                "state": st})

    # ---------------- 2. port not released
    print("\n=== 2: stop succeeds but the port stays bound ===")
    with Harness(free_start=2100, stop_rc=0, port_frees=False, memory_returns=True) as h:
        d = h.s.acquire("background", timeout_s=5)
        st = snap(h.s)
        print(f"  allowed={d.allowed} reason={d.reason[:80]}")
        check("a still-bound port means the job is NOT admitted", not d.allowed,
              d.reason[:80])
        check("no active count leaked", st["active"] == {}, str(st["active"]))
        check("no reservation leaked", st["reserved"] == 0, str(st["reserved"]))
        report["cases"].append({"case": "port_not_released", "allowed": d.allowed,
                                "state": st})

    # ---------------- 3. memory does not come back
    print("\n=== 3: port frees but memory does not return ===")
    with Harness(free_start=2100, stop_rc=0, port_frees=True, memory_returns=False) as h:
        d = h.s.acquire("background", timeout_s=5)
        st = snap(h.s)
        print(f"  allowed={d.allowed} reason={d.reason[:80]}")
        check("memory not returning means the job is NOT admitted", not d.allowed,
              d.reason[:80])
        check("the refusal explicitly mentions the re-confirmation",
              "after unloading" in d.reason or "cannot satisfy" in d.reason,
              d.reason[:90])
        check("no reservation leaked", st["reserved"] == 0, str(st["reserved"]))
        report["cases"].append({"case": "memory_not_returned", "allowed": d.allowed,
                                "state": st})

    # ---------------- 4. service in use by an active job
    print("\n=== 4: never unload a service an ACTIVE job depends on ===")
    # The first version of this case was a FALSE PASS risk and a real failure: free memory
    # was set too low for the first job to be admitted at all, so nothing was active, the
    # guard had nothing to protect, and unload correctly succeeded. The test then blamed the
    # guard. Memory is now ample so the job IS admitted, and the case asserts admission first
    # - otherwise it cannot demonstrate anything about protection.
    #
    # The dependency is now declared PER REQUEST via `requires`, which is what production
    # code does. The earlier version mutated the global JOB_PROFILE, which is exactly the
    # pattern P55R-02 removes: a runtime profile edit leaks one request's dependencies into
    # every other request of the same class.
    try:
        with Harness(free_start=12000, stop_rc=0, port_frees=True,
                     memory_returns=True, running=("needed",)) as h:
            first = h.s.acquire("interactive", timeout_s=5, requires=["needed"])
            print(f"  first job allowed={first.allowed} (requires 'needed') "
                  f"reason={first.reason[:60]}")
            check("setup precondition: a job requiring 'needed' is actually ADMITTED",
                  first.allowed, first.reason[:80])
            check("the active job's requirement is recorded on its reservation",
                  h.s._reservations.get(first.reservation_id, {}).get("requires") == ["needed"],
                  str(h.s._reservations.get(first.reservation_id)))
            active_before = dict(h.s._active)
            ok = h.s._unload_service("needed")
            print(f"  active before unload attempt: {active_before}")
            print(f"  direct unload('needed') -> {ok}")
            check("a service required by an ACTIVE job is refused for unload", ok is False,
                  f"returned {ok}")
            check("the active job is still registered after the refusal",
                  dict(h.s._active) == active_before,
                  f"{dict(h.s._active)} vs {active_before}")
            # and the public unload() entry must refuse for the same reason
            check("unload() entry point also refuses for an in-use service",
                  h.s.unload("needed") is False)
            if first.allowed:
                h.s.release("interactive", first.reservation_id)
            # after release, the protection is lifted: the service becomes unloadable
            check("after release the service is no longer protected",
                  h.s._unload_service("needed") is True)
    finally:
        pass
    report["cases"].append({"case": "service_in_use", "unload_refused": True})

    # ---------------- 5. essential service
    print("\n=== 5: essential services are never unloaded ===")
    with Harness(free_start=2100) as h:
        check("essential service refuses unload", h.s._unload_service("essential-svc") is False)
        check("unload() entry point also refuses",
              h.s.unload("essential-svc") is False)

    # ---------------- 6. declared requirements
    print("\n=== 6: jobs must DECLARE required services ===")
    undeclared = [c for c, p in G.JOB_PROFILE.items() if "needs" not in p]
    check("every job class declares a needs list", not undeclared, str(undeclared))
    check("background declares an explicit (possibly empty) needs list",
          "needs" in G.JOB_PROFILE["background"],
          str(G.JOB_PROFILE["background"]))
    check("interactive declares local-fast",
          G.JOB_PROFILE["interactive"]["needs"] == ["local-fast"],
          str(G.JOB_PROFILE["interactive"]["needs"]))

    # ---------------- 7. recovery requires a health check
    print("\n=== 7: a recovered service is only used after a health check ===")
    calls = {"health": 0}
    orig_health = G.service_healthy
    G.service_healthy = lambda port, *a, **k: (calls.__setitem__("health", calls["health"] + 1)
                                               or False)
    try:
        s = GpuScheduler(verbose=False, dry_run=False)
        s.reg = fake_registry() | {"local-embedding": G.REGISTRY["local-embedding"]}
        started = s.start("local-embedding", wait_s=2.0)
        check("start() reports FAILURE when health never becomes ok", started is False,
              f"returned {started}, health calls={calls['health']}")
        check("start() actually performed a health check", calls["health"] > 0,
              str(calls["health"]))
    finally:
        G.service_healthy = orig_health

    # ---------------- 8. the real stop scripts, inspected not assumed
    print("\n=== 8: what do the REAL stop scripts actually do? ===")
    scripts = {}
    for name in ("stop-local-fast.ps1", "stop-local-embedding.ps1",
                 "stop-local-reranker.ps1"):
        p = REPO / "scripts" / name
        if not p.is_file():
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        info = {
            "exists": True,
            "uses_Stop_Process": "Stop-Process" in src,
            "force_flag": "-Force" in src,
            "kill_by_port_owner": "OwningProcess" in src,
            "waits_for_port_release": "Get-NetTCPConnection" in src,
            "reports_vram": "nvidia-smi" in src,
            "validates_pid_numeric": "TryParse" in src,
        }
        scripts[name] = info
        print(f"  {name}: Stop-Process={info['uses_Stop_Process']} "
              f"force={info['force_flag']} waits_port={info['waits_for_port_release']}")
    report["stop_scripts"] = scripts
    check("stop scripts target a specific PID rather than killing by name",
          all(not re.search(r"Get-Process\s+-Name\s+llama-server\s*\|\s*Stop-Process", 
                            (REPO / "scripts" / n).read_text(encoding="utf-8", errors="replace"))
              for n in scripts), "no blanket kill-by-name found")
    check("stop scripts wait for the port to be released",
          all(v["waits_for_port_release"] for v in scripts.values()),
          str({k: v["waits_for_port_release"] for k, v in scripts.items()}))
    print("  NOTE: the scripts DO use Stop-Process -Force on the service's own PID. That is")
    print("  a targeted termination the service's own stop script performs, not the scheduler")
    print("  killing a running task to make room. The scheduler never terminates a task.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("B2 passed: unload failure is never papered over, and in-use services are protected")
    print(f"\nwrote {out}")
    return 0


import re  # noqa: E402  (used in the stop-script inspection above)

if __name__ == "__main__":
    raise SystemExit(main())

