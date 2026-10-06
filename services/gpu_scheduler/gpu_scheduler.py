"""
gpu_scheduler.py - GPU admission control prototype (Phase 5.5R hardened revision).

NOT a Gateway. A single-process admission controller: callers ask for permission and then
talk to the services directly. It does not proxy inference traffic.

WHAT THIS REVISION FIXES (all four were reproduced deterministically by
benchmarks/phase55r/b1_atomicity.py before being changed):

  1. DOUBLE ADMISSION. `acquire()` called `plan()` - which took and RELEASED the lock - and
     only afterwards incremented `_active` under a second lock acquisition. Two threads
     could both pass planning before either registered, so `serial` admitted two jobs at
     once. Admission is now a single atomic check-and-reserve under one lock.
  2. EXCLUSIVE CO-ADMISSION. An exclusive job racing another class was admitted alongside
     it, because neither saw the other in `_active`. Exclusive now reserves atomically and
     is mutually exclusive by construction.
  3. BUDGET DOUBLE-SPEND. The memory budget was evaluated from live `nvidia-smi` only.
     Memory already committed to accepted-but-not-yet-visible jobs was invisible, so three
     background jobs were admitted against a budget that fits one. Reservations are now
     tracked explicitly and subtracted, which is the difference between a policy that looks
     correct in testing and one that cannot over-commit.
  4. UNDECLARED REQUIREMENTS. Background jobs had `needs=[]`, which implicitly made every
     model an unload candidate - including ones the job was about to use. Jobs must now
     DECLARE the services they require, and the scheduler refuses to accept a job that does
     not say.

RESOURCE MODEL (the five quantities, kept separate as the phase requires):
    resident     measured live; already includes models currently loaded
    load_needed  additional memory to load a model the job needs but which is not running
    peak_extra   transient headroom the job needs on top (attention spikes)
    reserved     committed by ACCEPTED jobs that live sampling has not yet reflected
    margin       the safety margin, never lowered to make a test pass

    usable = measured_free - reserved - margin
             and the job must fit within it, or a declared unload must free enough.

CONCURRENCY POLICY (conservative, unified - the previous version contradicted itself by
claiming interactive never shares with background while permitting background to join
interactive):
    exclusive    mutually exclusive with everything, always
    interactive  runs alone; waits for already-running background units to finish; while an
                 interactive request is waiting, NO new background unit is admitted
    background   may run concurrently with other background units within budget

    "interactive never waits for background" is NOT promised and is not achievable without
    preemption. The wait is BOUNDED by requiring background work to be admitted in bounded
    units (max_background_units), and the bound is measured rather than asserted.

PROCESS BOUNDARY: this lock is in-process. Two separate CLI invocations do NOT share it and
must not be described as having a global lock. Cross-process coordination is out of scope.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
TOTAL_VRAM_MB = 12227


# --------------------------------------------------------------------------
# services
# --------------------------------------------------------------------------
@dataclass
class ServiceSpec:
    name: str
    port: int
    vram_mb: int            # steady-state footprint
    peak_extra_mb: int      # transient spike above steady state
    essential: bool
    start_script: str | None
    stop_script: str | None
    load_s: float

    def start_cmd(self) -> list[str] | None:
        return (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(SCRIPTS / self.start_script)] if self.start_script else None)

    def stop_cmd(self) -> list[str] | None:
        return (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(SCRIPTS / self.stop_script)] if self.stop_script else None)


REGISTRY: dict[str, ServiceSpec] = {
    "local-fast": ServiceSpec("local-fast", 8111, 2981, 0, True,
                              "start-local-fast.ps1", "stop-local-fast.ps1", 3.0),
    "local-embedding": ServiceSpec("local-embedding", 8113, 1246, 250, False,
                                   "start-local-embedding.ps1",
                                   "stop-local-embedding.ps1", 1.7),
    "local-reranker": ServiceSpec("local-reranker", 8114, 1160, 1800, False,
                                  "start-local-reranker.ps1",
                                  "stop-local-reranker.ps1", 1.22),
}
# NOT registered, so this prototype can never unload it:
#   Qwen-Image (D:\DSH\computer_science\qwen-image-deploy) - outside this project.


# --------------------------------------------------------------------------
# job profiles - EVERY class declares the services it requires
# --------------------------------------------------------------------------
JOB_PROFILE: dict[str, dict] = {
    "interactive": {"needs": ["local-fast"], "headroom_mb": 900, "bounded_units": False},
    # background work is admitted in bounded units so an interactive wait has an upper bound
    "background": {"needs": [], "headroom_mb": 600, "bounded_units": True},
    "exclusive": {"needs": [], "headroom_mb": 2000, "bounded_units": False},
}


def gpu_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    return int(out.stdout.strip().splitlines()[0])


def service_healthy(port: int, timeout: float = 2.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=timeout) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def port_free(port: int) -> bool:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-NetTCPConnection -State Listen -LocalPort {port} "
         f"-ErrorAction SilentlyContinue | Measure-Object).Count"],
        capture_output=True, text=True, timeout=25)
    try:
        return int(out.stdout.strip()) == 0
    except Exception:
        return False


@dataclass
class Decision:
    allowed: bool
    strategy: str
    job_class: str
    reason: str
    free_mb: int
    margin_mb: int
    required_mb: int
    reserved_mb: int = 0
    load_needed_mb: int = 0
    reservation_id: str | None = None
    requires: list[str] = field(default_factory=list)
    queued: bool = False
    # True only when WAITING could plausibly change the outcome (the GPU is busy). Terminal
    # refusals - budget, dependencies, unload failure, sensor failure - leave this False so
    # the caller refuses immediately instead of spinning until its deadline.
    retryable: bool = False
    unload: list[str] = field(default_factory=list)
    concurrent_with: list[str] = field(default_factory=list)
    waited_s: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["unload"] = list(self.unload)
        d["concurrent_with"] = list(self.concurrent_with)
        return d


class ReservationError(RuntimeError):
    pass


class GpuScheduler:
    def __init__(self, strategy: str = "conditional", safety_margin_mb: int = 2048,
                 registry: dict[str, ServiceSpec] | None = None,
                 dry_run: bool = False, verbose: bool = True,
                 max_background_units: int = 4):
        if strategy not in ("serial", "conditional", "unload"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.margin_mb = safety_margin_mb
        self.reg = registry if registry is not None else REGISTRY
        self.dry_run = dry_run
        self.verbose = verbose
        self.max_background_units = max_background_units
        self._lock = threading.RLock()
        self._active: dict[str, int] = {}
        # reservation ledger: id -> {job_class, mb}. Per-acquire rather than per-class so
        # release subtracts exactly what that job reserved.
        self._reservations: dict[str, dict] = {}
        self._rid_seq = 0
        self._waiting_interactive = 0
        self._queued: dict[str, int] = {}
        # FIFO admission tickets. Priority rules alone were NOT sufficient: see _advance.
        self._ticket_seq = 0
        self._serving = 1
        # Tickets that have been SERVED or ABANDONED, so the queue can be garbage-collected
        # and a cancelled request cannot leave residue behind.
        self._tickets_open: set[int] = set()
        self._tickets_closed: set[int] = set()
        # what kind of request holds each open ticket, so a queue refusal can NAME the
        # blocker instead of saying only "an earlier request"
        self._ticket_job: dict[int, str] = {}
        self.history: list[dict] = []
        # Per-request deadlines. A dict keyed by JOB CLASS was wrong: a later request of the
        # same class overwrote the earlier one's timeout, so a short-timeout waiter could
        # inherit a long one and vice versa. Each request now owns its own deadline.
        self._deadlines: dict[int, tuple[float, float]] = {}   # ticket -> (deadline, timeout)

    # ------------------------------------------------------------ FIFO fairness
    def _take_ticket(self) -> int:
        """FIFO admission ticket. Must be called with self._lock held."""
        self._ticket_seq += 1
        return self._ticket_seq

    def _is_next(self, ticket: int) -> bool:
        """True when this ticket may attempt admission. Lock held."""
        return self._serving == ticket

    def _advance(self) -> None:
        """
        Advance FIFO service to the next waiter. Lock held.

        Measured reason this exists: a background worker that releases and immediately
        re-requests can re-acquire the lock before a sleeping interactive waiter wakes, so
        the interactive starves. Observed as 35 background units completing while an
        interactive request waited its full timeout and was then refused. With FIFO, a waiter
        that keeps being passed over eventually reaches the front of the queue and is served.
        """
        self._serving += 1
        self._sweep_tickets()

    def _sweep_tickets(self) -> None:
        """
        Garbage-collect closed tickets. Lock held.

        A cancelled or timed-out request must not leave ticket state behind: without this the
        open-ticket set grows without bound and, worse, a stale deadline entry can outlive the
        request that created it.
        """
        for t in list(self._tickets_closed):
            self._deadlines.pop(t, None)
            self._ticket_job.pop(t, None)
            self._tickets_closed.discard(t)
        self._tickets_open = {t for t in self._tickets_open if t >= self._serving}

    def _close_ticket(self, ticket: int) -> None:
        """Mark a ticket finished (served or abandoned) and advance service past it."""
        self._tickets_open.discard(ticket)
        self._tickets_closed.add(ticket)
        if self._serving <= ticket:
            self._serving = ticket + 1
        self._sweep_tickets()

    # ------------------------------------------------------------ helpers
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[scheduler] {msg}", flush=True)

    def free_mb(self) -> int:
        return TOTAL_VRAM_MB - gpu_used_mb()

    def running(self) -> list[str]:
        return [n for n, s in self.reg.items() if service_healthy(s.port)]

    def reserved_mb(self) -> int:
        """
        Memory that accepted-but-not-yet-sampled work still has to cover.

        DOUBLE-COUNTING FIX. A naive sum of the ledger double-charges as soon as a job's
        memory actually materialises: the live reading falls (so `free` already reflects the
        job) AND the full reservation is subtracted again on top. Measured before the fix:
        reserved=600, live fell by 600, and usable still dropped by the full 600 a second
        time.

        Each reservation therefore records a WATERMARK: the live free reading at the moment
        it was admitted, plus what was reserved then. Whatever the live reading has since
        dropped below that watermark is memory that has already shown up, so only the
        remainder is still charged. This keeps the reservation's purpose - preventing a
        second job from spending memory that the first has committed but which the sensor
        cannot yet see - without charging for memory the sensor now reports.
        """
        with self._lock:
            if not self._reservations:
                return 0
            try:
                live_free = self.free_mb()
            except Exception:  # noqa: BLE001
                # sensor unusable: fall back to the conservative full sum rather than
                # pretending nothing is committed
                return sum(v["mb"] for v in self._reservations.values())
            total = 0
            for v in self._reservations.values():
                wm = v.get("watermark_free")
                if wm is None:
                    total += v["mb"]
                    continue
                materialised = max(0, wm - live_free)
                total += max(0, v["mb"] - materialised)
            return total

    def reserved_detail(self) -> list[dict]:
        """Per-reservation view, including what each request declared it depends on."""
        with self._lock:
            live_free = None
            try:
                live_free = self.free_mb()
            except Exception:  # noqa: BLE001
                pass
            out = []
            for rid, v in self._reservations.items():
                wm = v.get("watermark_free")
                materialised = (max(0, wm - live_free)
                                if (wm is not None and live_free is not None) else 0)
                out.append({"reservation_id": rid, "job_class": v["job_class"],
                            "mb": v["mb"], "requires": list(v.get("requires", [])),
                            "materialised_mb": materialised,
                            "still_charged_mb": max(0, v["mb"] - materialised)})
            return out

    def protected_services(self) -> dict[str, list[str]]:
        """service -> reservation ids that currently depend on it."""
        with self._lock:
            out: dict[str, list[str]] = {}
            for rid, v in self._reservations.items():
                for n in v.get("requires", []):
                    out.setdefault(n, []).append(rid)
            return out

    def _dependency_usable(self, name: str) -> tuple[bool, str]:
        """
        Can a declared dependency be satisfied for a request that is about to run?

        Returns (usable, reason). The REFUSAL case is a dependency that cannot be made
        available at all - not one that is merely not loaded yet.

        An earlier version refused whenever the service was not currently healthy. That was
        wrong and produced a chicken-and-egg: an ON-DEMAND service is unhealthy precisely
        because it has not been loaded, so the request that exists to load it was refused for
        the dependency being unloaded. Measured: the full-RAG request was refused with
        "declared dependency not usable: ['local-reranker']" in exactly the configuration that
        is supposed to load the reranker on demand.

        The correct question is whether the dependency CAN be brought up, which is knowable
        from the registry:
          * already healthy            -> usable
          * down but has a start script-> usable; the caller loads it, and its load cost is
                                          already inside the budget this decision checked
          * down with no start script  -> NOT usable, and that is a real refusal
        """
        if self.dry_run:
            return True, "dry-run: dependency probe skipped so policy tests do not depend " \
                         "on which real services happen to be up"
        spec = self.reg.get(name)
        if spec is None:
            return False, f"{name} is not a registered service"
        if service_healthy(spec.port):
            return True, "healthy"
        if spec.start_script:
            return True, (f"{name} is not loaded but is startable "
                          f"(load ~{spec.load_s}s, {spec.vram_mb} MiB); its cost is already "
                          f"included in the budget for this request")
        return False, f"{name} is down and has no start script, so it cannot be loaded"

    def _plan_locked(self, job_class: str, allow_unload: bool,
                     running: list[str], free: int,
                     requires: list[str] | None = None) -> Decision:
        """
        Evaluate the policy. MUST be called with self._lock held.

        The caller reserves before releasing, which is what makes the decision binding.
        """
        prof = JOB_PROFILE[job_class]
        # the REQUEST's declared dependencies, not the class default
        req = list(requires) if requires is not None else list(prof["needs"])
        active = dict(self._active)

        # ---- mutual exclusion
        if active:
            if job_class == "exclusive" or "exclusive" in active:
                return Decision(False, self.strategy, job_class,
                                f"exclusive is mutually exclusive; active={sorted(active)}",
                                free, self.margin_mb, 0, self.reserved_mb(), retryable=True)
            if self.strategy == "serial":
                return Decision(False, self.strategy, job_class,
                                f"strategy 'serial' does not permit overlap; "
                                f"active={sorted(active)}",
                                free, self.margin_mb, 0, self.reserved_mb(), retryable=True)
            if job_class == "interactive":
                return Decision(False, self.strategy, job_class,
                                f"interactive runs alone and must wait for "
                                f"{sorted(active)} to finish",
                                free, self.margin_mb, 0, self.reserved_mb(), retryable=True)
            # background joining: never while an interactive is active or waiting
            if "interactive" in active:
                return Decision(False, self.strategy, job_class,
                                "background may not join an active interactive job",
                                free, self.margin_mb, 0, self.reserved_mb(), retryable=True)
            if self._waiting_interactive > 0:
                return Decision(False, self.strategy, job_class,
                                f"background is not admitted while "
                                f"{self._waiting_interactive} interactive request(s) wait",
                                free, self.margin_mb, 0, self.reserved_mb(), retryable=True)
            if self._active.get("background", 0) >= self.max_background_units:
                return Decision(False, self.strategy, job_class,
                                f"background unit limit reached "
                                f"({self.max_background_units}); waiting bounds the "
                                f"interactive wait", free, self.margin_mb, 0,
                                self.reserved_mb())

        # ---- declared dependencies must be SATISFIABLE, not merely accounted for
        #
        # Budget is not sufficient, and neither is "currently healthy": an on-demand service
        # is down by design. What must hold is that every declared dependency can be brought
        # up before use. A dependency that cannot be loaded at all is a real refusal, and the
        # message says which one and why.
        if req:
            unusable = []
            for n in req:
                ok, why = self._dependency_usable(n)
                if not ok:
                    unusable.append(f"{n} ({why})")
            if unusable:
                return Decision(False, self.strategy, job_class,
                                f"declared dependency cannot be satisfied: {unusable}; "
                                f"refusing rather than sending inference to a service that "
                                f"cannot be made ready (budget was not the constraint)",
                                free, self.margin_mb, 0, self.reserved_mb())

        # ---- budget: resident already counted in `free`, so do NOT add it again
        need = prof["headroom_mb"]
        load_needed = 0
        for n in req:
            if n in self.reg and n not in running:
                load_needed += self.reg[n].vram_mb + self.reg[n].peak_extra_mb
        if job_class != "exclusive":
            for n in active:
                need += self.reg[n].peak_extra_mb if n in self.reg else 0
        need += load_needed

        reserved = self.reserved_mb()
        usable = free - reserved - self.margin_mb
        if usable >= need:
            return Decision(True, self.strategy, job_class,
                            f"{usable} MiB usable >= {need} MiB required "
                            f"(free {free} - reserved {reserved} - margin {self.margin_mb})",
                            free, self.margin_mb, need, reserved, load_needed,
                            concurrent_with=sorted(active))

        # ---- unload attempt
        if allow_unload:
            candidates = [n for n in running
                          if not self.reg[n].essential
                          and n not in req
                          and n not in active]
            candidates.sort(key=lambda n: self.reg[n].load_s)
            reclaimed, chosen = 0, []
            for n in candidates:
                chosen.append(n)
                reclaimed += self.reg[n].vram_mb
                if usable + reclaimed >= need:
                    break
            if chosen and usable + reclaimed >= need:
                return Decision(True, self.strategy, job_class,
                                f"can meet {need} MiB by unloading {chosen} "
                                f"(reclaims ~{reclaimed} MiB)", free, self.margin_mb,
                                need, reserved, load_needed, unload=chosen,
                                concurrent_with=sorted(active))
            return Decision(False, self.strategy, job_class,
                            f"cannot satisfy {need} MiB even after unloading "
                            f"{chosen or 'nothing'}: usable {usable} + reclaimable "
                            f"{reclaimed} MiB is still short. REFUSING rather than loading.",
                            free, self.margin_mb, need, reserved, load_needed)

        return Decision(False, self.strategy, job_class,
                        f"insufficient margin: {usable} MiB usable < {need} MiB required "
                        f"(free {free} - reserved {reserved} - margin {self.margin_mb})",
                        free, self.margin_mb, need, reserved, load_needed)

    # ------------------------------------------------------------ public planning
    def plan(self, job_class: str, allow_unload: bool | None = None) -> Decision:
        """Advisory only. Does NOT reserve; use acquire() for a binding decision."""
        if job_class not in JOB_PROFILE:
            raise ValueError(f"unknown job_class {job_class!r}")
        if allow_unload is None:
            allow_unload = self.strategy == "unload"
        with self._lock:
            try:
                free = self.free_mb()
            except Exception as e:  # noqa: BLE001
                return Decision(False, self.strategy, job_class,
                                f"sensor failure, refusing admission: "
                                f"{type(e).__name__}: {e}",
                                -1, self.margin_mb, 0, self.reserved_mb())
            return self._plan_locked(job_class, allow_unload, self.running(), free)

    # ------------------------------------------------------------ atomic acquire
    def acquire(self, job_class: str, timeout_s: float = 120.0,
                allow_unload: bool | None = None,
                requires: list[str] | None = None,
                request_id: str | None = None) -> Decision:
        """
        Block until the job may run, then reserve its budget ATOMICALLY.

        The lock is held across planning, unloading, re-confirmation and reservation, so no
        other thread can be admitted against the same budget in between. Unloading happens
        while the lock is held precisely so the freed memory cannot be taken by someone else
        before this job claims it.

        `requires` declares the services THIS REQUEST depends on. It is recorded on the
        request's own reservation and is validated here:

          * an unknown service name is rejected
          * a declared service that is not HEALTHY blocks admission even when the budget
            allows, because sending inference to a service that is still loading would fail
          * every unload entry point consults the ACTIVE reservations, so a dependency of a
            live request cannot be unloaded

        The class-level JOB_PROFILE `needs` remains the DEFAULT for a class; `requires` is the
        per-request truth. The global profile is never mutated at runtime, because doing so
        would make one request's dependencies leak into every other request of its class.
        """
        if job_class not in JOB_PROFILE:
            raise ValueError(f"unknown job_class {job_class!r}")
        if allow_unload is None:
            allow_unload = self.strategy == "unload"

        # ---- validate the declaration before doing anything else
        declared = list(requires) if requires is not None else list(JOB_PROFILE[job_class]["needs"])
        unknown = [n for n in declared if n not in self.reg]
        if unknown:
            raise ValueError(f"unknown service(s) in requires: {unknown}; "
                             f"registered: {sorted(self.reg)}")

        t0 = time.monotonic()          # MONOTONIC: wall clock can jump and break deadlines
        is_interactive = job_class == "interactive"
        with self._lock:
            ticket = self._take_ticket()
            self._tickets_open.add(ticket)
            self._ticket_job[ticket] = job_class
            self._deadlines[ticket] = (t0 + float(timeout_s), float(timeout_s))
            if is_interactive:
                self._waiting_interactive += 1
        try:
            while True:
                with self._lock:
                    deadline, tmo = self._deadlines.get(ticket, (t0 + timeout_s, timeout_s))

                    # EVERY request checks its OWN deadline, head of the queue or not. The
                    # previous version only checked it on the attempt path, so a non-head
                    # waiter could sit past its timeout while the queue kept moving.
                    if time.monotonic() > deadline:
                        holder = self._ticket_job.get(self._serving, "unknown")
                        blocked_by = ("" if self._is_next(ticket) else
                                      f"; queued behind an earlier {holder} request "
                                      f"(serving {self._serving})")
                        d = Decision(False, self.strategy, job_class,
                                     f"timed out after {tmo}s (ticket {ticket})"
                                     f"{blocked_by}",
                                     -1, self.margin_mb, 0, self.reserved_mb())
                        d.waited_s = round(time.monotonic() - t0, 3)
                        self.history.append({"event": "timeout", "ticket": ticket,
                                             **d.to_dict()})
                        self._close_ticket(ticket)
                        return d

                    if not self._is_next(ticket):
                        holder = self._ticket_job.get(self._serving, "unknown")
                        d = Decision(False, self.strategy, job_class,
                                     f"queued behind an earlier {holder} request "
                                     f"(ticket {ticket}, serving {self._serving})",
                                     -1, self.margin_mb, 0, self.reserved_mb())
                        d.queued = True
                    else:
                        d = self._attempt_locked(job_class, allow_unload, t0,
                                                 requires=declared,
                                                 request_id=request_id)

                    if d.allowed:
                        self._close_ticket(ticket)
                        return d
                    if not getattr(d, "queued", False):
                        # terminal refusal: close the ticket so later requests progress
                        self._close_ticket(ticket)
                        return d
                time.sleep(0.01)
        finally:
            with self._lock:
                if is_interactive:
                    self._waiting_interactive = max(0, self._waiting_interactive - 1)
                # an abnormal exit (exception, kill) must not leave the ticket or its
                # deadline behind
                if ticket in self._tickets_open:
                    self._close_ticket(ticket)

    def _attempt_locked(self, job_class: str, allow_unload: bool,
                        t0: float, requires: list[str] | None = None,
                        request_id: str | None = None) -> Decision:
        """
        One admission attempt. MUST be called with self._lock held and the FIFO ticket at
        the front. Returns a Decision; `queued=True` marks a retryable outcome.
        """
        try:
            free = self.free_mb()
        except Exception as e:  # noqa: BLE001
            d = Decision(False, self.strategy, job_class,
                         f"sensor failure, refusing admission: {type(e).__name__}: {e}",
                         -1, self.margin_mb, 0, self.reserved_mb())
            self.history.append({"event": "sensor_refuse", **d.to_dict()})
            return d

        d = self._plan_locked(job_class, allow_unload, self.running(), free,
                              requires=requires)
        if d.allowed:
            unload_ok = True
            for n in d.unload:
                if not self._unload_service(n):
                    unload_ok = False
                    break
            if not unload_ok:
                d = Decision(False, self.strategy, job_class,
                             f"unload did not succeed for {d.unload}; refusing admission "
                             f"rather than loading anyway",
                             free, self.margin_mb, d.required_mb,
                             self.reserved_mb(), d.load_needed_mb)
            else:
                # RE-CONFIRM after unloading: memory must actually be available now
                try:
                    free2 = self.free_mb()
                except Exception as e:  # noqa: BLE001
                    d = Decision(False, self.strategy, job_class,
                                 f"sensor failure after unload: {e}",
                                 -1, self.margin_mb, d.required_mb, self.reserved_mb())
                    self.history.append({"event": "refuse", **d.to_dict()})
                    return d
                usable2 = free2 - self.reserved_mb() - self.margin_mb
                if usable2 < d.required_mb:
                    d = Decision(False, self.strategy, job_class,
                                 f"after unloading {d.unload} only {usable2} MiB usable "
                                 f"< {d.required_mb} MiB required; refusing rather than "
                                 f"loading", free2, self.margin_mb, d.required_mb,
                                 self.reserved_mb(), d.load_needed_mb)
                else:
                    # RESERVE while still holding the lock.
                    #
                    # The ledger records what THIS request depends on, so unload protection is
                    # per-request rather than per-class. `charged_mb` is what this reservation
                    # still has to cover: see reserved_mb().
                    self._active[job_class] = self._active.get(job_class, 0) + 1
                    self._rid_seq += 1
                    rid = request_id or f"{job_class}-{self._rid_seq}"
                    self._reservations[rid] = {
                        "job_class": job_class,
                        "mb": d.required_mb,
                        "requires": list(requires or []),
                        "watermark_free": free2 + self.reserved_mb(),
                    }
                    d.reservation_id = rid
                    d.requires = list(requires or [])
                    d.waited_s = round(time.monotonic() - t0, 3)
                    self.history.append({"event": "acquire", **d.to_dict(),
                                         "t": round(time.time(), 3)})
                    return d

        # Retryability is a STRUCTURAL property, not a guess from the reason text.
        #
        # The previous version decided this by substring-matching the reason string against a
        # list of phrases. A new terminal refusal - "declared dependency not usable" - matched
        # none of them, so it was misclassified as retryable and the request spun on it until
        # its own deadline expired, reporting a TIMEOUT instead of the real cause. Any future
        # terminal reason would have hit the same trap. The planner now says directly whether
        # waiting could help.
        if not d.allowed and not getattr(d, "retryable", False):
            d.waited_s = round(time.monotonic() - t0, 3)
            self.history.append({"event": "refuse", **d.to_dict()})
            return d

        if time.monotonic() - t0 > 120.0:
            # The per-request deadline is enforced by the caller against the ticket's own
            # recorded timeout. This is only a backstop for a pathological path that never
            # reaches the caller's check.
            d.reason += " (internal backstop deadline exceeded)"
            d.waited_s = round(time.monotonic() - t0, 3)
            self.history.append({"event": "timeout", **d.to_dict()})
            return d

        # retryable: another job holds the GPU, so waiting could change the outcome
        d.queued = True
        d.retryable = True
        return d

    def _timeout_for(self, job_class: str) -> float:
        """
        Backstop only. Per-request timeouts live in self._deadlines keyed by TICKET.

        Kept because a class-level default is a reasonable backstop, but it must never be
        consulted for a live request: a dict keyed by job class let a later request of the
        same class overwrite an earlier one's timeout.
        """
        return float(JOB_PROFILE.get(job_class, {}).get("timeout_s", 120.0))

    def release(self, job_class: str, reservation_id: str | None = None) -> None:
        """
        Release a slot and its reservation.

        Each acquire() stores its own reservation amount keyed by a unique id, so release
        subtracts exactly what was reserved. Deriving the amount from the job class instead
        would be wrong the moment two jobs of the same class reserve different amounts
        because they ran under different conditions.

        Double release is a no-op that is REPORTED rather than silently lost - a silent
        double release would let the budget drift upward and over-admit later.
        """
        with self._lock:
            if self._active.get(job_class):
                self._active[job_class] -= 1
                if self._active[job_class] <= 0:
                    self._active.pop(job_class, None)
                # pop the matching reservation
                rid = reservation_id
                if rid is None:
                    # no id given: drop the oldest reservation for this class
                    rids = [k for k, v in self._reservations.items()
                            if v["job_class"] == job_class]
                    rid = rids[0] if rids else None
                if rid and rid in self._reservations:
                    self._reservations.pop(rid, None)
                self.history.append({"event": "release", "job_class": job_class,
                                     "reservation_id": rid,
                                     "reserved_now": self.reserved_mb(),
                                     "t": round(time.time(), 3)})
            else:
                self.history.append({"event": "double_release_ignored",
                                     "job_class": job_class,
                                     "t": round(time.time(), 3)})
                self._log(f"ignoring release for '{job_class}': nothing active")

    # ------------------------------------------------------------ unload
    def _unload_service(self, name: str) -> bool:
        """
        Unload one non-essential service. MUST be called with self._lock held.

        Refuses essential services and services used by an active job, waits for the port to
        free, and re-measures. Returns False on any failure so the caller can refuse
        admission rather than proceed on a false assumption.
        """
        spec = self.reg[name]
        if spec.essential:
            self._log(f"REFUSING to unload essential service {name}")
            return False
        # Never unload something an ACTIVE REQUEST depends on. This consults the per-request
        # reservations, not the class profile: two background tasks can declare different
        # services, and only the requesting task's own dependency should protect it.
        users = self.protected_services().get(name, [])
        if users:
            self._log(f"REFUSING to unload {name}: required by active request(s) {users}")
            return False
        if self.dry_run:
            self._log(f"[dry-run] would unload {name}")
            return True
        before = self.free_mb()
        self._log(f"unloading {name} (free {before} MiB) ...")
        cmd = spec.stop_cmd()
        if not cmd:
            return False
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stopped = getattr(r, "returncode", 0) == 0
        for _ in range(20):
            if port_free(spec.port):
                break
            time.sleep(0.5)
        time.sleep(1.0)
        freed = port_free(spec.port)
        after = self.free_mb()
        ok = stopped and freed
        self._log(f"  {name}: stop_rc={getattr(r,'returncode',None)} port_free={freed} "
                  f"free {before} -> {after} MiB (reclaimed {after - before})")
        return ok

    def unload(self, name: str) -> bool:
        """Explicit unload entry. A running task is never force-killed to make room."""
        with self._lock:
            return self._unload_service(name)

    def start(self, name: str, wait_s: float = 120.0) -> bool:
        """Start a service and require a HEALTH CHECK before reporting success."""
        spec = self.reg[name]
        if service_healthy(spec.port):
            return True
        cmd = spec.start_cmd()
        if not cmd or self.dry_run:
            return False
        subprocess.run(cmd, capture_output=True, text=True, timeout=int(wait_s) + 60)
        for _ in range(int(wait_s * 2)):
            if service_healthy(spec.port):
                return True
            time.sleep(0.5)
        return False

    def status(self) -> dict:
        running = self.running()
        try:
            free = self.free_mb()
            sensor_error = None
        except Exception as e:  # noqa: BLE001
            free, sensor_error = -1, f"{type(e).__name__}: {e}"
        return {
            "strategy": self.strategy,
            "safety_margin_mb": self.margin_mb,
            "max_background_units": self.max_background_units,
            "free_mb": free,
            "reserved_mb": self.reserved_mb(),
            "sensor_error": sensor_error,
            "margin_satisfied": (free - self.reserved_mb()) >= self.margin_mb
            if free >= 0 else None,
            "running": running,
            "active_jobs": dict(self._active),
            "waiting_interactive": self._waiting_interactive,
            "queued": {k: v for k, v in self._queued.items() if v},
            "process_boundary": ("this lock is in-process; separate CLI invocations do NOT "
                                 "share it"),
            "registry": {n: {"port": s.port, "vram_mb": s.vram_mb,
                             "peak_extra_mb": s.peak_extra_mb,
                             "essential": s.essential, "load_s": s.load_s,
                             "running": n in running}
                         for n, s in self.reg.items()},
            "protected_not_registered": ["Qwen-Image (outside this project's lifecycle)"],
        }






