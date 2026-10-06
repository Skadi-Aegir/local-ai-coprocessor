"""
p3_swap.py - staged handoff with REQUIRED unloads.

P55R-05 CHANGE: `allow_unload=True` only makes an unload PERMITTED. P55R-04 measured that this is
not enough - with the budget already sufficient the planner never unloaded anything, the embedding
model stayed resident, and the swap delivered no memory relief (`unloaded=[]`). This module adds an
explicit operation that REQUIRES each unload to actually happen and VERIFIES it:

    save_candidates_to_cpu(...)      -> candidates become plain Python data
    unload_required(name)            -> asserts the port freed AND the live reading moved

The run order P55R-05 asks for:
    embed+retrieve -> candidates to CPU -> UNLOAD embedding -> rerank
                   -> sorted results to CPU -> UNLOAD reranker -> pack+generate

Invariants (each asserted where it can be):
  I1  at most one stage of this pipeline holds a reservation at a time
  I2  a required unload is entered with zero live reservations of this pipeline
  I3  the scheduler refuses to unload a service named by an ACTIVE reservation
  I4  after any failure no reservation is left behind
  I5  candidates live on the CPU and survive every failure after stage 2
  I6  a REQUIRED unload either verifiably happens (port free AND free VRAM rose) or the stage
      fails - it is never silently skipped
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))

from gpu_scheduler import GpuScheduler, REGISTRY, port_free, service_healthy  # noqa: E402

MARGIN = 2048


def gpu_free() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class Stage(str, Enum):
    IDLE = "IDLE"
    EMBEDDING = "EMBEDDING"
    CANDIDATES_ON_CPU = "CANDIDATES_ON_CPU"
    SWAPPING = "SWAPPING"
    RERANKING = "RERANKING"
    RERANKED_ON_CPU = "RERANKED_ON_CPU"
    GENERATING = "GENERATING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class SwapResult:
    ok: bool
    stage: str = ""
    reason: str = ""
    unloaded: list[str] = field(default_factory=list)
    reservation_id: str | None = None
    waited_s: float = 0.0
    free_before: int | None = None
    free_after: int | None = None
    port_free_verified: bool = False


class StagedPipeline:
    def __init__(self, scheduler: GpuScheduler, *, log=None):
        self.s = scheduler
        self.state = Stage.IDLE
        self.candidates: list[dict] = []
        self.reranked: list[dict] = []
        self.rid: str | None = None
        self.trace: list[dict] = []
        self._log = log or (lambda m: None)
        self._own: set[str] = set()

    # ---------------------------------------------------------------- helpers
    def _note(self, event: str, **kw) -> None:
        self.trace.append({"t": round(time.monotonic(), 4), "state": self.state.value,
                           "event": event, **kw})
        self._log(f"{self.state.value:<18} {event} {kw if kw else ''}")

    def live_own_reservations(self) -> int:
        return sum(1 for r in self._own if r in self.s._reservations)

    def _live_free(self) -> int | None:
        """The scheduler's own live free reading, with a sensor failure reported as None."""
        try:
            return int(self.s.free_mb())
        except Exception:  # noqa: BLE001
            return None

    def _close(self, job_class: str) -> None:
        if self.rid and self.rid in self.s._reservations:
            self.s.release(job_class, self.rid)
        if self.rid:
            self._own.discard(self.rid)
        self.rid = None

    def _admit(self, job_class: str, requires: list[str], timeout_s: float,
               allow_unload: bool, request_id: str):
        t0 = time.monotonic()
        d = self.s.acquire(job_class, timeout_s=timeout_s, requires=requires,
                           allow_unload=allow_unload, request_id=request_id)
        return d, round(time.monotonic() - t0, 3)

    # ---------------------------------------------------------------- required unload
    def unload_required(self, name: str, timeout_s: float = 180.0) -> SwapResult:
        """
        Unload `name` and VERIFY it happened. Invariant I6.

        Verification is two-pronged and both must hold:
          * the port is free (the process stopped listening)
          * the live free reading ROSE by a material amount
        A port that frees while memory does not return is a failure, not a success - that exact
        case was injected and caught back in P55R-01's B2 work.

        Must be called with NO live reservation naming `name`, otherwise the scheduler refuses
        (invariant I3) - which is the correct behaviour and is surfaced, not worked around.
        """
        if self.live_own_reservations() != 0:
            return SwapResult(False, "I2", "unload_required entered with a live reservation")
        spec = self.s.reg[name]
        # Read the live free figure through the SCHEDULER, not nvidia-smi directly. Two reasons:
        # verification must be judged on the same source the admission decision used, otherwise a
        # divergence between the two could make an unload look successful to one and not the other;
        # and it keeps the operation testable with a stubbed reading.
        before = self._live_free()
        t0 = time.monotonic()
        ok = self.s._unload_service(name)
        waited = round(time.monotonic() - t0, 3)
        after = self._live_free()
        pf = port_free(spec.port)
        rose = (before is not None and after is not None and after > before + 16)
        self._note("unload_required", name=name, scheduler_ok=ok, port_free=pf,
                   free_before=before, free_after=after, rose=rose, waited_s=waited)
        if not ok:
            return SwapResult(False, "unload_refused",
                              f"the scheduler refused to unload {name}", [name], None, waited,
                              before, after, pf)
        if not pf:
            return SwapResult(False, "port_still_bound",
                              f"{name} still listening after unload", [], None, waited,
                              before, after, pf)
        if not rose:
            return SwapResult(False, "memory_not_returned",
                              f"{name} port freed but free VRAM did not rise "
                              f"({before} -> {after})", [name], None, waited, before, after, pf)
        return SwapResult(True, "unloaded", f"{name} unloaded and verified", [name], None,
                          waited, before, after, pf)

    # ---------------------------------------------------------------- stages
    def stage1_embed_retrieve(self, fn, timeout_s: float = 60.0) -> dict:
        self.state = Stage.EMBEDDING
        d, waited = self._admit("background", ["local-embedding"], timeout_s,
                                self.s.strategy == "unload", "swap-s1-embed")
        if not d.allowed:
            self.state = Stage.FAILED
            self._note("stage1_refused", reason=d.reason[:140])
            return {"ok": False, "reason": d.reason, "waited_s": waited}
        self.rid = d.reservation_id
        self._own.add(d.reservation_id)
        self._note("stage1_admitted", rid=d.reservation_id, waited_s=waited,
                   required_mb=d.required_mb)
        try:
            out = fn()
            self.candidates = out.get("candidates") or []
            self._note("stage1_executed", n=len(self.candidates), exec_s=out.get("exec_s"))
            if not self.candidates:
                self.state = Stage.FAILED
                return {"ok": False, "reason": "no candidates produced"}
            self.state = Stage.CANDIDATES_ON_CPU
            return {"ok": True, "candidates": self.candidates, "waited_s": waited}
        except Exception as e:  # noqa: BLE001
            self.state = Stage.FAILED
            self._note("stage1_error", error=f"{type(e).__name__}: {e}")
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        finally:
            self._close("background")          # stage 1 ENDS: protection ends at RELEASE
            self._note("stage1_closed", live_own=self.live_own_reservations())

    def stage3_unload_embed_then_admit_reranker(self, timeout_s: float = 60.0) -> SwapResult:
        """
        UNLOAD embedding (required), then admit the reranker.

        Order matters and is deliberate: the unload is performed FIRST, while this pipeline holds
        nothing, and the admission happens AFTER, so the decision is made against a live reading
        that already reflects the freed memory. The scheduler's own lock makes the admission
        atomic, so nothing can take the freed memory during the admission itself.
        """
        if self.live_own_reservations() != 0:
            self._note("swap_aborted", reason="I2 violated")
            return SwapResult(False, "I2", "swap entered with a live reservation")
        self.state = Stage.SWAPPING
        self._note("swap_begin", candidates_on_cpu=len(self.candidates))

        u = self.unload_required("local-embedding")
        if not u.ok:
            self.state = Stage.CANDIDATES_ON_CPU
            self._note("swap_unload_failed", reason=u.reason[:160])
            return u

        d, waited = self._admit("background", ["local-reranker"], timeout_s, True, "swap-s3-rr")
        if not d.allowed:
            self.state = Stage.CANDIDATES_ON_CPU
            self._note("swap_admit_refused", reason=d.reason[:180], waited_s=waited)
            return SwapResult(False, "admit_refused", d.reason, ["local-embedding"], None,
                              waited, u.free_before, u.free_after, True)
        self.rid = d.reservation_id
        self._own.add(d.reservation_id)
        self.state = Stage.RERANKING
        self._note("swap_done", rid=d.reservation_id, required_mb=d.required_mb, waited_s=waited)
        return SwapResult(True, "RERANKING", d.reason, ["local-embedding"], d.reservation_id,
                          waited, u.free_before, u.free_after, True)

    def stage4_rerank(self, fn) -> dict:
        if self.state is not Stage.RERANKING:
            return {"ok": False, "reason": f"stage4 entered in state {self.state.value}"}
        try:
            out = fn(self.candidates)
            self.reranked = out.get("reranked") or []
            self._note("stage4_executed", exec_s=out.get("exec_s"), n=len(self.reranked))
            return {"ok": True, **out}
        except Exception as e:  # noqa: BLE001
            self.state = Stage.FAILED
            self._note("stage4_error", error=f"{type(e).__name__}: {e}")
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        finally:
            self._close("background")          # reranker reservation ends here
            if self.state is Stage.RERANKING:
                self.state = Stage.RERANKED_ON_CPU
            self._note("stage4_closed", live_own=self.live_own_reservations())

    def stage5_unload_reranker(self) -> SwapResult:
        """UNLOAD the reranker (required) before generating. Invariant I6."""
        u = self.unload_required("local-reranker")
        if not u.ok:
            self._note("stage5_unload_failed", reason=u.reason[:160])
            return u
        self._note("stage5_unloaded_reranker", free_after=u.free_after)
        return u

    def stage6_generate(self, fn, timeout_s: float = 60.0) -> dict:
        self.state = Stage.GENERATING
        d, waited = self._admit("interactive", ["local-fast"], timeout_s, False, "swap-s6-gen")
        if not d.allowed:
            self.state = Stage.FAILED
            self._note("stage6_refused", reason=d.reason[:140])
            return {"ok": False, "reason": d.reason, "waited_s": waited}
        self.rid = d.reservation_id
        self._own.add(d.reservation_id)
        try:
            out = fn()
            self._note("stage6_executed", exec_s=out.get("exec_s"))
            self.state = Stage.DONE
            return {"ok": True, **out}
        except Exception as e:  # noqa: BLE001
            self.state = Stage.FAILED
            self._note("stage6_error", error=f"{type(e).__name__}: {e}")
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        finally:
            self._close("interactive")
            self._note("stage6_closed", live_own=self.live_own_reservations())

    def cleanup(self) -> None:
        with self.s._lock:
            for r in list(self._own):
                if r in self.s._reservations:
                    self.s.release(self.s._reservations[r]["job_class"], r)
        self._own.clear()
        self.rid = None
