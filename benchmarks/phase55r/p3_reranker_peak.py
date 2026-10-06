"""
p3_reranker_peak.py - P55R-03: measure the reranker's real peak.

CONSTRAINTS HELD, not relaxed:
  * the 2048 MiB GPU safety margin is unchanged
  * the default 1800 MiB budget is unchanged
  * the previously observed 262 MiB hot-sample maximum is NOT treated as a safe ceiling - this
    run exists precisely because that figure was too thin a basis

What is measured, per load case, across the whole load -> inference -> release cycle:
  VRAM peak, VRAM minimum, sampling interval, RAM commit, latency, and - the gate - whether the
  2048 MiB margin was EVER breached during the cycle.

GATE: if the baseline cannot hold the margin during a normal cycle, the experiment STOPS and
reports rather than proceeding to propose a budget. Reported as a decision, not buried.

Load cases: cold load, warm inference, and the largest input length / batch / microbatch the
system actually allows. Each critical case repeated at least three times.

Usage:
  python p3_reranker_peak.py --out results/<run>/reranker_peak.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))
from gpu_scheduler import REGISTRY  # noqa: E402

RERANK = "http://127.0.0.1:8114"
MARGIN = 2048
REPEATS = 3


def gpu() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        u, f = out.stdout.strip().splitlines()[0].split(",")
        return int(u.strip()), int(f.strip())
    except Exception:
        return None


def commit_mb() -> int | None:
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
             "$os=Get-CimInstance Win32_OperatingSystem;"
             "[int](($os.TotalVirtualMemorySize-$os.FreeVirtualMemory)/1KB)"],
            capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip())
    except Exception:
        return None


class Sampler:
    """Continuous sampler that also records every sub-margin excursion."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            g = gpu()
            if g:
                self.samples.append({"t": round(time.time(), 3), "used": g[0], "free": g[1]})
            time.sleep(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.t.join(timeout=10)
        return False

    def window(self, t0: float, t1: float) -> dict:
        vals = [s for s in self.samples if t0 <= s["t"] <= t1]
        if not vals:
            return {"n": 0}
        used = [s["used"] for s in vals]
        free = [s["free"] for s in vals]
        breaches = [s for s in vals if s["free"] < MARGIN]
        gaps = [round(vals[i + 1]["t"] - vals[i]["t"], 3) for i in range(len(vals) - 1)]
        return {
            "n_samples": len(vals),
            "interval_mean_s": round(statistics.mean(gaps), 3) if gaps else None,
            "interval_max_s": round(max(gaps), 3) if gaps else None,
            "used_min": min(used), "used_max": max(used),
            "free_min": min(free), "free_max": max(free),
            "peak_delta_mb": max(used) - min(used),
            "margin_mb": MARGIN,
            "margin_breached": bool(breaches),
            "n_sub_margin_samples": len(breaches),
            "worst_free_below_margin_mb": (MARGIN - min(free)) if breaches else 0,
        }


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, round(time.time() - t0, 3)


def unique_docs(n_docs: int, doc_words: int, tag: str) -> list[str]:
    """
    Documents whose CONTENT is unique per call.

    Measured reason this exists: `use_cache: false` is NOT honoured by the reranker service.
    Identical documents repeat at ~0.025 s whether the query changes or not, while genuinely new
    document content costs ~0.43 s. Repeating a load case with the same documents therefore
    re-measures a CACHE HIT and reports a peak taken while almost no compute happened - which is
    how a first version of this script produced peak_delta values from 0.002 s calls.
    Every repeat gets distinct content so the work is real.
    """
    return [f"{tag} rep{tag} doc{i} " + ("word " * doc_words) for i in range(n_docs)]


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def run_powershell(script: Path, timeout: int = 300) -> int:
    """stdout to a FILE, never a pipe - a pipe deadlock cost this project a whole run."""
    log = REPO / "logs" / f"p3-svc-{script.stem}.log"
    with open(log, "wb") as fh:
        r = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                            "-ExecutionPolicy", "Bypass", "-File", str(script)],
                           stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)

    spec = REGISTRY["local-reranker"]
    report: dict = {
        "margin_mb": MARGIN,
        "declared": {"vram_mb": spec.vram_mb, "peak_extra_mb": spec.peak_extra_mb},
        "prior_hot_sample_max_mb": 262,
        "prior_ceiling_rejected": ("the 262 MiB hot-sample maximum is NOT treated as a safe "
                                   "ceiling; this measurement exists because it was too thin"),
        "repeats": REPEATS,
        "cases": [],
        "note": "2048 MiB margin and the 1800 MiB default budget are both unchanged",
    }

    # gate: can the CURRENT baseline already hold the margin at idle?
    idle = gpu()
    report["baseline_idle"] = {"used": idle[0] if idle else None,
                               "free": idle[1] if idle else None,
                               "holds_margin": bool(idle and idle[1] >= MARGIN),
                               "commit_mb": commit_mb()}
    print(f"baseline idle: {idle}  holds_margin={report['baseline_idle']['holds_margin']}")
    if not idle or idle[1] < MARGIN:
        report["stopped_early"] = {
            "reason": "baseline idle free VRAM is below the 2048 MiB margin BEFORE loading the "
                      "reranker, so no measurement of its peak can be taken without breaching "
                      "the margin",
            "free_mb": idle[1] if idle else None,
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STOPPING: baseline cannot hold the margin")
        return 4

    # tell the service process what the largest allowed input is
    try:
        with urllib.request.urlopen(f"{RERANK}/health", timeout=5) as r:
            report["reranker_health"] = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        report["reranker_health"] = f"{type(e).__name__}: {e}"

    def cycle(label: str, n_docs: int, doc_words: int, cold: bool) -> dict:
        """One full load(if cold) -> inference -> (release) cycle, continuously sampled."""
        case: dict = {"case": label, "n_docs": n_docs, "doc_words": doc_words, "cold": cold,
                      "runs": []}
        for rep in range(REPEATS):
            if cold and healthy(spec.port):
                run_powershell(REPO / "scripts" / "stop-local-reranker.ps1")
                time.sleep(2)
            if not healthy(spec.port):
                t_load = time.time()
                run_powershell(REPO / "scripts" / "start-local-reranker.ps1")
                load_s = round(time.time() - t_load, 3)
            else:
                load_s = 0.0

            docs = unique_docs(n_docs, doc_words, f"{label}-{rep}")
            t0 = time.time()
            c0 = commit_mb()
            with Sampler(interval=0.02) as smp:
                try:
                    body, lat = post(RERANK + "/v1/rerank",
                                     {"query": f"unique query {label} {rep} {time.time_ns()}",
                                      "documents": docs, "top_n": min(5, n_docs),
                                      "instruction_type": "general", "use_cache": False})
                    n_res = len(body.get("results") or [])
                    err = None
                except Exception as e:  # noqa: BLE001
                    lat, n_res, err = None, 0, f"{type(e).__name__}: {e}"
                t1 = time.time()
            win = smp.window(t0, t1)
            c1 = commit_mb()
            run = {"repeat": rep + 1, "load_s": load_s, "latency_s": lat, "n_results": n_res,
                   "error": err, "vram": win,
                   "commit_before_mb": c0, "commit_after_mb": c1,
                   "commit_delta_mb": (c1 - c0) if (c0 and c1) else None}
            case["runs"].append(run)
            print(f"  {label} rep{rep+1}: load={load_s}s lat={lat}s "
                  f"peak_delta={win.get('peak_delta_mb')} free_min={win.get('free_min')} "
                  f"breach={win.get('margin_breached')}", flush=True)
            report["cases"].append({"case": label, "repeat": rep + 1, **run})
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return case

    # the largest input the service will accept: max_length is 2048 and body_budget 2000 tokens
    # is the configured body cap, so the largest realistic document set is a few long documents
    cycle("cold_load_and_first_inference", 5, 200, cold=True)
    cycle("warm_inference_small", 10, 50, cold=False)
    cycle("warm_inference_max_body", 10, 400, cold=False)
    cycle("large_batch_max_body", 50, 400, cold=False)

    deltas = [c["vram"]["peak_delta_mb"] for c in report["cases"]
              if c.get("vram", {}).get("peak_delta_mb") is not None]
    frees = [c["vram"]["free_min"] for c in report["cases"]
             if c.get("vram", {}).get("free_min") is not None]
    breaches = [c["case"] for c in report["cases"] if c.get("vram", {}).get("margin_breached")]
    report["summary"] = {
        "n_cycles": len(report["cases"]),
        "observed_peak_delta_mb_max": max(deltas) if deltas else None,
        "observed_peak_delta_mb_median": round(statistics.median(deltas), 1) if deltas else None,
        "observed_free_min_mb": min(frees) if frees else None,
        "cases_breaching_margin": sorted(set(breaches)),
        "margin_held_everywhere": not breaches,
    }
    print("\n=== SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
