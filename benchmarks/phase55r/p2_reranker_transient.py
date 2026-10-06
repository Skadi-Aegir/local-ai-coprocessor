"""
Measure the reranker's ACTUAL transient peak against its declared 1800 MiB.

The registry declares local-reranker steady=1160, transient=1800. The declared transient is
what makes all three models unable to co-reside: 7437 MiB declared vs 7464 MiB available after
the 2048 margin, i.e. 27 MiB of slack. If the declared transient is much larger than anything
actually observed, then refusals are being caused by the BUDGET FIGURE rather than by the
hardware, and that must be stated rather than blamed on the GPU.

This measures observed VRAM deltas at real batch sizes. It does NOT change the declared value:
lowering it to make a test pass is exactly what the task forbids. It reports the comparison.

Run: python p2_reranker_transient.py --out results/<run>/reranker_transient.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))
from gpu_scheduler import REGISTRY  # noqa: E402

RERANK = "http://127.0.0.1:8114"


def vram() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def peak_during(fn, seconds: float = 60.0) -> tuple[int, int]:
    """Sample continuously while fn runs; return (baseline, peak)."""
    base = vram()
    peak = base or 0
    t0 = time.time()
    import threading
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.is_set() and time.time() - t0 < seconds:
            v = vram()
            if v:
                peak = max(peak, v)
            time.sleep(0.1)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        fn()
    finally:
        done.set()
        th.join(timeout=5)
    return base or 0, peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    spec = REGISTRY["local-reranker"]
    report: dict = {
        "declared": {"steady_mb": spec.vram_mb, "transient_mb": spec.peak_extra_mb,
                     "total_mb": spec.vram_mb + spec.peak_extra_mb},
        "note": "declared values are NOT changed; this only compares them with observation",
        "runs": [],
    }
    print(f"declared: steady={spec.vram_mb} transient={spec.peak_extra_mb} "
          f"total={spec.vram_mb + spec.peak_extra_mb} MiB")

    if not _healthy():
        print("reranker is not up; start it first")
        return 4

    # baseline: reranker loaded but idle
    idle = vram()
    report["idle_used"] = idle
    print(f"idle (reranker up): {idle} MiB")

    for label, n_docs, doc_words in (("small (10 docs, short)", 10, 30),
                                     ("realistic (10 docs, ~250 words)", 10, 250),
                                     ("batch (50 docs, ~250 words)", 50, 250),
                                     ("batch (50 docs, ~1000 words)", 50, 1000)):
        docs = [("word " * doc_words) for _ in range(n_docs)]
        payload = json.dumps({"query": "what context size does local-fast use",
                              "documents": docs, "top_n": min(5, n_docs),
                              "instruction_type": "general", "use_cache": False}).encode()
        result = {}

        def call():
            req = urllib.request.Request(RERANK + "/v1/rerank", data=payload,
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=300) as r:
                result["body"] = json.loads(r.read().decode())
            result["s"] = round(time.time() - t0, 3)

        try:
            base, peak = peak_during(call)
            delta = peak - base
            entry = {"case": label, "n_docs": n_docs, "doc_words": doc_words,
                     "baseline_used": base, "peak_used": peak,
                     "observed_delta_mb": delta, "seconds": result.get("s"),
                     "n_results": len((result.get("body") or {}).get("results") or [])}
            print(f"  {label:<34} delta={delta:>5} MiB  ({result.get('s')}s)")
            report["runs"].append(entry)
        except Exception as e:  # noqa: BLE001
            print(f"  {label:<34} FAILED {type(e).__name__}: {e}")
            report["runs"].append({"case": label, "error": f"{type(e).__name__}: {e}"})

    observed = [r["observed_delta_mb"] for r in report["runs"] if "observed_delta_mb" in r]
    report["observed_max_delta_mb"] = max(observed) if observed else None
    report["comparison"] = {
        "declared_transient_mb": spec.peak_extra_mb,
        "observed_max_delta_mb": max(observed) if observed else None,
        "declared_exceeds_observed_by_mb": (spec.peak_extra_mb - max(observed))
        if observed else None,
        "verdict": ("the declared transient is materially larger than anything observed, so "
                    "refusals near the margin may be caused by the BUDGET FIGURE rather than "
                    "by the hardware; the declared value is left unchanged and this is "
                    "reported rather than used to relax the margin"
                    if observed and max(observed) < spec.peak_extra_mb * 0.7
                    else "the declared transient is consistent with observation"),
    }
    print(f"\nobserved max delta = {report['observed_max_delta_mb']} MiB "
          f"vs declared {spec.peak_extra_mb} MiB")
    print(f"verdict: {report['comparison']['verdict']}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(RERANK + "/health", timeout=5) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
