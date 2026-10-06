"""
vram_coresidency.py - VRAM under real load, not just at rest.

A steady-state reading is not enough for the reranker: its forward passes allocate
attention activations that grow with the square of sequence length, and Phase 5 already
saw a single forward pass attempt a 768 MiB allocation. So the number that matters is the
PEAK while all services are actually working.

Samples nvidia-smi in a background thread while driving:
  * reranker: full top-20 re-rank of a real query
  * embedding: a 32-chunk batch
  * local-fast: a generation request
Run with whatever combination of services is currently up; absent ones are skipped.

Usage:
  python vram_coresidency.py --out ../../benchmarks/reranker/results/coresidency_peak.json
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
sys.path.insert(0, str(HERE.parent / "retrieval"))

RERANK = "http://127.0.0.1:8114"
EMBED = "http://127.0.0.1:8113"
FAST = "http://127.0.0.1:8111"


def gpu_used() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def up(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=4) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


DOCS = [f"Document {i}. " + "Technical sentence about GPU memory and retrieval. " * 12
        for i in range(20)]
LONG_DOCS = ["filler sentence about nothing in particular. " * 40 for _ in range(20)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    services = {"reranker": up(RERANK), "embedding": up(EMBED), "local_fast": up(FAST)}
    print("services up:", services)
    active = [k for k, v in services.items() if v]
    if not active:
        print("no services up; nothing to measure")
        return 2

    at_rest = gpu_used()
    print(f"VRAM at rest: {at_rest} MiB")

    samples: list[tuple[float, int]] = []
    stop = threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            v = gpu_used()
            if v is not None:
                samples.append((time.time() - t0, v))
            time.sleep(0.15)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    counts = {"rerank_calls": 0, "embed_calls": 0, "fast_calls": 0}
    t_end = time.time() + args.seconds
    i = 0
    while time.time() < t_end:
        i += 1
        # worst case each round: a full pool re-rank, including an all-long-document round
        if services["reranker"]:
            docs = LONG_DOCS if i % 3 == 0 else DOCS
            try:
                post(RERANK + "/v1/rerank",
                     {"query": "how much VRAM does retrieval use", "documents": docs,
                      "top_n": 5, "instruction_type": "default"})
                counts["rerank_calls"] += 1
            except Exception as e:
                print(f"  rerank error: {type(e).__name__}: {e}")
        if services["embedding"]:
            try:
                post(EMBED + "/v1/embeddings",
                     {"model": "local-embedding",
                      "input": [f"chunk number {j} about retrieval" for j in range(32)]})
                counts["embed_calls"] += 1
            except Exception as e:
                print(f"  embed error: {type(e).__name__}: {e}")
        if services["local_fast"]:
            try:
                post(FAST + "/v1/chat/completions",
                     {"model": "local-fast",
                      "messages": [{"role": "user", "content": "Name three indexing strategies."}],
                      "max_tokens": 64, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": False}})
                counts["fast_calls"] += 1
            except Exception as e:
                print(f"  fast error: {type(e).__name__}: {e}")

    stop.set()
    th.join(timeout=10)

    vals = [v for _, v in samples]
    peak = max(vals)
    total = 12227
    res = {
        "services": services,
        "at_rest_mb": at_rest,
        "peak_mb": peak,
        "peak_delta_mb": peak - at_rest,
        "median_mb": int(statistics.median(vals)),
        "p95_mb": sorted(vals)[min(len(vals) - 1, int(0.95 * len(vals)))],
        "samples": len(vals),
        "free_at_peak_mb": total - peak,
        "meets_2gb_target": (total - peak) >= 2048,
        "workload": counts,
        "seconds": args.seconds,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nsamples {res['samples']}  workload {counts}")
    print(f"  at rest     {res['at_rest_mb']} MiB")
    print(f"  median      {res['median_mb']} MiB")
    print(f"  p95         {res['p95_mb']} MiB")
    print(f"  PEAK        {res['peak_mb']} MiB   (+{res['peak_delta_mb']} over rest)")
    print(f"  free at peak {res['free_at_peak_mb']} MiB")
    print(f"  >=2GB free at peak: {res['meets_2gb_target']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
