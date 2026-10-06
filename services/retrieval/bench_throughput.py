"""
bench_throughput.py - embedding throughput and co-residency measurements.

Measures through the LIVE HTTP service so the numbers describe the deployed system,
not an in-process shortcut.

  * throughput at batch 1 / 8 / 32 / 64 (chunks per second, latency, tokens where known)
  * VRAM attributable to the embedding service
  * optional concurrent load against a llama-server worker, to quantify contention

Usage:
  python bench_throughput.py --out ..\\..\\benchmarks\\retrieval\\results\\embedding_throughput.json
  python bench_throughput.py --coresident-url http://127.0.0.1:8111 --coresident-model local-fast
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))

EMBED_URL = "http://127.0.0.1:8113"

# Realistic chunks drawn from this project's own text, so token lengths are
# representative rather than synthetic.
SAMPLE_TEXTS = [
    "local-fast runs Qwen3.5-4B at an 8192 token context and occupies about 2981 MiB of VRAM. "
    "It reaches roughly 130 tokens per second during generation with a first-token latency near 0.13 seconds.",
    "Concurrent inference of the 4B and 9B models reduces aggregate throughput by 47 percent, because the GPU "
    "serialises the work anyway and adds memory-bandwidth contention on top.",
    "The VRAM safety margin is configured as min_free_vram_mb. The launcher refuses to start when free memory "
    "falls below it, so the desktop and other services keep working.",
    "结构化任务必须使用 deterministic preset：temperature 设为 0，保证 JSON 输出可被程序稳定解析。"
    "温度升高会显著降低 schema 完整率。",
    "The chunking strategy decides whether a retrieved passage is semantically complete. A function split across "
    "three windows is not useful context even when every window has exactly 500 tokens.",
    "Thinking mode costs roughly 17 times the latency and 20 times the tokens on a trivial extraction task for no "
    "quality gain, and on a hard task it can consume the entire output budget without producing an answer.",
    "Metadata is stored in SQLite rather than inside FAISS, so project filtering, file removal and incremental "
    "updates have a single authority to consult.",
    "Incremental indexing compares a content hash per file. When nothing changed, a rebuild embeds zero chunks and "
    "completes in milliseconds instead of seconds.",
]


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def nvidia(query: str) -> float | None:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def embed(texts: list[str], timeout: float = 600.0) -> dict:
    return http_json(f"{EMBED_URL}/v1/embeddings",
                     {"model": "local-embedding", "input": texts}, timeout)


def bench_batches(counts=(1, 8, 32, 64), rounds: int = 5) -> list[dict]:
    out = []
    for n in counts:
        # build a pool of `n` texts by cycling the samples
        pool = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(n)]
        # warm up this batch size
        embed(pool)
        lat, vram_peak, util_peak = [], 0.0, 0.0
        for _ in range(rounds):
            v0 = nvidia("memory.used")
            u0 = nvidia("utilization.gpu")
            t0 = time.time()
            r = embed(pool)
            dt = time.time() - t0
            v1 = nvidia("memory.used")
            u1 = nvidia("utilization.gpu")
            lat.append(dt)
            if v1:
                vram_peak = max(vram_peak, v1)
            if u0 is not None and u1 is not None:
                util_peak = max(util_peak, u1)
        med = statistics.median(lat)
        out.append({
            "batch_size": n,
            "rounds": rounds,
            "median_seconds": round(med, 4),
            "min_seconds": round(min(lat), 4),
            "chunks_per_s": round(n / med, 1) if med else None,
            "ms_per_chunk": round(1000 * med / n, 2) if n else None,
            "vram_peak_mb": vram_peak,
            "gpu_util_peak": util_peak,
            "dimension": r["local"]["dimension"],
        })
        print(f"  batch {n:3}: {out[-1]['median_seconds']:7.4f}s  "
              f"{out[-1]['chunks_per_s']:8.1f} chunks/s  "
              f"{out[-1]['ms_per_chunk']:6.2f} ms/chunk  vram={vram_peak:.0f} MiB", flush=True)
    return out


def coresident_test(kind: str, worker_url: str, model: str, seconds: int = 8) -> dict:
    """
    Run embedding batches while a llama-server worker generates continuously.

    Returns embedding throughput under load plus the worker's own generation rate,
    so the contention can be quantified from both sides.
    """
    result: dict = {"kind": kind, "worker_url": worker_url, "model": model}
    stop = threading.Event()
    worker_stats: list[dict] = []

    def worker_load():
        prompt = ("Write a detailed technical explanation of database indexing, "
                  "covering B-trees, hash indexes, and when each is appropriate.")
        while not stop.is_set():
            try:
                t0 = time.time()
                r = http_json(f"{worker_url}/v1/chat/completions", {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256, "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=600)
                t = r.get("timings", {})
                worker_stats.append({
                    "wall_s": round(time.time() - t0, 3),
                    "gen_tps": round(t.get("predicted_per_second", 0), 2),
                })
            except Exception as e:  # noqa: BLE001
                worker_stats.append({"error": f"{type(e).__name__}: {e}"})
                break

    th = threading.Thread(target=worker_load, daemon=True)
    th.start()
    time.sleep(1.0)     # let the worker get going

    pool = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(32)]
    embed(pool)         # warm
    lat = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        t0 = time.time()
        embed(pool)
        lat.append(time.time() - t0)

    stop.set()
    th.join(timeout=30)

    med = statistics.median(lat) if lat else 0
    result.update({
        "embedding_batches": len(lat),
        "embedding_batch_size": 32,
        "embedding_median_s": round(med, 4),
        "embedding_chunks_per_s": round(32 / med, 1) if med else None,
        "worker_requests": len(worker_stats),
        "worker_gen_tps_samples": [w.get("gen_tps") for w in worker_stats if w.get("gen_tps")],
        "worker_errors": [w["error"] for w in worker_stats if "error" in w],
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--coresident-url", default=None,
                    help="llama-server base URL, e.g. http://127.0.0.1:8111")
    ap.add_argument("--coresident-model", default="local-fast")
    ap.add_argument("--kind", default="local-fast")
    args = ap.parse_args()

    try:
        h = http_json(f"{EMBED_URL}/health")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: embedding service unreachable at {EMBED_URL}: {e}")
        return 2

    print(f"service: {h}")
    vram_idle = nvidia("memory.used")
    print(f"gpu used with embedding resident: {vram_idle} MiB\n")

    report = {
        "service": h,
        "gpu_used_with_embedding_mb": vram_idle,
        "batches": [],
        "coresident": None,
    }

    print("=== embedding alone: batch throughput ===")
    report["batches"] = bench_batches()

    best = max(report["batches"], key=lambda b: b["chunks_per_s"] or 0)
    report["best_batch_size"] = best["batch_size"]
    report["best_chunks_per_s"] = best["chunks_per_s"]

    # a fixed corpus is served; report its size too
    try:
        st = http_json(f"{EMBED_URL}/stats")
        report["stats"] = st
    except Exception:
        pass

    if args.coresident_url:
        print(f"\n=== co-residency: embedding + {args.kind} generating ===")
        report["coresident"] = coresident_test(args.kind, args.coresident_url,
                                               args.coresident_model)
        c = report["coresident"]
        print(f"  embedding under load: {c['embedding_chunks_per_s']} chunks/s "
              f"(median {c['embedding_median_s']}s for 32)")
        print(f"  worker gen tps samples: {c['worker_gen_tps_samples']}")
        report["gpu_used_coresident_mb"] = nvidia("memory.used")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    print("\n=== EMBEDDING THROUGHPUT ===")
    print(f"{'batch':>6} {'median_s':>10} {'chunks/s':>10} {'ms/chunk':>10} {'VRAM_MiB':>10}")
    for b in report["batches"]:
        print(f"{b['batch_size']:>6} {b['median_seconds']:>10.4f} {b['chunks_per_s']:>10.1f} "
              f"{b['ms_per_chunk']:>10.2f} {b['vram_peak_mb']:>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
