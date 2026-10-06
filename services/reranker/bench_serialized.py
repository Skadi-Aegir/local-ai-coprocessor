"""
bench_serialized.py - serialized vs concurrent GPU scheduling for the Phase 5 pipeline.

Success criterion 13 asks for a serialized-GPU-scheduling recommendation. Phase 4 showed
concurrent embedding + generation loses throughput, but Phase 5 adds the reranker, so the
recommendation must be re-measured on the CURRENT pipeline rather than inherited.

Two workloads, the same total work each time:

  SERIALIZED   embedding batches finish, THEN reranking runs
  CONCURRENT   embedding and reranking issue at the same time from separate threads

Reported per workload, so neither is credited with the other's idle time. The embedding
service and the reranker are separate processes on the same GPU, which is exactly the
situation the recommendation applies to.

Usage:
  python bench_serialized.py --out ../../benchmarks/reranker/results/serialized_vs_concurrent.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "retrieval"))

EMBED = "http://127.0.0.1:8113"
RERANK = "http://127.0.0.1:8114"

EMBED_TEXTS = [f"chunk {i}: " + "technical sentence about retrieval and ranking. " * 10
               for i in range(32)]
RERANK_DOCS = [f"document {i}: " + "candidate passage about GPU memory and indexing. " * 12
               for i in range(20)]
RERANK_QUERY = "how does retrieval use GPU memory"


def post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def do_embed(n: int = 4) -> tuple[int, float]:
    t0 = time.time()
    for _ in range(n):
        post(f"{EMBED}/v1/embeddings",
             {"model": "local-embedding", "input": EMBED_TEXTS})
    return n, time.time() - t0


def do_rerank(n: int = 4) -> tuple[int, float]:
    """
    Rerank with the cache DISABLED and a unique query each time.

    A first version reused the same query, so every call after the first was a cache hit:
    four queries completed in 0.043 s and the concurrency comparison was meaningless
    because there was almost no second workload left to overlap with. Cache hits are
    reported separately as their own result; they must not be allowed to masquerade as
    GPU contention measurements.
    """
    t0 = time.time()
    for i in range(n):
        post(f"{RERANK}/v1/rerank",
             {"query": f"{RERANK_QUERY} (variant {time.time_ns()}-{i})",
              "documents": RERANK_DOCS, "top_n": 5,
              "instruction_type": "general", "use_cache": False})
    return n, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    # warm both services
    do_embed(1)
    do_rerank(1)

    report: dict = {"rounds": args.rounds, "reps_per_round": args.reps}

    # ---------------- serialized
    print("=== SERIALIZED (embedding finishes, then reranking) ===")
    emb_times, rr_times, wall_serial = [], [], []
    for _ in range(args.rounds):
        t0 = time.time()
        _, te = do_embed(args.reps)
        _, tr = do_rerank(args.reps)
        wall_serial.append(time.time() - t0)
        emb_times.append(te)
        rr_times.append(tr)
    s_emb = statistics.median(emb_times)
    s_rr = statistics.median(rr_times)
    s_wall = statistics.median(wall_serial)
    print(f"  embedding {args.reps} batches : {s_emb:.3f}s")
    print(f"  rerank    {args.reps} queries: {s_rr:.3f}s")
    print(f"  total wall                    : {s_wall:.3f}s")

    # ---------------- concurrent
    print("\n=== CONCURRENT (both issued simultaneously) ===")
    wall_conc, c_emb, c_rr = [], [], []
    for _ in range(args.rounds):
        box: dict = {}

        def th_embed():
            box["e"] = do_embed(args.reps)

        def th_rerank():
            box["r"] = do_rerank(args.reps)

        t0 = time.time()
        a = threading.Thread(target=th_embed)
        b = threading.Thread(target=th_rerank)
        a.start(); b.start(); a.join(); b.join()
        wall_conc.append(time.time() - t0)
        c_emb.append(box["e"][1])
        c_rr.append(box["r"][1])
    cc_emb = statistics.median(c_emb)
    cc_rr = statistics.median(c_rr)
    cc_wall = statistics.median(wall_conc)
    print(f"  embedding {args.reps} batches : {cc_emb:.3f}s  (vs {s_emb:.3f}s serialized)")
    print(f"  rerank    {args.reps} queries: {cc_rr:.3f}s  (vs {s_rr:.3f}s serialized)")
    print(f"  total wall                    : {cc_wall:.3f}s  (vs {s_wall:.3f}s serialized)")

    report["serialized"] = {
        "embedding_s": round(s_emb, 3), "rerank_s": round(s_rr, 3),
        "wall_s": round(s_wall, 3),
        "sum_of_parts_s": round(s_emb + s_rr, 3),
    }
    report["concurrent"] = {
        "embedding_s": round(cc_emb, 3), "rerank_s": round(cc_rr, 3),
        "wall_s": round(cc_wall, 3),
        "embedding_slowdown_pct": round(100 * (cc_emb - s_emb) / s_emb, 1) if s_emb else None,
        "rerank_slowdown_pct": round(100 * (cc_rr - s_rr) / s_rr, 1) if s_rr else None,
        "wall_change_pct": round(100 * (cc_wall - s_wall) / s_wall, 1) if s_wall else None,
    }

    # Concurrency is only worth it if total wall time drops. Each INDIVIDUAL workload is
    # expected to slow down; what matters is whether the overlap wins overall.
    gain = 100 * (s_wall - cc_wall) / s_wall if s_wall else 0
    report["wall_time_gain_pct"] = round(gain, 1)
    report["verdict"] = (
        "concurrent wins on wall time" if gain > 5
        else "serialized recommended: concurrency does not reduce total wall time")

    print(f"\n  wall-time gain from concurrency: {gain:+.1f}%")
    print(f"  VERDICT: {report['verdict']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
