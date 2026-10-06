"""
bench_reranker_params.py - max_length, batch, cache and load/unload measurements.

Runs IN-PROCESS rather than through the HTTP service, because max_length and batch size
are construction-time settings: measuring them via HTTP would mean five full service
restarts and five model loads, and the load itself is one of the things being measured.

max_length is swept 512/1024/2048/4096 on REAL chunked corpus documents, so the truncation
question is answered on the content this pipeline actually sees, not on synthetic filler.

Quality is measured with the multi-source ground truth: the reranker reorders a dense
top-20 pool and we record where the correct source lands.

Usage:
  python bench_reranker_params.py --out ../../benchmarks/reranker/results/params.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "retrieval"))

from benchmark_data import ALL_SETS, valid_sources  # noqa: E402
from embedder import Embedder, EmbedderConfig  # noqa: E402
from index_store import IndexStore  # noqa: E402
from reranker import GENERAL_INSTRUCTION, Reranker, RerankerConfig  # noqa: E402
from retriever import Retriever, RetrieveConfig  # noqa: E402

INDEX = (Path(__file__).resolve().parents[2] / r"cache\retrieval_bench\indexes\d1024_structured")
DIM = 1024
POOL = 20
TOPK = 5


def vram():
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2**20, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", default="20")
    args = ap.parse_args()
    pool = int(args.pools)

    report: dict = {}

    # ---------------- load / unload cost (cold, measured before anything is warm)
    print("=== load / unload cost ===")
    v0 = vram()
    t0 = time.time()
    rr = Reranker(RerankerConfig(max_length=4096, batch_size=20))
    cold_load = time.time() - t0
    v1 = vram()
    print(f"  cold load {cold_load:.2f}s   vram {v0} -> {v1} MiB  (delta {v1-v0:.0f})")

    # first request latency (includes CUDA kernel autotuning / lazy init)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    rr.score_pairs([("warmup", "a warmup document")], batch_size=1, use_cache=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    first_req = time.time() - t0
    t0 = time.time()
    rr.score_pairs([("warmup", "a warmup document")], batch_size=1, use_cache=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    warm_req = time.time() - t0
    print(f"  first request {first_req*1000:.0f} ms   warm request {warm_req*1000:.0f} ms")

    report["load"] = {
        "cold_load_s": round(cold_load, 3),
        "vram_before_mb": v0, "vram_after_mb": v1,
        "vram_delta_mb": round((v1 - v0), 1) if v0 and v1 else None,
        "first_request_ms": round(first_req * 1000, 1),
        "warm_request_ms": round(warm_req * 1000, 1),
        "steady_vram_mb": v1,
    }

    # ---------------- build a real query/pool set from the corpus
    emb = Embedder(EmbedderConfig(dimension=DIM, batch_size=32, max_seq_length=2048))
    store = IndexStore.load(INDEX, DIM)
    retr = Retriever(emb, store, RetrieveConfig(top_k=pool, instruction=None))
    queries = [q for s in ("doc", "code", "mixed") for q in ALL_SETS[s]]

    cases = []
    for q in queries:
        hits = retr.retrieve(q["q"], top_k=pool, instruction=None)
        if not hits:
            continue
        cases.append({
            "query": q["q"], "type": q["type"], "valid": set(valid_sources(q)),
            "docs": [h.content for h in hits],
            "paths": [h.source_path for h in hits],
        })
    lens = []
    for c in cases:
        for d in c["docs"]:
            lens.append(len(rr.tokenizer.encode(d, add_special_tokens=False)))
    print(f"\n  {len(cases)} queries with a top-{pool} pool")
    print(f"  candidate document token lengths: min={min(lens)} median="
          f"{int(statistics.median(lens))} p90={sorted(lens)[int(.9*len(lens))]} max={max(lens)}")
    report["candidate_token_lengths"] = {
        "min": min(lens), "median": int(statistics.median(lens)),
        "p90": sorted(lens)[int(0.9 * len(lens))], "max": max(lens), "n": len(lens),
    }

    # ---------------- max_length sweep
    print("\n=== max_length sweep (quality + latency) ===")
    print(f"{'max_len':>8} {'body':>6} {'trunc':>7} {'R@1':>6} {'R@5':>6} {'MRR':>7} "
          f"{'mean s':>8} {'p95 s':>7} {'vram':>7}")
    ml_rows = []
    for ml in (512, 1024, 2048, 4096):
        rr.cfg.max_length = ml
        rr.body_budget = ml - len(rr.prefix_tokens) - len(rr.suffix_tokens)
        rr.cache.clear()

        # warm the new shape
        rr.score_pairs([(cases[0]["query"], cases[0]["docs"][0])], batch_size=1,
                       use_cache=False)

        ranks, lat, trunc_total = [], [], 0
        for c in cases:
            t0 = time.time()
            res = rr.rerank(c["query"], c["docs"], instruction=GENERAL_INSTRUCTION,
                            batch_size=pool, top_n=None)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            lat.append(time.time() - t0)
            trunc_total += getattr(rr, "last_truncated", 0)
            ordered = [c["paths"][x.index] for x in res]
            rank = next((i + 1 for i, p in enumerate(ordered) if p in c["valid"]), None)
            ranks.append(rank)

        n = len(ranks)
        r1 = 100 * sum(1 for r in ranks if r == 1) / n
        r5 = 100 * sum(1 for r in ranks if r and r <= TOPK) / n
        mrr = sum(1 for r in ranks if r) / n
        mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / n
        p95 = sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))]
        row = {
            "max_length": ml, "body_budget": rr.body_budget,
            "truncated_pairs": trunc_total,
            "recall@1": round(r1, 1), "recall@5": round(r5, 1), "mrr": round(mrr, 4),
            "mean_s": round(statistics.mean(lat), 4),
            "median_s": round(statistics.median(lat), 4),
            "p95_s": round(p95, 4),
            "vram_mb": vram(),
        }
        ml_rows.append(row)
        print(f"{ml:>8} {rr.body_budget:>6} {trunc_total:>7} {r1:>6.1f} {r5:>6.1f} "
              f"{mrr:>7.4f} {row['mean_s']:>8.4f} {row['p95_s']:>7.4f} {row['vram_mb']:>7}")
    report["max_length_sweep"] = ml_rows

    # ---------------- batch sweep at the chosen max_length
    print("\n=== batch sweep (pairs/s, VRAM, utilisation) ===")
    rr.cfg.max_length = 2048
    rr.body_budget = 2048 - len(rr.prefix_tokens) - len(rr.suffix_tokens)
    # a fixed workload of 64 pairs, scored at each batch size
    flat = []
    for c in cases[:4]:
        for d in c["docs"]:
            flat.append((c["query"], d))
    flat = flat[:64]
    print(f"  workload: {len(flat)} query-document pairs")
    print(f"{'batch':>6} {'passes':>7} {'mean s':>8} {'pairs/s':>9} {'p95 s':>7} {'vram':>7}")
    bs_rows = []
    for bs in (1, 4, 8, 16, 32):
        rr.cache.clear()
        rr.cfg.batch_size = bs
        lat, passes = [], []
        for rep in range(3):
            t0 = time.time()
            rr.score_pairs(flat, batch_size=bs, use_cache=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            lat.append(time.time() - t0)
            passes.append(getattr(rr, "last_batches", None))
        med = statistics.median(lat)
        row = {
            "batch_size": bs, "pairs": len(flat),
            "median_s": round(med, 4),
            "pairs_per_s": round(len(flat) / med, 1) if med else None,
            "p95_s": round(sorted(lat)[-1], 4),
            "forward_passes": passes[0],
            "vram_mb": vram(),
        }
        bs_rows.append(row)
        print(f"{bs:>6} {str(passes[0]):>7} {row['median_s']:>8.4f} "
              f"{row['pairs_per_s']:>9.1f} {row['p95_s']:>7.4f} {row['vram_mb']:>7}")
    report["batch_sweep"] = bs_rows

    # ---------------- the realistic workload: 1 query x N documents
    print("\n=== realistic workload: 1 query x N documents ===")
    print(f"{'pool':>6} {'passes':>7} {'mean s':>8} {'R@5':>6} {'vram':>7}")
    pool_rows = []
    for n in (5, 10, 20, 50):
        rr.cache.clear()
        rr.cfg.batch_size = n
        ranks, lat = [], []
        for c in cases:
            docs = c["docs"][:n]
            paths = c["paths"][:n]
            t0 = time.time()
            res = rr.rerank(c["query"], docs, instruction=GENERAL_INSTRUCTION,
                            batch_size=n, top_n=None)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            lat.append(time.time() - t0)
            ordered = [paths[x.index] for x in res]
            rank = next((i + 1 for i, p in enumerate(ordered) if p in c["valid"]), None)
            ranks.append(rank)
        r5 = 100 * sum(1 for r in ranks if r and r <= TOPK) / len(ranks)
        row = {"pool": n, "median_s": round(statistics.median(lat), 4),
               "mean_s": round(statistics.mean(lat), 4),
               "recall@5": round(r5, 1), "vram_mb": vram(),
               "forward_passes": getattr(rr, "last_batches", None)}
        pool_rows.append(row)
        print(f"{n:>6} {str(row['forward_passes']):>7} {row['mean_s']:>8.4f} "
              f"{r5:>6.1f} {row['vram_mb']:>7}")
    report["pool_sweep"] = pool_rows

    # ---------------- cache effectiveness
    print("\n=== cache ===")
    rr.cache.clear()
    t0 = time.time()
    rr.score_pairs(flat, batch_size=20, use_cache=False)
    cold = time.time() - t0
    t0 = time.time()
    rr.score_pairs(flat, batch_size=20, use_cache=True)
    warm = time.time() - t0
    t0 = time.time()
    rr.score_pairs(flat, batch_size=20, use_cache=True)
    hot = time.time() - t0
    print(f"  cold (no cache)     {cold*1000:.1f} ms")
    print(f"  warm (fill cache)   {warm*1000:.1f} ms")
    print(f"  hot  (all hits)     {hot*1000:.1f} ms   speedup {cold/hot:.0f}x")
    report["cache"] = {
        "pairs": len(flat),
        "cold_ms": round(cold * 1000, 1),
        "fill_ms": round(warm * 1000, 1),
        "hot_ms": round(hot * 1000, 1),
        "speedup": round(cold / hot, 1) if hot else None,
        "stats": rr.cache.stats(),
    }

    # ---------------- unload
    gc.collect()
    v_before_unload = vram()
    rr.close()
    time.sleep(1)
    v_after = vram()
    print(f"\n=== unload ===\n  vram {v_before_unload} -> {v_after} MiB "
          f"(released {v_before_unload - v_after:.0f})")
    report["unload"] = {"vram_before_mb": v_before_unload, "vram_after_mb": v_after,
                        "released_mb": round(v_before_unload - v_after, 1)}

    store.close()
    emb.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
