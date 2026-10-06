"""
retrieval_bench.py - Phase 4 retrieval benchmark with the metric sweeps.

Sweeps, as the specification requires:
  * dimension     256 / 512 / 1024 (Matryoshka)
  * instruction   on / off (query side only)
  * chunking      structured / fixed

Metrics: Recall@1, Recall@3, Recall@5, MRR, nDCG@5.

Ground truth is a file path per query (see benchmark_data.py). A retrieved chunk counts
as relevant when it comes from the expected file - that is the decision a caller
actually makes. Requiring an exact chunk id would measure chunk-boundary luck rather
than retrieval quality.

Each index is cached on disk and REUSED when present, because embedding ~150 chunks per
configuration is the slow part; the FAISS search itself is microseconds. A run can
therefore sweep many query configurations over the same index cheaply, and a rerun with
a changed query set costs nothing.

Run:
  python retrieval_bench.py --build           # build all indexes
  python retrieval_bench.py --eval            # evaluate (builds any missing index)
  python retrieval_bench.py --eval --sets doc
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark_data import ALL_SETS, PROJECT, materialise_corpus  # noqa: E402
from embedder import Embedder, EmbedderConfig  # noqa: E402
from index_builder import IndexBuilder  # noqa: E402
from index_store import IndexStore  # noqa: E402
from retriever import CODE_INSTRUCT, DOC_INSTRUCT  # noqa: E402

WORK = (Path(__file__).resolve().parents[2] / r"cache\retrieval_bench")
CORPUS = WORK / "corpus"
INDEX_ROOT = WORK / "indexes"

DIMENSIONS = [256, 512, 1024]
STRATEGIES = ["structured", "fixed"]


def index_dir(dim: int, strategy: str) -> Path:
    return INDEX_ROOT / f"d{dim}_{strategy}"


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_indexes(dims=DIMENSIONS, strategies=STRATEGIES, force: bool = False,
                  verbose: bool = True) -> dict:
    if not CORPUS.exists() or force:
        n = materialise_corpus(CORPUS)
        print(f"corpus materialised: {n} files -> {CORPUS}")
    out = {}
    # One Embedder per DIMENSION, released before the next dimension is loaded.
    # Constructing one per (dim, strategy) pair leaked a whole model each time; on this
    # 32 GB machine that exhausted the commit limit and the next load died with
    # "OSError: [WinError 1455] page file too small", which points nowhere near the
    # real cause. Reuse within a dimension keeps the peak at one model.
    for dim in dims:
        emb = None
        try:
            for strategy in strategies:
                d = index_dir(dim, strategy)
                marker = d / "built.json"
                if marker.exists() and not force:
                    info = json.loads(marker.read_text(encoding="utf-8"))
                    if verbose:
                        print(f"  reuse d{dim} {strategy}: {info['chunks']} chunks")
                    out[(dim, strategy)] = info
                    continue
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                if emb is None:
                    emb = Embedder(EmbedderConfig(dimension=dim, batch_size=32,
                                                  max_seq_length=2048))
                store = IndexStore(d, dim)
                builder = IndexBuilder(emb, store, strategy=strategy, max_tokens=900)
                t0 = time.time()
                rep = builder.build(PROJECT, CORPUS)
                info = {
                    "dimension": dim, "strategy": strategy, "chunks": rep.chunks_embedded,
                    "files": rep.files_new, "seconds": round(time.time() - t0, 2),
                    "vram_mb": emb.vram_mb(),
                }
                store.close()
                marker.write_text(json.dumps(info), encoding="utf-8")
                out[(dim, strategy)] = info
                if verbose:
                    print(f"  built d{dim} {strategy}: {rep.chunks_embedded} chunks "
                          f"from {rep.files_new} files in {info['seconds']}s", flush=True)
        finally:
            if emb is not None:
                emb.close()
    return out


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(hit_flags: list[bool], k: int) -> float:
    """Binary-gain nDCG@k with a single relevant document."""
    gains = [1.0 if f else 0.0 for f in hit_flags[:k]]
    ideal = [1.0] + [0.0] * (len(gains) - 1) if gains else []
    d = dcg(gains)
    i = dcg(ideal)
    return (d / i) if i > 0 else 0.0


def evaluate_query(retriever, q: dict, k_max: int = 5) -> dict:
    expected = q["file"]
    qtype = q["type"]
    # code-concept and code queries use the code instruction; everything else the
    # general retrieval instruction. instruction=None is handled by the caller.
    instr = CODE_INSTRUCT if qtype.startswith("code") else DOC_INSTRUCT
    hits = retriever.retrieve(q["q"], top_k=k_max, instruction=instr)
    flags = [h.source_path == expected for h in hits]
    rank = next((i + 1 for i, f in enumerate(flags) if f), None)
    return {
        "query": q["q"], "type": qtype, "expected": expected,
        "rank": rank,
        "recall_at_1": bool(flags[:1] and flags[0]),
        "recall_at_3": any(flags[:3]),
        "recall_at_5": any(flags[:5]),
        "rr": (1.0 / rank) if rank else 0.0,
        "ndcg_at_5": ndcg_at_k(flags, 5),
        "top_paths": [h.source_path for h in hits[:5]],
        "top_scores": [round(h.score, 4) for h in hits[:5]],
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    return {
        "n": n,
        "recall@1": round(100 * sum(r["recall_at_1"] for r in rows) / n, 1),
        "recall@3": round(100 * sum(r["recall_at_3"] for r in rows) / n, 1),
        "recall@5": round(100 * sum(r["recall_at_5"] for r in rows) / n, 1),
        "mrr": round(sum(r["rr"] for r in rows) / n, 4),
        "ndcg@5": round(sum(r["ndcg_at_5"] for r in rows) / n, 4),
    }


def evaluate_config(dim: int, strategy: str, sets: list[str], instruction: bool,
                    verbose: bool = False) -> dict:
    store = IndexStore.load(index_dir(dim, strategy), dim)
    emb = Embedder(EmbedderConfig(dimension=dim, batch_size=32, max_seq_length=2048))
    from retriever import Retriever, RetrieveConfig
    retr = Retriever(emb, store, RetrieveConfig(top_k=5, instruction=None))
    try:
        all_rows, per_set = [], {}
        for s in sets:
            rows = []
            for q in ALL_SETS[s]:
                if not instruction:
                    # force no instruction regardless of query type
                    hits = retr.retrieve(q["q"], top_k=5, instruction=None)
                    flags = [h.source_path == q["file"] for h in hits]
                    rank = next((i + 1 for i, f in enumerate(flags) if f), None)
                    rows.append({
                        "query": q["q"], "type": q["type"], "expected": q["file"],
                        "rank": rank,
                        "recall_at_1": bool(flags[:1] and flags[0]),
                        "recall_at_3": any(flags[:3]), "recall_at_5": any(flags[:5]),
                        "rr": (1.0 / rank) if rank else 0.0,
                        "ndcg_at_5": ndcg_at_k(flags, 5),
                        "top_paths": [h.source_path for h in hits[:5]],
                        "top_scores": [round(h.score, 4) for h in hits[:5]],
                    })
                else:
                    rows.append(evaluate_query(retr, q))
            per_set[s] = {"metrics": aggregate(rows), "rows": rows}
            all_rows.extend(rows)

        return {
            "dimension": dim, "strategy": strategy, "instruction": instruction,
            "sets": sets, "overall": aggregate(all_rows), "per_set": per_set,
        }
    finally:
        # Release the model even if a query raises; each config loads a fresh one and
        # the sweep runs 12 of them back to back.
        store.close()
        emb.close()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild indexes")
    ap.add_argument("--dims", default=",".join(map(str, DIMENSIONS)))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--sets", default="doc,code,mixed")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / r"benchmarks\retrieval\results"))
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    sets = [x.strip() for x in args.sets.split(",") if x.strip()]

    if args.build or args.eval:
        print("=== building/validating indexes ===")
        build_indexes(dims, strategies, force=args.force)

    if not args.eval:
        return 0

    print("\n=== evaluating ===")
    results = []
    for strategy in strategies:
        for dim in dims:
            for instruction in (True, False):
                t0 = time.time()
                r = evaluate_config(dim, strategy, sets, instruction)
                r["seconds"] = round(time.time() - t0, 2)
                results.append(r)
                o = r["overall"]
                print(f"  d{dim:<4} {strategy:<10} instr={str(instruction):<5} "
                      f"R@1={o['recall@1']:5.1f}  R@3={o['recall@3']:5.1f}  "
                      f"R@5={o['recall@5']:5.1f}  MRR={o['mrr']:.3f}  "
                      f"nDCG@5={o['ndcg@5']:.3f}  ({r['seconds']}s)", flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "retrieval_sweep.json"
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")

    # compact table for the report
    print("\n" + "=" * 96)
    print(f"{'dim':>5} {'chunking':<11} {'instr':<6} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'nDCG@5':>8}")
    print("=" * 96)
    for r in sorted(results, key=lambda x: (-x["overall"]["recall@5"], -x["overall"]["mrr"])):
        o = r["overall"]
        print(f"{r['dimension']:>5} {r['strategy']:<11} {str(r['instruction']):<6} "
              f"{o['recall@1']:>7.1f} {o['recall@3']:>7.1f} {o['recall@5']:>7.1f} "
              f"{o['mrr']:>7.3f} {o['ndcg@5']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
