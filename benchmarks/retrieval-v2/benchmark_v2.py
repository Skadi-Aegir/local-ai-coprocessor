"""
benchmark_v2.py - Phase 5.5 extension of the retrieval benchmark.

The 39 original queries are NOT redefined here; they remain authoritative in
`services/retrieval/benchmark_data.py` and are re-exported unchanged. This module adds 44
new human-labelled queries and the machinery to evaluate the combined set.

Categories, chosen to cover what the original set under-tested:

  causal            asks WHY a decision was made. Phase 5 found these are where the
                    reranker fails, because it prefers passages that DESCRIBE a mechanism
                    over passages that EXPLAIN a rationale (hn19 inverted under every
                    instruction tested).
  cross_doc         the answer only exists by combining two or more sources. A single
                    chunk is insufficient, so top-1 accuracy is not the right target.
  version_conflict  two sources state different values for the same fact, and only one is
                    current. Retrieval RELEVANCE and factual RECENCY are different things
                    and are labelled separately.
  near_duplicate    several sources state nearly the same thing; all are relevant, so this
                    tests ranking rather than recall.

DEV / HELD-OUT SPLIT
  Every new query carries `split`: "dev" or "test". Tuning (instructions, pools, rules) may
  only look at dev. The held-out set is scored once, at the end, so a tuned result cannot
  be an artefact of fitting the queries that were used to choose the settings.

LABELLING RULE
  `sources` lists every file that genuinely answers the query, verified by reading the
  file. Files that merely mention the topic are NOT included: widening ground truth until
  everything passes is the standard-lowering this project forbids. Where a query has only
  one true source, the list is a singleton.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from benchmark_data import ALL_SETS, valid_sources  # noqa: E402

# --------------------------------------------------------------------------
# NEW QUERIES
# --------------------------------------------------------------------------
NEW_QUERIES: list[dict] = [
    # ================= CAUSAL (why was this decided?) — 14 =================
    dict(id="c01", split="dev", category="causal",
         q="Why was Q4_K_M selected instead of the larger Q5_K_M quantisation?",
         sources=["benchmarks/local-smart-results.md", "docs/models/local-smart.md",
                  "docs/models/local-fast.md"]),
    dict(id="c02", split="dev", category="causal",
         q="Why is thinking mode kept disabled for the fast worker?",
         sources=["benchmarks/local-fast-results.md", "docs/models/local-fast.md",
                  "docs/local-fast-api.md"]),
    dict(id="c03", split="dev", category="causal",
         q="Why was the small worker not made resident by default?",
         sources=["docs/escalation-policy.md", "benchmarks/local-smart-results.md",
                  "docs/models/local-smart.md"]),
    dict(id="c04", split="dev", category="causal",
         q="Why does the reranker need a token budget instead of a fixed batch size?",
         sources=["docs/models/local-reranker.md", "docs/reranking-architecture.md"]),
    dict(id="c05", split="dev", category="causal",
         q="Why was a candidate pool of ten chosen over twenty?",
         sources=["docs/reranking-architecture.md", "benchmarks/reranker-results.md"]),
    dict(id="c06", split="dev", category="causal",
         q="Why is structure-aware chunking preferred over fixed windows?",
         sources=["docs/retrieval-architecture.md", "benchmarks/retrieval-results.md"]),
    dict(id="c07", split="dev", category="causal",
         q="Why must the reranker not manage the FAISS index itself?",
         sources=["docs/reranking-architecture.md", "docs/models/local-reranker.md"]),
    dict(id="c08", split="test", category="causal",
         q="Why does the launcher refuse to start when free VRAM is too low?",
         sources=["docs/local-fast-api.md", "config/local-fast.yaml",
                  "docs/models/local-fast.md"]),
    dict(id="c09", split="test", category="causal",
         q="Why is the context window capped well below what the model supports?",
         sources=["docs/models/local-fast.md", "config/local-fast.yaml",
                  "docs/local-fast-api.md"]),
    dict(id="c10", split="test", category="causal",
         q="Why does incremental indexing compare file hashes rather than timestamps?",
         sources=["docs/retrieval-architecture.md"]),
    dict(id="c11", split="test", category="causal",
         q="Why was the embedding service given its own virtual environment?",
         sources=["docs/models/local-embedding.md"]),
    dict(id="c12", split="test", category="causal",
         q="Why did the project stop treating the small worker as the default escalation tier?",
         sources=["docs/escalation-policy.md", "benchmarks/local-smart-results.md"]),
    dict(id="c13", split="test", category="causal",
         q="Why is a reranker needed at all when dense retrieval already returns candidates?",
         sources=["docs/reranking-architecture.md", "benchmarks/reranker-results.md"]),
    dict(id="c14", split="test", category="causal",
         q="Why does the reranker score for a higher VRAM peak when running alone?",
         sources=["docs/models/local-reranker.md", "benchmarks/reranker-results.md"]),

    # ================= CROSS-DOCUMENT INTEGRATION — 12 =================
    dict(id="x01", split="dev", category="cross_doc",
         q="Which port serves each local service, and where is each port configured?",
         sources=["config/local-fast.yaml", "config/local-embedding.yaml",
                  "config/local-reranker.yaml", "config/local-smart.yaml"]),
    dict(id="x02", split="dev", category="cross_doc",
         q="How is VRAM shared between the generative worker and the retrieval services?",
         sources=["benchmarks/embedding-results.md", "benchmarks/reranker-results.md",
                  "docs/reranking-architecture.md"]),
    dict(id="x03", split="dev", category="cross_doc",
         q="What has to be running for a full retrieval-augmented answer to succeed?",
         sources=["docs/reranking-architecture.md", "docs/local-fast-api.md"]),
    dict(id="x04", split="dev", category="cross_doc",
         q="Which documents record the measured throughput of each local model?",
         sources=["benchmarks/local-fast-results.md", "benchmarks/embedding-results.md",
                  "benchmarks/reranker-results.md"]),
    dict(id="x05", split="dev", category="cross_doc",
         q="How do the licensing terms differ across the models this project deploys?",
         sources=["docs/models/local-fast.md", "docs/models/local-embedding.md",
                  "docs/models/local-reranker.md"]),
    dict(id="x06", split="dev", category="cross_doc",
         q="What are the failure modes if a local service is stopped mid-request?",
         sources=["docs/local-fast-api.md", "docs/reranking-architecture.md",
                  "scripts/stop-local-fast.ps1"]),
    dict(id="x07", split="test", category="cross_doc",
         q="Which safety margins are enforced, and by which component?",
         sources=["config/local-fast.yaml", "docs/local-fast-api.md",
                  "scripts/start-local-fast.ps1"]),
    dict(id="x08", split="test", category="cross_doc",
         q="How does a caller discover which retrieval configuration is active?",
         sources=["docs/reranking-architecture.md", "config/local-reranker.yaml",
                  "docs/retrieval-architecture.md"]),
    dict(id="x09", split="test", category="cross_doc",
         q="What evidence exists that retrieval actually improved final answer quality?",
         sources=["benchmarks/reranker-results.md", "benchmarks/retrieval-results.md"]),
    dict(id="x10", split="test", category="cross_doc",
         q="Which components keep a pid file, and what happens if it is stale?",
         sources=["scripts/stop-local-fast.ps1", "scripts/stop-local-embedding.ps1",
                  "scripts/stop-local-reranker.ps1"]),
    dict(id="x11", split="test", category="cross_doc",
         q="How is the embedding service's health verified beyond a simple ping?",
         sources=["scripts/check-local-embedding.ps1", "docs/models/local-embedding.md"]),
    dict(id="x12", split="test", category="cross_doc",
         q="What is the full path from a raw file on disk to a packed context block?",
         sources=["docs/retrieval-architecture.md", "docs/reranking-architecture.md"]),

    # ================= VERSION CONFLICT — 10 =================
    # `sources` = files that contain the CURRENT answer. `superseded` = files that state a
    # value which is no longer correct. Relevance and recency are labelled separately, so a
    # retriever that surfaces the stale file is scored as recency-wrong, not relevance-wrong.
    dict(id="v01", split="dev", category="version_conflict",
         q="What is the current default candidate pool size for retrieval?",
         sources=["docs/reranking-architecture.md", "config/local-reranker.yaml"],
         superseded=["benchmarks/reranker-results.md"],
         note="Phase 5 measured pool 10 beating the originally suggested 20"),
    dict(id="v02", split="dev", category="version_conflict",
         q="What is the current recommended embedding batch size?",
         sources=["docs/models/local-embedding.md", "config/local-embedding.yaml"],
         superseded=[]),
    dict(id="v03", split="dev", category="version_conflict",
         q="How many chunks does the served retrieval index currently hold?",
         sources=["docs/models/local-embedding.md"],
         superseded=[],
         note="index size changed repeatedly during Phase 4 and 5"),
    dict(id="v04", split="dev", category="version_conflict",
         q="What is the current strict Recall@5 for dense retrieval?",
         sources=["benchmarks/retrieval-results.md"],
         superseded=[],
         note="Phase 4 reported 74.4, Phase 5 re-measured 66.7 on a smaller index"),
    dict(id="v05", split="dev", category="version_conflict",
         q="Is the small worker still part of the default escalation path?",
         sources=["docs/escalation-policy.md"],
         superseded=["docs/models/local-smart.md"]),
    dict(id="v06", split="test", category="version_conflict",
         q="What is the current value of the embedding dimension default?",
         sources=["config/local-embedding.yaml", "docs/models/local-embedding.md"],
         superseded=[]),
    dict(id="v07", split="test", category="version_conflict",
         q="What is the current recommended reranker max length?",
         sources=["config/local-reranker.yaml", "docs/models/local-reranker.md"],
         superseded=["benchmarks/reranker-results.md"],
         note="512 measured faster but was deliberately not adopted"),
    dict(id="v08", split="test", category="version_conflict",
         q="Which quantisation is actually deployed for the fast worker now?",
         sources=["docs/models/local-fast.md", "config/local-fast.yaml"],
         superseded=["benchmarks/local-smart-results.md"]),
    dict(id="v09", split="test", category="version_conflict",
         q="What is the current claim about how many queries the retrieval benchmark holds?",
         sources=[],
         superseded=["benchmarks/retrieval-results.md"],
         note="39 in Phase 4/5, extended in Phase 5.5; no corpus file states the new count"),
    dict(id="v10", split="test", category="version_conflict",
         q="What is the current port for the reranker service?",
         sources=["config/local-reranker.yaml", "docs/models/local-reranker.md"],
         superseded=[]),

    # ================= NEAR-DUPLICATE / HIGH-SIMILARITY NEGATIVES — 8 =================
    # Every listed source is genuinely relevant; several say almost the same thing. This
    # tests ranking and deduplication, not recall.
    dict(id="n01", split="dev", category="near_duplicate",
         q="How much VRAM does the four billion parameter worker occupy?",
         sources=["benchmarks/local-fast-results.md", "benchmarks/local-smart-results.md",
                  "docs/models/local-fast.md"]),
    dict(id="n02", split="dev", category="near_duplicate",
         q="What is the embedding service's memory footprint?",
         sources=["benchmarks/embedding-results.md", "docs/models/local-embedding.md"]),
    dict(id="n03", split="dev", category="near_duplicate",
         q="What load time does the reranker have?",
         sources=["docs/models/local-reranker.md", "benchmarks/reranker-results.md"]),
    dict(id="n04", split="dev", category="near_duplicate",
         q="What is the measured generation throughput of the fast worker?",
         sources=["benchmarks/local-fast-results.md", "docs/models/local-fast.md",
                  "docs/escalation-policy.md"]),
    dict(id="n05", split="test", category="near_duplicate",
         q="How long does the embedding model take to load?",
         sources=["docs/models/local-embedding.md", "benchmarks/embedding-results.md"]),
    dict(id="n06", split="test", category="near_duplicate",
         q="What is the throughput of the reranker in pairs per second?",
         sources=["benchmarks/reranker-results.md"]),
    dict(id="n07", split="test", category="near_duplicate",
         q="How much free VRAM remains when the small worker runs with the embedding service?",
         sources=["benchmarks/embedding-results.md", "benchmarks/local-smart-results.md"]),
    dict(id="n08", split="test", category="near_duplicate",
         q="What is the token reduction achieved by retrieval?",
         sources=["benchmarks/retrieval-results.md", "docs/retrieval-architecture.md"]),
]


# --------------------------------------------------------------------------
# combined views
# --------------------------------------------------------------------------

def original_queries() -> list[dict]:
    """The 39 Phase 4/5 queries, unchanged, tagged with their original source list."""
    out = []
    for setname, qs in ALL_SETS.items():
        for i, q in enumerate(qs):
            out.append({
                "id": f"orig_{setname}_{i:02d}",
                "split": "test",          # never tuned on: already published
                "category": q["type"],
                "set": setname,
                "q": q["q"],
                "sources": valid_sources(q),
                "original": True,
            })
    return out


def all_queries(splits: tuple[str, ...] | None = None,
                categories: tuple[str, ...] | None = None) -> list[dict]:
    qs = original_queries() + [
        {**q, "original": False} for q in NEW_QUERIES
    ]
    if splits:
        qs = [q for q in qs if q["split"] in splits]
    if categories:
        qs = [q for q in qs if q["category"] in categories]
    return qs


def summary() -> dict:
    qs = all_queries()
    by_split: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for q in qs:
        by_split[q["split"]] = by_split.get(q["split"], 0) + 1
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    new = [q for q in qs if not q["original"]]
    return {
        "total": len(qs),
        "original": len(qs) - len(new),
        "new": len(new),
        "by_split": by_split,
        "by_category": by_cat,
        "new_by_category": {c: sum(1 for q in new if q["category"] == c)
                            for c in sorted({q["category"] for q in new})},
        "new_by_split": {s: sum(1 for q in new if q["split"] == s)
                         for s in sorted({q["split"] for q in new})},
    }


if __name__ == "__main__":
    import json
    s = summary()
    print(json.dumps(s, indent=1))
    # verify every labelled source exists on disk
    missing = []
    for q in all_queries():
        for src in q["sources"]:
            if not (REPO / src).is_file():
                missing.append((q["id"], src))
    print(f"\nlabelled sources missing from the repo: {len(missing)}")
    for m in missing:
        print("  ", m)
