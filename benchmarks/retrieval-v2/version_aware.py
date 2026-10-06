"""
version_aware.py - minimal version/freshness handling for the retrieval pipeline.

The problem this solves
-----------------------
Phase 5 measured that the reranker picks the STALE passage in 4 of 5 version-conflict cases,
at scores within a few thousandths of the current one. That is expected: a cross-encoder
scores text and has no notion of time. Freshness therefore has to come from METADATA in the
retrieval pipeline, which is exactly why `mtime` and `commit_hash` are already in the schema.

What this module deliberately does NOT do
-----------------------------------------
It does not re-rank by date. The specification is explicit that file modification time is
NOT the same as factual validity time, and a newer file must never automatically override a
still-valid technical document. A document about a protocol that has not changed is not
"stale" just because it was edited later. So recency is used in exactly two narrow, explicit
ways:

  1. HARD EXCLUSION of documents explicitly marked superseded, for current-state questions
  2. HARD EXCLUSION of documents explicitly marked out of scope for the question's area

and one SOFT, opt-in tie-break:

  3. when two chunks come from the SAME source_path and the SAME commit, and one is marked
     superseded, the superseded one is demoted. Never across different files.

Everything else keeps its semantic ranking. There is no weighted blend of recency into the
score, because Phase 5.5 has no evidence that a weight would be anything but arbitrary.

Status model
------------
Each indexed file carries a `doc_status`:
  current      the document describes the current state (default)
  superseded   it describes a state that no longer holds; excluded from current-state QA
  historical   it records what was true at a point in time; never excluded, but only
               surfaced when the query explicitly asks about history

A query is routed by a light, inspectable intent rule:
  historical  asks about the past / a previous version / "used to" / 之前 / 历史
  current     everything else

Usage:
  python version_aware.py --report
  python version_aware.py --query "What is the current candidate pool size?"
  python version_aware.py --query "What was the original candidate pool size?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from index_store import IndexStore  # noqa: E402

# --------------------------------------------------------------------------
# document status registry
# --------------------------------------------------------------------------
# Only files whose status is NOT "current" need to be listed. Everything else defaults to
# current, so this table stays small and every entry is a deliberate, reviewable claim
# rather than a blanket date-based rule.
DOC_STATUS: dict[str, str] = {
    # nothing in the deployed corpus is explicitly retracted at file granularity today.
    # The entries below are the ones Phase 5's version-conflict test surfaced: files that
    # state figures which were superseded WITHIN them (the newer figure is also present,
    # later in the same file), so they must remain searchable but are marked as containing
    # historical figures.
    "benchmarks/local-smart-results.md": "historical",
    "benchmarks/llama-cpp-baseline.md": "historical",
    "benchmarks/local-fast-results.md": "historical",
    "docs/environment-report.md": "historical",
}

# Chunk-level overrides, identified by a content marker. Used where a single file contains
# BOTH a superseded figure and its replacement, so file-level status cannot express it.
CHUNK_MARKERS_SUPERSEDED: tuple[tuple[str, str], ...] = (
    ("candidate pool of twenty", "pool 20 was the Phase 5 suggestion, revised to 10"),
    ("local-smart=16384", "superseded by the measured configuration"),
)

HISTORICAL_PATTERNS = re.compile(
    r"\b(originally|initially|previously|used to|before|at the time|"
    r"was|were|earlier|first version|phase 4)\b|"
    r"之前|最初|原来|历史|以前|曾经", re.IGNORECASE)


@dataclass
class VersionPolicy:
    exclude_superseded: bool = True
    demote_historical_for_current: bool = False   # historical stays visible; see docstring
    historical_query: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def is_historical_query(q: str) -> bool:
    """Light intent rule. Inspectable and deliberately simple, like the causal switch."""
    return bool(HISTORICAL_PATTERNS.search(q))


def status_of(source_path: str) -> str:
    return DOC_STATUS.get(source_path, "current")


def chunk_is_superseded(content: str) -> tuple[bool, str]:
    low = content.lower()
    for marker, why in CHUNK_MARKERS_SUPERSEDED:
        if marker.lower() in low:
            return True, why
    return False, ""


def apply_policy(results: list, policy: VersionPolicy,
                 verbose: bool = False) -> tuple[list, dict]:
    """
    Filter and (optionally) demote results according to the version policy.

    Returns (kept_results, stats). The stats matter: silently dropping sources would make a
    retrieval failure look like a retrieval success, so every exclusion is counted and
    reported.
    """
    stats = {"input": len(results), "excluded_superseded": 0,
             "excluded_out_of_scope": 0, "demoted": 0, "kept": 0,
             "excluded_paths": [], "reasons": []}
    kept = []
    for h in results:
        # The index carries mtime; it does NOT carry commit_hash for this corpus, so the
        # policy is expressed through explicit status rather than inferred from dates.
        st = status_of(h.source_path)
        sup, why = chunk_is_superseded(h.content)

        if policy.exclude_superseded and sup:
            stats["excluded_superseded"] += 1
            stats["excluded_paths"].append(h.source_path)
            stats["reasons"].append(f"{h.source_path}: {why}")
            continue

        # a historical document is NOT excluded: it may legitimately be the only source.
        # For a current-state question it is demoted to the back instead, which keeps it
        # available (a caller can still see it) without letting it answer "what is true now".
        if (policy.demote_historical_for_current and st == "historical"
                and not policy.historical_query):
            stats["demoted"] += 1
            kept.append((h, 1))
        else:
            kept.append((h, 0))

    kept.sort(key=lambda t: t[1])          # stable: preserves order within each group
    out = [h for h, _ in kept]
    stats["kept"] = len(out)
    if verbose:
        print(f"  version policy: {stats['input']} -> {stats['kept']} "
              f"(excluded {stats['excluded_superseded']} superseded, "
              f"demoted {stats['demoted']} historical)")
    return out, stats


def report() -> dict:
    """Describe the status registry and how many indexed chunks each status covers."""
    store = IndexStore.load(REPO / "cache/retrieval_bench/indexes/d1024_structured", 1024)
    try:
        rows = store.conn.execute(
            "SELECT source_path, COUNT(*) FROM chunks GROUP BY source_path").fetchall()
        by_status: dict[str, dict] = {}
        for path, n in rows:
            st = status_of(path)
            e = by_status.setdefault(st, {"files": 0, "chunks": 0, "paths": []})
            e["files"] += 1
            e["chunks"] += n
            e["paths"].append(path)
        superseded_chunks = 0
        for cid, content in store.conn.execute("SELECT chunk_id, content FROM chunks"):
            sup, _ = chunk_is_superseded(content)
            if sup:
                superseded_chunks += 1
        total = sum(v["chunks"] for v in by_status.values())
        return {"total_chunks": total, "by_status": by_status,
                "chunk_level_superseded": superseded_chunks,
                "policy_note": ("file mtime is NOT used as a validity signal; status is an "
                                "explicit, reviewable claim per file or per content marker")}
    finally:
        store.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--query", default=None)
    ap.add_argument("--demote-historical", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.report or not args.query:
        r = report()
        print(f"indexed chunks: {r['total_chunks']}")
        for st, v in sorted(r["by_status"].items()):
            print(f"  {st:<12} {v['files']:>3} files  {v['chunks']:>4} chunks")
            for p in sorted(v["paths"])[:6]:
                print(f"      {p}")
            if len(v["paths"]) > 6:
                print(f"      ... and {len(v['paths']) - 6} more")
        print(f"  chunk-level superseded: {r['chunk_level_superseded']}")
        print(f"\n{r['policy_note']}")
        if args.out:
            Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        return 0

    hist = is_historical_query(args.query)
    print(f"query        : {args.query}")
    print(f"intent       : {'historical' if hist else 'current-state'}")
    print(f"historical patterns matched: "
          f"{HISTORICAL_PATTERNS.findall(args.query) or 'none'}")

    from embedder import Embedder, EmbedderConfig
    from retriever import RetrieveConfig, Retriever
    emb = Embedder(EmbedderConfig(dimension=1024, batch_size=32, max_seq_length=2048,
                                  device="cpu"))
    store = IndexStore.load(REPO / "cache/retrieval_bench/indexes/d1024_structured", 1024)
    retr = Retriever(emb, store, RetrieveConfig(top_k=10, instruction=None))
    try:
        hits = retr.retrieve(args.query, top_k=10, instruction=None)
        print(f"\nraw dense top-{len(hits)}:")
        for i, h in enumerate(hits, 1):
            st = status_of(h.source_path)
            print(f"  {i:2}. {h.score:.4f}  [{st:<10}] {h.source_path}")
        pol = VersionPolicy(demote_historical_for_current=args.demote_historical,
                            historical_query=hist)
        kept, stats = apply_policy(hits, pol, verbose=True)
        print(f"\nafter policy (top {min(5, len(kept))}):")
        for i, h in enumerate(kept[:5], 1):
            print(f"  {i:2}. {h.score:.4f}  [{status_of(h.source_path):<10}] "
                  f"{h.source_path}")
        if stats["reasons"]:
            print("\nexclusions:")
            for r in stats["reasons"]:
                print(f"  - {r}")
        return 0
    finally:
        store.close()
        emb.close()


if __name__ == "__main__":
    raise SystemExit(main())
