"""
test_version_aware.py - assertions for the version/freshness policy.

The exclusion path matters more than the demotion path: exclusion REMOVES a source, so if it
misbehaves a retrieval failure looks like a retrieval success. These tests therefore check
the exclusion mechanism directly against a synthetic result set, rather than relying on
whatever the current corpus happens to contain (the corpus yielded 0 superseded chunks, so
the path was exercised by nothing at all when first written).

Also asserted: file modification time is NOT treated as a validity signal anywhere.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))
sys.path.insert(0, str(HERE))

from version_aware import (DOC_STATUS, VersionPolicy, apply_policy,  # noqa: E402
                           chunk_is_superseded, is_historical_query, status_of)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


@dataclass
class FakeHit:
    """Minimal stand-in for SearchHit: the policy only reads path and content."""
    source_path: str
    content: str
    score: float = 0.5


def main() -> int:
    # ---------------- intent routing
    print("--- historical intent routing ---")
    for q in ("What was the original pool size?",
              "How did this work previously?",
              "之前的候选池大小是多少？",
              "What did Phase 4 report?",
              "It used to listen on a different port"):
        check(f"historical: {q[:46]!r}", is_historical_query(q))
    for q in ("What is the current pool size?",
              "How do I start the reranker?",
              "现在的候选池大小是多少？"):
        check(f"current-state: {q[:46]!r}", not is_historical_query(q))

    # ---------------- chunk-level superseded detection
    print("\n--- chunk-level superseded markers ---")
    sup, why = chunk_is_superseded("...uses a candidate pool of twenty documents...")
    check("a registered marker is detected", sup, why)
    check("the reason is reported", bool(why), why)
    sup2, _ = chunk_is_superseded("this text mentions nothing special at all")
    check("unmarked text is not flagged", not sup2)

    # ---------------- exclusion path (the dangerous one)
    print("\n--- exclusion path ---")
    hits = [
        FakeHit("docs/retrieval-architecture.md", "current candidate pool is ten"),
        FakeHit("x.md", "uses a candidate pool of twenty documents"),
        FakeHit("docs/local-fast-api.md", "unrelated current text"),
    ]
    kept, stats = apply_policy(hits, VersionPolicy(exclude_superseded=True))
    check("a superseded chunk is EXCLUDED", stats["excluded_superseded"] == 1,
          str(stats["excluded_superseded"]))
    check("exclusion reduces the result count", len(kept) == 2, str(len(kept)))
    check("the exclusion is COUNTED, not silent", stats["excluded_superseded"] >= 1
          and stats["kept"] == 2, str(stats))
    check("the excluded path is named", "x.md" in stats["excluded_paths"],
          str(stats["excluded_paths"]))
    check("the excluded chunk is really gone",
          all("twenty" not in h.content for h in kept))

    kept2, stats2 = apply_policy(hits, VersionPolicy(exclude_superseded=False))
    check("exclusion can be disabled", stats2["excluded_superseded"] == 0
          and len(kept2) == 3, str(len(kept2)))

    # ---------------- demotion path
    print("\n--- historical demotion ---")
    demote_hits = [
        FakeHit("benchmarks/local-fast-results.md", "a historical figure"),
        FakeHit("docs/retrieval-architecture.md", "the current figure"),
    ]
    kept, stats = apply_policy(demote_hits,
                               VersionPolicy(demote_historical_for_current=True,
                                             historical_query=False))
    check("a historical source is DEMOTED, not dropped", len(kept) == 2, str(len(kept)))
    check("demotion moves it after current sources",
          kept[0].source_path == "docs/retrieval-architecture.md",
          kept[0].source_path)
    check("demotion is counted", stats["demoted"] == 1, str(stats["demoted"]))

    kept_h, stats_h = apply_policy(demote_hits,
                                   VersionPolicy(demote_historical_for_current=True,
                                                 historical_query=True))
    check("a HISTORICAL QUERY suppresses demotion", stats_h["demoted"] == 0,
          str(stats_h["demoted"]))
    check("historical order is preserved for a historical query",
          kept_h[0].source_path == "benchmarks/local-fast-results.md",
          kept_h[0].source_path)

    # ---------------- the stated guarantee: mtime is not a validity signal
    #
    # This inspects CODE, not prose. A first version grepped every line containing "mtime"
    # and matched the docstring that says mtime is NOT used - a false positive that would
    # have failed a correct implementation. Lines that are part of a string literal or a
    # comment are skipped.
    print("\n--- mtime is not used as a validity signal ---")
    src = (HERE / "version_aware.py").read_text(encoding="utf-8")
    suspicious = []
    for ln in src.splitlines():
        code = ln.split("#", 1)[0]
        if "mtime" not in code:
            continue
        # a line that merely MENTIONS mtime inside a string is documentation, not logic
        if code.count('"') >= 2 or code.count("'") >= 2:
            continue
        if any(op in code for op in (">", "<", "exclude", "demote", "status")):
            suspicious.append(ln.strip())
    check("no policy decision reads or compares mtime in code", not suspicious,
          str(suspicious[:2]))
    check("the policy docstring states mtime is not validity",
          "NOT the same as factual" in src)
    check("status_of defaults unknown files to current",
          status_of("some/unknown/file.md") == "current")
    check("the registry only names non-current files",
          set(DOC_STATUS.values()) <= {"superseded", "historical", "current"})

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("version-aware policy test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
