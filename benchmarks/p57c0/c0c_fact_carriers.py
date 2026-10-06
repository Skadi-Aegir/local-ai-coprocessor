"""
P57-C0 step 2c - validate the requirement sets against the REAL current index.

The earlier diagnosis asserted "required file X was missing from the top-5". That is only a failure if
X is genuinely the place the fact lives. This script asks the corpus itself:

  for each fact, which of the 54 INDEXED files contain it, and was any of them in the retrieved top-5?

A fact is GATHERED when at least one indexed chunk that carries it appears in the retrieved set,
whatever file it is in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402

man = json.load(open(IDX / "manifest.json", encoding="utf-8"))
indexed = set(man["included_files"])
print(f"index {man['index_version']}: {len(indexed)} indexed files, {man['n_chunks']} chunks")

# does the index even contain the new Phase 6 files?
for f in ("config/local-ocr.yaml", "docs/models/local-ocr.md", "services/ocr/server.py"):
    print(f"  indexed? {f}: {f in indexed}")

FACTS = {
    "d01.fast_port": ["8111"],
    "d01.embed_port": ["8113"],
    "d05.char_cap": ["8000"],
    "d06.reranker_port": ["8114"],
    "d06.ocr_port": ["8116"],
}
# d02/d03/d07 are prose facts: use the distinctive phrase, not a generic word
PHRASE_FACTS = {
    "d02.torch_reserved": "torch_reserved",
    "d02.margin_policy": "2048",
    "d03.policy_threshold": "escalat",
    "d07.softmax_score": "softmax",
}

store = IndexStore.load(IDX, 1024)
try:
    def files_with(needles: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in needles:
            for sp, c in store.conn.execute(
                    "SELECT source_path, COUNT(*) c FROM chunks WHERE content LIKE ? "
                    "GROUP BY source_path", (f"%{n}%",)).fetchall():
                out[sp] = out.get(sp, 0) + c
        return out

    diag = json.load(open(REPO / "benchmarks/p57c0/results/diagnosis.json", encoding="utf-8"))
    result: dict = {"index_version": man["index_version"], "facts": {}}

    allfacts = {**{k: v for k, v in FACTS.items()},
                **{k: [v] for k, v in PHRASE_FACTS.items()}}
    for name, needles in allfacts.items():
        carriers = files_with(needles)
        result["facts"][name] = {"needles": needles, "carrier_files": carriers}
        print(f"\n{name}  needles={needles}")
        print(f"   carried by {len(carriers)} indexed files: "
              f"{sorted(carriers, key=lambda x: -carriers[x])[:6]}")

    # now: for each probe, was each fact reachable from the top-5 / top-10 that were retrieved?
    print("\n\n=== GATHERED-BY-CONTENT check (file-agnostic) ===")
    for pid, p in diag["probes"].items():
        if pid.endswith("DIAGNOSTIC_ONLY"):
            continue
        top5 = [(r["source_path"], r["start_line"]) for r in p["ranks"][:5]]
        top10 = [(r["source_path"], r["start_line"]) for r in p["ranks"][:10]]
        print(f"\n{pid}: {p['question'][:70]}")
        for f in p["required_files"]:
            print(f"   my requirement {f}: "
                  f"in top5={any(s == f for s, _ in top5)}  "
                  f"in top10={any(s == f for s, _ in top10)}")
finally:
    store.close()

(REPO / "benchmarks/p57c0/results/fact_carriers.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote results/fact_carriers.json")
