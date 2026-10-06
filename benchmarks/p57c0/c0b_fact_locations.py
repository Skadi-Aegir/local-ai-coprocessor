"""
P57-C0 step 2b - check MY OWN required-file list before blaming the retriever.

A requirement set is only valid if the named file is really needed. If a fact is also stated in a file
that IS retrieved, the requirement is redundant and the "incomplete" verdict would be my error, not
the retriever's.

For each fact, search the whole corpus for which files contain it, and note whether any of those files
was retrieved at top-5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from index_store import IndexStore  # noqa: E402

# The literal fact strings that actually answer each question.
FACTS = {
    "d01.fast_port": ["8111"],
    "d01.embed_port": ["8113"],
    "d02.torch_reserved_meaning": ["torch_reserved", "reserved_mb", "vram_mb"],
    "d02.margin": ["2048"],
    "d03.escalation_threshold": ["evidence", "0.30", "floor", "threshold"],
    "d05.char_cap": ["8000", "max_chars_per_chunk"],
    "d06.reranker_port": ["8114"],
    "d06.ocr_port": ["8116"],
    "d07.score_softmax": ["softmax", "logit"],
}

store = IndexStore.load(IDX, 1024)
try:
    where: dict[str, dict] = {}
    for name, needles in FACTS.items():
        files: dict[str, int] = {}
        for n in needles:
            rows = store.conn.execute(
                "SELECT source_path, COUNT(*) c FROM chunks WHERE content LIKE ? "
                "GROUP BY source_path", (f"%{n}%",)).fetchall()
            for sp, c in rows:
                files[sp] = files.get(sp, 0) + c
        where[name] = files
        print(f"\n{name}   (needles: {needles})")
        if not files:
            print("   *** NOT FOUND IN THE CORPUS AT ALL ***")
            continue
        for sp, c in sorted(files.items(), key=lambda x: -x[1])[:8]:
            print(f"   {c:>3} chunks  {sp}")
    (REPO / "benchmarks/p57c0/results/fact_locations.json").write_text(
        json.dumps(where, ensure_ascii=False, indent=2), encoding="utf-8")
finally:
    store.close()

# Cross-check against the diagnosis: which of the retrieved top-5 files carry the fact?
diag = json.load(open(REPO / "benchmarks/p57c0/results/diagnosis.json", encoding="utf-8"))
print("\n\n=== is each fact reachable from the TOP-5 that was actually retrieved? ===")
for pid, p in diag["probes"].items():
    top5 = [r["source_path"] for r in p["ranks"][:5]]
    print(f"\n{pid}  top5 files:")
    for f in top5:
        print(f"   {f}")
    print(f"  required by me: {p['required_files']}")
