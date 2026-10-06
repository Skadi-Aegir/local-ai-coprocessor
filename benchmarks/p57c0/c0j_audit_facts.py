"""
P57-C0 step 3b - audit the DISCRIMINATING POWER of each fact's literals.

A fact that matches 64 chunks does not identify evidence; it just happens to be true somewhere. This
lists, for each fact, the carriers that are actually ABOUT the question's subject, so the frozen set
can record honest citation targets instead of accidental substring hits.

Suspects: bare numbers ("22", "24", "2000", "8000") and short common words ("truncat").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402

# Fact -> the file(s) that are genuinely about the subject, from the earlier mechanism work.
EXPECTED_HOME = {
    "v03.unload_figures": ["docs/staged-lifecycle-design.md",
                           "docs/phase55r-evidence-consolidation.md"],
    "v03.cycle_latency": ["docs/staged-lifecycle-design.md",
                          "docs/phase55r-evidence-consolidation.md",
                          "docs/production-integration-design.md"],
    "v04.why_raised": ["services/retrieval/retriever.py"],
    "v04.char_cap": ["services/retrieval/retriever.py"],
}

LITERALS = {
    "v03.unload_figures": ["1368", "1392", "2591", "2595"],
    "v03.cycle_latency": ["22", "24"],
    "v04.why_raised": ["2000", "truncat"],
    "v04.char_cap": ["8000", "max_chars_per_chunk"],
}

store = IndexStore.load(IDX, 1024)
try:
    for name, lits in LITERALS.items():
        print(f"\n{'='*80}\n{name}   literals={lits}")
        home = EXPECTED_HOME.get(name, [])
        for lit in lits:
            rows = store.conn.execute(
                "SELECT source_path, start_line, end_line, content FROM chunks "
                "WHERE content LIKE ? ORDER BY source_path", (f"%{lit}%",)).fetchall()
            in_home = [r for r in rows if r[0] in home]
            print(f"  {lit!r:<22} {len(rows):>3} carrier(s);  {len(in_home)} in the expected home "
                  f"file(s) {home}")
            for r in in_home[:4]:
                snippet = " ".join((r[3] or "").split())
                idx = snippet.find(lit)
                ctx = snippet[max(0, idx - 55):idx + 55]
                print(f"       {r[0]}:{r[1]}-{r[2]}   ...{ctx}...")
            if not in_home and rows:
                r = rows[0]
                snippet = " ".join((r[3] or "").split())
                idx = snippet.find(lit)
                print(f"       (no home hit; first carrier elsewhere) {r[0]}:{r[1]}  "
                      f"...{snippet[max(0, idx-45):idx+45]}...")
finally:
    store.close()
