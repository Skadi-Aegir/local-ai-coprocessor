"""
P57-C0R step 0b - what IS the correct answer to v02, and is it reachable at k=5?

v02 asks: "How do the embedding service memory figures and the scheduler margin relate to each other?"
Required facts:
  torch_reserved_meaning  - carriers: services/retrieval/embedder.py, services/retrieval/server.py,
                            docs/models/local-embedding.md
  margin_policy           - the 2048 MiB margin

This prints the carrier chunks for the FIRST fact so the correct content is on record, then shows
where they rank for this question.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402

print("=== carrier chunks for the FIRST required fact (torch_reserved meaning) ===")
st = IndexStore.load(IDX, 1024)
carriers = []
try:
    for lit in ("torch_reserved", "reserved_mb"):
        for cid, sp, s, e, body in st.conn.execute(
                "SELECT chunk_id, source_path, start_line, end_line, content FROM chunks "
                "WHERE content LIKE ?", (f"%{lit}%",)).fetchall():
            carriers.append((cid, sp, s, e, body))
    uniq = {}
    for cid, sp, s, e, body in carriers:
        uniq[cid] = (sp, s, e, body)
    print(f"  {len(uniq)} distinct carrier chunks")
    for cid, (sp, s, e, body) in sorted(uniq.items(), key=lambda x: (x[1][0], x[1][1]))[:6]:
        print(f"\n  --- {sp}:{s}-{e} ---")
        print("  " + " ".join(body.split())[:560])
finally:
    st.close()

print("\n\n=== where do those carriers rank for v02's question? ===")
r = ASK.Retriever(IDX, embedder="service")
try:
    hits = r.search("How do the embedding service memory figures and the scheduler margin relate to "
                    "each other?", top_k=40)
finally:
    r.close()
rank = {}
for i, h in enumerate(hits, 1):
    if h.get("chunk_id") in uniq:
        rank.setdefault(h["chunk_id"], i)
if rank:
    for cid, rk in sorted(rank.items(), key=lambda x: x[1]):
        sp, s, e, _ = uniq[cid]
        print(f"  rank {rk:>3}  {sp}:{s}-{e}")
else:
    print("  NONE of the first fact's carriers appear in the top 40 -> a RANKING failure for that fact")

print("\n=== the second fact (margin policy) is widely carried; check the pack had it ===")
print("  From step 0: '2048' and 'GiB' and 'VRAM margin' all present in the pack -> fact 2 WAS available")
