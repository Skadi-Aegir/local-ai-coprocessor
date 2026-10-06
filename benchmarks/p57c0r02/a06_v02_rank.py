"""
Where does the chunk that actually EXPLAINS the embedding memory figure rank for v02's question?

The adjudication found that at k=8 and k=10 the pack contains config/local-embedding.yaml, which states
the values and distinguishes two figures, but NOT docs/models/local-embedding.md:90-112, the passage
that says what the figure IS ("cudaMemGetInfo total - free, not PyTorch's attribution to this
process"). This locates that passage's rank, which decides whether v02 is a depth or a ranking case.
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

CANON = ("docs/models/local-embedding.md", 90, 112)
Q = "How do the embedding service memory figures and the scheduler margin relate to each other?"

st = IndexStore.load(IDX, 1024)
cid = None
try:
    row = st.conn.execute(
        "SELECT chunk_id, content FROM chunks WHERE source_path=? AND start_line=?",
        (CANON[0], CANON[1])).fetchone()
    if row:
        cid, body = row
        print(f"canonical explanation chunk: {CANON[0]}:{CANON[1]}-{CANON[2]}  id={cid}")
        print("  text: " + " ".join(body.split())[:420])
    else:
        print(f"chunk {CANON} not found by exact start_line; listing the file's chunks")
        for c, s, e in st.conn.execute(
                "SELECT chunk_id, start_line, end_line FROM chunks WHERE source_path=? "
                "ORDER BY start_line", (CANON[0],)).fetchall():
            print(f"    {s}-{e}  {c}")
finally:
    st.close()

print()
r = ASK.Retriever(IDX, embedder="service")
try:
    hits = r.search(Q, top_k=60)
finally:
    r.close()
rank = next((i for i, h in enumerate(hits, 1) if h.get("chunk_id") == cid), None)
print(f"  its rank for v02's question, searching to depth 60: {rank}")
if rank is None:
    print("  -> not in the top 60: a RANKING failure, no depth would reach it")
elif rank <= 10:
    print(f"  -> within the k<=10 ceiling: a DEPTH case")
else:
    print(f"  -> rank {rank} is beyond any bounded k: a RANKING failure")

print("\n  for comparison, the chunk that WAS supplied at k=8/10 (the config file with the comment):")
for i, h in enumerate(hits, 1):
    if h.get("source_path") == "config/local-embedding.yaml":
        print(f"    config/local-embedding.yaml rank {i}  start_line={h.get('start_line')}")
        break
