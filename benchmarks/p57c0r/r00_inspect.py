"""
P57-C0R step 0 - INSPECT the actual pack and answer for v02 before designing a corrected measurement.

The stored record kept only answer_head (400 chars) and a boolean facts_in_pack, so the previous claim
cannot be re-checked from it. This does ONE live request at k=5 and prints:
  * which chunks were actually supplied to the model, with file and line range
  * each supplied chunk's FULL content
  * the FULL answer text
  * whether the literals I previously used as fact detectors appear in the pack or the answer

Nothing is written; this is reconnaissance for the corrected scorer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402

Q_V02 = "How do the embedding service memory figures and the scheduler margin relate to each other?"

r = ASK.Retriever(IDX, embedder="service")
try:
    rec = ASK.answer(Q_V02, retriever=r, top_k=5, measure_vram=False)
finally:
    r.close()

pack = rec.get("pack") or {}
supplied = pack.get("supplied") or []
print("=" * 90)
print("ONE REQUEST at k=5 - pack and answer come from the SAME call")
print("=" * 90)
print(f"  prompt_tokens      : {pack.get('prompt_tokens')} / {pack.get('budget_tokens')}")
print(f"  sources supplied   : {len(supplied)}   omitted: {len(pack.get('omitted') or [])}")
print(f"  outcome            : {'refused' if rec.get('refused') else 'answered'}")
print(f"  citations          : {[(c['source_path'], c['start_line'], c['end_line']) for c in (rec.get('citations') or [])]}")
print(f"  citation_verify    : {rec.get('citation_verification')}")

print("\n--- SUPPLIED SOURCES (what the model actually saw) ---")
for s in supplied:
    print(f"  [{s.get('source')}] {s.get('source_path')}:{s.get('start_line')}-{s.get('end_line')}"
          f"  chunk={s.get('chunk_id')}")

print("\n--- FULL ANSWER ---")
print(rec.get("answer") or "(none)")

print("\n--- LITERAL PROBES: previously-used detectors ---")
ans = (rec.get("answer") or "")
# the pack's actual text, by reading the supplied chunk contents from the index
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402
st = IndexStore.load(IDX, 1024)
try:
    pack_text_parts = []
    for s in supplied:
        row = st.conn.execute("SELECT content FROM chunks WHERE chunk_id=?",
                              (s.get("chunk_id"),)).fetchone()
        pack_text_parts.append(row[0] if row else "")
    pack_text = "\n".join(pack_text_parts)
finally:
    st.close()

for lit in ("torch_reserved", "reserved_mb", "vram_mb", "2048", "allocator", "reserved",
            "cache", "attribution", "GiB", "VRAM margin"):
    print(f"  {lit!r:18} in pack={lit.lower() in pack_text.lower():<6} "
          f"in answer={lit.lower() in ans.lower()}")

print("\n--- the supplied chunks that look most relevant to the first required fact ---")
for s in supplied:
    row = None
    st = IndexStore.load(IDX, 1024)
    try:
        row = st.conn.execute("SELECT content FROM chunks WHERE chunk_id=?",
                              (s.get("chunk_id"),)).fetchone()
    finally:
        st.close()
    body = (row[0] if row else "") or ""
    low = body.lower()
    score = sum(1 for w in ("torch", "reserved", "allocator", "vram", "memory") if w in low)
    print(f"\n  === {s.get('source_path')}:{s.get('start_line')}-{s.get('end_line')} "
          f"(keyword score {score}) ===")
    print("  " + " ".join(body.split())[:700])
