"""
P57-C0 step 2e - name the mechanism: for each UNGATHERED fact, which indexed chunks actually carry it,
and how do those chunks rank against the question?

This separates two very different causes:
  (A) DEPTH  - the carrier ranks just past the cut, so a larger k fixes it
  (B) RANKING- the carrier ranks far down or not at all, so a larger k does NOT fix it and the
      question's wording does not reach the chunk that holds the fact
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

MISSING = {
    "d01": {"fact": "embed_port 8113",
            "question": "Which components does a question pass through before reaching local-fast, "
                        "and which ports do they use?",
            "needles": ["8113"]},
    "d02": {"fact": "torch_reserved meaning",
            "question": "How do the embedding service memory figures and the scheduler margin "
                        "relate to each other?",
            "needles": ["torch_reserved", "reserved_mb"]},
    "d03": {"fact": "escalation threshold 0.30 / evidence floor",
            "question": "What does the escalation policy require when local retrieval is weak, and "
                        "where is that threshold configured?",
            "needles": ["0.30", "evidence floor", "EVIDENCE_FLOOR"]},
}
DEEP = 60

store = IndexStore.load(IDX, 1024)
r = ASK.Retriever(IDX, embedder="service")
rep = {"note": "carriers of each ungarnered fact, and their rank for that question", "facts": {}}
try:
    for pid, info in MISSING.items():
        print(f"\n{'='*90}\n{pid}: {info['fact']}")
        # all indexed chunks that carry the fact
        carriers = []
        for n in info["needles"]:
            for cid, sp, s, e in store.conn.execute(
                    "SELECT chunk_id, source_path, start_line, end_line FROM chunks "
                    "WHERE content LIKE ?", (f"%{n}%",)).fetchall():
                carriers.append((cid, sp, s, e, n))
        uniq = {}
        for cid, sp, s, e, n in carriers:
            uniq[cid] = (sp, s, e)
        print(f"  the fact is carried by {len(uniq)} indexed chunks:")
        for cid, (sp, s, e) in sorted(uniq.items(), key=lambda x: x[1][0])[:20]:
            print(f"     {sp}:{s}-{e}")

        # rank those carriers for this question, out to DEEP
        hits = r.search(info["question"], top_k=DEEP)
        rank_of = {}
        for i, h in enumerate(hits, 1):
            if h.get("chunk_id") in uniq:
                rank_of.setdefault(h["chunk_id"], i)
        print(f"\n  their rank for this question (searched to depth {DEEP}):")
        if not rank_of:
            print("     *** NONE of the carrier chunks appear even in the top "
                  f"{DEEP} ***  -> not a depth problem")
        else:
            for cid, rk in sorted(rank_of.items(), key=lambda x: x[1]):
                sp, s, e = uniq[cid]
                band = ("would be fixed by k=8" if rk <= 8 else
                        "would be fixed by k=10" if rk <= 10 else
                        f"needs k>={rk}")
                print(f"     rank {rk:>3}  {sp}:{s}-{e}   {band}")
        rep["facts"][pid] = {"fact": info["fact"], "question": info["question"],
                             "n_carrier_chunks": len(uniq),
                             "carriers": {cid: {"source_path": v[0], "start_line": v[1],
                                                "end_line": v[2]} for cid, v in uniq.items()},
                             "rank_of_carriers": {cid: rk for cid, rk in rank_of.items()},
                             "appears_in_top": max(rank_of.values()) if rank_of else None}
finally:
    r.close()
    store.close()

print(f"\n\n{'='*90}\n=== VERDICT PER FACT ===")
for pid, v in rep["facts"].items():
    top = v["appears_in_top"]
    if top is None:
        print(f"  {pid}: RANKING problem - no carrier in the top {DEEP}. Larger k cannot help.")
    elif top <= 10:
        print(f"  {pid}: DEPTH problem - best carrier at rank {top}. Larger k can help.")
    else:
        print(f"  {pid}: RANKING problem - best carrier at rank {top}, beyond any bounded k.")

p = REPO / "benchmarks/p57c0/results/mechanism.json"
p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {p}")
