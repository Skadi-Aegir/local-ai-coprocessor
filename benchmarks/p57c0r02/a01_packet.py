"""
P57-C0R-02 step 1 - assemble the offline adjudication packet.

No model calls. Everything comes from:
  * benchmarks/p57c0r/results/r02_scored.json   the 18 stored records (full answers, supplied source
                                                IDs with file and line range, citations)
  * the frozen devset_v1 manifest               the questions and their required facts
  * vector/qa/current                           the fixed index, from which each supplied chunk's
                                                text is fetched by ID

This step only GATHERS and de-duplicates: for every (question, k) it resolves the supplied chunk IDs to
actual text and prints the answers. The adjudication itself is done by reading this output, not by a
keyword rule.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
SRC = REPO / "benchmarks/p57c0r/results/r02_scored.json"
SET = REPO / "benchmarks/p57c0/devset_v1"
OUT = REPO / "benchmarks/p57c0r02"
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402

d = json.loads(SRC.read_text(encoding="utf-8"))
man = json.loads((SET / "manifest.json").read_text(encoding="utf-8"))
facts_by_q = {e["id"]: e for e in man["entries"] if e["class"].endswith("answerable")}

st = IndexStore.load(IDX, 1024)
packet: dict = {"index_version": d["index_version"], "questions": {}}
try:
    for qid, q in d["probes"].items():
        entry = {"question": q["question"],
                 "required_facts": [{"name": f["name"], "n_carriers": f["n_carriers"],
                                     "canonical": f["canonical"]}
                                    for f in facts_by_q[qid]["resolved_facts"]],
                 "by_k": {}}
        for k in ("5", "8", "10"):
            b = q["by_k"][k]
            supplied = []
            for s in b["supplied"]:
                row = st.conn.execute(
                    "SELECT content, source_path, start_line, end_line FROM chunks WHERE chunk_id=?",
                    (s["chunk_id"],)).fetchone()
                text = (row[0] if row else "") or ""
                supplied.append({
                    "source": s["source"], "chunk_id": s["chunk_id"],
                    "source_path": s["source_path"], "start_line": s["start_line"],
                    "end_line": s["end_line"],
                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "chars": len(text),
                    "text": text,
                })
            entry["by_k"][k] = {
                "answer": b["answer"],
                "answer_sha256": hashlib.sha256((b["answer"] or "").encode()).hexdigest(),
                "contains_refusal_sentence": b["contains_refusal_sentence"],
                "prompt_tokens": b["prompt_tokens"],
                "supplied": supplied,
                "citations": b["citations"],
            }
        packet["questions"][qid] = entry
finally:
    st.close()

OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "packet.json"
p.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"index {packet['index_version']}; {len(packet['questions'])} questions")
for qid, q in packet["questions"].items():
    print(f"\n  {qid}: {q['question'][:80]}")
    print(f"    required facts: {[f['name'] for f in q['required_facts']]}")
    for k in ("5", "8", "10"):
        b = q["by_k"][k]
        print(f"      k={k:>2}: {len(b['supplied'])} chunks, {len(b['citations'])} citations, "
              f"answer {len(b['answer'])} chars, refusal_in_text={b['contains_refusal_sentence']}")
print(f"\nwrote {p}  ({p.stat().st_size} bytes)")
print("\nNOTE: the stored per-fact booleans are NOT trusted here. They are the artifact under")
print("      correction. Adjudication reads the text in packet.json.")
