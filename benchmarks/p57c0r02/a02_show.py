"""Print the full adjudication material for one question so it can be read, not keyword-matched."""

import json
import sys
from pathlib import Path

P = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r02/packet.json")
d = json.loads(P.read_text(encoding="utf-8"))

qid = sys.argv[1] if len(sys.argv) > 1 else "v02"
ks = sys.argv[2].split(",") if len(sys.argv) > 2 else ["5", "8", "10"]
q = d["questions"][qid]

print("#" * 100)
print(f"# {qid}: {q['question']}")
print(f"# required facts: {[f['name'] for f in q['required_facts']]}")
print("#" * 100)

for k in ks:
    b = q["by_k"][k]
    print("\n" + "=" * 100)
    print(f"=== {qid} k={k}   prompt_tokens={b['prompt_tokens']}   "
          f"refusal_sentence_in_answer={b['contains_refusal_sentence']}")
    print("=" * 100)
    print("\n--- SUPPLIED SOURCES (what the model saw) ---")
    for i, s in enumerate(b["supplied"], 1):
        print(f"  [{i}] {s['source_path']}:{s['start_line']}-{s['end_line']}  "
              f"sha={s['content_sha256'][:16]}  {s['chars']} chars")
    print("\n--- CITATIONS ---")
    for c in b["citations"]:
        print(f"  [{c.get('source')}] {c['source_path']}:{c['start_line']}-{c['end_line']}  "
              f"{c['verification']}")
    print("\n--- FULL ANSWER ---")
    print(b["answer"] or "(empty)")
    print("\n--- SUPPLIED CHUNK TEXT ---")
    for i, s in enumerate(b["supplied"], 1):
        print(f"\n  ===== [{i}] {s['source_path']}:{s['start_line']}-{s['end_line']} "
              f"(sha {s['content_sha256'][:16]}) =====")
        body = " ".join(s["text"].split())
        print("  " + body[:1500])
