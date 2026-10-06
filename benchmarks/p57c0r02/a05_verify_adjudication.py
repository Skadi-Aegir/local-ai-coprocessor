"""Verify the v03/v04/v05/v06 adjudication claims against the actual packet text.

For each, print: which required-fact carriers are supplied, whether the answer states the fact, and the
citations - so the adjudication in a04 is checked, not assumed.
"""
import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r02/packet.json").read_text(encoding="utf-8"))

CARRIER_HINTS = {
    "v03": {"unload_figures": ["1368", "2591"], "cycle_latency": ["24.4", "22-24", "22\u201324"]},
    "v04": {"char_cap": ["8000"], "why_raised": ["Raised from 2000", "capped every chunk at 2000"]},
    "v05": {"reranker_port": ["8114"], "ocr_port": ["8116"]},
    "v06": {"score_method": ["softmax", "logit"]},
}

for qid, facts in CARRIER_HINTS.items():
    q = d["questions"][qid]
    print("#" * 96)
    print(f"# {qid}: {q['question'][:84]}")
    print("#" * 96)
    for k in ("5", "8", "10"):
        b = q["by_k"][k]
        print(f"\n  --- k={k}  refusal={b['contains_refusal_sentence']}  "
              f"supplied={len(b['supplied'])} ---")
        pack = " || ".join(" ".join(s["text"].split()) for s in b["supplied"])
        ans = b["answer"] or ""
        for fname, needles in facts.items():
            psrc = [s["source_path"] + f":{s['start_line']}"
                    for s in b["supplied"]
                    if any(n.lower() in " ".join(s["text"].split()).lower() for n in needles)]
            ahit = [n for n in needles if n.lower() in ans.lower()]
            print(f"     {fname:<16} chunk_has_it={psrc or 'NO'}   answer_has={ahit or 'NO'}")
        print(f"     citations: {[(c['source_path'], c['start_line']) for c in b['citations']]}")
        print(f"     answer: {' '.join(ans.split())[:260]}")
