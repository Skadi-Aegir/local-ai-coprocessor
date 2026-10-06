"""P57-C0R - read v01's full answer at k=10 and k=8, and v02's at k=8, from the stored records."""

import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r/results/r01_measured.json")
               .read_text(encoding="utf-8"))

for qid, k in (("v01", "10"), ("v01", "8"), ("v02", "8")):
    b = d["probes"][qid]["by_k"][k]
    print("=" * 88)
    print(f"{qid} k={k}   outcome={b['outcome']}   verdict={b['verdict_marker_based']}")
    print("=" * 88)
    print(f"  facts in pack   : {b['facts_in_pack']}/{len(b['facts'])}")
    print(f"  facts in answer : {b['facts_in_answer']}/{len(b['facts'])}")
    for n, f in b["facts"].items():
        print(f"    {n:<24} pack={f['in_pack']} answer={f['in_answer']} "
              f"markers={f['pack_markers_matched']}")
    print(f"  citations: {[(c['source_path'], c['start_line'], c['end_line'], c['verification']) for c in b['citations']]}")
    print("\n  FULL ANSWER:")
    print("  " + (b["answer"] or "(none)").replace("\n", "\n  "))
    print()
