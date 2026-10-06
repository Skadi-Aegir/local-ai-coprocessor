"""
P57-C0R - verify the two claims that changed most, directly from the corrected per-k records.

CLAIM 1: v02's retracted statement. The old record said v02 answered with 0 of 2 required facts and
         that both facts' content was absent. Check what the CORRECTED measurement shows per k, and
         whether the first fact's content really reaches the pack at k=8 and k=10.

CLAIM 2: v01. Its answer-level verdict moves around with k (incorrect_refusal at 5, partial at 8,
         incorrect at 10) while its pack stays at 1/2. Check whether the missing fact is a RANKING
         problem - i.e. absent from the pack at every k - because if so, no bounded k can fix it.
"""

from __future__ import annotations

import json
from pathlib import Path

P = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r/results/r01_measured.json")
d = json.loads(P.read_text(encoding="utf-8"))

print("=" * 88)
print("CLAIM 1 - v02, per k, from the corrected measurement")
print("=" * 88)
v02 = d["probes"]["v02"]
print(f"  question: {v02['question']}")
print(f"  required facts: {v02['required_facts']}")
for k in ("5", "8", "10"):
    b = v02["by_k"][k]
    print(f"\n  k={k}   outcome={b['outcome']}  verdict={b['verdict_marker_based']}  "
          f"prompt_tokens={b['prompt_tokens']}")
    for name, f in b["facts"].items():
        print(f"    {name:<24} in_pack={f['in_pack']}  in_answer={f['in_answer']}")
        print(f"        pack markers   : {f['pack_markers_matched']}")
        print(f"        answer markers : {f['answer_markers_matched']}")
    print(f"    supplied: {[s['source_path'] for s in b['supplied']]}")

print("\n  --- did the FIRST fact's content reach the pack? ---")
for k in ("5", "8", "10"):
    f = v02["by_k"][k]["facts"]["torch_reserved_meaning"]
    print(f"    k={k}: in_pack={f['in_pack']}  matched={f['pack_markers_matched']}")

print("\n" + "=" * 88)
print("CLAIM 2 - v01: is the missing fact a RANKING problem?")
print("=" * 88)
v01 = d["probes"]["v01"]
print(f"  question: {v01['question']}")
for k in ("5", "8", "10"):
    b = v01["by_k"][k]
    print(f"\n  k={k}  outcome={b['outcome']}  verdict={b['verdict_marker_based']}")
    for name, f in b["facts"].items():
        print(f"    {name:<14} in_pack={f['in_pack']}  in_answer={f['in_answer']}  "
              f"pack_markers={f['pack_markers_matched']}  carriers={f['carrier_count']}")
    print(f"    supplied: {[(s['source_path'], s['start_line']) for s in b['supplied']]}")

print("\n  --- summary of which fact is missing ---")
for k in ("5", "8", "10"):
    facts = v01["by_k"][k]["facts"]
    missing = [n for n, f in facts.items() if f["in_pack"] is False]
    print(f"    k={k}: missing from pack = {missing or 'none'}")

print("\n" + "=" * 88)
print("OVERALL, CORRECTED (facts in the pack, one request per measurement)")
print("=" * 88)
for k in ("5", "8", "10"):
    s = d["summary"][k]
    print(f"  k={k:>2}: {len(s['complete'])}/6   incomplete={s['incomplete']}"
          f"   max prompt_tokens={s['max_prompt_tokens']}")
print("\n  OLD (uncorrected) figures were 2/6 -> 3/6 -> 4/6. Those came from a detector keyed on the")
print("  wrong fact name, so its v02 marker set never applied. The corrected figures are above.")
