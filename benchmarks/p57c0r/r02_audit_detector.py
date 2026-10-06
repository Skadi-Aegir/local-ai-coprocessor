"""
P57-C0R - audit the v02 first-fact detector, which is now TOO LOOSE.

Step 0 established, from the actual pack text, that v02's pack contains NO explanation of what the
embedding service memory figure means: torch_reserved, reserved_mb, vram_mb, allocator, reserved all
absent from the pack. But the corrected run reports pack=1/2 for v02, because the marker set I gave
that fact includes the bare word "margin" - which belongs to the OTHER fact.

A detector that is too loose is as bad as one that is too strict: it turns a real failure into an
apparent success. This prints exactly which markers matched and where, so the error is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

P = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r/results/r01_measured.json")
d = json.loads(P.read_text(encoding="utf-8"))
v02 = d["probes"]["v02"]

print("=== v02 per-k fact detail, as measured ===")
for k in ("5", "8", "10"):
    b = v02["by_k"][k]
    print(f"\n  k={k}")
    for name, f in b["facts"].items():
        print(f"    {name}")
        print(f"       in_pack={f['in_pack']}  in_answer={f['in_answer']}  detect={f['detect']}")
        print(f"       pack markers matched  : {f['pack_markers_matched']}")
        print(f"       answer markers matched: {f['answer_markers_matched']}")
        print(f"       carrier_count={f['carrier_count']}  carriers_recorded="
              f"{len(f['carrier_chunks_full'])}")

print("\n\n=== is 'margin' a valid marker for 'what the memory figure MEANS'? ===")
print("  NO. v02 has two required facts:")
print("    fact 1 = the embedding service memory figures  (what the number MEANS)")
print("    fact 2 = the scheduler margin policy (2048 MiB)")
print("  'margin' belongs to fact 2. Accepting it for fact 1 makes fact 1 appear satisfied whenever")
print("  fact 2 is, which collapses two facts into one and hides a genuine gap.")

print("\n=== what the pack text actually contains for fact 1, for the record ===")
for k in ("5",):
    for s in v02["by_k"][k]["supplied"]:
        print(f"    supplied: {s['source_path']}:{s['start_line']}-{s['end_line']}")

print("\n=== CORRECTED marker set for fact 1 (must mean WHAT THE FIGURE IS) ===")
corrected = ["cudaMemGetInfo", "torch attribution", "VRAM delta", "vram_mb", "reserved_mb",
             "attribution"]
print(f"  any_of = {corrected}")
print("  Step 0 measured every one of these as ABSENT from v02's k=5 pack, so fact 1 is NOT in the pack.")
