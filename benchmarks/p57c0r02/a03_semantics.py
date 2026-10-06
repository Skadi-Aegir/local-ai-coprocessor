"""Does any supplied chunk at any k explain WHAT the embedding memory figure IS?"""
import json
from pathlib import Path

PROBES = ["cudaMemGetInfo", "torch attribution", "torch_reserved", "reserved_mb",
          "attribution", "VRAM delta", "vram_mb", "expected_vram_mb", "measured_vram_delta"]

d = json.loads((Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r02/packet.json").read_text(encoding="utf-8"))
v02 = d["questions"]["v02"]

for k in ("5", "8", "10"):
    b = v02["by_k"][k]
    print("=" * 96)
    print(f"v02 k={k} - where do the semantic probes appear in the SUPPLIED chunks?")
    print("=" * 96)
    found_any = False
    for s in b["supplied"]:
        t = " ".join(s["text"].split())
        hits = []
        for p in PROBES:
            i = t.lower().find(p.lower())
            if i >= 0:
                hits.append((p, t[max(0, i - 80):i + 110]))
        if hits:
            found_any = True
            print(f"\n  {s['source_path']}:{s['start_line']}-{s['end_line']}  "
                  f"sha={s['content_sha256'][:16]}")
            for p, ctx in hits:
                print(f"     {p!r}: ...{ctx}...")
    if not found_any:
        print("  NONE of the probes appear in any supplied chunk")
    print(f"\n  ALSO: does the ANSWER contain 'vram_mb' / the numbers?")
    ans = b["answer"].lower()
    for p in ("vram_mb", "expected_vram_mb", "1246", "2360", "torch_reserved", "cudamemgetinfo"):
        print(f"     {p!r} in answer: {p in ans}")
