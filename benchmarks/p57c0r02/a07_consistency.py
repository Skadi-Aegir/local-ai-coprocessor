"""
P57-C0R-02 consistency check: bind the adjudication to the stored evidence and confirm the
documentation reflects it.

Checks that the adjudication's readings match packet.json, that no verdict was forced, that the docs
carry the corrected figures, and that nothing in the QA path or the frozen set moved.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
P02 = R / "benchmarks/p57c0r02"

fails: list[str] = []
n = 0


def ck(name: str, ok: bool, detail: str = "") -> None:
    global n
    n += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        fails.append(name)


packet = json.loads((P02 / "packet.json").read_text(encoding="utf-8"))
adj = json.loads((P02 / "adjudication.json").read_text(encoding="utf-8"))
results = (P02 / "RESULTS.md").read_text(encoding="utf-8")
errata = (P02 / "ERRATA-append.md").read_text(encoding="utf-8")
review = (R / "benchmarks/p57c0/REVIEW_REQUEST.md").read_text(encoding="utf-8")
issues = (R / "OPEN_ISSUES.md").read_text(encoding="utf-8")

print("=== inputs are the stored records, not new model output ===")
ck("packet covers 6 questions", len(packet["questions"]) == 6, str(len(packet["questions"])))
ck("packet records chunk content hashes",
   all(s["content_sha256"] for q in packet["questions"].values()
       for b in q["by_k"].values() for s in b["supplied"]))
ck("18 records present",
   sum(len(q["by_k"]) for q in packet["questions"].values()) == 18)
ck("adjudication covers the same questions", set(adj["questions"]) == set(packet["questions"]))

print("\n=== the v02 reading matches the packet text ===")
v02k8 = packet["questions"]["v02"]["by_k"]["8"]
v02k10 = packet["questions"]["v02"]["by_k"]["10"]
ck("v02 k=8 does NOT supply the canonical explanation chunk",
   not any(s["source_path"] == "docs/models/local-embedding.md" and s["start_line"] == 90
           for s in v02k8["supplied"]))
ck("v02 k=10 does NOT supply it either",
   not any(s["source_path"] == "docs/models/local-embedding.md" and s["start_line"] == 90
           for s in v02k10["supplied"]))
cfg8 = [s for s in v02k8["supplied"] if s["source_path"] == "config/local-embedding.yaml"]
ck("v02 k=8 DOES supply the config file", len(cfg8) == 1)
ck("that config chunk contains the distinguishing comment",
   "torch's own attribution" in " ".join(cfg8[0]["text"].split()))
ck("but does NOT contain the explanation phrases",
   not any(p in " ".join(cfg8[0]["text"]).lower()
           for p in ("cudamemgetinfo", "not pytorch's attribution", "torch_reserved")))
ck("v02 k=5 supplies neither the config nor the explanation",
   not any(s["source_path"] in ("config/local-embedding.yaml", "docs/models/local-embedding.md")
           for s in packet["questions"]["v02"]["by_k"]["5"]["supplied"]))

print("\n=== v02 is undecided, not forced ===")
f = adj["questions"]["v02"]["by_k"]["8"]["torch_reserved_meaning"]
ck("v02 k=8 memory-figure fact is 'uncertain', not 'yes'",
   f["chunk_contains_fact"] == "uncertain" and f["answer_states_fact"] == "uncertain",
   f"{f['chunk_contains_fact']}/{f['answer_states_fact']}")
ck("v02 k=10 likewise uncertain",
   adj["questions"]["v02"]["by_k"]["10"]["torch_reserved_meaning"]["answer_states_fact"] == "uncertain")
ck("v02 k=5 is a definite 'no'",
   adj["questions"]["v02"]["by_k"]["5"]["torch_reserved_meaning"]["answer_states_fact"] == "no")
ck("v02 verdict is partial at every k",
   all(adj["questions"]["v02"]["by_k"][k]["verdict"] == "partial" for k in ("5", "8", "10")))
ck("every fact carries a written reason",
   all(v.get("reason") for q in adj["questions"].values() for b in q["by_k"].values()
       for v in b.values() if isinstance(v, dict)))

print("\n=== v01 k=10: chunk HAS the fact, answer does NOT state it ===")
f = adj["questions"]["v01"]["by_k"]["10"]["fast_port"]
ck("v01 k=10 fast_port chunk=yes answer=no", f["chunk_contains_fact"] == "yes"
   and f["answer_states_fact"] == "no", f"{f['chunk_contains_fact']}/{f['answer_states_fact']}")
ck("v01 k=10 embed_port absent from the pack",
   adj["questions"]["v01"]["by_k"]["10"]["embed_port"]["chunk_contains_fact"] == "no")

print("\n=== v04 correction is in place ===")
ck("v04 k=10 why_raised answer=yes (was wrongly 'no')",
   adj["questions"]["v04"]["by_k"]["10"]["why_raised"]["answer_states_fact"] == "yes")
ck("v04 k=10 verdict complete", adj["questions"]["v04"]["by_k"]["10"]["verdict"] == "complete")

print("\n=== verdict table matches the documents ===")
for k, expect in (("5", 2), ("8", 3), ("10", 4)):
    comp = sum(1 for q in adj["questions"].values() if q["by_k"][k]["verdict"] == "complete")
    ck(f"k={k}: complete answers = {expect}", comp == expect, str(comp))
ck("RESULTS carries 2/6 -> 3/6 -> 4/6", "2/6 → 3/6 → 4/6" in results)
ck("ERRATA R10-R15 recorded", all(f"## R{i} " in errata for i in range(10, 16)),
   str([i for i in range(10, 16) if f"## R{i} " not in errata]))
ck("the false-pass defect is recorded (R11)", "FALSE PASS" in errata)
ck("REVIEW has section 4h", "## 4h." in review)
ck("REVIEW states 4h supersedes 4g's verdicts", "supersedes the marker verdicts in §4g" in review)
ck("OPEN_ISSUES P2-11 now shows the adjudicated verdicts",
   "adjudicated verdict" in issues)
ck("P2-11 remains RETRACTED", "P2-11 — RETRACTED" in issues)
ck("P2-12 still listed separately", "P2-12 — NEW" in issues)
ck("P2-12 not fixed this round", "out of scope for" in issues)

print("\n=== nothing that must not move, moved ===")
changed = subprocess.run(["git", "diff", "--name-only", "f3f1509", "HEAD"], cwd=str(R),
                         stdout=subprocess.PIPE).stdout.decode().split()
code = [c for c in changed if c.startswith(("cli/", "services/", "config/"))]
ck("no cli/, services/ or config/ change since f3f1509", not code, str(code))
d = R / "benchmarks/p57c0/devset_v1"
m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
payload = json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"},
                     ensure_ascii=False, sort_keys=True).encode()
ck("frozen manifest hash unchanged",
   hashlib.sha256(payload).hexdigest() == m["manifest_sha256"])
ck("false-pass first attempt preserved",
   (P02 / "adjudication.first-attempt-false-pass.json").is_file())
ck("P57-C0 and P57-C0R results preserved",
   (R / "benchmarks/p57c0/results/answerable_run.json").is_file()
   and (R / "benchmarks/p57c0r/results/r02_scored.json").is_file())

print("\n" + "=" * 72)
print(f"CHECKS: {n}   FAILURES: {len(fails)}")
for x in fails:
    print(f"  ! {x}")
print("=" * 72)
sys.exit(1 if fails else 0)
