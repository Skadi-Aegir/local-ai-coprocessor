"""
P57-C0R - bind the CORRECTED numbers to the stored data, and check that the retraction is complete.

A retraction that leaves the old figure standing elsewhere in the docs is worse than no retraction. So
this checks two things:
  A. every figure in the corrected narrative is derivable from results/r02_scored.json
  B. the retracted claim is actually marked as retracted wherever it appears, and no document still
     asserts it as current

Read-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
P57R = R / "benchmarks/p57c0r"
REVIEW = R / "benchmarks/p57c0/REVIEW_REQUEST.md"
ISSUES = R / "OPEN_ISSUES.md"

fails: list[str] = []
n = 0


def ck(name: str, ok: bool, detail: str = "") -> None:
    global n
    n += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        fails.append(name)


scored = json.loads((P57R / "results/r02_scored.json").read_text(encoding="utf-8"))
raw = json.loads((P57R / "results/r01_measured.json").read_text(encoding="utf-8"))
review = REVIEW.read_text(encoding="utf-8")
issues = ISSUES.read_text(encoding="utf-8")
errata = (P57R / "ERRATA.md").read_text(encoding="utf-8")

print("=== A. corrected figures derive from the stored data ===")
for k, expect in (("5", 2), ("8", 4), ("10", 5)):
    got = len(scored["summary_final"][k]["facts_all_in_pack"])
    ck(f"k={k}: facts all in pack = {expect}/6", got == expect, f"got {got}")
    ck(f"k={k}: figure {expect}/6 stated in ERRATA", f"{expect}/6" in errata)
ck("progression 2 -> 4 -> 5 stated in ERRATA", "2/6 → 4/6 → 5/6" in errata or "2 → 4 → 5" in errata)
ck("old progression 2/6 -> 3/6 -> 4/6 recorded as the error",
   "2/6 → 3/6 → 4/6" in errata)
ck("v01 is the only k=10 incomplete question",
   scored["summary_final"]["10"]["facts_all_in_pack"] == ["v02", "v03", "v04", "v05", "v06"],
   str(scored["summary_final"]["10"]["facts_all_in_pack"]))

print("\n=== v02 corrected ===")
v02 = scored["probes"]["v02"]
for k, exp_pack, exp_verdict in (("5", 1, "partial"), ("8", 2, "complete"), ("10", 2, "complete")):
    b = v02["by_k"][k]
    ck(f"v02 k={k}: facts in pack {exp_pack}/2", b["facts_in_pack"] == exp_pack,
       str(b["facts_in_pack"]))
    ck(f"v02 k={k}: verdict {exp_verdict}", b["verdict_final"] == exp_verdict, b["verdict_final"])
ck("v02 k=5 memory-figure fact absent", v02["by_k"]["5"]["facts"]["torch_reserved_meaning"]["in_pack"] is False)
ck("v02 k=8 memory-figure fact present",
   v02["by_k"]["8"]["facts"]["torch_reserved_meaning"]["in_pack"] is True)
ck("v02 k=8 markers come from the real passage",
   "vram_mb" in v02["by_k"]["8"]["facts"]["torch_reserved_meaning"]["pack_markers_matched"],
   str(v02["by_k"]["8"]["facts"]["torch_reserved_meaning"]["pack_markers_matched"]))

print("\n=== v01 still a ranking failure ===")
v01 = scored["probes"]["v01"]
for k in ("5", "8", "10"):
    ck(f"v01 k={k}: 8113 absent from pack",
       v01["by_k"][k]["facts"]["embed_port"]["in_pack"] is False)
    ck(f"v01 k={k}: 8111 present in pack", v01["by_k"][k]["facts"]["fast_port"]["in_pack"] is True)

print("\n=== refusal-with-reasoning handling ===")
mine = {"v01": {"8": "incorrect_refusal", "10": "incorrect_refusal"}}
for q, ks in mine.items():
    for k, v in ks.items():
        ck(f"{q} k={k} re-scored to {v}", scored["probes"][q]["by_k"][k]["verdict_final"] == v,
           scored["probes"][q]["by_k"][k]["verdict_final"])
        ck(f"{q} k={k} flagged as containing the refusal sentence",
           scored["probes"][q]["by_k"][k]["contains_refusal_sentence"] is True)
nref = sum(1 for q in scored["probes"].values() for b in q["by_k"].values()
           if b["contains_refusal_sentence"])
ck("6 of 18 answers contain the refusal sentence", nref == 6, f"got {nref}")

print("\n=== B. the retraction is complete ===")
ck("P2-11 is marked RETRACTED", "P2-11 — RETRACTED" in issues)
ck("P2-11 keeps the original claim visible (strikethrough)",
   "~~**Evidence:**" in issues)
ck("P2-12 filed for the refusal-detector mislabel", "P2-12 — NEW" in issues)
ck("REVIEW section 4e marked superseded", "SUPERSEDED by §4g" in review)
ck("REVIEW contains the corrected section 4g", "## 4g." in review)
ck("REVIEW no longer asserts the 2/3/4 progression as current",
   "2 → 3 → 4 of 6" not in review)
ck("REVIEW carries the corrected progression", "2 → 4 → 5 of 6" in review)
ck("REVIEW retracts the 0/2 v02 row explicitly", "RETRACTED" in review)
ck("ERRATA records all nine items R1-R9",
   all(f"## R{i} " in errata for i in range(1, 10)),
   str([i for i in range(1, 10) if f"## R{i} " not in errata]))

print("\n=== C. nothing in the QA path changed ===")
import subprocess
changed = subprocess.run(["git", "diff", "--name-only", "a96190a", "HEAD"], cwd=str(R),
                         stdout=subprocess.PIPE).stdout.decode().split()
code = [c for c in changed if c.startswith(("cli/", "services/", "config/"))]
ck("no cli/, services/ or config/ change since a96190a", not code, str(code))

print("\n=== D. frozen set and raw data preserved ===")
import hashlib
d = R / "benchmarks/p57c0/devset_v1"
m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
payload = json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"},
                     ensure_ascii=False, sort_keys=True).encode()
ck("frozen manifest hash unchanged",
   hashlib.sha256(payload).hexdigest() == m["manifest_sha256"])
ck("superseded first corrected attempt preserved",
   (P57R / "results/r01_measured.first-attempt.json").is_file())
ck("P57-C0 raw results preserved",
   (R / "benchmarks/p57c0/results/answerable_run.json").is_file())

print("\n" + "=" * 72)
print(f"CHECKS: {n}   FAILURES: {len(fails)}")
for f in fails:
    print(f"  ! {f}")
print("=" * 72)
sys.exit(1 if fails else 0)
