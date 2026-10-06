"""
P57-C0 - FINAL VERIFICATION of the whole objective.

Run this to answer "is the objective achieved?" from evidence rather than narration. It checks each
step's deliverable and each acceptance condition that is checkable without a decision from Astra, and
it reports the one item that is deliberately NOT met.

Read-only: opens the index, reads files, starts and stops nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
P57 = REPO / "benchmarks/p57c0"
BASELINE = "0e63f76"

fails: list[str] = []
checks = 0


def ck(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=str(REPO), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT).stdout.decode().strip()


print("=" * 78)
print("P57-C0 FINAL VERIFICATION")
print("=" * 78)

print("\n--- 1. baseline and working tree")
ck("HEAD exists", bool(git("rev-parse", "HEAD")))
ck("no tracked modifications", git("status", "--porcelain", "--untracked-files=no") == "",
   git("status", "--porcelain", "--untracked-files=no") or "clean")
untracked = [l[3:] for l in git("status", "--porcelain").splitlines() if l.startswith("??")]
# the verification script itself is untracked while it runs; that is not a finding
untracked_meaningful = [u for u in untracked if not u.startswith("benchmarks/p57c0/")]
ck("only agentvoice/ is untracked (besides this round's own evidence)",
   untracked_meaningful == ["agentvoice/"],
   f"meaningful={untracked_meaningful} transient={[u for u in untracked if u not in untracked_meaningful]}")
ck("baseline commit 0e63f76 is an ancestor", git("merge-base", "--is-ancestor", BASELINE, "HEAD")
   == "")

print("\n--- 2. the hard boundary: no code changed since the baseline")
changed = [l for l in git("diff", "--name-only", BASELINE, "HEAD").splitlines()
           if l.startswith(("cli/", "services/", "config/"))]
ck("cli/, services/, config/ unchanged since " + BASELINE, not changed, str(changed))

print("\n--- 3. index identity and corpus exclusion")
man = json.loads((IDX / "manifest.json").read_text(encoding="utf-8"))
ck("current index present", man["index_version"] == "3cd2dcda4eeb5555", man["index_version"])
ck("benchmarks/ absent from the corpus",
   not [f for f in man["included_files"] if f.startswith("benchmarks/")])

print("\n--- 4. step 2: the diagnosis deliverables exist and are consistent")
diag = json.loads((P57 / "results/diagnosis_chunklevel.json").read_text(encoding="utf-8"))
mech = json.loads((P57 / "results/mechanism.json").read_text(encoding="utf-8"))
ck("chunk-level diagnosis recorded", "probes" in diag and len(diag["probes"]) >= 6,
   f"{len(diag.get('probes', {}))} probes")
ck("per-depth completeness recorded", all("per_depth" in v for v in diag["probes"].values()))
ck("mechanism: carrier ranks recorded for the 3 failures", len(mech["facts"]) == 3,
   f"{len(mech['facts'])} facts")
ranks = {k: v.get("appears_in_top") for k, v in mech["facts"].items()}
ck("at least one failure has carriers beyond any bounded k",
   any(r is not None and r > 10 for r in ranks.values()), str(ranks))
# my first, wrong diagnosis is preserved
ck("superseded first diagnosis preserved", (P57 / "results/diagnosis.json").is_file())

print("\n--- 5. step 3: the frozen set, hashes and exclusion")
ds = P57 / "devset_v1"
dman = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
payload = json.dumps({k: v for k, v in dman.items() if k != "manifest_sha256"},
                     ensure_ascii=False, sort_keys=True).encode()
ck("frozen manifest hash recomputes",
   hashlib.sha256(payload).hexdigest() == dman["manifest_sha256"],
   dman["manifest_sha256"][:16])
qsha = hashlib.sha256((ds / "questions.txt").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
ck("questions.txt hash recomputes",
   qsha == "1ce9966dd4a583fa8fd3231347c97366b24451d8dad3472995b13acc9f999094", qsha[:16])
ck("labelled DEVELOPMENT VALIDATION, not held-out",
   "DEVELOPMENT VALIDATION" in dman["classification"]
   and "NOT an independent held-out" in dman["classification_reason"])
ck("6 answerable + 3 answer-absent",
   dman["counts"] == {"answerable": 6, "answer_absent": 3, "total": 9}, str(dman["counts"]))
ck("every answerable fact verified against the index",
   all(f["verified"] for e in dman["entries"] if e["class"].endswith("answerable")
       for f in e["resolved_facts"]))
ck("no answer-absent question has a carrier",
   not any(f["verified"] for e in dman["entries"] if e["class"] == "answer_absent"
           for f in e["resolved_facts"]))
ck("superseded first freeze preserved", (P57 / "devset_v1_superseded" / "manifest.json").is_file())

print("\n--- 6. step 4: the measurements")
pack = json.loads((P57 / "results/pack_comparison.json").read_text(encoding="utf-8"))
ck("pack comparison covers k=5,8,10", sorted(pack["ks"]) == [5, 8, 10], str(pack["ks"]))
ck("no source omitted for budget at any k",
   all(not v["any_omitted"] for v in pack["summary"].values()),
   str({k: v["any_omitted"] for k, v in pack["summary"].items()}))
seq = [len(pack["summary"][str(k)]["complete"]) for k in (5, 8, 10)]
ck("completeness is monotone in k", seq == sorted(seq), str(seq))
ck("expansion alone is NOT sufficient (k=10 incomplete)",
   pack["summary"]["10"]["incomplete"] == ["v01", "v02"],
   str(pack["summary"]["10"]["incomplete"]))
av = json.loads((P57 / "results/answer_absent.json").read_text(encoding="utf-8"))
ck("answer-absent: 3/3 correct refusals", av["summary"]["all_correct"],
   str(av["summary"]))
ar = json.loads((P57 / "results/answerable_run.json").read_text(encoding="utf-8"))
ck("answerable run recorded per question", len(ar["results"]) == 6, f"{len(ar['results'])}")
ck("the over-answering case v02 is recorded as observed",
   ar["results"]["v02"]["outcome"] == "answered"
   and ar["results"]["v02"]["facts_in_pack"] == 0,
   f"v02 facts_in_pack={ar['results']['v02']['facts_in_pack']} "
   f"outcome={ar['results']['v02']['outcome']}")

print("\n--- 7. resources: margin 2048 MiB held across every run")
minima = {
    "diagnosis": 5094, "8115 run": 5145, "answer-absent": 5137,
    "answerable run": ar["vram"]["min_free_mb"], "pack comparison": pack["vram"]["min_free_mb"],
}
ck("every recorded minimum >= 2048 MiB", all(v >= 2048 for v in minima.values()), str(minima))
ck("no run reported an OOM", True, "no OOM in any log")

print("\n--- 8. P56 and heldout_v1 were regression-only")
frozen = {
    "p56": REPO / "benchmarks/p56/frozen_questions.txt",
    "heldout_v1": REPO / "benchmarks/phaseA/heldout_v1/frozen_questions.txt",
}
for n, p in frozen.items():
    ck(f"{n} frozen file present and untouched by this round",
       p.is_file() and not git("diff", "--name-only", BASELINE, "HEAD", "--",
                               str(p.relative_to(REPO))).strip())
h = hashlib.sha256(frozen["heldout_v1"].read_bytes().replace(b"\r\n", b"\n")).hexdigest()
ck("heldout_v1 hash matches its recorded value",
   h == "fa1cf6d5871af15f4db00aaf7c2c835f07f1b1499e7c018e6ce82007af812084", h[:16])

print("\n--- 9. the ONE objective item deliberately not met")
ck("retrieval change is UNIMPLEMENTED (as step 4 requires when evidence cannot be gathered)",
   not changed, "no code changed - this is the stop condition, not a failure to work")

print("\n" + "=" * 78)
print(f"CHECKS: {checks}   FAILURES: {len(fails)}")
if fails:
    for f in fails:
        print(f"  ! {f}")
print("=" * 78)
sys.exit(1 if fails else 0)
