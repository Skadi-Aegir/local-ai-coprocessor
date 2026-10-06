"""
P57-C0 - verify that the REPORT's numbers match the STORED measurements.

A reviewer should not have to open four JSON files to check that a summary table is honest. This
script re-derives every number quoted in REVIEW_REQUEST.md from the recorded results and fails if any
quoted figure cannot be reproduced.

Read-only. It changes nothing and is safe to run at any time.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

P57 = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0")
REPORT = P57 / "REVIEW_REQUEST.md"

fails: list[str] = []
checked = 0


def ck(name: str, ok: bool, detail: str = "") -> None:
    global checked
    checked += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        fails.append(name)


pack = json.loads((P57 / "results/pack_comparison.json").read_text(encoding="utf-8"))
arun = json.loads((P57 / "results/answerable_run.json").read_text(encoding="utf-8"))
absent = json.loads((P57 / "results/answer_absent.json").read_text(encoding="utf-8"))
chunk = json.loads((P57 / "results/diagnosis_chunklevel.json").read_text(encoding="utf-8"))
mech = json.loads((P57 / "results/mechanism.json").read_text(encoding="utf-8"))
b5 = json.loads(Path(r"<repo>/benchmarks/p57c0/results/b_reverify.json")
                .read_text(encoding="utf-8"))
supp = json.loads((P57 / "results/supplementary.json").read_text(encoding="utf-8"))
text = REPORT.read_text(encoding="utf-8")

print("=== pack comparison table (section 4c) ===")
for k in ("5", "8", "10"):
    s = pack["summary"][k]
    n = len(s["complete"])
    ck(f"k={k} complete count is {n}/6", n == len(pack["probes"]) - len(s["incomplete"]),
       f"complete={n} incomplete={s['incomplete']}")
    ck(f"k={k} quoted '{n} / 6' appears in the report", f"**{n} / 6**" in text or f"{n}/6" in text)
    ck(f"k={k} no source omitted for budget is quoted correctly",
       s["any_omitted"] is False and "omitted for budget" in text)
tokens = {k: pack["summary"][k]["prompt_tokens_max"] for k in ("5", "8", "10")}
for k, v in tokens.items():
    ck(f"k={k} max prompt tokens {v} appears in the report", str(v) in text)
ck("k=10 headroom sentence matches the data", "6565 / 7616" in text and tokens["10"] == 6565)

print("\n=== completeness is monotone, as the report claims ===")
seq = [len(pack["summary"][k]["complete"]) for k in ("5", "8", "10")]
ck("monotone 2,3,4", seq == [2, 3, 4], str(seq))
ck("report states the progression 2 -> 3 -> 4", "2 → 3 → 4" in text or "2 -> 3 -> 4" in text)

print("\n=== the two failures named in the report ===")
ck("k=10 incomplete is exactly v01,v02", pack["summary"]["10"]["incomplete"] == ["v01", "v02"],
   str(pack["summary"]["10"]["incomplete"]))
ck("report names v01 and v02 as the failed items", "v01" in text and "v02" in text)

print("\n=== answer-absent (section 4c) ===")
ck("3/3 correct refusals in the data", absent["summary"]["all_correct"]
   and absent["summary"]["n_correct_refusals"] == 3, str(absent["summary"]))
ck("report states 3 / 3", "3 / 3 CORRECT REFUSALS" in text or "3/3" in text)

print("\n=== answerable run (section 4e) ===")
rows = arun["results"]
ck("6 questions recorded", len(rows) == 6, str(len(rows)))
ans = [k for k, v in rows.items() if v["outcome"] == "answered"]
ref = [k for k, v in rows.items() if v["outcome"] == "refused"]
ck("3 answered / 3 refused as the report says", len(ans) == 3 and len(ref) == 3,
   f"answered={ans} refused={ref}")
v02 = rows["v02"]
ck("v02 answered with 0/2 facts in pack, as quoted",
   v02["outcome"] == "answered" and v02["facts_in_pack"] == 0
   and v02["facts_total"] == 2,
   f"{v02['facts_in_pack']}/{v02['facts_total']} {v02['outcome']}")
ck("v02 citations all verified", v02["citation_verification"]
   and all(s == "verified" for s in v02["citation_verification"]),
   str(v02["citation_verification"]))
ck("the quoted v02 answer fragment matches the stored answer",
   "2 GiB (2048 MiB) VRAM m" in v02["answer_head"], v02["answer_head"][:70])
ck("report states 3 answered / 3 incorrect refusals", "3 answered, 3 incorrect refusals" in text)

print("\n=== diagnosis: the rank claims in section 1.2 ===")
for fact, expect in (("d01", 38), ("d02", 15)):
    ranks = sorted(mech["facts"][fact]["rank_of_carriers"].values())
    ck(f"{fact} best carrier rank is {expect}", ranks and min(ranks) == expect, str(ranks))
    ck(f"{fact} quoted rank {expect} appears in the report", str(expect) in text)
ck("d03 has zero carriers", mech["facts"]["d03"]["n_carrier_chunks"] == 0)
ck("report states d03's fact is absent from the corpus",
   "not in the corpus" in text or "absent from the corpus" in text)
answerable_probes = {k: v for k, v in chunk["probes"].items()
                     if not k.endswith("DIAGNOSTIC_ONLY")}
for d, expect in (("5", 2), ("8", 4), ("10", 4)):
    n = sum(1 for v in answerable_probes.values() if v["per_depth"][d]["complete"])
    ck(f"chunk-level depth {d} complete count is {expect}", n == expect,
       f"got {n} over {len(answerable_probes)} answerable probes")

print("\n=== resources quoted in section 4f ===")
mins = {
    "pack comparison": pack["vram"]["min_free_mb"],
    "answerable run": arun["vram"]["min_free_mb"],
    "answer-absent": absent["vram"]["min_free_mb"],
    "8115 run": b5["vram_overall"]["min_across_scenarios"],
}
for name, v in mins.items():
    ck(f"{name} minimum {v} MiB >= 2048", v >= 2048, str(v))
    ck(f"{name} minimum {v} appears in the report", str(v) in text)
ck("supplementary checks all passed", supp["all_three_ok"] is True, str(supp["all_three_ok"]))
ck("8115 seven scenarios consistent",
   b5["limits_verified"]["max_concurrency_enforced"] and
   b5["scenarios"]["service_unavailable"]["http"] == 503,
   f"503={b5['scenarios']['service_unavailable']['http']}")

print("\n" + "=" * 70)
print(f"NUMBERS CHECKED: {checked}   MISMATCHES: {len(fails)}")
for f in fails:
    print(f"  ! {f}")
print("=" * 70)
sys.exit(1 if fails else 0)
