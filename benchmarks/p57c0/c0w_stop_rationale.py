"""
P57-C0 - record a correction to the stop rationale, and state the decision this round needs.

CORRECTION: I hypothesised that raising k might convert v02's refusal into an unsupported answer, i.e.
that expansion would WORSEN the over-answering. Checking the stored data: it does not. v02 answers at
k=5 with 0/2 facts already. So the over-answering is present at the CURRENT setting and is not caused
by, nor worsened by, raising k. My hypothesis was wrong and is withdrawn.

WHAT REMAINS TRUE, and is the actual reason to hold:
  1. "reliably gathers" is FALSE - v01 and v02 stay incomplete at every k up to the ceiling
  2. v02 is answered without its required evidence at EVERY k (0/2), so a candidate-count change would
     show "improvement" on the pack table while the user-visible defect is untouched
  3. headroom at k=10 is 6565/7616 - about 1051 tokens of slack
"""

import json
from pathlib import Path

p = json.loads(Path(r"<repo>/benchmarks/p57c0/results/pack_comparison.json")
               .read_text(encoding="utf-8"))
ar = json.loads(Path(r"<repo>/benchmarks/p57c0/results/answerable_run.json")
                .read_text(encoding="utf-8"))

print("=== does the CURRENT setting already over-answer? ===")
v02 = ar["results"]["v02"]
print(f"  k=5 (current setting, independent run): outcome={v02['outcome']}  "
      f"facts_in_pack={v02['facts_in_pack']}/{v02['facts_total']}  "
      f"citations={v02['citation_verification']}")
print(f"  -> the over-answering EXISTS AT k=5. Raising k neither causes nor worsens it.")

print("\n=== so is k an effective lever for what the round wants? ===")
for k in ("5", "8", "10"):
    s = p["summary"][k]
    print(f"  k={k:>2}: {len(s['complete'])}/6 complete, incomplete={s['incomplete']}")
print("  v01 and v02 are untouched by k. Two of the six questions cannot be helped by this lever.")

print("\n=== headroom at the ceiling ===")
for k in ("5", "8", "10"):
    t = p["summary"][k]["prompt_tokens_max"]
    print(f"  k={k:>2}: max prompt_tokens={t}/7616  slack={7616 - t}")

print("\n=== the decision this round needs from Astra ===")
print("  (A) fact-oriented second retrieval pass + merge")
print("  (B) k=8 only")
print("  (C) fix the over-answering (P2-11) first")
print("  A, B and C are mutually independent; none can be chosen from measurement alone.")
