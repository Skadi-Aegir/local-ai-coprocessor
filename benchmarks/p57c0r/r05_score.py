"""
P57-C0R - final corrected scorer.

Two classifier defects found in r01 by reading the actual answers:

DEFECT A - "refusal + reasoning" is not detected as a refusal.
  cli/ask.py decides `refused` by EXACT match against the refusal sentence, so when the model writes
  the refusal sentence FOLLOWED BY an explanation, refused stays False and the record looks answered.
  Measured: v01 at k=8 and k=10 are both "I could not find enough evidence... **Reasoning:** ...".
  The consequence is worse than a wrong label: the reasoning is then scanned for fact markers, so the
  model's own restatement of the question ("listens on port 8111") registers as a covered fact.
  This scorer therefore treats an answer CONTAINING the refusal sentence as a refusal.

DEFECT B - marker hits inside a refusal's reasoning must not count as covered facts.
  Consequence of A. Once the answer is a refusal, the answer-side fact markers are recorded but the
  verdict is an incorrect refusal (for an answerable question), regardless of what the prose contains.

This is a SCORER correction. It does not change cli/ask.py, and it does not change the frozen set.
It also means the earlier verdict columns in r01 are superseded.

Reads r01_measured.json and rewrites the verdict layer; re-runs nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402

SRC = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r/results/r01_measured.json")
OUT = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r/results/r02_scored.json")
REFUSAL = ASK.REFUSAL


def contains_refusal(answer: str) -> bool:
    return REFUSAL.lower() in (answer or "").lower()


def main() -> int:
    if OUT.exists():
        print(f"REFUSING to overwrite {OUT}")
        return 3
    d = json.loads(SRC.read_text(encoding="utf-8"))
    d["scorer_corrections"] = {
        "defect_A": ("an answer CONTAINING the refusal sentence is a refusal, because cli/ask.py's "
                     "exact-match detector misses 'refusal + reasoning'"),
        "defect_B": ("fact markers found inside a refusal's reasoning are not covered facts; the "
                     "verdict for an answerable question is then an incorrect refusal"),
        "source": "reads r01_measured.json; re-runs no model call",
    }
    n_ref = 0
    for qid, q in d["probes"].items():
        for k, b in q["by_k"].items():
            ans = b["answer"] or ""
            refusal_text = contains_refusal(ans)
            b["contains_refusal_sentence"] = refusal_text
            b["api_refused_flag"] = b["outcome"] == "refused"
            if refusal_text:
                n_ref += 1
                b["verdict_final"] = "incorrect_refusal"
                b["note"] = ("the answer contains the refusal sentence, so it is a refusal; the "
                             "reasoning prose is not counted as covering any fact")
                b["facts_in_answer_for_verdict"] = 0
            else:
                b["facts_in_answer_for_verdict"] = b["facts_in_answer"]
                nf = len(q["required_facts"])
                if b["facts_in_answer"] == nf:
                    b["verdict_final"] = "complete"
                elif b["facts_in_answer"] > 0:
                    b["verdict_final"] = "partial"
                else:
                    b["verdict_final"] = "incorrect"

    print("=== verdicts: as r01 classified them, vs corrected ===")
    print(f"  {'q':<5}{'k':<4}{'r01 verdict':<20}{'corrected':<20}{'contains refusal':<18}")
    for qid, q in d["probes"].items():
        for k in ("5", "8", "10"):
            b = q["by_k"][k]
            mark = "  <-- CHANGED" if b["verdict_marker_based"] != b["verdict_final"] else ""
            print(f"  {qid:<5}{k:<4}{b['verdict_marker_based']:<20}{b['verdict_final']:<20}"
                  f"{str(b['contains_refusal_sentence']):<18}{mark}")

    print(f"\n  answers containing the refusal sentence: {n_ref} of "
          f"{sum(len(q['by_k']) for q in d['probes'].values())}")

    print("\n=== corrected overall per k ===")
    d["summary_final"] = {}
    for k in ("5", "8", "10"):
        counts: dict[str, list[str]] = {}
        for qid, q in d["probes"].items():
            v = q["by_k"][k]["verdict_final"]
            counts.setdefault(v, []).append(qid)
        pack_ok = [qid for qid, q in d["probes"].items()
                   if q["by_k"][k]["facts_in_pack"] == len(q["required_facts"])]
        d["summary_final"][k] = {"verdicts": counts, "facts_all_in_pack": pack_ok}
        print(f"  k={k:>2}: facts all in pack {len(pack_ok)}/6 {pack_ok}")
        print(f"        verdicts: {json.dumps(counts)}")

    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
