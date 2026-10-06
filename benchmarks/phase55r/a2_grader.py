"""
a2_grader.py - P55R-01: counter-examples for the answer grader, plus corrected scoring.

Entirely offline. No model calls.

WHY THIS EXISTS
  The grader is a keyword matcher: an answer scores a fact when every keyword in a group
  appears in the text, and a hallucination is flagged when a forbidden string appears. That
  is cheap and repeatable, but it cannot tell these apart:

    * quoting a wrong number in order to DENY it ("the answer is not 4096") is scored as a
      hallucination even though the model behaved correctly
    * explaining why an alternative is wrong mentions the wrong value and is penalised
    * giving every required fact AND inventing an extra one scores fully correct
    * refusing when the evidence IS present is indistinguishable from refusing correctly
    * "A or B is acceptable" and "A and B are both required" are encoded identically

  The historical scoring is NOT changed. The original grader is preserved and the corrected
  grader is reported alongside it, so every published figure remains comparable while the
  direction of its error becomes visible.

Usage:
  python a2_grader.py --out ../results/<run-id>/a2_grader.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------------------
# the ORIGINAL grader, copied verbatim in behaviour from the Phase 4/5 harnesses
# --------------------------------------------------------------------------
def grade_original(answer: str, facts: list[list[str]], bad: list[list[str]]) -> dict:
    low = answer.lower()
    hits = [all(k.lower() in low for k in grp) for grp in facts if grp]
    halluc = [all(k.lower() in low for k in grp) for grp in bad if grp]
    refused = ("not in sources" in low or "i do not know" in low or "\u4e0d\u77e5\u9053" in low
               or "not provided" in low or "no information" in low)
    return {"fact_hits": sum(1 for h in hits if h),
            "fact_total": len(hits),
            "all_facts": bool(hits) and all(hits),
            "hallucination_flags": sum(1 for h in halluc if h),
            "refused": refused}


# --------------------------------------------------------------------------
# the CORRECTED grader
# --------------------------------------------------------------------------
NEGATION_CUES = ("not ", "isn't", "is not", "was not", "never", "incorrect", "wrong",
                 "no,", "rather than", "instead of", "does not", "doesn't",
                 "\u4e0d\u662f", "\u9519\u8bef", "\u5e76\u975e")
# Contrastive correction cues. Added after a counter-example FAILED: the sentence
# "Some sources say 4096, but that applies to a different model." corrects the value rather
# than asserting it, and the first version of this list missed it and counted a
# hallucination. These are deliberately specific phrases - a bare "but" would excuse almost
# any assertion and make the grader blind to real errors, which the third counter-example
# exists to catch.
CORRECTION_CUES = ("but that", "but it", "but this", "that applies to", "applies to a",
                   "a different", "refers to", "instead", "rather than", "however",
                   "although", "is not the", "was not the", "\u800c\u662f", "\u4f46\u90a3")


def _mentioned_positively(answer: str, value: str) -> bool:
    """
    True when `value` appears WITHOUT a nearby negation or correction.

    A window around the match is inspected rather than the whole answer, because a single
    negation elsewhere should not excuse a genuine assertion. This is deliberately simple
    and auditable; it is not a claim to solve natural-language negation.
    """
    low = answer.lower()
    v = value.lower()
    start = 0
    while True:
        i = low.find(v, start)
        if i < 0:
            return False
        window = low[max(0, i - 60): i + len(v) + 40]
        if not any(cue in window for cue in NEGATION_CUES + CORRECTION_CUES):
            return True
        start = i + 1


def grade_corrected(answer: str, facts: list[list[str]], bad: list[list[str]],
                    *, mode: str = "and", min_extra_sentences: int = 0) -> dict:
    """
    Corrected grader with explicit semantics.

    mode="and"  every group in `facts` is required (the historical meaning)
    mode="or"   ANY group in `facts` is sufficient (used where several answers are valid)

    Hallucination flags now require the forbidden value to be ASSERTED, not merely
    mentioned, so quoting a wrong number in order to deny it is no longer a false positive.
    """
    low = answer.lower()
    groups = [g for g in facts if g]
    per_group = [all(k.lower() in low for k in g) for g in groups]
    if mode == "or":
        all_facts = any(per_group) if per_group else False
    else:
        all_facts = bool(per_group) and all(per_group)

    halluc = 0
    halluc_detail = []
    for grp in bad:
        if not grp:
            continue
        # a forbidden value counts only if it is asserted rather than denied
        if all(k.lower() in low for k in grp):
            if any(_mentioned_positively(answer, k) for k in grp):
                halluc += 1
                halluc_detail.append(grp)

    refused = ("not in sources" in low or "i do not know" in low or "\u4e0d\u77e5\u9053" in low
               or "not provided" in low or "no information" in low
               or "cannot determine" in low or "\u65e0\u6cd5\u786e\u5b9a" in low)

    # unsupported extra numbers: a digit-bearing claim not present in the expected facts.
    # Trailing punctuation is stripped so "8192." and "8192" are not treated as different
    # values - the first version reported sentence-final digits as unsupported, which is an
    # artefact of the regex rather than a finding.
    def _norm_digits(text: str) -> set[str]:
        return {m.rstrip(".,;:") for m in re.findall(r"\d[\d.,]*", text)}

    expected_digits = _norm_digits(" ".join(k for g in groups for k in g))
    said_digits = _norm_digits(answer)
    unsupported = sorted(said_digits - expected_digits)

    return {
        "fact_hits": sum(1 for h in per_group if h),
        "fact_total": len(per_group),
        "all_facts": all_facts,
        "mode": mode,
        "hallucination_flags": halluc,
        "hallucination_detail": halluc_detail,
        "refused": refused,
        "unsupported_numbers": unsupported,
    }


# --------------------------------------------------------------------------
# counter-examples
# --------------------------------------------------------------------------
CASES = [
    dict(
        name="denies a quoted wrong number",
        answer="The context size is not 4096; the correct value is 8192.",
        facts=[["8192"]], bad=[["4096"]],
        expect_corrected={"all_facts": True, "hallucination_flags": 0},
        note="must NOT be a hallucination: the wrong value is being denied",
    ),
    dict(
        name="explains why another answer is wrong",
        answer="It is 8192. Some sources say 4096, but that applies to a different model.",
        facts=[["8192"]], bad=[["4096"]],
        expect_corrected={"all_facts": True, "hallucination_flags": 0},
        note="mentioning a wrong value while correcting it is not an error",
    ),
    dict(
        name="asserts the wrong number (must still be flagged)",
        answer="The context size is 4096.",
        facts=[["8192"]], bad=[["4096"]],
        expect_corrected={"all_facts": False, "hallucination_flags": 1},
        note="the corrected grader must not become blind to real errors",
    ),
    dict(
        name="all required facts plus an invented one",
        answer="It uses 8192 tokens and occupies 2981 MiB and runs at 999 tokens/s.",
        facts=[["8192"], ["2981"]], bad=[[]],
        expect_corrected={"all_facts": True},
        note="ALL_FACTS is true, but unsupported_numbers must expose the fabrication",
    ),
    dict(
        name="correct refusal when evidence is absent",
        answer="NOT IN SOURCES",
        facts=[["8192"]], bad=[[]],
        expect_corrected={"refused": True, "all_facts": False},
        note="refusing is correct here",
    ),
    dict(
        name="incorrect refusal when the evidence WAS provided",
        answer="I do not know.",
        facts=[["8192"]], bad=[[]],
        expect_corrected={"refused": True, "all_facts": False},
        note="identical to the previous case under both graders - the grader cannot tell "
             "them apart, which is why refusal rate alone is never evidence of improvement",
    ),
    dict(
        name="OR semantics: either value is acceptable",
        answer="Roughly 130 tokens per second.",
        facts=[["130"], ["131"]], bad=[[]],
        mode="or",
        expect_corrected={"all_facts": True},
        note="with mode='or' one acceptable answer suffices",
    ),
    dict(
        name="AND semantics: same text, both required",
        answer="Roughly 130 tokens per second.",
        facts=[["130"], ["131"]], bad=[[]],
        mode="and",
        expect_corrected={"all_facts": False},
        note="with mode='and' the same text fails - the encoding of the requirement, not "
             "the answer, decides the score",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"cases": [], "note": "offline; historical grader preserved unchanged"}
    print("=== grader counter-examples ===\n")
    print(f"{'case':<44} {'orig':<22} {'corrected':<26}")
    print("-" * 96)

    for c in CASES:
        orig = grade_original(c["answer"], c["facts"], c["bad"])
        corr = grade_corrected(c["answer"], c["facts"], c["bad"],
                               mode=c.get("mode", "and"))
        exp = c["expect_corrected"]
        ok = all(corr[k] == v for k, v in exp.items())

        o = f"facts={orig['fact_hits']}/{orig['fact_total']} hall={orig['hallucination_flags']}"
        n = (f"facts={corr['fact_hits']}/{corr['fact_total']} "
             f"hall={corr['hallucination_flags']} "
             f"unsup={len(corr['unsupported_numbers'])}")
        print(f"{c['name']:<44} {o:<22} {n:<26}")
        print(f"    {c['note']}")
        if corr["unsupported_numbers"]:
            print(f"    unsupported numbers: {corr['unsupported_numbers']}")
        report["cases"].append({
            "name": c["name"], "answer": c["answer"],
            "original_grader": orig, "corrected_grader": corr,
            "expected_corrected": exp, "matched_expectation": ok, "note": c["note"],
        })
        check(f"corrected grader behaves as specified: {c['name']}", ok,
              f"expected {exp}, got "
              f"{ {k: corr[k] for k in exp} }")

    # ---------------- the two false-positive directions are demonstrated
    print("\n=== does the original grader produce false positives? ===")
    fp = report["cases"][0]
    check("the ORIGINAL grader flags a denied wrong number as a hallucination",
          fp["original_grader"]["hallucination_flags"] == 1
          and fp["corrected_grader"]["hallucination_flags"] == 0,
          f"orig={fp['original_grader']['hallucination_flags']} "
          f"corr={fp['corrected_grader']['hallucination_flags']}")

    fn = report["cases"][3]
    check("both graders call 'all facts plus a fabricated number' fully correct, "
          "so fabricated extras are invisible to the score",
          fn["original_grader"]["all_facts"] is True
          and fn["corrected_grader"]["all_facts"] is True,
          "unsupported_numbers is what exposes it, not all_facts")

    # ---------------- summary of what this means for published figures
    report["implications"] = {
        "direction_of_error": ("the original grader can only OVER-count hallucinations "
                               "(it flags denied values) and can only OVER-count "
                               "correctness (fabricated extras are invisible)"),
        "consequence": ("published hallucination counts are upper bounds and published "
                        "accuracy is an upper bound. Neither is retracted; both are now "
                        "labelled."),
        "refusal_not_evidence": ("correct and incorrect refusals are indistinguishable "
                                 "under both graders, which is why a falling refusal rate "
                                 "must never be read as improving accuracy"),
        "historical_scoring": "preserved and reported separately; not overwritten",
    }
    print("\n=== implications for already-published figures ===")
    for k, v in report["implications"].items():
        print(f"  {k}: {v}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("grader counter-examples pass; historical scoring preserved separately")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
