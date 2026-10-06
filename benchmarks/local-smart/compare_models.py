"""
compare_models.py - Phase 3: direct local-fast (4B) vs local-smart (9B) head-to-head.

Reads the result JSONs produced by repair_suite.py, tasks_c_review.py, run_phase3.py
and run_bench.py for both models and prints the comparison table the Phase 3 report
requires, with a winner per row.

Usage: python compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(name: str):
    p = R / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k) if isinstance(cur, dict) else None
    return default if cur is None else cur


rows: list[tuple[str, str, str, str]] = []

# ---- Test B: bug repair
r4, r9 = load("repair_4b.json"), load("repair_9b.json")
if r4 and r9:
    rows.append(("B bug repair pass@1", f"{r4['pass_at_1_rate']}% ({r4['pass_at_1']}/{len(r4['fixtures'])})",
                 f"{r9['pass_at_1_rate']}% ({r9['pass_at_1']}/{len(r9['fixtures'])})",
                 "4B" if r4["pass_at_1"] > r9["pass_at_1"] else ("9B" if r9["pass_at_1"] > r4["pass_at_1"] else "tie")))
    rows.append(("B bug repair pass@2", f"{r4['pass_at_2_rate']}%", f"{r9['pass_at_2_rate']}%",
                 "tie" if r4["pass_at_2"] == r9["pass_at_2"] else
                 ("4B" if r4["pass_at_2"] > r9["pass_at_2"] else "9B")))
    rows.append(("B scope creep", str(r4["scope_creep_count"]), str(r9["scope_creep_count"]),
                 "tie" if r4["scope_creep_count"] == r9["scope_creep_count"] else
                 ("4B" if r4["scope_creep_count"] < r9["scope_creep_count"] else "9B")))

# ---- Test C: code review
v4, v9 = load("review_4b.json"), load("review_9b.json")
if v4 and v9:
    rows.append(("C review recall", f"{v4['recall_pct']}% ({v4['true_positives']}/{v4['planted_total']})",
                 f"{v9['recall_pct']}% ({v9['true_positives']}/{v9['planted_total']})",
                 "4B" if v4["recall_pct"] > v9["recall_pct"] else ("9B" if v9["recall_pct"] > v4["recall_pct"] else "tie")))
    rows.append(("C review precision", f"{v4['precision_pct']}%", f"{v9['precision_pct']}%",
                 "tie" if v4["precision_pct"] == v9["precision_pct"] else
                 ("4B" if v4["precision_pct"] > v9["precision_pct"] else "9B")))
    rows.append(("C false positives", str(v4["false_positives"]), str(v9["false_positives"]),
                 "tie" if v4["false_positives"] == v9["false_positives"] else
                 ("4B" if v4["false_positives"] < v9["false_positives"] else "9B")))

# ---- Tests E-H
p4, p9 = load("phase3_4b.json"), load("phase3_9b.json")
if p4 and p9:
    e4, e9 = get(p4, "tests", "E"), get(p9, "tests", "E")
    if e4 and e9:
        rows.append(("E hierarchy violations", str(len(e4.get("mechanical_violations") or [])),
                     str(len(e9.get("mechanical_violations") or [])),
                     "tie" if len(e4.get("mechanical_violations") or []) == len(e9.get("mechanical_violations") or []) else
                     ("4B" if len(e4.get("mechanical_violations") or []) < len(e9.get("mechanical_violations") or []) else "9B")))
        rows.append(("E bug actually fixed", str(e4.get("bug_fixed")), str(e9.get("bug_fixed")),
                     "tie" if e4.get("bug_fixed") == e9.get("bug_fixed") else "9B" if e9.get("bug_fixed") else "4B"))

    f4, f9 = get(p4, "tests", "F"), get(p9, "tests", "F")
    if f4 and f9:
        rows.append(("F current state correct", str(f4.get("current_state_correct")), str(f9.get("current_state_correct")),
                     "tie" if f4.get("current_state_correct") == f9.get("current_state_correct") else
                     ("4B" if f4.get("current_state_correct") else "9B")))
        rows.append(("F obsolete identified",
                     f"{f4.get('obsolete_identified')}/{f4.get('obsolete_total')}",
                     f"{f9.get('obsolete_identified')}/{f9.get('obsolete_total')}",
                     "4B" if (f4.get('obsolete_identified') or 0) > (f9.get('obsolete_identified') or 0) else
                     ("9B" if (f9.get('obsolete_identified') or 0) > (f4.get('obsolete_identified') or 0) else "tie")))
        rows.append(("F conflict identified", str(f4.get("conflict_identified")), str(f9.get("conflict_identified")),
                     "tie" if f4.get("conflict_identified") == f9.get("conflict_identified") else
                     ("4B" if f4.get("conflict_identified") else "9B")))

    g4, g9 = get(p4, "tests", "G"), get(p9, "tests", "G")
    if g4 and g9:
        rows.append(("G json valid", f"{g4.get('json_rate')}%", f"{g9.get('json_rate')}%",
                     "tie" if g4.get("json_rate") == g9.get("json_rate") else
                     ("4B" if (g4.get("json_rate") or 0) > (g9.get("json_rate") or 0) else "9B")))
        rows.append(("G schema match", f"{g4.get('schema_rate')}%", f"{g9.get('schema_rate')}%",
                     "4B" if (g4.get("schema_rate") or 0) > (g9.get("schema_rate") or 0) else
                     ("9B" if (g9.get("schema_rate") or 0) > (g4.get("schema_rate") or 0) else "tie")))
        rows.append(("G reasoning leaks", str(g4.get("leaks")), str(g9.get("leaks")),
                     "tie" if g4.get("leaks") == g9.get("leaks") else
                     ("4B" if (g4.get("leaks") or 0) < (g9.get("leaks") or 0) else "9B")))

    h4, h9 = get(p4, "tests", "H"), get(p9, "tests", "H")
    if h4 and h9:
        rows.append(("H routing accuracy", f"{h4.get('accuracy')}%", f"{h9.get('accuracy')}%",
                     "4B" if (h4.get("accuracy") or 0) > (h9.get("accuracy") or 0) else
                     ("9B" if (h9.get("accuracy") or 0) > (h4.get("accuracy") or 0) else "tie")))
        rows.append(("H under-escalations", str(h4.get("under_escalations")), str(h9.get("under_escalations")),
                     "4B" if (h4.get("under_escalations") or 0) < (h9.get("under_escalations") or 0) else
                     ("9B" if (h9.get("under_escalations") or 0) < (h4.get("under_escalations") or 0) else "tie")))

# ---- Test A / F handoff via run_bench
a4, a9 = load("p3_4b_AF.json"), load("p3_9b_AEF.json")
if a4 and a9:
    g4 = get(a4, "tests", "A", "grade", default={})
    g9 = get(a9, "tests", "A", "grade", default={})
    rows.append(("A summary facts", f"{g4.get('facts_hit')}/{g4.get('facts_total')}",
                 f"{g9.get('facts_hit')}/{g9.get('facts_total')}",
                 "4B" if (g4.get("facts_hit") or 0) > (g9.get("facts_hit") or 0) else
                 ("9B" if (g9.get("facts_hit") or 0) > (g4.get("facts_hit") or 0) else "tie")))
    rows.append(("A hallucinations", str(len(g4.get("hallucination_markers") or [])),
                 str(len(g9.get("hallucination_markers") or [])),
                 "tie" if len(g4.get("hallucination_markers") or []) == len(g9.get("hallucination_markers") or []) else
                 ("4B" if len(g4.get("hallucination_markers") or []) < len(g9.get("hallucination_markers") or []) else "9B")))
    h4 = get(a4, "tests", "F", default={})
    h9 = get(a9, "tests", "F", default={})
    rows.append(("F handoff sections", f"{h4.get('sections_ok')}/{h4.get('sections_total')}",
                 f"{h9.get('sections_ok')}/{h9.get('sections_total')}",
                 "tie" if h4.get("sections_ok") == h9.get("sections_ok") else
                 ("4B" if (h4.get("sections_ok") or 0) > (h9.get("sections_ok") or 0) else "9B")))

w = max((len(r[0]) for r in rows), default=20)
print("=" * (w + 42))
print(f"{'Test':<{w}}  {'local-fast (4B)':<18} {'local-smart (9B)':<18} Winner")
print("=" * (w + 42))
wins = {"4B": 0, "9B": 0, "tie": 0}
for label, a, b, win in rows:
    wins[win] = wins.get(win, 0) + 1
    print(f"{label:<{w}}  {a:<18} {b:<18} {win}")
print("=" * (w + 42))
print(f"rows won:  4B={wins['4B']}  9B={wins['9B']}  tie={wins['tie']}")
print()
print("NOTE: this counts ROWS, not importance. Several rows are ties or near-ties;")
print("read the individual metrics rather than the tally.")
