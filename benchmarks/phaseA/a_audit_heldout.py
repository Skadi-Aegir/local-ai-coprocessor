"""
a_audit_heldout.py - mechanical audit of a run of the Phase A held-out set (heldout_v1).

Reads the records `cli/ask.py --json --out` wrote and the pre-registration (heldout_v1/expected.json)
and reports, per question, only what can be decided mechanically:

  outcome       answered / refused / error, against the expected outcome
  evidence      was an expected source file RETRIEVED (candidates), SHOWN to the model (the pack's
                supplied sources), CITED - and does a cited range overlap the expected lines
  verification  the citation statuses
  key facts     which expected fact groups appear in the answer text. A heuristic: mention is not
                assertion (OPEN_ISSUES P2-1), so this flags, it does not grade
  superseded    whether a superseded value appears in the answer
  refusal cause for a refused ANSWERABLE question: not retrieved / retrieved but not shown /
                shown but not used / citation unverified / no valid citation. The cause is
                recorded beside the outcome and never re-labels it: it stays an incorrect refusal

It never decides support. That verdict is human and is written per question in review.md.

It refuses to audit a run whose questions are not exactly the frozen set in order, whose frozen file
no longer matches the recorded hash, or whose index does not meet the run conditions.

Usage:
  python a_audit_heldout.py --answers heldout_v1/results/run1.json
  python a_audit_heldout.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from provenance import HASH_SCHEME, content_hash  # noqa: E402

SET_DIR = HERE / "heldout_v1"


def load_expected() -> dict:
    return json.loads((SET_DIR / "expected.json").read_text(encoding="utf-8"))


def check_frozen(exp: dict) -> str | None:
    f = REPO / exp["frozen_file"]
    if content_hash(f.read_bytes()) != exp["frozen_sha256_lf"]:
        return f"{exp['frozen_file']} no longer matches its recorded hash; the set is not frozen"
    return None


def check_index(index_dir: Path, records: list[dict]) -> list[str]:
    """The run conditions that can be read from the index manifest and the records."""
    problems = []
    try:
        m = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"cannot read {index_dir}/manifest.json: {e}"]
    if m.get("hash_scheme") != HASH_SCHEME:
        problems.append(f"index hash_scheme is {m.get('hash_scheme')!r}, not {HASH_SCHEME}")
    if m.get("index_version_scheme") != 2 or not m.get("chunks_sha256"):
        problems.append("the index predates the 2026-10-05 builder (no index_version_scheme 2 / "
                        "chunks_sha256): it was not rebuilt after the corpus corrections")
    versions = {r.get("index_version") for r in records}
    if versions != {m.get("index_version")}:
        problems.append(f"records name index version(s) {sorted(map(str, versions))}, the "
                        f"manifest {m.get('index_version')}")
    prompts = {r.get("prompt_version") for r in records}
    if prompts != {"qa-v2"}:
        problems.append(f"prompt version(s) {sorted(map(str, prompts))}, expected qa-v2")
    return problems


def _overlap(a: tuple[int, int], b: list[int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def audit_one(q: dict, rec: dict) -> dict:
    text = (rec.get("answer") or "") if not rec.get("refused") else ""
    low = text.lower()
    outcome = "error" if rec.get("error") else ("refused" if rec.get("refused") else "answered")
    files = {s["path"]: s["lines"] for s in q.get("sources", [])}
    retrieved = sorted({c["source_path"] for c in rec.get("candidates") or []
                        if c.get("source_path") in files})
    shown = sorted({s["source_path"] for s in (rec.get("pack") or {}).get("supplied") or []
                    if s.get("source_path") in files})
    cits = rec.get("citations") or []
    cited = sorted({c["source_path"] for c in cits if c.get("source_path") in files})
    cited_overlap = any(_overlap((c["start_line"], c["end_line"]), files[c["source_path"]])
                        for c in cits if c.get("source_path") in files)
    row = {"id": q["id"], "category": q["category"], "expected": q["expected_outcome"],
           "outcome": outcome, "error_kind": rec.get("error_kind"),
           "evidence_retrieved": retrieved, "evidence_shown": shown, "evidence_cited": cited,
           "cited_range_overlaps_expected": cited_overlap,
           "citation_status": [(c.get("verification") or {}).get("status") for c in cits],
           "other_files_cited": sorted({c["source_path"] for c in cits} - set(files))}
    if q["expected_outcome"] == "refuse":
        row["verdict"] = {"refused": "correct_refusal", "error": "error"}.get(
            outcome, "answered_an_answer_absent_question")
        return row
    groups = q.get("key_facts") or []
    present = [any(alt.lower() in low for alt in g) for g in groups]
    row["key_fact_groups_present"] = f"{sum(present)}/{len(groups)}"
    row["key_fact_groups_missing"] = [g for g, ok in zip(groups, present) if not ok]
    row["superseded_value_present"] = [s for s in q.get("superseded") or [] if s.lower() in low]
    if outcome == "error":
        row["verdict"] = "error"
    elif outcome == "refused":
        row["verdict"] = "incorrect_refusal"
        if rec.get("stale_or_unverifiable_citations"):
            cause = "citation_unverified"
        elif rec.get("unsupported_draft") is not None:
            cause = "no_valid_citation"
        elif not retrieved:
            cause = "evidence_not_retrieved"
        elif not shown:
            cause = "evidence_retrieved_not_shown"
        else:
            cause = "evidence_shown_not_used"
        row["refusal_cause"] = cause
    else:
        row["verdict"] = "answered_needs_support_review"
    return row


def audit(exp: dict, records: list[dict]) -> dict:
    rows = [audit_one(q, r) for q, r in zip(exp["questions"], records)]
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        counts.setdefault(r["category"], {})
        counts[r["category"]][r["verdict"]] = counts[r["category"]].get(r["verdict"], 0) + 1
    return {"set": exp["set"], "frozen_sha256_lf": exp["frozen_sha256_lf"],
            "index_version": records[0].get("index_version") if records else None,
            "counts_by_category": counts, "questions": rows,
            "note": "mechanical audit only; the support verdict is human (review.md)"}


def print_table(rep: dict) -> None:
    print(f"\n| id | category | outcome | verdict | retrieved / shown / cited | key facts | notes |")
    print("|---|---|---|---|---|---|---|")
    for r in rep["questions"]:
        path = (f"{'Y' if r['evidence_retrieved'] else '-'} / {'Y' if r['evidence_shown'] else '-'}"
                f" / {'Y' if r['evidence_cited'] else '-'}") if r["expected"] == "answer" else "n/a"
        notes = []
        if r.get("refusal_cause"):
            notes.append(r["refusal_cause"])
        if r.get("superseded_value_present"):
            notes.append(f"SUPERSEDED: {r['superseded_value_present']}")
        if r.get("error_kind"):
            notes.append(r["error_kind"])
        print(f"| {r['id']} | {r['category']} | {r['outcome']} | {r['verdict']} | {path} | "
              f"{r.get('key_fact_groups_present', 'n/a')} | {'; '.join(notes)} |")
    print(f"\ncounts by category: {json.dumps(rep['counts_by_category'])}")


def selftest() -> int:
    """Synthetic records through every verdict path; no run, no service."""
    exp = load_expected()
    fails = []

    def rec_for(q, **kw):
        base = {"question": q["question"], "index_version": "x", "prompt_version": "qa-v2",
                "candidates": [], "citations": [], "pack": {"supplied": []}}
        base.update(kw)
        return base
    q_ans = next(q for q in exp["questions"] if q["expected_outcome"] == "answer")
    q_abs = next(q for q in exp["questions"] if q["expected_outcome"] == "refuse")
    src = q_ans["sources"][0]
    hit = {"source_path": src["path"], "start_line": src["lines"][0], "end_line": src["lines"][1]}
    cases = [
        (q_ans, rec_for(q_ans, refused=False, answer=" ".join(g[0] for g in q_ans["key_facts"]),
                        candidates=[hit], pack={"supplied": [hit]},
                        citations=[dict(hit, source=1, verification={"status": "verified"})]),
         "answered_needs_support_review"),
        (q_ans, rec_for(q_ans, refused=True, answer="r"), "incorrect_refusal"),
        (q_abs, rec_for(q_abs, refused=True, answer="r"), "correct_refusal"),
        (q_abs, rec_for(q_abs, refused=False, answer="12 %"), "answered_an_answer_absent_question"),
        (q_ans, rec_for(q_ans, error="x", error_kind="upstream_timeout"), "error"),
    ]
    for q, r, want in cases:
        got = audit_one(q, r)
        if got["verdict"] != want:
            fails.append((q["id"], want, got["verdict"]))
    r = audit_one(q_ans, rec_for(q_ans, refused=True, answer="r", candidates=[hit]))
    if r.get("refusal_cause") != "evidence_retrieved_not_shown":
        fails.append(("cause", r.get("refusal_cause")))
    r = audit_one(q_ans, rec_for(q_ans, refused=True, answer="r", candidates=[hit],
                                 pack={"supplied": [hit]}))
    if r.get("refusal_cause") != "evidence_shown_not_used":
        fails.append(("cause2", r.get("refusal_cause")))
    full = cases[0][1]
    if audit_one(q_ans, full)["key_fact_groups_present"] != f"{len(q_ans['key_facts'])}/" \
                                                             f"{len(q_ans['key_facts'])}":
        fails.append("key facts")
    if check_frozen(exp) is not None:
        fails.append(check_frozen(exp))
    print("FAILED: " + str(fails) if fails else "audit self-test passed")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers")
    ap.add_argument("--index", default=str(REPO / "vector/qa/current"))
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.answers:
        ap.error("--answers is required")

    exp = load_expected()
    problem = check_frozen(exp)
    if problem:
        print(f"REFUSING to audit: {problem}")
        return 3
    records = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    asked = [r.get("question", "").lstrip("﻿") for r in records]
    frozen = [q["question"] for q in exp["questions"]]
    if asked != frozen:
        print("REFUSING to audit: the run's questions are not the frozen set, in order")
        return 3
    conditions = check_index(Path(args.index), records)
    if conditions:
        print("REFUSING to audit: the run does not meet the set's run conditions:")
        for c in conditions:
            print(f"  {c}")
        return 3

    rep = audit(exp, records)
    print_table(rep)
    out = Path(args.out) if args.out else Path(args.answers).with_name(
        Path(args.answers).stem + "_audit.json")
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
