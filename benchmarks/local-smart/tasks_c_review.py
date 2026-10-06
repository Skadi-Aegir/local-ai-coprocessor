"""
tasks_c_review.py - Phase 3 Test C: code review scoring.

Scores a model's review of a deliberately flawed patch against a labelled inventory:
  - true positives   : planted issues the model found
  - missed           : planted issues it did not find
  - false positives  : things it flagged that are not problems

Matching is keyword-based over the whole normalised review text, which is robust to
wording and language (the fixtures accept Chinese and English phrasings).

Usage:
  python tasks_c_review.py --base-url http://127.0.0.1:8111 --model local-fast --out results/review_4b.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))
from common import Client, extract_json, thinking_leak  # noqa: E402

REVIEW_DIR = HERE / "fixtures" / "review"

SYSTEM = """You are a meticulous code reviewer.

Review the proposed patch and reply with ONLY a JSON object, no prose, no code fences:
{
  "issues": [
    {"title": "<short title>", "file": "<file>", "severity": "critical|high|medium|low",
     "detail": "<why it is a problem>"}
  ],
  "severity": "critical|high|medium|low",
  "needs_cloud_review": true
}

"severity" at the top level is the highest severity among the issues you list.
Set "needs_cloud_review" to true only if the patch needs a stronger reviewer than you.

Report ONLY real problems. Do not invent issues, and do not report things that are
correct or unchanged. An empty issues list is a valid answer if the patch is clean."""

SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "file": {"type": "string"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "detail": {"type": "string"},
                },
                "required": ["title", "file", "severity", "detail"],
                "additionalProperties": False,
            },
        },
        "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
        "needs_cloud_review": {"type": "boolean"},
    },
    "required": ["issues", "severity", "needs_cloud_review"],
    "additionalProperties": False,
}

GRAMMAR = r'''
root     ::= "{" ws "\"issues\"" ws ":" ws issues "," ws "\"severity\"" ws ":" ws sev "," ws
                 "\"needs_cloud_review\"" ws ":" ws bool ws "}"
issues   ::= "[" ws (issue (ws "," ws issue)*)? ws "]"
issue    ::= "{" ws "\"title\"" ws ":" ws jstr "," ws "\"file\"" ws ":" ws jstr "," ws
                 "\"severity\"" ws ":" ws sev "," ws "\"detail\"" ws ":" ws jstr ws "}"
sev      ::= "\"critical\"" | "\"high\"" | "\"medium\"" | "\"low\""
bool     ::= "true" | "false"
jstr     ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
ws       ::= [ \t\n]*
'''.strip()


def load_expected(name: str):
    """
    Load the ground-truth module for a review fixture.

    Fixture files are named review1_limiter.md / review2_orders.md but the expected
    modules are review1_expected.py / review2_expected.py, so only the leading
    "reviewN" part is used. (An earlier version appended the whole stem and raised
    FileNotFoundError.)
    """
    prefix = name.split("_", 1)[0]          # review1_limiter -> review1
    path = REVIEW_DIR / f"{prefix}_expected.py"
    if not path.is_file():
        raise FileNotFoundError(f"ground truth missing for {name}: {path}")
    spec = importlib.util.spec_from_file_location(f"exp_{prefix}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalise(obj: dict | None, raw: str) -> str:
    """Flatten the review to lowercase text for keyword matching."""
    parts = [raw or ""]
    if isinstance(obj, dict):
        for it in obj.get("issues") or []:
            if isinstance(it, dict):
                parts.append(str(it.get("title", "")))
                parts.append(str(it.get("detail", "")))
                parts.append(str(it.get("file", "")))
                parts.append(str(it.get("severity", "")))
    return " ".join(parts).lower()


def hit(text: str, alternatives: list[list[str]]) -> bool:
    for group in alternatives:
        if all(k.lower() in text for k in group):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--model", default="local-fast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grammar", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    c = Client(args.base_url, timeout=900)
    c.health()

    targets = sorted(p for p in REVIEW_DIR.glob("review*.md"))
    rows = []
    for md in targets:
        name = md.stem
        exp = load_expected(name)
        text = md.read_text(encoding="utf-8")
        print(f"--- {name} ---", flush=True)
        t0 = time.time()
        kw = {"grammar": GRAMMAR} if args.grammar else {}
        r = c.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": text + "\n\nReview the patch."}],
            model=args.model, max_tokens=2000, temperature=0.0,
            enable_thinking=args.thinking, **kw,
        )
        parsed = extract_json(r.text)
        norm = normalise(parsed, r.text)

        found, missed = [], []
        for issue in exp.EXPECTED:
            (found if hit(norm, issue["matches"]) else missed).append(
                {"id": issue["id"], "severity_at_least": issue["severity_at_least"],
                 "desc": issue["desc"]}
            )
        fps = [{"id": n["id"], "desc": n["desc"]}
               for n in exp.NOT_ISSUES if hit(norm, n["matches"])]

        n_issues_reported = len((parsed or {}).get("issues") or []) if isinstance(parsed, dict) else 0
        rows.append({
            "fixture": name,
            "json_ok": parsed is not None,
            "n_issues_reported": n_issues_reported,
            "true_positives": len(found),
            "planted_total": len(exp.EXPECTED),
            "recall": round(100 * len(found) / len(exp.EXPECTED), 1),
            "found": found,
            "missed": missed,
            "false_positives": fps,
            "top_severity": (parsed or {}).get("severity") if isinstance(parsed, dict) else None,
            "needs_cloud_review": (parsed or {}).get("needs_cloud_review") if isinstance(parsed, dict) else None,
            "latency_s": round(time.time() - t0, 2),
            "gen_tps": round(r.gen_tps, 2),
            "leak": thinking_leak(r.text),
            "raw": r.text[:2500],
        })
        print(f"    tp {len(found)}/{len(exp.EXPECTED)}  fp {len(fps)}  "
              f"json={parsed is not None}  {round(time.time()-t0,1)}s")

    tot_found = sum(r["true_positives"] for r in rows)
    tot_planted = sum(r["planted_total"] for r in rows)
    tot_fp = sum(len(r["false_positives"]) for r in rows)
    report = {
        "model": args.model,
        "grammar": args.grammar,
        "fixtures": rows,
        "true_positives": tot_found,
        "planted_total": tot_planted,
        "recall_pct": round(100 * tot_found / tot_planted, 1) if tot_planted else 0,
        "false_positives": tot_fp,
        "precision_pct": round(100 * tot_found / (tot_found + tot_fp), 1) if (tot_found + tot_fp) else 0,
        "json_ok": sum(1 for r in rows if r["json_ok"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.model} code review ===")
    print(f"  recall    {tot_found}/{tot_planted} = {report['recall_pct']}%")
    print(f"  precision {report['precision_pct']}%  (false positives: {tot_fp})")
    print(f"  json ok   {report['json_ok']}/{len(rows)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
