"""
repair_suite.py - Phase 3 Test B: executable bug repair (the core Phase 3 metric).

Flow per fixture:
  pristine copy -> model proposes a patch -> apply -> run that project's pytest
  -> also run the hidden demonstration that exposes the bug -> record pass/fail

Metrics: pass@1, pass@2 (one feedback retry), tests passed, and whether the model
touched files outside its stated repair (scope creep).

A repair only counts as successful when BOTH:
  * the project's own tests still pass (no regression), and
  * the hidden demonstration passes (the actual bug is gone).

Passing the shipped tests alone is NOT success - those tests pass against the buggy
code by construction (see verify_fixtures.py), so they cannot detect the fix.

Usage:
  python repair_suite.py --task A --base-url http://127.0.0.1:8111 --model local-fast
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))
from common import Client, extract_json, thinking_leak  # noqa: E402
from verify_fixtures import DEMO, FIX  # noqa: E402

PROJECTS = HERE / "fixtures" / "projects"


# --------------------------------------------------------------------------
# patch format
# --------------------------------------------------------------------------

PATCH_SYSTEM = """You are a precise software engineer fixing a real bug.

Reply with ONLY a JSON object, no prose and no code fences, in exactly this shape:
{
  "files": ["relative/path.py", ...],
  "edits": [
    {"file": "relative/path.py", "find": "<exact existing text>", "replace": "<new text>"}
  ],
  "explanation": "<one or two sentences: root cause and why this fixes it>"
}

Rules:
- "find" must be an EXACT substring of the current file contents, including indentation.
- Prefer the smallest possible edit. Do not reformat, rename, or restructure.
- Only edit files you list in "files".
- Do not add dependencies."""

PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "string"}},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "find": {"type": "string"},
                    "replace": {"type": "string"},
                },
                "required": ["file", "find", "replace"],
                "additionalProperties": False,
            },
        },
        "explanation": {"type": "string"},
    },
    "required": ["files", "edits", "explanation"],
    "additionalProperties": False,
}

PATCH_GRAMMAR = r'''
root      ::= "{" ws "\"files\"" ws ":" ws strarr "," ws "\"edits\"" ws ":" ws edits "," ws
                  "\"explanation\"" ws ":" ws jstring ws "}"
strarr    ::= "[" ws (jstring (ws "," ws jstring)*)? ws "]"
edits     ::= "[" ws (edit (ws "," ws edit)*)? ws "]"
edit      ::= "{" ws "\"file\"" ws ":" ws jstring "," ws "\"find\"" ws ":" ws jstring "," ws
                  "\"replace\"" ws ":" ws jstring ws "}"
jstring   ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
ws        ::= [ \t\n]*
'''.strip()


def read_project_files(root: Path, max_bytes: int = 12000) -> str:
    """Render every source file in a fixture so the model can see the whole project."""
    parts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in (".py", ".md", ".json", ".txt", ".csv"):
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n... [truncated]\n"
        parts.append(f"===== FILE: {rel} =====\n{text}")
    return "\n\n".join(parts)


def apply_patch(root: Path, patch: dict) -> tuple[bool, list[str], set[str]]:
    """Apply find/replace edits. Returns (ok, messages, set_of_files_touched)."""
    msgs: list[str] = []
    touched: set[str] = set()
    edits = patch.get("edits") or []
    if not edits:
        return False, ["patch contained no edits"], touched

    for i, ed in enumerate(edits, 1):
        f = ed.get("file", "")
        find = ed.get("find", "")
        repl = ed.get("replace", "")
        target = (root / f).resolve()
        # path safety: stay inside the project
        if not str(target).startswith(str(root.resolve())):
            msgs.append(f"edit {i}: path escapes project ({f})")
            return False, msgs, touched
        if not target.is_file():
            msgs.append(f"edit {i}: file not found ({f})")
            return False, msgs, touched
        text = target.read_text(encoding="utf-8")
        if find == "" :
            msgs.append(f"edit {i}: empty find string")
            return False, msgs, touched
        if text.count(find) == 0:
            msgs.append(f"edit {i}: find text not present in {f}")
            return False, msgs, touched
        if text.count(find) > 1:
            msgs.append(f"edit {i}: find text is ambiguous in {f} ({text.count(find)} matches)")
            return False, msgs, touched
        target.write_text(text.replace(find, repl, 1), encoding="utf-8")
        touched.add(f)
        msgs.append(f"edit {i}: applied to {f}")

    return True, msgs, touched


def run_pytest(root: Path, timeout: int = 120) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=root, capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def run_demo(root: Path, code: str, timeout: int = 120) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=root,
                                     delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], cwd=root,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# per-fixture task framing
# --------------------------------------------------------------------------

TASK_BRIEF = {
    "proj1_pagination": (
        "Users report that GET /items?page=1 returns the second page of results, while "
        "page=0 returns the first page. Find the root cause and fix it.",
        ["pagination.py", "service.py", "storage.py", "validators.py", "test_service.py"],
    ),
    "proj2_importer": (
        "The batch importer eventually fails with 'Too many open files' (Errno 24) after "
        "processing many files. It only happens when some inputs are malformed. "
        "Find the root cause and fix it.",
        ["batch.py", "reader.py", "sink.py", "test_import.py", "SYMPTOM.md"],
    ),
    "proj3_report": (
        "monthly_report() under-reports totals whenever a customer appears in more than "
        "one region. Per-region rows are correct. Find the root cause and fix it.",
        ["report.py", "currency.py", "loader.py", "test_report.py", "SYMPTOM.md"],
    ),
}


def run_fixture(client: Client, model: str, name: str, args) -> dict:
    src = PROJECTS / name
    brief, allowed = TASK_BRIEF[name]
    listing = read_project_files(src)

    user = (
        f"PROJECT FILES:\n\n{listing}\n\n"
        f"PROBLEM REPORT:\n{brief}\n\n"
        f"Propose the minimal fix as JSON in the required format."
    )
    messages = [
        {"role": "system", "content": PATCH_SYSTEM},
        {"role": "user", "content": user},
    ]

    attempts = []
    result: dict = {"fixture": name, "attempts": [], "pass_at_1": False, "pass_at_2": False}

    for attempt in (1, 2):
        t0 = time.time()
        kw = {"grammar": PATCH_GRAMMAR} if args.grammar else {}
        r = client.chat(messages, model=model, max_tokens=3000,
                        temperature=0.0 if args.deterministic else 0.2,
                        enable_thinking=args.thinking, **kw)
        rec: dict = {
            "attempt": attempt,
            "latency_s": round(time.time() - t0, 2),
            "gen_tps": round(r.gen_tps, 2),
            "raw": r.text[:1500],
            "leak": thinking_leak(r.text),
            "error": r.error,
        }
        patch = extract_json(r.text)
        rec["json_ok"] = patch is not None

        if patch is None:
            rec["applied"] = False
            rec["note"] = "model did not return parseable JSON"
            attempts.append(rec)
            messages.append({"role": "assistant", "content": r.text[:2000]})
            messages.append({"role": "user", "content":
                             "That was not valid JSON. Reply with ONLY the JSON object."})
            continue

        rec["files_declared"] = patch.get("files")
        rec["explanation"] = (patch.get("explanation") or "")[:400]

        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / name
            shutil.copytree(src, work)
            ok, msgs, touched = apply_patch(work, patch)
            rec["applied"] = ok
            rec["apply_msgs"] = msgs
            rec["files_touched"] = sorted(touched)
            # scope creep: touched files the model did not declare, or outside the task
            declared = set(patch.get("files") or [])
            rec["undeclared_touches"] = sorted(touched - declared)
            rec["out_of_scope"] = sorted(t for t in touched if t not in allowed)

            if ok:
                pt_rc, pt_out = run_pytest(work)
                demo_rc, demo_out = run_demo(work, DEMO[name])
                rec["tests_pass"] = (pt_rc == 0)
                rec["bug_fixed"] = (demo_rc == 0)
                rec["tests_tail"] = pt_out.strip().splitlines()[-1][:200] if pt_out.strip() else ""
                rec["demo_tail"] = demo_out.strip().splitlines()[-1][:200] if demo_out.strip() else ""
                rec["success"] = bool(pt_rc == 0 and demo_rc == 0)
                result["tests_pass"] = rec["tests_pass"]
                result["bug_fixed"] = rec["bug_fixed"]
            else:
                rec["success"] = False

        attempts.append(rec)
        if rec.get("success"):
            break
        # feed the failure back for a second attempt
        messages.append({"role": "assistant", "content": r.text[:2000]})
        messages.append({"role": "user", "content":
                         f"That patch failed: {rec.get('apply_msgs') or rec.get('tests_tail') or 'tests failed'}. "
                         f"Propose a corrected patch as JSON."})

    result["attempts"] = attempts
    result["pass_at_1"] = bool(attempts and attempts[0].get("success"))
    result["pass_at_2"] = bool(any(a.get("success") for a in attempts))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="A", help="label for the report (e.g. A or B)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--model", default="local-fast")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default=",".join(TASK_BRIEF))
    ap.add_argument("--grammar", action="store_true")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    c = Client(args.base_url, timeout=600)
    try:
        c.health()
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: server unreachable at {args.base_url}: {e}")
        return 2

    names = [n.strip() for n in args.fixtures.split(",") if n.strip()]
    print(f"model={args.model}  fixtures={len(names)}  grammar={args.grammar}")

    rows = []
    for n in names:
        print(f"--- {n} ---", flush=True)
        try:
            row = run_fixture(c, args.model, n, args)
        except Exception as e:  # noqa: BLE001
            row = {"fixture": n, "fatal": f"{type(e).__name__}: {e}"}
        rows.append(row)
        s = row.get("pass_at_1"), row.get("pass_at_2")
        print(f"    pass@1={s[0]}  pass@2={s[1]}  "
              f"touched={row.get('attempts', [{}])[-1].get('files_touched')}")
        time.sleep(0.5)

    n = len(rows)
    p1 = sum(1 for r in rows if r.get("pass_at_1"))
    p2 = sum(1 for r in rows if r.get("pass_at_2"))
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "grammar": args.grammar,
        "fixtures": rows,
        "pass_at_1": p1, "pass_at_1_rate": round(100 * p1 / n, 1) if n else 0,
        "pass_at_2": p2, "pass_at_2_rate": round(100 * p2 / n, 1) if n else 0,
        "scope_creep_count": sum(1 for r in rows if any(
            a.get("out_of_scope") for a in r.get("attempts", []))),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.model} bug repair ===")
    print(f"  pass@1 {p1}/{n} = {report['pass_at_1_rate']}%")
    print(f"  pass@2 {p2}/{n} = {report['pass_at_2_rate']}%")
    print(f"  scope creep incidents: {report['scope_creep_count']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
