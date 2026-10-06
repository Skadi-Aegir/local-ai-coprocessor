"""
test_repair_harness.py - offline self-test for the bug-repair harness.

Feeds the KNOWN-GOOD reference fixes (from verify_fixtures) through the real patch
application path and asserts that:
  * the patch applies cleanly
  * the project's pytest still passes
  * the hidden demonstration now passes (the bug is actually gone)

This proves the harness can recognise a correct fix. Without it, a 0% pass rate from
a model would be indistinguishable from a broken harness - which is exactly the class
of bug that already bit the Phase 2 grader once.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))

from repair_suite import apply_patch, run_demo, run_pytest  # noqa: E402
from verify_fixtures import DEMO, FIX  # noqa: E402

PROJ = HERE / "fixtures" / "projects"
FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {label}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def main() -> int:
    print("--- patch application accepts the known-good reference fixes ---")
    for name in sorted(FIX):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / name
            shutil.copytree(PROJ / name, work)

            edits = [{"file": f, "find": old, "replace": new} for f, old, new in FIX[name]]
            patch = {
                "files": [f for f, _, _ in FIX[name]],
                "edits": edits,
                "explanation": "reference fix",
            }
            ok, msgs, touched = apply_patch(work, patch)
            check(f"{name}: reference patch applies", ok, "; ".join(msgs)[:160])

            pt_rc, pt_out = run_pytest(work)
            check(f"{name}: pytest passes after reference fix", pt_rc == 0,
                  "" if pt_rc == 0 else pt_out.strip().splitlines()[-1][:160])

            dm_rc, dm_out = run_demo(work, DEMO[name])
            check(f"{name}: bug is gone after reference fix", dm_rc == 0,
                  "" if dm_rc == 0 else dm_out.strip().splitlines()[-1][:160])

    # --- negative controls: a bad patch must be REJECTED, not silently accepted ---
    print("\n--- negative controls ---")

    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "proj1_pagination"
        shutil.copytree(PROJ / "proj1_pagination", work)

        ok, msgs, _ = apply_patch(work, {"files": ["pagination.py"], "edits": [
            {"file": "pagination.py", "find": "this text does not exist", "replace": "x"}]})
        check("patch with absent find text is rejected", not ok, "; ".join(msgs)[:120])

        ok, msgs, _ = apply_patch(work, {"files": ["pagination.py"], "edits": [
            {"file": "nope.py", "find": "a", "replace": "b"}]})
        check("patch targeting a missing file is rejected", not ok, "; ".join(msgs)[:120])

        ok, msgs, _ = apply_patch(work, {"files": ["../escape.py"], "edits": [
            {"file": "../escape.py", "find": "a", "replace": "b"}]})
        check("patch escaping the project root is rejected", not ok, "; ".join(msgs)[:120])

        ok, msgs, _ = apply_patch(work, {"files": [], "edits": []})
        check("empty patch is rejected", not ok, "; ".join(msgs)[:120])

        # ambiguous find (a string occurring many times) must be refused
        ok, msgs, _ = apply_patch(work, {"files": ["pagination.py"], "edits": [
            {"file": "pagination.py", "find": "e", "replace": "E"}]})
        check("ambiguous find text is rejected", not ok, "; ".join(msgs)[:120])

    # a patch that applies but does NOT fix the bug must not count as success
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "proj1_pagination"
        shutil.copytree(PROJ / "proj1_pagination", work)
        ok, _, _ = apply_patch(work, {"files": ["pagination.py"], "edits": [
            {"file": "pagination.py", "find": "    end = min(start + per_page, total)",
             "replace": "    end = min(start + per_page, total)  # harmless comment"}]})
        check("cosmetic patch applies", ok)
        dm_rc, _ = run_demo(work, DEMO["proj1_pagination"])
        check("cosmetic patch does NOT count as a fix", dm_rc != 0)

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("repair harness verified: accepts correct fixes, rejects bad patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
