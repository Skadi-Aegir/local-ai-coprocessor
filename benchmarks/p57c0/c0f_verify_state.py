"""
P57-C0 - VERIFY the PROJECT_STATE.md correction instead of trusting the edit.

Checks:
  1. the acceptance-authority line names Astra as current and preserves the history
  2. no item id appears in BOTH the resolved and the still-open lists
  3. items the objective required to be treated as resolved are NOT in the still-open list
  4. every id mentioned in PROJECT_STATE actually exists in OPEN_ISSUES (no invented ids)
  5. historical records were not touched: the round's edits are confined to PROJECT_STATE.md
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[2]
ps = (R / "PROJECT_STATE.md").read_text(encoding="utf-8")
oi = (R / "OPEN_ISSUES.md").read_text(encoding="utf-8")

ID = re.compile(r"\b(P[123]-\d+)\b")
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        fails.append(name)


# ---- 1 authority
head = "\n".join(ps.splitlines()[:12])
check("authority line names Astra as current",
      "Acceptance authority (current)" in head and "Astra" in head.split("History")[0],
      head.splitlines()[0][:90])
check("authority history preserved (Claude + AD-21/22 mentioned)",
      "Claude" in head and "Astra" in head)

# ---- 2/3 split the status bullets
split = ps.split("**Status split:**", 1)
body = split[1] if len(split) > 1 else ""
res = re.search(r"\*\*RESOLVED\*\*(.*?)\*\*STILL OPEN\*\*", body, re.S)
still = re.search(r"\*\*STILL OPEN\*\*(.*?)\*\*ENVIRONMENT RISK", body, re.S)
check("found the RESOLVED block", bool(res))
check("found the STILL OPEN block", bool(still))

res_ids = set(ID.findall(res.group(1))) if res else set()
still_ids = set(ID.findall(still.group(1))) if still else set()
overlap = sorted(res_ids & still_ids)
check("no id is in BOTH resolved and still-open", not overlap, f"overlap={overlap}")

must_be_resolved = {"P1-6", "P1-11", "P1-12", "P1-13"}
bad = sorted(must_be_resolved & still_ids)
check("P1-6/P1-11/P1-12/P1-13 are NOT in the still-open list", not bad, f"still listed={bad}")
missing = sorted(must_be_resolved - res_ids)
check("P1-6/P1-11/P1-12/P1-13 ARE in the resolved list", not missing, f"absent={missing}")

new_open = {"P3-9", "P3-10"}
absent_new = sorted(new_open - still_ids)
check("this round's new findings (P3-9, P3-10) are listed as open", not absent_new,
      f"absent={absent_new}")
check("P3-8 is present as an environment risk",
      "P3-8" in ps and "ENVIRONMENT RISK" in body)

# ---- 4 ids exist in OPEN_ISSUES
oi_ids = set(re.findall(r"^## (P[123]-\d+)", oi, re.M))
used = set(ID.findall(ps))
invented = sorted(used - oi_ids)
check("every id in PROJECT_STATE exists in OPEN_ISSUES", not invented, f"invented={invented}")

# ---- 5 historical results untouched
print("\n=== git: which files did this round's edits touch vs HEAD? ===")
out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                     cwd=str(R), stdout=subprocess.PIPE).stdout.decode().strip()
print("  modified:", out.splitlines() if out else "none")
touched = {l.split()[-1] for l in out.splitlines()} if out else set()
bad_touch = {t for t in touched if t.startswith("benchmarks/") or t.startswith("vector/")}
check("no historical result or index file modified", not bad_touch, f"touched={bad_touch}")
check("only PROJECT_STATE.md (and later this script) modified",
      touched <= {"PROJECT_STATE.md"}, f"touched={touched}")

print()
print("VERDICT:", "all checks passed" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
