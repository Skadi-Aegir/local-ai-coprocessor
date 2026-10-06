"""
verify_fixtures.py - prove each Test A / Test B fixture is actually well-formed.

For every project this asserts BOTH directions:
  1. the shipped tests PASS against the buggy code  (so the bug is genuinely hidden)
  2. a demonstration script FAILS against the buggy code (so the bug is genuine)
  3. the same demonstration PASSES once the reference fix is applied

A fixture that does not satisfy all three is useless for benchmarking: either the bug
is not real, or it is not hidden, and the comparison between models would measure
nothing. This must pass before any model is asked to fix anything.

Usage: python verify_fixtures.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = HERE / "fixtures" / "projects"

FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {label}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def run_pytest(project_dir: Path, timeout: int = 90) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=project_dir, capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def run_script(project_dir: Path, code: str, timeout: int = 60) -> tuple[int, str]:
    """Run a small python snippet inside the project (so imports resolve)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=project_dir,
                                     delete=False, encoding="utf-8") as fh:
        fh.write(code)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], cwd=project_dir,
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    finally:
        os.unlink(path)


# --------------------------------------------------------------------------
# demonstrations: exit 0 == bug NOT present, non-zero == bug present
# --------------------------------------------------------------------------

DEMO = {
    "proj1_pagination": """
from service import ItemService
from storage import Store
svc = ItemService(Store())
out = svc.list_items({"page": 1, "per_page": 10})
names = [i["name"] for i in out["items"]]
assert names[0] == "item-1", f"page 1 must start at item-1, got {names[0]}"
assert len(names) == 10, f"page 1 must return 10 items, got {len(names)}"
p2 = [i["name"] for i in svc.list_items({"page": 2, "per_page": 10})["items"]]
assert p2[0] == "item-11", f"page 2 must start at item-11, got {p2[0]}"
print("PASS")
""",
    "proj2_importer": """
import os, sys, tempfile
from batch import import_paths
from sink import Sink

# Cross-platform descriptor-leak probe. The `resource` module is Unix-only, so
# instead of reading RLIMIT_NOFILE we detect the leak by BEHAVIOUR: a leaked
# descriptor is never reclaimed, so the process's handle count grows monotonically
# with the number of malformed files. We measure that growth directly.

def handle_count():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        # declare the signature explicitly: without argtypes/restype the 64-bit
        # HANDLE is truncated to a C int and the call fails
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.GetCurrentProcess.argtypes = []
        k.GetProcessHandleCount.restype = wintypes.BOOL
        k.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        n = wintypes.DWORD(0)
        if not k.GetProcessHandleCount(k.GetCurrentProcess(), ctypes.byref(n)):
            raise OSError(f"GetProcessHandleCount failed: {ctypes.get_last_error()}")
        return n.value
    return len(os.listdir("/proc/self/fd"))

N = 400
with tempfile.TemporaryDirectory() as d:
    paths = []
    for i in range(N):
        p = os.path.join(d, f"bad{i}.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("[1, 2, 3]")   # array, not object -> raises AFTER open()
        paths.append(p)

    before = handle_count()
    out = os.path.join(d, "out.jsonl")
    with Sink(out) as s:
        summary = import_paths(paths, s)
    after = handle_count()

    assert summary["ok"] == 0, summary["ok"]
    assert len(summary["failed"]) == N, len(summary["failed"])

    growth = after - before
    # A correct implementation opens and closes each file, so handles return to
    # roughly their starting level. A leak grows by ~N.
    assert growth < N * 0.25, f"descriptor leak: handles grew by {growth} over {N} files"
    print(f"PASS (handle growth {growth} over {N} files)")
""",
    "proj3_report": """
from report import monthly_report
rows = [
    {"region": "north", "customer_id": "c1", "amount": 10.00},
    {"region": "south", "customer_id": "c1", "amount": 5.00},
]
out = monthly_report(rows)
assert out["by_customer"]["c1"] == 15.00, f"c1 must total 15.00, got {out['by_customer']['c1']}"
assert out["grand_total"] == 15.00, f"grand total must be 15.00, got {out['grand_total']}"
assert out["by_region"]["north"] == 10.00
assert out["by_region"]["south"] == 5.00
print("PASS")
""",
}

# reference fixes, applied to a COPY so the pristine fixtures stay buggy
FIX = {
    "proj1_pagination": [("pagination.py", "    start = page * per_page",
                          "    start = (page - 1) * per_page")],
    "proj2_importer": [("reader.py",
                        '''    fh = open(path, "r", encoding="utf-8")
    raw = fh.read()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    fh.close()
    return data''',
                        '''    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data''')],
    "proj3_report": [("report.py", "        by_customer[customer] = cents",
                      "        by_customer[customer] = by_customer.get(customer, 0) + cents")],
}


def main() -> int:
    import platform
    print(f"platform: {platform.system()}  python: {sys.version.split()[0]}\n")

    for name in sorted(DEMO):
        src = PROJECTS / name
        print(f"===== {name} =====")
        check(f"{name}: project exists", src.is_dir())

        # 1. shipped tests pass against the buggy code
        rc, out = run_pytest(src)
        check(f"{name}: shipped tests pass on BUGGY code", rc == 0,
              "" if rc == 0 else out.strip().splitlines()[-1][:160])

        # 2. demonstration fails against the buggy code
        rc_bug, out_bug = run_script(src, DEMO[name])
        check(f"{name}: demonstrated bug is REAL (demo fails)",
              rc_bug != 0, f"demo unexpectedly passed")

        # 3. demonstration passes with the reference fix
        with tempfile.TemporaryDirectory() as td:
            copy = Path(td) / name
            shutil.copytree(src, copy)
            for fname, old, new in FIX[name]:
                fp = copy / fname
                text = fp.read_text(encoding="utf-8")
                if old not in text:
                    check(f"{name}: fix target found in {fname}", False, "anchor text missing")
                    continue
                fp.write_text(text.replace(old, new, 1), encoding="utf-8")
            rc_fix, out_fix = run_script(copy, DEMO[name])
            check(f"{name}: reference fix makes demo PASS", rc_fix == 0,
                  "" if rc_fix == 0 else out_fix.strip().splitlines()[-1][:160])
            rc_pt, out_pt = run_pytest(copy)
            check(f"{name}: shipped tests still pass AFTER fix", rc_pt == 0,
                  "" if rc_pt == 0 else out_pt.strip().splitlines()[-1][:160])
        print()

    print("=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("all fixtures verified: bugs are real AND hidden, fixes work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
