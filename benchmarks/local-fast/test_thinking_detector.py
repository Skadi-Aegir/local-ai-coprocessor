"""Offline check for the thinking-mode leak detector."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import thinking_leak  # noqa: E402

CASES = [
    ('{"a": 1}', None),
    ("Paris", None),
    ("", None),
    ("Thinking Process:\n1. analyse", "Thinking Process:"),
    ("<think>reasoning</think>Answer", "<think>"),
    ("[Start thinking]", "[Start thinking]"),
    ("1. point one\n2. point two", None),
    ("```json\n{}\n```", None),
]

fails = 0
for text, expect in CASES:
    got = thinking_leak(text)
    ok = got == expect
    if not ok:
        fails += 1
    print(f"{'OK  ' if ok else 'FAIL'} {text[:36]!r:40} -> {got!r}")

print()
print("all detector cases passed" if not fails else f"FAILED ({fails})")
raise SystemExit(1 if fails else 0)
