"""
test_leak_fault_injection.py - prove the thinking-leak detector actually FIRES.

A detector that never triggers is decorative. The mock server answers any prompt
containing the sentinel "LEAKTEST" in thinking style, so this drives that through
the real client path and asserts the detector reports it.

Requires the mock server:
  python mock_server.py --port 8199
  python test_leak_fault_injection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import Client, thinking_leak  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199"


def main() -> int:
    c = Client(BASE)

    leaked = c.chat([{"role": "user", "content": "LEAKTEST say hi"}], max_tokens=32)
    marker = thinking_leak(leaked.text)
    print(f"injected reply : {leaked.text!r}")
    print(f"detector       : {marker!r}")
    if marker is None:
        print("FAIL: detector did not fire on an injected thinking block")
        return 1
    print("OK   detector fires on injected thinking block")

    clean = c.chat([{"role": "user", "content": "say hi"}], max_tokens=32)
    marker2 = thinking_leak(clean.text)
    print(f"clean reply    : {clean.text!r}")
    print(f"detector       : {marker2!r}")
    if marker2 is not None:
        print("FAIL: detector reported a leak on a clean reply")
        return 1
    print("OK   detector stays quiet on a clean reply")

    print("\nfault injection passed: the leak check is real, not decorative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
