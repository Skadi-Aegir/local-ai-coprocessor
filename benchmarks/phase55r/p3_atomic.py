"""
p3_atomic.py - crash-safe JSON persistence for the P55R experiments.

REQUIRED BY P55R-04: "a write method that does not overwrite old results and preserves the last
complete JSON as far as possible when it crashes."

Why this is not simply `path.write_text(...)`:
  `write_text` truncates the target FIRST and then writes. A crash between those two steps leaves a
  truncated, unparseable file - which is exactly how P55R-02's mode B result was lost in spirit:
  the run died before the single write, and had it died DURING the write the file would have been
  corrupt instead of absent. Two mechanisms here:
    1. write to a temp file in the same directory, fsync, then os.replace() - atomic on Windows
       and POSIX, so the target is either the old complete file or the new complete file
    2. before replacing, keep the previous target as `<name>.prev.json`, so even a bug in the new
       payload cannot destroy the last known-good record
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_json(path: str | Path, payload, *, keep_previous: bool = True) -> Path:
    """
    Write `payload` as JSON to `path` atomically, preserving the previous version.

    Returns the path written. Never leaves a partially written target.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    prev = p.with_name(p.stem + ".prev" + p.suffix)

    data = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())

    if keep_previous and p.exists():
        try:
            # copy the CURRENT complete file aside before it is replaced
            prev.write_bytes(p.read_bytes())
        except Exception:  # noqa: BLE001
            pass                      # a failure to keep a backup must not block the real write

    os.replace(tmp, p)                # atomic
    return p
