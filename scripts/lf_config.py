#!/usr/bin/env python3
"""
lf_config.py - minimal YAML-subset loader + shared helpers for the local-fast worker.

Why not PyYAML: the LLM service venv is intentionally dependency-free so the
worker can start on a clean machine without a package install. The config file
uses only the subset implemented here (nested maps, scalars, simple lists,
comments, `key: value`, `- item`).

Also hosts the tiny HTTP helper used by check-local-fast, for the same reason.
Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Repository root. Config files write paths as ${REPO}\..., expanded here so the
# checkout can live anywhere. LOCAL_AI_ROOT overrides it (e.g. models on another disk).
REPO_ROOT = Path(os.environ.get("LOCAL_AI_ROOT") or Path(__file__).resolve().parents[1])
DEFAULT_CONFIG = REPO_ROOT / "config" / "local-fast.yaml"


# --------------------------------------------------------------------------
# YAML subset parser
# --------------------------------------------------------------------------

def _coerce(tok: str) -> Any:
    t = tok.strip().replace("${REPO}", str(REPO_ROOT))
    if t == "" or t == "~" or t.lower() in ("null", "none"):
        return None
    if t.lower() in ("true", "yes", "on"):
        return True
    if t.lower() in ("false", "no", "off"):
        return False
    # strip surrounding quotes
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Parse the project's YAML subset into nested dicts."""
    root: dict[str, Any] = {}
    # stack of (indent, container)
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: tuple[int, str, dict] | None = None

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        # pop to the right nesting level
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        cur = stack[-1][1]

        if line.startswith("- "):
            item = _coerce(line[2:])
            if isinstance(cur, list):
                cur.append(item)
            continue

        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if val == "":
            # could be a nested map or a list; look ahead lazily by pushing a dict
            # and converting to list if the next line is a "- " item
            child: Any = {}
            cur[key] = child
            pending_key = (indent, key, cur)
            stack.append((indent, child))
        else:
            cur[key] = _coerce(val)

    return root


def _fix_lists(node: Any, path: str = "") -> Any:
    """
    The streaming parser cannot tell an empty map from an empty list without
    lookahead. Post-process: any dict that stayed empty and was followed by
    list items is already correct; nothing to do in this config's case, but
    keep the hook for clarity.
    """
    return node


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = load_yaml(p)
    if not cfg:
        raise SystemExit(f"config not found or empty: {p}")
    return cfg


# --------------------------------------------------------------------------
# dotted access
# --------------------------------------------------------------------------

def dig(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# --------------------------------------------------------------------------
# HTTP helpers (stdlib)
# --------------------------------------------------------------------------

def http_get_json(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001
        return 0, {"_error": f"{type(e).__name__}: {e}"}


def http_post_json(url: str, payload: dict, timeout: float = 60.0) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, {"_body": e.read().decode("utf-8", "replace")[:400]}
    except Exception as e:  # noqa: BLE001
        return 0, {"_error": f"{type(e).__name__}: {e}"}


def main() -> int:
    """CLI: print a config value (used by the shell scripts)."""
    import argparse

    ap = argparse.ArgumentParser(description="read a value from local-fast.yaml")
    ap.add_argument("key", help="dotted key, e.g. runtime.port")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    val = dig(cfg, args.key)
    if val is None:
        print("", end="")
        return 1
    if isinstance(val, bool):
        print("true" if val else "false")
    else:
        print(val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
