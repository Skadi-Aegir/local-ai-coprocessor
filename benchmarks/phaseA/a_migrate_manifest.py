"""
a_migrate_manifest.py - re-express an existing index manifest's file hashes in the
line-ending-independent scheme `sha256-lf-v1`, WITHOUT rebuilding the index.

WHY. The Phase A manifest (index 259fae05785b433c) recorded raw-byte SHA256s. They match the
working copy the index was built from and nothing else: 0/48 in a fresh worktree of the same
commit, because core.autocrlf=true checks the files out as CRLF, and every citation there was
reported `stale`. Rebuilding would re-embed every chunk through 8113 - the very operation that
triggers OPEN_ISSUES P1-11. The chunks do not depend on line endings (verified: identical chunk
tables, byte for byte, for LF and CRLF copies of all 48 files), so the index itself is still
correct; only the way its sources are identified has to change.

HOW IT STAYS HONEST
  * Every file is first re-hashed RAW in --source-root (the working copy the index was built from)
    and must equal the recorded hash, and its line count must equal the recorded one. Only then is
    its LF-normalised hash computed, so each new hash is a hash of exactly the bytes that were
    indexed. One mismatch aborts the migration and nothing is written.
  * index_version, the validation block and every other field are left as they were. The old
    hashes are kept under `file_hashes_raw_legacy`, and the migration is recorded with its basis.
  * Each file's status against the manifest's own git_commit is recorded as well.
  * Nothing is written without --write.

Usage:
  python a_migrate_manifest.py --source-root <repo>                 # dry run
  python a_migrate_manifest.py --source-root <repo> --write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from provenance import (HASH_SCHEME, content_hash, file_git_status,  # noqa: E402
                        line_count, raw_hash, tree_blobs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "vector/qa/current/manifest.json"))
    ap.add_argument("--source-root", required=True,
                    help="the working copy the index was BUILT from (its raw bytes are checked)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    path = Path(args.manifest)
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("hash_scheme") == HASH_SCHEME:
        print(f"{path}: already uses {HASH_SCHEME}; nothing to do")
        return 0

    src = Path(args.source_root)
    blobs = tree_blobs(src, m.get("git_commit", ""))
    new_hashes: dict[str, str] = {}
    status: dict[str, str] = {}
    had_crlf: list[str] = []
    problems: list[str] = []
    for rel, recorded in sorted(m["file_hashes"].items()):
        f = src / rel
        if not f.is_file():
            problems.append(f"{rel}: missing in {src}")
            continue
        raw = f.read_bytes()
        if raw_hash(raw) != recorded:
            problems.append(f"{rel}: raw bytes in {src} no longer match the recorded hash")
            continue
        if line_count(raw) != m["file_lines"].get(rel):
            problems.append(f"{rel}: line count {line_count(raw)} != recorded "
                            f"{m['file_lines'].get(rel)}")
            continue
        new_hashes[rel] = content_hash(raw)
        status[rel] = file_git_status(rel, raw, blobs)
        if b"\r\n" in raw:
            had_crlf.append(rel)

    if problems:
        print("ABORTED - the source root does not hold the bytes that were indexed:")
        for p in problems:
            print(f"  {p}")
        return 4

    not_at_commit = sorted(f for f, s in status.items() if s != "clean")
    migrated = dict(m)
    migrated["file_hashes"] = new_hashes
    migrated["hash_scheme"] = HASH_SCHEME
    migrated["file_hashes_raw_legacy"] = m["file_hashes"]
    migrated["file_git_status"] = status
    migrated["corpus_files_not_at_commit"] = not_at_commit
    migrated.setdefault("migrations", []).append({
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "benchmarks/phaseA/a_migrate_manifest.py",
        "change": (f"file_hashes re-expressed as {HASH_SCHEME} (CRLF folded to LF before "
                   f"hashing); the raw hashes are kept in file_hashes_raw_legacy"),
        "basis": (f"each of the {len(new_hashes)} files was re-hashed raw in {src} and found "
                  f"equal to its recorded hash and line count before it was normalised"),
        "files_with_crlf_in_the_indexed_bytes": had_crlf,
        "corpus_files_not_at_git_commit": not_at_commit,
        "index_rebuilt": False,
        "index_version_unchanged": True,
    })

    print(f"{len(new_hashes)} files verified against {src}; "
          f"{len(had_crlf)} had CRLF in the indexed bytes: {had_crlf}")
    print(f"corpus files not at {m.get('git_commit', '?')[:12]}: {not_at_commit or 'none'}")
    if not args.write:
        print("dry run: --write not given, nothing written")
        return 0
    path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
