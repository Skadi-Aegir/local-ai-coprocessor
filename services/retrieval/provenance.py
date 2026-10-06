"""
provenance.py - content hashing and Git provenance for indexed source files.

WHY THIS EXISTS. The Phase A manifest hashed the RAW BYTES of each source file. This repository
is checked out with core.autocrlf=true and has no .gitattributes, so the same commit is LF in one
working copy and CRLF in another. Measured on 2026-10-05: the manifest matched 48/48 files in the
main checkout and 0/48 in a fresh worktree of the same commit, which turned every citation into a
false `stale` and made the QA path refuse every answer. A line ending changes no line's text, so
it must not change a file's identity.

  content_hash(raw)   sha256 of the bytes with CRLF folded to LF          scheme `sha256-lf-v1`
  raw_hash(raw)       sha256 of the bytes exactly as on disk              scheme `sha256-raw`
                      (what manifests written before the scheme field recorded)
  git_blob_id(data)   the object id Git gives these bytes as a blob

Only the CRLF pair is folded. A lone CR is content, and Git does not convert it either.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

HASH_SCHEME = "sha256-lf-v1"
LEGACY_HASH_SCHEME = "sha256-raw"


def normalise_newlines(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def content_hash(raw: bytes) -> str:
    return hashlib.sha256(normalise_newlines(raw)).hexdigest()


def raw_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def line_count(raw: bytes) -> int:
    """Line count exactly as the builder and the chunker number lines."""
    return len(raw.decode("utf-8", errors="replace").splitlines())


def match_recorded(recorded: str, raw: bytes, scheme: str | None) -> str | None:
    """
    Decide whether the bytes on disk are the bytes a manifest recorded.

    Returns the basis of the match, or None when the file's content has changed:
      "hash"                         equal under the manifest's own scheme
      "hash_after_lf_normalisation"  legacy raw-byte manifest; the recorded bytes were LF and
                                     this copy differs from them ONLY in line endings
    """
    if scheme == HASH_SCHEME:
        return "hash" if content_hash(raw) == recorded else None
    if raw_hash(raw) == recorded:
        return "hash"
    if content_hash(raw) == recorded:
        return "hash_after_lf_normalisation"
    return None


def git_blob_id(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str | None:
    """Run git in `repo`; None when git is unavailable or the command fails."""
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", errors="replace").strip()


def tree_blobs(repo: Path, commit: str) -> dict[str, str] | None:
    """Map every path in `commit` to its blob id, from one `git ls-tree` call."""
    try:
        r = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", "--full-tree", commit],
                           capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    out: dict[str, str] = {}
    for entry in r.stdout.split(b"\0"):
        if not entry:
            continue
        meta, _, path = entry.partition(b"\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == b"blob":
            out[path.decode("utf-8", errors="replace")] = parts[2].decode()
    return out


def file_git_status(rel: str, raw: bytes, blobs: dict[str, str] | None) -> str:
    """
    Per-file provenance against a commit: "clean", "modified", "untracked" or "unknown".

    A repo-wide dirty flag cannot say WHICH files differ from the commit. Phase A's build was
    flagged dirty, yet the dirt was outside the corpus; this records it per file. A file is
    clean when its LF-normalised bytes (what autocrlf commits) or its raw bytes (a file committed
    with CRLF) are exactly the blob in the commit.
    """
    if blobs is None:
        return "unknown"
    blob = blobs.get(rel)
    if blob is None:
        return "untracked"
    if git_blob_id(normalise_newlines(raw)) == blob or git_blob_id(raw) == blob:
        return "clean"
    return "modified"
