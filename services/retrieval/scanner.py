"""
scanner.py - walk a project tree and decide what needs (re)embedding.

Incremental indexing is the point of this module. Re-embedding a whole repository on
every run would be slow and would also churn the index, so the scanner:
  * skips files whose content hash is unchanged since the last index
  * reports deleted files so their chunks can be removed
  * never reads a file into memory just to hash it twice

File hashes are SHA-1 over the raw bytes. Content hashes for chunks are computed
separately in chunker.py, so a whitespace-only edit changes the file hash and forces a
re-chunk, which is correct: line numbers may have moved even if the text looks equal.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Directories never worth indexing.
DEFAULT_EXCLUDES = {
    ".git", ".svn", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".venv", "venv", "env", ".idea", ".vscode",
    "dist", "build", "target", ".next", ".cache", "site-packages",
    ".tox", "htmlcov", ".ruff_cache",
}

# Extensions we index. Anything else is ignored rather than embedded as noise.
TEXT_EXTENSIONS = {
    ".py", ".md", ".markdown", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".ps1", ".sh", ".bash", ".bat", ".cmd",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".java", ".go", ".rs",
    ".sql", ".html", ".css", ".scss",
    ".dockerfile", ".gitignore", ".env.example",
}

MAX_FILE_BYTES = 2_000_000      # skip anything larger; almost certainly data, not source
BINARY_SNIFF_BYTES = 8192


@dataclass
class ScannedFile:
    project: str
    rel_path: str
    abs_path: str
    size: int
    mtime: str
    file_hash: str
    status: str          # new | changed | unchanged | deleted

    def to_dict(self) -> dict:
        return {
            "project": self.project, "rel_path": self.rel_path,
            "abs_path": self.abs_path, "size": self.size, "mtime": self.mtime,
            "file_hash": self.file_hash, "status": self.status,
        }


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def looks_binary(path: Path) -> bool:
    """A NUL byte in the first few KB is a reliable binary signal."""
    try:
        with path.open("rb") as fh:
            head = fh.read(BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in head


def is_indexable(path: Path, excludes: set[str]) -> bool:
    if any(part in excludes for part in path.parts):
        return False
    name = path.name.lower()
    if name in ("dockerfile", "makefile", "cmakelists.txt"):
        return True
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return False
    except OSError:
        return False
    return not looks_binary(path)


def scan_project(
    root: str | Path, project: str, *, known: dict[str, str] | None = None,
    excludes: set[str] | None = None, include: list[str] | None = None,
) -> dict[str, list[ScannedFile]]:
    """
    Walk `root` and classify every indexable file.

    `known` maps rel_path -> file_hash from the previous index. Files absent from it
    are new; files whose hash differs are changed; files in `known` that are no longer
    on disk come back under the "deleted" key.
    """
    root = Path(root).resolve()
    known = known or {}
    excludes = set(excludes or DEFAULT_EXCLUDES)
    out: dict[str, list[ScannedFile]] = {"new": [], "changed": [], "unchanged": [], "deleted": []}
    seen: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        # prune in place so os.walk does not descend into excluded trees
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            if not is_indexable(p, excludes):
                continue
            if include and not any(rel.endswith(s) for s in include):
                continue
            seen.add(rel)
            try:
                st = p.stat()
                fh = hash_file(p)
            except OSError:
                continue
            prev = known.get(rel)
            status = "new" if prev is None else ("unchanged" if prev == fh else "changed")
            out[status].append(ScannedFile(
                project=project, rel_path=rel, abs_path=str(p), size=st.st_size,
                mtime=_iso(st.st_mtime), file_hash=fh, status=status,
            ))

    for rel in sorted(set(known) - seen):
        out["deleted"].append(ScannedFile(
            project=project, rel_path=rel, abs_path="", size=0, mtime="",
            file_hash=known[rel], status="deleted",
        ))
    return out


def read_text(path: str | Path) -> str:
    """Read a text file defensively; the scanner already filtered binaries."""
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return p.read_text(encoding="utf-8", errors="replace")
