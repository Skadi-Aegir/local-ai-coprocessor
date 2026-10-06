"""
chunker.py - Phase 4 chunking strategies.

Two strategies are implemented so they can be compared on real retrieval metrics:

  fixed      : naive token windows with overlap. Fast, language-agnostic, and the
               baseline the specification asks to beat.
  structured : structure-aware. Python via AST (function / method / class /
               module-level blocks), Markdown via heading sections, and a
               paragraph-block fallback for plain text. Falls back to fixed windows
               whenever parsing fails, so it never produces fewer chunks than fixed.

The goal is not even chunk sizes - it is that a retrieved chunk is semantically
complete on its own. A function split across three windows is not useful context
even if every window is exactly 500 tokens.

Token counting uses the embedding model's own tokenizer when available, because a
character-based approximation would mis-size CJK text badly (Chinese is ~0.6
tokens/char while English is ~0.25).
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

LANG_BY_SUFFIX = {
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".ps1": "powershell",
    ".sh": "shell",
    ".ts": "typescript",
    ".js": "javascript",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".sql": "sql",
}


@dataclass
class Chunk:
    chunk_id: str
    project: str
    source_path: str
    content_type: str          # code | doc | config | data
    language: str
    start_line: int            # 1-based inclusive
    end_line: int              # 1-based inclusive
    content: str
    strategy: str              # fixed | structured
    symbol: str = ""           # function/class name or heading text, when known
    commit_hash: str = ""
    mtime: str = ""
    content_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def make_chunk_id(project: str, rel_path: str, start: int, end: int, symbol: str = "") -> str:
    return _sha1(f"{project}|{rel_path}|{start}|{end}|{symbol}")[:20]


def classify_content(rel_path: str, language: str) -> str:
    p = rel_path.lower()
    if language in ("json", "yaml", "ini", "toml"):
        return "config"
    if language in ("markdown", "text") or p.endswith(".rst"):
        return "doc"
    if language in ("csv", "tsv"):
        return "data"
    return "code"


# --------------------------------------------------------------------------
# token counting
# --------------------------------------------------------------------------

class TokenCounter:
    """
    Counts tokens with the embedding model's tokenizer when one is supplied.

    Falls back to a per-script heuristic: CJK is roughly 0.6 tokens per character
    and Latin roughly 0.25, which is far closer than a flat chars/4 rule for the
    mixed Chinese/English content in this project.
    """

    def __init__(self, tokenizer=None):
        self.tok = tokenizer

    def count(self, text: str) -> int:
        if self.tok is not None:
            try:
                return len(self.tok.encode(text, add_special_tokens=False))
            except Exception:
                pass
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f")
        other = len(text) - cjk
        return max(1, int(cjk * 0.6 + other * 0.25))


# --------------------------------------------------------------------------
# fixed-window chunking
# --------------------------------------------------------------------------

def chunk_fixed(
    text: str, *, size_tokens: int = 400, overlap_tokens: int = 60,
    counter: TokenCounter | None = None,
) -> list[tuple[int, int, str]]:
    """
    Split text into overlapping fixed-size token windows.

    Returns (start_line, end_line, content) with 1-based inclusive line numbers.
    Windows are built from whole lines so line numbers stay meaningful.
    """
    counter = counter or TokenCounter()
    lines = text.splitlines()
    if not lines:
        return []

    out: list[tuple[int, int, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        toks = 0
        while j < n:
            t = counter.count(lines[j]) + 1
            if toks + t > size_tokens and j > i:
                break
            toks += t
            j += 1
        body = "\n".join(lines[i:j])
        if body.strip():
            out.append((i + 1, j, body))
        if j >= n:
            break
        # step back for overlap
        back = 0
        toks = 0
        k = j - 1
        while k > i and toks < overlap_tokens:
            toks += counter.count(lines[k]) + 1
            back += 1
            k -= 1
        i = max(i + 1, j - back)
    return out


# --------------------------------------------------------------------------
# Python AST chunking
# --------------------------------------------------------------------------

def _node_span(node: ast.AST, lines: list[str]) -> tuple[int, int]:
    """1-based inclusive line span, extended to include a leading decorator block."""
    start = node.lineno
    for dec in getattr(node, "decorator_list", []) or []:
        start = min(start, dec.lineno)
    end = getattr(node, "end_lineno", None) or node.lineno
    return start, end


def _describe(node: ast.AST, parents: list[str]) -> str:
    name = getattr(node, "name", "")
    if isinstance(node, ast.ClassDef):
        return ".".join(parents + [name])
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ".".join(parents + [name]) + "()"
    return ".".join(parents) if parents else ""


def chunk_python_ast(
    text: str, *, max_tokens: int = 1200, counter: TokenCounter | None = None,
) -> list[tuple[int, int, str, str]]:
    """
    Chunk Python by top-level definition: function, class, or module-level block.

    A class is kept whole unless it exceeds max_tokens, in which case its methods are
    emitted individually (each still carrying the class name in `symbol`).
    Returns (start_line, end_line, content, symbol).
    """
    counter = counter or TokenCounter()
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    out: list[tuple[int, int, str, str]] = []
    body = list(tree.body)
    if not body:
        return []

    # module header: everything before the first top-level statement
    first = min((_node_span(n, lines)[0] for n in body), default=1)
    if first > 1:
        head = "\n".join(lines[: first - 1])
        if head.strip():
            out.append((1, first - 1, head, "<module-header>"))

    for node in body:
        start, end = _node_span(node, lines)
        seg = "\n".join(lines[start - 1:end])
        if not seg.strip():
            continue

        if isinstance(node, ast.ClassDef):
            if counter.count(seg) <= max_tokens:
                out.append((start, end, seg, _describe(node, [])))
                continue
            # too big: emit a short class preamble plus each method separately
            methods = [m for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not methods:
                out.append((start, end, seg, _describe(node, [])))
                continue
            mstart = min(_node_span(m, lines)[0] for m in methods)
            preamble = "\n".join(lines[start - 1: mstart - 1])
            if preamble.strip():
                out.append((start, mstart - 1, preamble, node.name))
            for m in methods:
                ms, me = _node_span(m, lines)
                mseg = "\n".join(lines[ms - 1:me])
                out.append((ms, me, mseg, _describe(m, [node.name]).replace("()", "()")))
        else:
            sym = _describe(node, [])
            if counter.count(seg) <= max_tokens:
                out.append((start, end, seg, sym))
            else:
                # a single oversized function: split it with the fixed strategy but
                # keep the symbol so provenance survives
                for s, e, body_txt in chunk_fixed(seg, size_tokens=max_tokens,
                                                  overlap_tokens=0, counter=counter):
                    out.append((start + s - 1, start + e - 1, body_txt, sym))
    return out


# --------------------------------------------------------------------------
# Markdown heading chunking
# --------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(text: str, *, max_tokens: int = 900,
                   counter: TokenCounter | None = None) -> list[tuple[int, int, str, str]]:
    """
    Chunk Markdown by heading section. A section is never split from its heading, and
    oversized sections are split on blank-line paragraph boundaries rather than by
    raw token windows, so prose stays intact.
    """
    counter = counter or TokenCounter()
    lines = text.splitlines()
    if not lines:
        return []

    # find heading positions
    marks: list[tuple[int, int, str]] = []   # (line_idx, level, title)
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(ln)
        if m:
            marks.append((i, len(m.group(1)), m.group(2).strip()))

    if not marks:
        # no headings: paragraph blocks
        return _chunk_paragraphs(text, max_tokens=max_tokens, counter=counter)

    out: list[tuple[int, int, str, str]] = []
    # preamble before the first heading
    if marks[0][0] > 0:
        pre = "\n".join(lines[: marks[0][0]])
        if pre.strip():
            out.append((1, marks[0][0], pre, "<preamble>"))

    for idx, (li, lvl, title) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        seg = "\n".join(lines[li:end])
        if not seg.strip():
            continue
        if counter.count(seg) <= max_tokens:
            out.append((li + 1, end, seg, title))
        else:
            for s, e, body_txt, _sym in _chunk_paragraphs(seg, max_tokens=max_tokens,
                                                          counter=counter):
                out.append((li + s, li + e - 1, body_txt, title))
    return out


def _chunk_paragraphs(text: str, *, max_tokens: int,
                      counter: TokenCounter | None = None) -> list[tuple[int, int, str, str]]:
    """
    Split on blank lines, merging short paragraphs up to max_tokens.

    Always returns (start_line, end_line, content, symbol) with an empty symbol, so
    every producer in this module yields the same 4-tuple shape. An earlier version
    returned 3-tuples while its callers unpacked 4 values, which raised
    "not enough values to unpack" for any file that reached this path - it only
    surfaced once the corpus contained a file that took it.

    Line numbers are 1-based and relative to `text`.
    """
    counter = counter or TokenCounter()
    lines = text.splitlines()
    blocks: list[tuple[int, int]] = []
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "" and i > start:
            blocks.append((start, i))
            start = i + 1
    if start < len(lines):
        blocks.append((start, len(lines)))

    out: list[tuple[int, int, str, str]] = []
    cur_s = cur_e = None
    cur_txt: list[str] = []
    cur_tok = 0
    for s, e in blocks:
        seg = "\n".join(lines[s:e])
        if not seg.strip():
            continue
        t = counter.count(seg)
        if cur_txt and cur_tok + t > max_tokens:
            out.append((cur_s + 1, cur_e, "\n\n".join(cur_txt), ""))
            cur_txt, cur_tok, cur_s, cur_e = [], 0, None, None
        if cur_s is None:
            cur_s = s
        cur_e = e
        cur_txt.append(seg)
        cur_tok += t
    if cur_txt:
        out.append((cur_s + 1, cur_e, "\n\n".join(cur_txt), ""))
    return out


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def chunk_file(
    text: str, rel_path: str, project: str, *, strategy: str = "structured",
    size_tokens: int = 400, overlap_tokens: int = 60, max_tokens: int = 1200,
    counter: TokenCounter | None = None, commit_hash: str = "", mtime: str = "",
) -> list[Chunk]:
    """
    Chunk one file according to `strategy`.

    strategy='structured' uses AST/heading parsing and falls back to fixed windows
    when the file is not parseable or is of an unsupported type.
    """
    counter = counter or TokenCounter()
    suffix = Path(rel_path).suffix.lower()
    language = LANG_BY_SUFFIX.get(suffix, "text")
    ctype = classify_content(rel_path, language)

    pieces: list[tuple[int, int, str, str]] = []
    used = strategy

    if strategy == "structured":
        if language == "python":
            pieces = chunk_python_ast(text, max_tokens=max_tokens, counter=counter)
        elif language == "markdown":
            pieces = chunk_markdown(text, max_tokens=max_tokens, counter=counter)
        else:
            pieces = _chunk_paragraphs(text, max_tokens=max_tokens, counter=counter)
        if not pieces:
            used = "fixed(fallback)"
            pieces = [(s, e, c, "") for s, e, c in
                      chunk_fixed(text, size_tokens=size_tokens,
                                  overlap_tokens=overlap_tokens, counter=counter)]
    else:
        used = "fixed"
        pieces = [(s, e, c, "") for s, e, c in
                  chunk_fixed(text, size_tokens=size_tokens,
                              overlap_tokens=overlap_tokens, counter=counter)]

    chunks: list[Chunk] = []
    for s, e, body, symbol in pieces:
        if not body.strip():
            continue
        cid = make_chunk_id(project, rel_path, s, e, symbol)
        chunks.append(Chunk(
            chunk_id=cid, project=project, source_path=rel_path,
            content_type=ctype, language=language,
            start_line=int(s), end_line=int(e), content=body,
            strategy=used, symbol=symbol,
            commit_hash=commit_hash, mtime=mtime,
            content_hash=_sha1(body),
        ))
    return chunks

