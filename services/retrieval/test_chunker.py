"""
test_chunker.py - asserting self-test for the chunking strategies.

The property that actually matters is semantic completeness: a chunk should be a
whole function / class / heading section, not an arbitrary window. These tests check
that directly, plus the invariants that line numbers are correct and that every
chunk's content really is the source text at those lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chunker import (  # noqa: E402
    TokenCounter, chunk_file, chunk_fixed, chunk_markdown, chunk_python_ast,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


PY_SRC = '''"""Module docstring."""
import os

CONST = 1


def alpha(x):
    """Alpha."""
    return x + 1


class Widget:
    """A widget."""

    def __init__(self, n):
        self.n = n

    def render(self):
        return f"w{self.n}"


def beta(y):
    return y * 2
'''

MD_SRC = """# Title

Intro paragraph.

## Section One

Some text for section one.
More text.

## Section Two

Text for section two.

### Sub Two A

Nested content here.
"""


def main() -> int:
    counter = TokenCounter()

    # ---------------- Python AST
    print("--- Python AST chunking ---")
    pieces = chunk_python_ast(PY_SRC, max_tokens=200, counter=counter)
    symbols = [p[3] for p in pieces]
    print(f"  symbols: {symbols}")
    # NOTE: the module docstring is itself the FIRST top-level statement, so there is
    # no header before it and a "<module-header>" chunk must NOT appear. What must hold
    # is that module-level statements are chunked rather than lost. The original
    # version of this test asserted the wrong thing (and a preamble in a Markdown
    # fixture that starts with a heading - also impossible).
    joined = "\n".join(p[2] for p in pieces)
    check("module docstring is captured", '"""Module docstring."""' in joined)
    check("module-level import is captured", "import os" in joined)
    check("module-level assignment is captured", "CONST = 1" in joined)
    check("finds top-level function alpha", any(s == "alpha()" for s in symbols), str(symbols))
    check("finds class Widget", any(s == "Widget" for s in symbols), str(symbols))
    check("finds top-level function beta", any(s == "beta()" for s in symbols), str(symbols))

    # a file WITH leading content before the first statement must produce a header chunk
    hdr_src = "# a comment banner\n# second line\nimport os\n\n\ndef f():\n    return 1\n"
    hdr = chunk_python_ast(hdr_src, max_tokens=200, counter=counter)
    hdr_syms = [p[3] for p in hdr]
    check("leading comment banner becomes a module header chunk",
          any(s == "<module-header>" for s in hdr_syms), str(hdr_syms))

    # a small class should stay whole
    check("small class is kept as ONE chunk",
          sum(1 for s in symbols if s == "Widget") == 1, str(symbols))
    wid = next(p for p in pieces if p[3] == "Widget")
    check("whole-class chunk contains both methods",
          "__init__" in wid[2] and "render" in wid[2])

    # line numbers must address the real source
    lines = PY_SRC.splitlines()
    ok_lines = True
    for s, e, body, sym in pieces:
        actual = "\n".join(lines[s - 1:e])
        if actual != body:
            ok_lines = False
            print(f"    line mismatch for {sym}: {s}-{e}")
    check("every chunk's line range matches its content", ok_lines)

    # ---------------- oversized class splits into methods
    print("\n--- oversized class splits into methods ---")
    big = "class Big:\n" + "\n".join(
        f"    def m{i}(self):\n        return {i}\n" for i in range(40)
    )
    bp = chunk_python_ast(big, max_tokens=80, counter=counter)
    bsym = [p[3] for p in bp]
    check("oversized class yields method chunks",
          any(s.startswith("Big.m") for s in bsym), str(bsym[:5]))
    check("method chunks still name the class",
          all(s.startswith("Big") for s in bsym if "m" in s))

    # ---------------- unparseable python falls back
    print("\n--- fallback on syntax error ---")
    bad_src = "def broken(:\n    pass\n" * 30
    ch = chunk_file(bad_src, "bad.py", "test", strategy="structured",
                    size_tokens=50, overlap_tokens=0, counter=counter)
    check("unparseable python still yields chunks", len(ch) > 0, str(len(ch)))
    check("fallback is recorded in strategy",
          all("fallback" in c.strategy or c.strategy == "fixed" for c in ch),
          str({c.strategy for c in ch}))

    # ---------------- Markdown headings
    print("\n--- Markdown heading chunking ---")
    mp = chunk_markdown(MD_SRC, max_tokens=100, counter=counter)
    titles = [p[3] for p in mp]
    print(f"  titles: {titles}")
    check("captures heading titles", "Section One" in titles and "Section Two" in titles,
          str(titles))
    check("captures nested heading", "Sub Two A" in titles, str(titles))
    # MD_SRC starts with a heading, so there is genuinely NO preamble to emit.
    # Assert the inverse on a fixture that does have one.
    pre_src = "Loose intro text.\n\n# Title\n\nBody.\n"
    pre_titles = [p[3] for p in chunk_markdown(pre_src, max_tokens=100, counter=counter)]
    check("preamble emitted when text precedes the first heading",
          "<preamble>" in pre_titles, str(pre_titles))
    check("no spurious preamble when the file starts with a heading",
          "<preamble>" not in titles, str(titles))
    ok_md = True
    md_lines = MD_SRC.splitlines()
    for s, e, body, t in mp:
        if "\n".join(md_lines[s - 1:e]) != body:
            ok_md = False
    check("markdown line ranges match content", ok_md)
    check("section heading stays with its body",
          all(t == "<preamble>" or md_lines[s - 1].startswith("#")
              for s, e, body, t in mp),
          "first line of each section must be its heading")

    # ---------------- fixed chunking invariants
    print("\n--- fixed chunking ---")
    fx = chunk_fixed(PY_SRC, size_tokens=40, overlap_tokens=10, counter=counter)
    check("fixed chunking produces chunks", len(fx) > 1, str(len(fx)))
    check("fixed windows respect line numbers",
          all("\n".join(PY_SRC.splitlines()[s - 1:e]) == b for s, e, b in fx))

    # ---------------- strategy comparison on the same file
    print("\n--- structured vs fixed on the same file ---")
    c_struct = chunk_file(PY_SRC, "m.py", "test", strategy="structured", counter=counter)
    c_fixed = chunk_file(PY_SRC, "m.py", "test", strategy="fixed",
                         size_tokens=60, overlap_tokens=0, counter=counter)
    check("structured chunks carry symbols",
          any(c.symbol for c in c_struct), str([c.symbol for c in c_struct]))
    check("fixed chunks carry no symbols", all(not c.symbol for c in c_fixed))
    check("both produce stable chunk ids",
          {c.chunk_id for c in chunk_file(PY_SRC, "m.py", "test", strategy="structured",
                                          counter=counter)} ==
          {c.chunk_id for c in c_struct})

    # ---------------- metadata fields present
    print("\n--- metadata completeness ---")
    sample = c_struct[0]
    required = ["chunk_id", "project", "source_path", "content_type", "language",
                "start_line", "end_line", "content_hash", "strategy"]
    d = sample.to_dict()
    missing = [k for k in required if not d.get(k) and d.get(k) != 0]
    check("required metadata present", not missing, str(missing))
    check("content_type classified as code", sample.content_type == "code",
          sample.content_type)
    md_chunk = chunk_file(MD_SRC, "d.md", "test", strategy="structured", counter=counter)[0]
    check("markdown classified as doc", md_chunk.content_type == "doc", md_chunk.content_type)

    # ---------------- non-code / non-markdown falls through to paragraphs
    # Regression: _chunk_paragraphs used to return 3-tuples while every caller
    # unpacked 4 values, raising "not enough values to unpack" for ANY file that
    # reached the paragraph path. The original tests never exercised it because they
    # only used .py and .md fixtures - it surfaced only when the real corpus gained a
    # .ps1 script.
    print("\n--- paragraph fallback (non-code, non-markdown) ---")
    ps_src = ("# a powershell comment\n"
              "param([int]$Port = 8113)\n\n"
              "function Start-Thing {\n"
              "  Write-Host \"starting on $Port\"\n"
              "}\n\n"
              "Start-Thing\n")
    for suffix in (".ps1", ".json", ".yaml", ".txt"):
        ch = chunk_file(ps_src, f"probe{suffix}", "test", strategy="structured",
                        max_tokens=60, counter=counter)
        check(f"{suffix} produces chunks via the paragraph path", len(ch) > 0, str(len(ch)))
        check(f"{suffix} chunks carry the 4-tuple shape",
              all(isinstance(c.start_line, int) and isinstance(c.content, str)
                  and isinstance(c.symbol, str) for c in ch))
    # and the same shape through chunk_markdown's oversized-section path
    big_md = "# H\n\n" + "\n\n".join(f"paragraph number {i} " + "x" * 40 for i in range(40))
    mdc = chunk_markdown(big_md, max_tokens=60, counter=counter)
    check("oversized markdown section splits into paragraphs", len(mdc) > 1, str(len(mdc)))
    check("split markdown sections keep the heading as their symbol",
          all(sym == "H" for _s, _e, _b, sym in mdc), str({s for *_x, s in mdc}))

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("chunker self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
