"""
B2 - capture the REAL-content half of the frozen OCR fixture set.

Content is REAL project material, gathered by actually running the commands or reading the files -
not invented text. It is then rendered to an image with PIL so that the ground truth is EXACT rather
than hand-transcribed: the same strings that are drawn are written into the manifest. A hand
transcription step would put a human error inside the ground truth, which is the one place an OCR
benchmark cannot tolerate one.

Content is read and rendered in Python, NOT through the PowerShell console: the console re-encodes
non-ASCII and would have put mojibake into the images (observed while preparing this: an em dash came
back as three replacement characters through the pipeline, while the file itself is valid UTF-8).

Every capture is checked for personal information before it is accepted: no user name, no email
address, no drive path containing a user profile, no browser or chat content. The check is
programmatic and its result is recorded in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/ocr/fixtures"
IMAGES = OUT / "images"
CAP = OUT / "capture"
FONT = Path(r"C:\Windows\Fonts\consola.ttf")
SIZE = 16
BG = (252, 252, 252)
FG = (28, 28, 32)
PAD = 14
WRAP = 74


def run(args: list[str]) -> list[str]:
    r = subprocess.run(args, cwd=str(REPO), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=120)
    return r.stdout.decode("utf-8", errors="replace").rstrip("\n").splitlines()


def read_lines(rel: str, skip: int, count: int) -> list[str]:
    t = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    return t.splitlines()[skip:skip + count]


def personal_info(lines: list[str]) -> list[str]:
    """Any hit means the capture is rejected. Deliberately broad."""
    user = Path.home().name
    pats = {
        "user_name": re.escape(user),
        "users_path": r"(?i)[A-Z]:\\Users\\",
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "openai_or_anthropic_key": r"(?i)sk-[A-Za-z0-9]{10,}",
        "browser": r"(?i)(chrome|edge|firefox|bookmark)",
        "chat": r"(?i)(wechat|telegram|discord|slack)",
    }
    hits = []
    for name, pat in pats.items():
        for i, ln in enumerate(lines, 1):
            m = re.search(pat, ln)
            if m:
                hits.append({"pattern": name, "line": i, "matched": m.group(0)[:40]})
    return hits


def render(cid: str, lines: list[str], title: str) -> tuple[Path, list[str]]:
    """Wrap deterministically so the ground truth is exactly what is drawn."""
    wrapped: list[str] = []
    for ln in lines:
        if not ln.strip():
            wrapped.append("")
            continue
        for part in textwrap.wrap(ln, width=WRAP,
                                  subsequent_indent="    ",
                                  break_long_words=True, break_on_hyphens=False) or [""]:
            wrapped.append(part)
    wrapped = [title, ""] + wrapped

    font = ImageFont.truetype(str(FONT), SIZE)
    tmp = Image.new("RGB", (10, 10))
    td = ImageDraw.Draw(tmp)
    line_h = int(SIZE * 1.5)
    widths = [td.textbbox((0, 0), ln, font=font)[2] for ln in wrapped] or [10]
    w = max(widths) + PAD * 2
    h = line_h * len(wrapped) + PAD * 2
    img = Image.new("RGB", (max(w, 60), max(h, 30)), BG)
    d = ImageDraw.Draw(img)
    y = PAD
    for i, ln in enumerate(wrapped):
        d.text((PAD, y), ln, fill=(90, 90, 100) if i == 0 else FG, font=font)
        y += line_h
    p = IMAGES / f"{cid}.png"
    img.save(p)
    return p, wrapped


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


CAPTURES = [
    ("real_nvidia_smi", "real/terminal", "nvidia-smi memory query",
     lambda: [l for l in run(["nvidia-smi", "--query-gpu=name,memory.used,memory.free",
                              "--format=csv"])]),
    ("real_git_log", "real/terminal", "git log --oneline (project history)",
     lambda: [l for l in run(["git", "log", "--format=%h %s", "-8"])]),
    ("real_pip_paddle", "real/terminal", "pip list: OCR packages",
     lambda: [l for l in run([str(REPO / "services/ocr/venv/Scripts/python.exe"),
                              "-m", "pip", "list"]) if "paddle" in l.lower()]),
    ("real_server_py", "real/editor", "services/retrieval/server.py",
     lambda: read_lines("services/retrieval/server.py", 105, 16)),
    ("real_local_fast_yaml", "real/editor", "config/local-fast.yaml",
     lambda: read_lines("config/local-fast.yaml", 30, 16)),
    ("real_local_ocr_md", "real/editor", "docs/models/local-ocr.md",
     lambda: read_lines("docs/models/local-ocr.md", 55, 14)),
]


def main() -> int:
    manifest = OUT / "manifest.json"
    if not manifest.is_file():
        print("ERROR: run make_fixtures.py first (the frozen manifest must exist)")
        return 2
    man = json.loads(manifest.read_text(encoding="utf-8"))
    if any(e.get("kind") == "real" for e in man["entries"]):
        print("REFUSING: the manifest already holds real captures; it is frozen")
        return 3

    IMAGES.mkdir(parents=True, exist_ok=True)
    CAP.mkdir(parents=True, exist_ok=True)

    accepted, rejected = [], []
    for cid, cat, title, getter in CAPTURES:
        try:
            lines = getter()
        except Exception as e:  # noqa: BLE001
            rejected.append({"id": cid, "error": f"{type(e).__name__}: {e}"})
            print(f"  {cid}: FAILED {type(e).__name__}: {e}")
            continue
        if not lines:
            rejected.append({"id": cid, "error": "no content"})
            print(f"  {cid}: SKIPPED (no content)")
            continue
        hits = personal_info(lines)
        (CAP / f"{cid}.txt").write_text("\n".join(lines), encoding="utf-8")
        if hits:
            rejected.append({"id": cid, "personal_info": hits})
            print(f"  {cid}: REJECTED -- personal information found: {hits}")
            continue
        path, drawn = render(cid, lines, title)
        accepted.append({
            "id": cid, "category": cat, "kind": "real",
            "file": f"images/{path.name}", "sha256": sha256_file(path),
            "truth_lines": drawn, "truth": " ".join(drawn),
            "source": {"title": title, "capture": f"capture/{cid}.txt",
                       "personal_info_check": "passed -- no user name, email, profile path, "
                                              "browser or chat content"},
        })
        print(f"  {cid:<22} {len(drawn):>3} lines  {path.stat().st_size:>7} bytes  "
              f"PI check passed")

    man["entries"].extend(accepted)
    man["real_count"] = len(accepted)
    man["real_rejected"] = rejected
    man["requirements"]["min_real"] = 6
    man["requirements"]["met"] = (len(accepted) >= 6
                                  and man["synthetic_count"] >= 24
                                  and all(v >= 4 for v in man["category_counts"].values()))
    man["total_count"] = len(man["entries"])
    cats: dict[str, int] = {}
    for e in man["entries"]:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    man["all_category_counts"] = cats

    manifest.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreal captures accepted: {len(accepted)}  rejected: {len(rejected)}")
    print(f"total images: {man['total_count']}  categories {cats}")
    print(f"requirements met: {man['requirements']['met']}")
    print(f"FROZEN manifest sha256: {sha256_file(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
