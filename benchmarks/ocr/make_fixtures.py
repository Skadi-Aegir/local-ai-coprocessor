"""
B2 - render the synthetic half of the frozen OCR fixture set.

Ground truth is EXACT by construction: the same strings that are drawn are written into the manifest,
so there is no transcription step to get wrong. Everything is rendered with fonts that ship with
Windows - no font file is committed.

Categories (at least 4 images each, at least 24 synthetic total):
    chinese        Chinese text
    english        English text
    mixed          Chinese and English on the same image
    code           terminal / code style: dark background, monospace, prompt glyphs
    small          12 px text
    lowcontrast    grey-on-grey, deliberately hard
    ui             normal UI size, light background

The manifest is written by make_fixtures.py and then FROZEN: B3 reads it and never rewrites it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks/ocr/fixtures"
IMAGES = OUT / "images"

FONTS = Path(r"C:\Windows\Fonts")
MONO = FONTS / "consola.ttf"          # Consolas
MONO_B = FONTS / "consolab.ttf"
UI = FONTS / "segoeui.ttf"            # Segoe UI
UI_B = FONTS / "segoeuib.ttf"
CJK = FONTS / "msyh.ttc"              # Microsoft YaHei
CJK_B = FONTS / "msyhbd.ttc"

# (id, category, lines, font, size, fg, bg, pad)
CASES: list[tuple] = [
    # ---------------- chinese (4) ----------------
    ("syn_zh_01", "chinese", ["本地人工智能协处理器"], CJK, 30, (0, 0, 0), (255, 255, 255), 16),
    ("syn_zh_02", "chinese", ["本项目只在本地运行", "不向公网发送任何数据"], CJK, 28, (0, 0, 0), (255, 255, 255), 16),
    ("syn_zh_03", "chinese", ["阶段六：本地文字识别服务", "只使用处理器，不占用显存"], CJK, 26, (20, 20, 20), (250, 250, 250), 18),
    ("syn_zh_04", "chinese", ["显卡余量必须保持在两千零四十八兆字节以上"], CJK_B, 30, (0, 0, 0), (255, 255, 255), 16),
    # ---------------- english (4) ----------------
    ("syn_en_01", "english", ["Local OCR runs on the CPU only."], UI, 30, (0, 0, 0), (255, 255, 255), 16),
    ("syn_en_02", "english", ["The service listens on 127.0.0.1:8116", "and returns text, blocks and confidence."], UI, 26, (0, 0, 0), (255, 255, 255), 16),
    ("syn_en_03", "english", ["VRAM margin: 2048 MiB", "Device: cpu", "Tier: mobile"], MONO, 28, (10, 10, 10), (255, 255, 255), 18),
    ("syn_en_04", "english", ["Screenshots are read top to bottom,", "then left to right."], UI_B, 30, (0, 0, 0), (255, 255, 255), 16),
    # ---------------- mixed (4) ----------------
    ("syn_mx_01", "mixed", ["本地服务 local-ocr", "端口 port 8116"], CJK, 28, (0, 0, 0), (255, 255, 255), 16),
    ("syn_mx_02", "mixed", ["状态 status: ok", "设备 device: cpu"], CJK, 26, (0, 0, 0), (255, 255, 255), 16),
    ("syn_mx_03", "mixed", ["识别结果 confidence 0.982", "耗时 elapsed 0.256 s"], CJK, 26, (0, 0, 0), (255, 255, 255), 16),
    ("syn_mx_04", "mixed", ["测试 test 通过 passed", "余量 margin 已守住 held"], CJK_B, 28, (0, 0, 0), (255, 255, 255), 16),
    # ---------------- code / terminal (4) ----------------
    ("syn_code_01", "code", ["PS D:\\MultiAgent> git log --oneline -1",
                             "b637590 Phase 6 B1: local-ocr environment"], MONO, 24, (220, 220, 220), (12, 12, 12), 16),
    ("syn_code_02", "code", ["$ python -m pip list | findstr paddle",
                             "paddleocr    3.7.0", "paddlepaddle 3.3.1"], MONO, 24, (200, 255, 200), (16, 16, 16), 16),
    ("syn_code_03", "code", ["def add(a, b):", "    return a + b"], MONO, 26, (240, 240, 240), (10, 10, 16), 18),
    ("syn_code_04", "code", ["nvidia-smi --query-gpu=memory.free",
                             "--format=csv,noheader", "5379 MiB"], MONO_B, 24, (255, 230, 180), (18, 12, 12), 16),
    # ---------------- small 12 px (4) ----------------
    ("syn_sm_01", "small", ["small twelve pixel text line one", "and a second line at the same size"], UI, 12, (0, 0, 0), (255, 255, 255), 10),
    ("syn_sm_02", "small", ["本地小字号十二像素测试", "第二行同样大小"], CJK, 12, (0, 0, 0), (255, 255, 255), 10),
    ("syn_sm_03", "small", ["127.0.0.1:8116", "margin 2048 MiB"], MONO, 12, (0, 0, 0), (255, 255, 255), 10),
    ("syn_sm_04", "small", ["port 8116 device cpu tier mobile", "confidence 0.982 elapsed 0.256 s"], UI, 12, (30, 30, 30), (250, 250, 250), 10),
    # ---------------- low contrast (4) ----------------
    ("syn_lc_01", "lowcontrast", ["low contrast grey on grey"], UI, 28, (150, 150, 150), (200, 200, 200), 16),
    ("syn_lc_02", "lowcontrast", ["低对比度测试文本"], CJK, 28, (160, 160, 160), (205, 205, 205), 16),
    ("syn_lc_03", "lowcontrast", ["margin 2048 MiB", "device cpu"], MONO, 24, (140, 140, 150), (195, 195, 205), 16),
    ("syn_lc_04", "lowcontrast", ["barely visible second line"], UI, 26, (170, 170, 170), (210, 210, 210), 16),
    # ---------------- normal UI size (4) ----------------
    ("syn_ui_01", "ui", ["Settings", "Device: cpu", "Model: PP-OCRv5 mobile"], UI, 20, (20, 20, 20), (245, 245, 245), 14),
    ("syn_ui_02", "ui", ["设置 Settings", "端口 Port 8116"], CJK, 20, (20, 20, 20), (245, 245, 245), 14),
    ("syn_ui_03", "ui", ["OCR result", "confidence 0.992", "elapsed 1.191 s"], UI, 20, (20, 20, 20), (250, 250, 250), 14),
    ("syn_ui_04", "ui", ["状态 Status: ok", "模型 Model: server"], CJK, 20, (20, 20, 20), (248, 248, 248), 14),
]

LINE_SPACING = 1.45


def render(case) -> tuple[Path, list[str]]:
    cid, cat, lines, font_path, size, fg, bg, pad = case
    font = ImageFont.truetype(str(font_path), size)
    tmp = Image.new("RGB", (10, 10))
    td = ImageDraw.Draw(tmp)
    widths, heights = [], []
    for ln in lines:
        box = td.textbbox((0, 0), ln, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    line_h = int(max(heights) * LINE_SPACING) if heights else size
    w = max(widths) + pad * 2
    h = line_h * len(lines) + pad * 2
    img = Image.new("RGB", (max(w, 40), max(h, 24)), bg)
    d = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        d.text((pad, y), ln, fill=fg, font=font)
        y += line_h
    path = IMAGES / f"{cid}.png"
    img.save(path)
    return path, list(lines)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    manifest = OUT / "manifest.json"
    if manifest.exists():
        print(f"REFUSING to overwrite the frozen manifest {manifest}")
        return 3
    IMAGES.mkdir(parents=True, exist_ok=True)

    entries = []
    for case in CASES:
        path, lines = render(case)
        entries.append({
            "id": case[0],
            "category": case[1],
            "kind": "synthetic",
            "file": f"images/{path.name}",
            "sha256": sha256_file(path),
            "truth_lines": lines,
            "truth": " ".join(lines),
            "render": {"font": Path(case[3]).name, "size_px": case[4],
                       "fg": list(case[5]), "bg": list(case[6]), "pad": case[7]},
        })
        print(f"  {case[0]:<12} {case[1]:<12} {path.stat().st_size:>7} bytes  "
              f"{len(lines)} line(s)")

    by_cat: dict[str, int] = {}
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1

    man = {
        "set": "ocr_fixtures_v1",
        "frozen": True,
        "frozen_note": ("Frozen from creation. B3 reads this file and never rewrites it. "
                        "Changing any image or truth string means a NEW set under a new name."),
        "synthetic_count": len(entries),
        "category_counts": by_cat,
        "requirements": {"min_per_category": 4, "min_synthetic_total": 24},
        "scoring": {
            "cer": "Levenshtein(pred, truth) / len(truth), on whitespace-normalised strings",
            "normalisation": ("collapse every run of whitespace to a single space, then strip; "
                              "no punctuation equivalence, no case folding"),
            "line_recall": ("fraction of truth lines that appear COMPLETE inside the recognised "
                            "text after the same normalisation"),
        },
        "entries": entries,
    }
    manifest.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsynthetic: {len(entries)} images, categories {by_cat}")
    print(f"manifest sha256: {sha256_file(manifest)}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
