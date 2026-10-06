"""
B1 smoke - prove the CPU OCR stack actually runs on the staged models, and that it can be pointed
at <repo>\\models\\ocr\\ explicitly.

The three preprocessing stages the task excludes are turned OFF:
  use_doc_orientation_classify=False   (no document rotation classification)
  use_doc_unwarping=False              (no page dewarping)
  use_textline_orientation=False       (no text-line rotation classification)
These are for photographed or scanned pages. Screenshots are already upright, and the task explicitly
excludes table recognition and PDF text-layer work.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models/ocr"
EVID = REPO / "benchmarks/ocr/results/OCR-01-env"
# keep any incidental download inside models/ocr, never on C:
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS / "_rt_cache")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402


def make_probe(path: Path) -> None:
    """A tiny high-contrast English line, rendered with a system font (no font file committed)."""
    img = Image.new("RGB", (760, 90), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 32)
    d.text((14, 26), "local-ocr smoke test 8116", fill="black", font=font)
    img.save(path)


def main() -> int:
    out = EVID / "b1_smoke.json"
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    probe = EVID / "b1_probe.png"
    make_probe(probe)

    rep: dict = {"probe": str(probe), "models": {}}
    for tier in ("mobile", "server"):
        kwargs = dict(
            text_detection_model_name=f"PP-OCRv5_{tier}_det",
            text_recognition_model_name=f"PP-OCRv5_{tier}_rec",
            text_detection_model_dir=str(MODELS / f"PP-OCRv5_{tier}_det"),
            text_recognition_model_dir=str(MODELS / f"PP-OCRv5_{tier}_rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            # oneDNN MUST be off on this build. Measured with the default (oneDNN on):
            #   NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute not
            #   support [pir::ArrayAttribute<pir::DoubleAttribute>]
            #   (onednn_instruction.cc:118)
            # Both tiers loaded fine and failed only at predict, so this is a oneDNN kernel/IR
            # incompatibility in paddlepaddle 3.3.1, not a bad model or a bad download. With oneDNN
            # disabled the same models return the expected text at confidence 0.982.
            # FLAGS_use_mkldnn=0 is equivalent; the API flag is used because it is visible here.
            enable_mkldnn=False,
        )
        print(f"\n--- {tier}: loading")
        t0 = time.time()
        try:
            ocr = PaddleOCR(**kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"    LOAD FAILED: {type(e).__name__}: {e}")
            rep["models"][tier] = {"load_error": f"{type(e).__name__}: {e}"}
            continue
        load_s = round(time.time() - t0, 2)
        print(f"    loaded in {load_s}s")

        t0 = time.time()
        try:
            res = ocr.predict(str(probe))
        except Exception as e:  # noqa: BLE001
            print(f"    PREDICT FAILED: {type(e).__name__}: {e}")
            rep["models"][tier] = {"load_s": load_s, "predict_error": f"{type(e).__name__}: {e}"}
            continue
        infer_s = round(time.time() - t0, 3)

        texts, scores = [], []
        for r in res:
            j = r.json if hasattr(r, "json") else r
            if callable(j):
                j = j()
            d = j.get("res", j) if isinstance(j, dict) else {}
            texts.extend(d.get("rec_texts") or [])
            scores.extend(d.get("rec_scores") or [])
        joined = " ".join(texts)
        print(f"    {infer_s}s  texts={texts}  scores={[round(s,3) for s in scores]}")
        rep["models"][tier] = {"load_s": load_s, "predict_s": infer_s,
                               "rec_texts": texts, "rec_scores": scores,
                               "joined": joined,
                               "matched_truth": "local-ocr smoke test 8116" in joined}

    rep["preprocessing_disabled"] = {"doc_orientation": False, "unwarping": False,
                                     "textline_orientation": False}
    rep["device"] = "cpu"
    rep["cache_home"] = str(MODELS / "_rt_cache")

    # remove anything the run cached under models/ocr
    rt = MODELS / "_rt_cache"
    if rt.exists():
        import shutil
        shutil.rmtree(rt, ignore_errors=True)
    rep["rt_cache_removed"] = not rt.exists()

    home_px = Path.home() / ".paddlex"
    rep["c_user_paddlex_files"] = (sum(1 for f in home_px.rglob("*") if f.is_file())
                                   if home_px.exists() else 0)

    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nrt cache removed: {rep['rt_cache_removed']}   C: .paddlex files: {rep['c_user_paddlex_files']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
