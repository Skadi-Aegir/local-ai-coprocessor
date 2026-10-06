"""
B3 - OCR engine benchmark: PP-OCRv5 mobile vs server, in-process, CPU only.

Reads the FROZEN manifest (never writes it) and applies the scoring rules pre-registered in
benchmarks/ocr/questions.md:
    CER           = Levenshtein(pred, truth) / len(truth)
    normalisation = collapse whitespace runs to one space, then strip. Nothing else.
    line recall   = fraction of truth lines appearing COMPLETE in the normalised prediction

Records per image and per category, plus load time, RSS, minimum available commit memory, CPU thread
setting and nvidia-smi before/during/after. The OCR process must not occupy VRAM; the numbers are
reported as they come out, whatever they show.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "benchmarks/ocr/fixtures"
MODELS = REPO / "models/ocr"
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS / "_bench_cache")
# oneDNN off: required on this build (see docs/models/local-ocr.md section 3)
os.environ["FLAGS_use_mkldnn"] = "0"

import psutil  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.split())


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    p, t = norm(pred), norm(truth)
    if not t:
        return 0.0 if not p else 1.0
    return lev(p, t) / len(t)


def line_recall(pred: str, truth_lines: list[str]) -> float:
    p = norm(pred)
    if not truth_lines:
        return 1.0
    hit = sum(1 for ln in truth_lines if norm(ln) and norm(ln) in p)
    return hit / len(truth_lines)


def nvidia_free() -> int | None:
    import subprocess
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                            "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, timeout=30)
        u, f = o.stdout.decode().strip().split(",")
        return int(f)
    except Exception:  # noqa: BLE001
        return None


def commit_free_mb() -> int:
    return int(psutil.virtual_memory().available / 1024 / 1024)


def extract(res) -> tuple[str, list[float], list[list]]:
    texts, scores, boxes = [], [], []
    for r in res:
        j = r.json if hasattr(r, "json") else r
        if callable(j):
            j = j()
        d = j.get("res", j) if isinstance(j, dict) else {}
        texts.extend(d.get("rec_texts") or [])
        scores.extend(d.get("rec_scores") or [])
        for b in (d.get("rec_polys") or d.get("dt_polys") or d.get("rec_boxes") or []):
            try:
                boxes.append([[float(x), float(y)] for x, y in b])
            except Exception:  # noqa: BLE001
                pass
    return "\n".join(texts), [float(s) for s in scores], boxes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out_p = Path(args.out)
    if out_p.exists():
        print(f"REFUSING to overwrite {out_p}")
        return 3
    out_p.parent.mkdir(parents=True, exist_ok=True)

    man = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    man_sha = hashlib.sha256((FIX / "manifest.json").read_bytes()).hexdigest()
    expected = "7ebb5cc4f3b3c7ab5667ca20130f7eb53f9e757f1d184b697fc8200f29d3ffa6"
    if man_sha != expected:
        print(f"STOP: frozen manifest hash changed. got {man_sha}, expected {expected}")
        return 4

    n_cpu = os.cpu_count()
    try:
        import paddle
        paddle.set_num_threads(args.threads)
    except Exception:  # noqa: BLE001
        pass
    proc = psutil.Process()

    rep: dict = {
        "manifest_sha256": man_sha, "n_images": len(man["entries"]),
        "device": "cpu", "threads_requested": args.threads, "os_cpu_count": n_cpu,
        "python": platform.python_version(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
        "onednn": "disabled (FLAGS_use_mkldnn=0)",
        "scoring": man["scoring"],
        "nvidia_free_before_mb": nvidia_free(),
        "commit_free_before_mb": commit_free_mb(),
        "tiers": {},
    }
    print(f"frozen manifest OK ({man_sha[:16]}), {len(man['entries'])} images")
    print(f"threads={args.threads} cpu_count={n_cpu} "
          f"nvidia_free_before={rep['nvidia_free_before_mb']} MiB "
          f"commit_free_before={rep['commit_free_before_mb']} MiB")

    vram_samples: list[int] = []
    commit_samples: list[int] = []

    for tier in ("mobile", "server"):
        print(f"\n===== tier: {tier} =====")
        rss_idle = proc.memory_info().rss / 1024 / 1024
        t0 = time.time()
        ocr = PaddleOCR(
            text_detection_model_name=f"PP-OCRv5_{tier}_det",
            text_recognition_model_name=f"PP-OCRv5_{tier}_rec",
            text_detection_model_dir=str(MODELS / f"PP-OCRv5_{tier}_det"),
            text_recognition_model_dir=str(MODELS / f"PP-OCRv5_{tier}_rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
        load_s = round(time.time() - t0, 2)
        rss_loaded = proc.memory_info().rss / 1024 / 1024
        print(f"  load {load_s}s   rss idle {rss_idle:.0f} MB -> loaded {rss_loaded:.0f} MB")

        rows = []
        peak_rss = rss_loaded
        for e in man["entries"]:
            img = FIX / e["file"]
            t0 = time.time()
            try:
                res = ocr.predict(str(img))
                pred, scores, boxes = extract(res)
                err = None
            except Exception as ex:  # noqa: BLE001
                pred, scores, boxes, err = "", [], [], f"{type(ex).__name__}: {ex}"
            dt = round(time.time() - t0, 3)
            peak_rss = max(peak_rss, proc.memory_info().rss / 1024 / 1024)
            v = nvidia_free()
            if v:
                vram_samples.append(v)
            commit_samples.append(commit_free_mb())
            rows.append({
                "id": e["id"], "category": e["category"], "kind": e["kind"],
                "file": e["file"], "seconds": dt,
                "cer": round(cer(pred, e["truth"]), 6),
                "line_recall": round(line_recall(pred, e["truth_lines"]), 4),
                "n_blocks": len(scores),
                "mean_conf": round(statistics.fmean(scores), 4) if scores else None,
                "pred": pred[:2000], "truth": e["truth"][:2000],
                "error": err,
            })
            print(f"    {e['id']:<22} {e['category']:<14} cer={rows[-1]['cer']:.4f} "
                  f"recall={rows[-1]['line_recall']:.2f} {dt:>6.3f}s")

        def stats(sel):
            cers = [r["cer"] for r in sel]
            lats = [r["seconds"] for r in sel]
            recs = [r["line_recall"] for r in sel]
            return {
                "n": len(sel),
                "cer_median": round(statistics.median(cers), 6) if cers else None,
                "cer_mean": round(statistics.fmean(cers), 6) if cers else None,
                "line_recall_mean": round(statistics.fmean(recs), 4) if recs else None,
                "seconds_p50": round(statistics.median(lats), 4) if lats else None,
                "seconds_p95": (round(sorted(lats)[min(len(lats) - 1,
                                                        int(round(0.95 * (len(lats) - 1))))], 4)
                                if lats else None),
                "errors": sum(1 for r in sel if r["error"]),
            }

        cats = {}
        for c in sorted({r["category"] for r in rows}):
            cats[c] = stats([r for r in rows if r["category"] == c])
        rep["tiers"][tier] = {
            "load_s": load_s, "rss_idle_mb": round(rss_idle), "rss_after_load_mb": round(rss_loaded),
            "rss_peak_mb": round(peak_rss),
            "overall": stats(rows), "per_category": cats, "images": rows,
        }
        print(f"  overall: median CER {rep['tiers'][tier]['overall']['cer_median']:.4f}  "
              f"mean {rep['tiers'][tier]['overall']['cer_mean']:.4f}  "
              f"p50 {rep['tiers'][tier]['overall']['seconds_p50']}s  "
              f"p95 {rep['tiers'][tier]['overall']['seconds_p95']}s  rss peak {peak_rss:.0f} MB")
        del ocr

    rep["nvidia_free_after_mb"] = nvidia_free()
    rep["commit_free_min_mb"] = min(commit_samples) if commit_samples else None
    rep["nvidia_free_min_during_mb"] = min(vram_samples) if vram_samples else None
    rep["nvidia_free_samples"] = len(vram_samples)

    # ---- apply the PRE-REGISTERED selection rule
    mo, se = rep["tiers"]["mobile"]["overall"], rep["tiers"]["server"]["overall"]
    d = abs(mo["cer_median"] - se["cer_median"])
    if d >= 0.01:
        winner = "mobile" if mo["cer_median"] < se["cer_median"] else "server"
        why = f"lower overall CER by {d*100:.2f} percentage points"
    else:
        winner = "mobile" if (mo["seconds_p95"] or 0) < (se["seconds_p95"] or 0) else "server"
        why = (f"CER within 1 pp ({d*100:.2f} pp), so the lower p95 latency decides "
               f"({mo['seconds_p95']}s vs {se['seconds_p95']}s)")
    rep["selection"] = {
        "rule": ("lower overall CER wins; if within 1 percentage point, lower p95 latency wins"),
        "cer_gap_pp": round(d * 100, 3), "winner": winner, "why": why,
    }
    rep["thresholds"] = {
        "clear_text_median_cer_max_pct": 2.0,
        "categories_over_10pct_must_be_analysed": [],
    }
    for tier, t in rep["tiers"].items():
        for c, s in t["per_category"].items():
            if c in ("english", "mixed", "ui") and s["cer_median"] is not None \
                    and s["cer_median"] > 0.02:
                rep["thresholds"].setdefault("clear_text_failures", []).append(
                    {"tier": tier, "category": c, "cer_median": s["cer_median"]})
            if s["cer_median"] is not None and s["cer_median"] > 0.10:
                rep["thresholds"]["categories_over_10pct_must_be_analysed"].append(
                    {"tier": tier, "category": c, "cer_median": s["cer_median"]})

    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWINNER by the pre-registered rule: {winner} -- {why}")
    print(f"nvidia free: before {rep['nvidia_free_before_mb']} min-during "
          f"{rep['nvidia_free_min_during_mb']} after {rep['nvidia_free_after_mb']} MiB")
    print(f"commit free min: {rep['commit_free_min_mb']} MB")
    print(f"wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
