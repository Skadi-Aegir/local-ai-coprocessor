"""
B3 follow-up - test whether the measured CER is a BENCHMARK RECONSTRUCTION ARTIFACT.

HYPOTHESIS (mine, before checking): the missing spaces might be my fault. The benchmark joined
PaddleOCR's blocks with a newline, which could have thrown away a horizontal gap the detector found.
That would have made the benchmark, not the model, responsible for the error.

RESULT: **the hypothesis is WRONG. The artifact is exactly 0.000 pp on both tiers.**

Why it is zero: whitespace normalisation collapses a run of whitespace to a single space, so
`a\\nb` and `a b` normalise to the same string. Replacing the newline with a space therefore cannot
change any CER - the two scorings are identical by construction. The newline was never the problem.

WHAT THE ERROR ACTUALLY IS: measured by re-running the model and printing the raw blocks
(`_inspect_blocks.py`, output kept in benchmarks/ocr/results/OCR-03-bench/block_inspection.txt):

    syn_mx_01  block 0: '本地服务local-ocr'   x[15,251] y[22,51]     <- ONE line, ONE block
               block 1: '端口port 8116'        x[11,212] y[64,99]
    syn_en_03  block 0: 'VRAM margin:2048 MiB'                      <- colon-space dropped
               block 1: 'Device: cpu'      (this one KEEPS its space)
               block 2: 'Tier:mobile'                          <- colon-space dropped

So the missing space is INSIDE a single recognised block: the detector found the line correctly and
the RECOGNISER omitted the space. It is not a block-join artifact at all.

The behaviour is real and worth stating: **PP-OCRv5 drops spaces at CJK/Latin boundaries and around
punctuation in this material**, and it is inconsistent - `Device: cpu` keeps its space while
`Tier:mobile` loses its. It costs CER on exactly the categories that mix scripts or punctuation.
The tier CER gap is 0.000 pp either way, so the pre-registered selection outcome is unaffected.

This script re-scores the SAME stored predictions. It does NOT re-run OCR, and it changes no frozen
file.
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
B = R / "benchmarks/ocr/results/OCR-03-bench/bench.json"
OUT = R / "benchmarks/ocr/results/OCR-03-bench/reconstruction_effect.json"

d = json.load(open(B, encoding="utf-8"))


def norm(s):
    return " ".join(s.split())


def lev(a, b):
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


def cer(pred, truth):
    p, t = norm(pred), norm(truth)
    if not t:
        return 0.0 if not p else 1.0
    return lev(p, t) / len(t)


# The stored prediction joins blocks with "\n". Rebuild the same text, but where a line break was
# really a horizontal gap between two blocks on one visual line, use a space instead.
with_space = {}
for tier in ("mobile", "server"):
    for r in d["tiers"][tier]["images"]:
        # a newline in the stored pred marks a block boundary; keep line structure but let the
        # normaliser see an intra-line boundary as a space rather than nothing
        with_space.setdefault(tier, {})[r["id"]] = r["pred"].replace("\n", " ")

rep = {"note": ("Re-scores the SAME stored predictions. No OCR was re-run and no frozen file was "
                "changed."), "tiers": {}}
for tier in ("mobile", "server"):
    rows = d["tiers"][tier]["images"]
    a = [cer(r["pred"], r["truth"]) for r in rows]
    b = [cer(with_space[tier][r["id"]], r["truth"]) for r in rows]
    import statistics
    rep["tiers"][tier] = {
        "as_measured_cer_mean": round(statistics.fmean(a), 6),
        "as_measured_cer_median": round(statistics.median(a), 6),
        "gap_preserving_cer_mean": round(statistics.fmean(b), 6),
        "gap_preserving_cer_median": round(statistics.median(b), 6),
        "artifact_pp": round((statistics.fmean(a) - statistics.fmean(b)) * 100, 3),
        "per_image": [{"id": r["id"], "as_measured": round(x, 6), "gap_preserving": round(y, 6)}
                      for r, x, y in zip(rows, a, b)],
    }

ma = rep["tiers"]["mobile"]["as_measured_cer_median"]
sa = rep["tiers"]["server"]["as_measured_cer_median"]
mb = rep["tiers"]["mobile"]["gap_preserving_cer_median"]
sb = rep["tiers"]["server"]["gap_preserving_cer_median"]
rep["conclusion"] = {
    "as_measured_gap_pp": round(abs(ma - sa) * 100, 3),
    "gap_preserving_gap_pp": round(abs(mb - sb) * 100, 3),
    "selection_unchanged": (
        "the pre-registered rule is applied to the as-measured figures and is NOT recalculated; the "
        "gap-preserving figures are reported alongside for interpretation"),
}
OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

print("CER, as measured (blocks joined with a newline) vs gap-preserving (same blocks, space where "
      "the gap was):\n")
print(f"  {'tier':<8}{'as-meas mean':>14}{'as-meas med':>13}"
      f"{'gap-pres mean':>15}{'gap-pres med':>14}{'artifact':>10}")
for tier in ("mobile", "server"):
    t = rep["tiers"][tier]
    print(f"  {tier:<8}{t['as_measured_cer_mean']:>14.4f}{t['as_measured_cer_median']:>13.4f}"
          f"{t['gap_preserving_cer_mean']:>15.4f}{t['gap_preserving_cer_median']:>14.4f}"
          f"{t['artifact_pp']:>9.3f}pp")
print(f"\n  tier gap as measured  : {rep['conclusion']['as_measured_gap_pp']} pp")
print(f"  tier gap gap-preserving: {rep['conclusion']['gap_preserving_gap_pp']} pp")
print(f"\nwrote {OUT}")
