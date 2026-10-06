# OCR benchmark results — PP-OCRv5 mobile vs server (CPU)

**Phase 6, step B3.** Rules, sample set and thresholds were pre-registered in
`benchmarks/ocr/questions.md` **before** this ran. Nothing was adjusted afterwards.

**Attribution:** this document was written by the construction side (DSH), which also authored the
sample set and the pre-registered rules. That is the same independence weakness recorded for
`heldout_v1`: the fixer wrote the questions. The numbers below are the construction side's own
measurements of its own set, and the mechanical parts were re-run rather than trusted.

| item | value |
|---|---|
| sample set | `ocr_fixtures_v1`, **34 images** (28 synthetic + 6 real), 9 categories |
| frozen manifest SHA256 | `7ebb5cc4f3b3c7ab5667ca20130f7eb53f9e757f1d184b697fc8200f29d3ffa6` (verified unchanged after measuring) |
| device | **CPU**, `paddlepaddle 3.3.1` CPU wheel, `is_compiled_with_cuda() == False` |
| oneDNN | **disabled** (`FLAGS_use_mkldnn=0`); required on this build, see `docs/models/local-ocr.md` §3 |
| threads | 8 requested, 24 logical cores available |
| records | `benchmarks/ocr/results/OCR-03-bench/bench.json` |

## Overall

| tier | median CER | mean CER | line recall | p50 | p95 | load | RSS idle → peak |
|---|---:|---:|---:|---:|---:|---:|---|
| **mobile** | **0.0000** | **0.0104** | 0.799 | **0.404 s** | **6.294 s** | 0.48 s | 289 → 450 MB |
| server | 0.0000 | 0.0195 | 0.746 | 0.868 s | 12.484 s | 0.61 s | 450 → 650 MB |

Errors: **0** for both tiers.

**The selection rule fired on its latency branch, not its accuracy branch.** Both tiers have a median
CER of **exactly 0.0000**, so the gap is **0.00 percentage points** — within the 1 pp threshold that
was pre-registered as "CER too close to call, let p95 latency decide". Mobile has the lower p95
(**6.294 s vs 12.484 s**), so:

> **Pre-registered winner: `mobile`.**

Mobile also wins on mean CER (0.0104 vs 0.0195), line recall (0.799 vs 0.746), p50, load time and
peak RSS. **The two tiers are not equivalent in cost**: server's p95 is about twice mobile's and its
peak RSS is 200 MB higher, for no accuracy gain on this set.

## Per category

| category | n | mobile CER med / mean | mobile recall | server CER med / mean | server recall |
|---|---:|---|---:|---|---:|
| chinese | 4 | 0.0000 / 0.0000 | 1.000 | 0.0000 / 0.0125 | 0.750 |
| english | 4 | 0.0000 / 0.0109 | 0.833 | 0.0000 / 0.0000 | 1.000 |
| mixed | 4 | **0.0317** / 0.0344 | 0.500 | 0.0000 / 0.0179 | 0.875 |
| code | 4 | 0.0139 / 0.0162 | 0.667 | 0.0324 / 0.0256 | 0.667 |
| small (12 px) | 4 | 0.0000 / 0.0000 | 1.000 | 0.0000 / 0.0000 | 1.000 |
| lowcontrast | 4 | 0.0000 / 0.0000 | 1.000 | 0.0000 / 0.0000 | 1.000 |
| ui | 4 | 0.0000 / 0.0083 | 0.875 | **0.0533** / 0.0658 | 0.333 |
| real/terminal | 3 | 0.0050 / 0.0044 | 0.708 | 0.0242 / 0.0223 | 0.448 |
| real/editor | 3 | 0.0165 / 0.0207 | 0.515 | 0.0397 / 0.0361 | 0.504 |

**12 px text and low contrast both scored a perfect median CER of 0.0000 with full line recall on
both tiers** — the two categories expected to be hardest were the easiest, because the material is
cleanly rendered rather than photographed. That is a property of this set, not a general claim.

## Pre-registered thresholds — honest result

**Threshold 1 — "median CER <= 2 % on synthetic clear text at normal size and high contrast"** (the
`english`, `mixed`, `ui` categories). **Two failures, one per tier:**

| tier | category | median CER | verdict |
|---|---|---:|---|
| mobile | `mixed` | 3.17 % | **FAIL** |
| server | `ui` | 5.33 % | **FAIL** |
| mobile | `english` / `ui` | 0.00 % | pass |
| server | `english` / `mixed` | 0.00 % | pass |

**The failures are reported as failures. No sample, truth string, rule or threshold was changed, and
no parameter was retuned and re-tested on this set.** Note the failures are *not* in the same
category for both tiers, which on 4 images per category is what a small-sample effect looks like —
one image changing its outcome moves the category median. This is stated as a caveat, **not** as an
excuse: the threshold is missed on the terms it was written.

**Threshold 2 — "any category with median CER > 10 % must be analysed separately":**
**no category exceeded 10 %** on either tier (`categories_over_10pct_must_be_analysed: []`), so no
separate analysis is owed on that rule.

## What the errors actually are: the recogniser drops spaces

The single most common error is **not** a mis-recognised character. PP-OCRv5 **omits spaces** at
CJK/Latin boundaries and around punctuation, and does so inconsistently. Measured from the raw blocks
(`results/OCR-03-bench/block_inspection.txt`):

```
syn_mx_01  block 0: '本地服务local-ocr'   x[15,251] y[22,51]   truth: '本地服务 local-ocr …'
           block 1: '端口port 8116'        x[11,212] y[64,99]   truth: '端口 port 8116'
syn_en_03  block 0: 'VRAM margin:2048 MiB'                     truth: 'VRAM margin: 2048 MiB'
           block 1: 'Device: cpu'        <- this one KEEPS its space
           block 2: 'Tier:mobile'        <- this one loses it
```

The detector found each line correctly — the missing space is **inside a single recognised block**,
so it is a recogniser behaviour, not a detection or reconstruction problem.

**A hypothesis of mine was tested and disproved.** I first suspected the benchmark was at fault for
joining blocks with a newline and so discarding a horizontal gap. Measured
(`b3_reconstruction.py`, `results/OCR-03-bench/reconstruction_effect.json`): **the artifact is exactly
0.000 pp on both tiers**, because whitespace normalisation already collapses `a\nb` and `a b` to the
same string, so the two scorings are identical by construction. The hypothesis was wrong and is
recorded as wrong rather than quietly dropped.

**Consequence:** this behaviour costs CER precisely in `mixed` and in punctuation-heavy material,
which is where the threshold failures above are.

## Resource behaviour

* **VRAM: no measurable effect.** nvidia-smi free before **5391 MiB**, minimum during **5334 MiB**,
  after **5383 MiB** (68 samples). The spread is ordinary background noise on this machine and there
  is no mechanism for a VRAM cost: the Paddle build has **no CUDA support at all**
  (`is_compiled_with_cuda() == False`) and the device is `cpu`. Reported as measured, not smoothed.
* **Commit memory** free before 7283 MB, minimum during **6824 MB** — comfortably above the 4 GB
  stop condition.
* **RSS**: mobile 289 MB idle → 450 MB peak; server 450 MB → 650 MB peak.

## What this does and does not support

* It supports: a per-category comparison of the two PP-OCRv5 tiers on 34 images, and the
  pre-registered selection of **mobile**.
* It does **not** support any general OCR accuracy claim. 34 images, mostly rendered rather than
  photographed, with rotation/perspective/dewarping excluded by scope.
* It must not be combined with the P56 or `heldout_v1` question results, and it is not an accuracy
  figure for the `local-ocr` service as deployed (that is B5's functional check).
