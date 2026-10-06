# Phase 6 OCR — frozen sample set and PRE-REGISTERED rules

**Everything on this page was written BEFORE any OCR measurement was taken.** Nothing here was
adjusted after seeing a result. The sample set, the truth strings, the scoring rules, the selection
rule and the thresholds are all fixed at this point.

## 1. The frozen set

| item | value |
|---|---|
| set id | `ocr_fixtures_v1` |
| **manifest SHA256** | `7ebb5cc4f3b3c7ab5667ca20130f7eb53f9e757f1d184b697fc8200f29d3ffa6` |
| manifest | `benchmarks/ocr/fixtures/manifest.json` |
| images | **34** — 28 synthetic + 6 real |
| synthetic generator | `benchmarks/ocr/make_fixtures.py` (committed) |
| real capture + render | `benchmarks/ocr/make_real_fixtures.py` (committed) |
| raw captured text | `benchmarks/ocr/fixtures/capture/*.txt` |

**Frozen means frozen.** B3 reads the manifest and never writes it. Changing any image or any truth
string means a **new set under a new name** — not an edit to this one. If the set has to change, the
previous results stay and the new set is measured separately.

### Categories and counts

| category | count | what it is |
|---|---:|---|
| `chinese` | 4 | Chinese-only text |
| `english` | 4 | English-only text |
| `mixed` | 4 | Chinese and English on one image |
| `code` | 4 | terminal / code style: dark background, monospace, prompt glyphs |
| `small` | 4 | 12 px text |
| `lowcontrast` | 4 | grey-on-grey, deliberately hard |
| `ui` | 4 | normal UI size, light background |
| `real/terminal` | 3 | real command output |
| `real/editor` | 3 | real project files |

Requirements met and checked in the manifest: at least 4 per category (synthetic), at least 24
synthetic, at least 6 real. `manifest["requirements"]["met"] == true`.

**Fonts:** rendered with fonts that ship with Windows (`consola`, `consolab`, `segoeui`, `segoeuib`,
`msyh`, `msyhbd`). **No font file is committed** — only the font *name* and size are recorded.

**Ground truth is exact, not transcribed.** The same strings that are drawn are written into the
manifest, so there is no human transcription step that could put an error inside the truth. For the
six real captures, the content is genuine project material gathered by running the command or reading
the file, and the drawn text is derived from those same bytes.

**Personal-information check.** Every real capture is scanned for the user name, `X:\Users\` profile
paths, email addresses, API-key-shaped strings, and browser/chat product names, before it is accepted.
All six passed. The result is recorded per capture in the manifest. Two candidate capture kinds were
rejected by design rather than by the checker: the default `nvidia-smi` process list (its paths carry
the user name) and anything from a browser or chat window.

## 2. PRE-REGISTERED scoring rules

* **CER** = Levenshtein distance between the recognised text and the truth, divided by the number of
  truth characters.
* **Normalisation before comparing:** collapse every run of whitespace to a single space, then strip
  leading and trailing whitespace. **Nothing else** — no case folding, no punctuation equivalence
  (Chinese and English punctuation are NOT treated as interchangeable), no character-class
  substitutions.
* **Line recall** = the fraction of truth lines that appear **complete** inside the recognised text
  after the same normalisation.
* Per-image CER, line recall and latency are recorded; per category and overall we report the
  **median and mean CER**, and latency **p50 and p95**.

## 3. PRE-REGISTERED selection rule (fixed before measuring)

> **The tier with the lower overall CER wins. If the two tiers' overall CER are within 1 percentage
> point of each other, the tier with the lower p95 latency wins.**

This rule is not to be changed after seeing the results.

## 4. PRE-REGISTERED thresholds

* **Synthetic clear text at normal size and high contrast** (the `english`, `mixed` and `ui`
  categories, which are 20 px and above on a light background): **median CER must be <= 2 %.**
* **Any category whose median CER exceeds 10 %** must be **analysed separately in the report**, with
  a stated cause.
* If a threshold is missed, it is **reported as missed**. The samples and truth are **not** adjusted,
  and parameters are **not** retuned and re-tested on this same set.

## 5. What this set can and cannot support

* It is **34 images**. It supports per-category counts and medians. It supports **no** general claim
  about OCR accuracy on arbitrary material.
* The 28 synthetic images are rendered, not photographed. They test rendering-to-text, not camera
  capture, and they deliberately exclude rotation, perspective and page curvature (the three
  preprocessing stages are disabled by scope).
* The 6 real captures are genuine project output but are few, and they are rendered rather than
  screen-grabbed from a live window, so they do not exercise subpixel anti-aliasing.
* No accuracy figure from this set may be presented as a general OCR quality claim, and none may be
  combined with the P56 or heldout_v1 question results.
