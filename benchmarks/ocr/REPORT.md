# Phase 6 (local OCR) — construction report

**Construction:** DSH ｜ **Command and acceptance:** Claude ｜ **Repo:** `D:\MultiAgent` (master)
**Baseline:** `5989730` ｜ **This round's commits:** `d29d2dc` (Part A), `b637590` (B1),
`0d8b175` (B2+B3), and the B4+B5 and B6 commits that follow.

**No PASS verdict is given here.** Each acceptance condition is listed with satisfied / not satisfied
and an evidence path. The judgement is the commander's.

---

## 1. Acceptance conditions, one by one

### Part A

| # | condition | satisfied | evidence |
|---|---|---|---|
| A-1 | the truncation fix has offline case coverage, all passing | **YES** | `cli/test_qa_path.py` **89 checks, 0 failures** (was 68; 21 new). New cases pin the 512 cap, the derived 7616 budget, that `max_tokens=512` is actually **sent**, `"stop"` not flagged, `"length"` flagged **and the text still returned**, the citation-gate refusal still carrying both fields, the model-refusal and error paths, and the HTTP + `/health` surface |
| A-2 | in the real-machine regression every question's `finish_reason` is recorded, and every `"length"` is marked | **YES** | `benchmarks/qa_hardening/results/QH-01/{p56,heldout_v1}.json` — all 22 records carry `finish_reason` and `answer_truncated`; the single `"length"` (p56 q05) is flagged and the CLI printed the truncation line |
| A-3 | verdicts match the previous round | **YES** | heldout **12 of 12 unchanged** (h01–h09 answered, h10–h12 refused); p56 7 answered / 3 refused / 0 errors, same as prior |
| A-4 | all citations `verified` | **YES** | no citation in either regression run is other than `verified` |
| A-5 | the margin held | **YES** | 185 samples at 200 ms, **free_min 5123 MiB** |

### Part B

| # | condition | satisfied | evidence |
|---|---|---|---|
| B-1 | scripts start and stop the service correctly | **YES** | `start-local-ocr.ps1` -> `READY` in ~8 s (load 6.789 s); `stop-local-ocr.ps1` -> `STOPPED`, port released, process gone, pid file removed. **A parse error in the stop script was found and fixed** — see §4 |
| B-2 | `/health`, `/v1/ocr` and the structured errors work as specified; loopback only | **YES** | `check-local-ocr.ps1` -> **`RESULT: HEALTHY`**, `bind : 127.0.0.1`, `bind : OK loopback only`; `/v1/ocr` returned 1 block in 0.242 s conf 0.9815; non-image -> **415**, non-object JSON -> **400** |
| B-3 | the OCR process's VRAM change is within noise | **YES** | benchmark: free 5391 before / **5334 min during** / 5383 after MiB (68 samples). Coexistence: **used delta +48 MiB** across the whole test. There is no mechanism for a VRAM cost — the Paddle build has **no CUDA support** (`is_compiled_with_cuda() == False`) |
| B-4 | coexistence holds the margin | **YES** | 8111+8113+8116 running, QA smoke + 20 OCR requests, **free_min 5341 MiB**, margin 2048 held |
| B-5 | memory and commit memory both recorded | **YES** | RSS mobile 289 → 450 MB, server 450 → 650 MB (bench); 332 MB resident (service); commit free min **6824 MB**, 7664 MB after. Above the 4 GB stop condition throughout |
| B-6 | accuracy reported per the pre-registered thresholds and selection rule | **YES** | `benchmarks/ocr-results.md`. Selection rule -> **`mobile`**. **Two thresholds MISSED and reported as missed** (§3) |
| B-7 | the fixture set's frozen hash is consistent | **YES** | manifest SHA256 `7ebb5cc4f3b3c7ab5667ca20130f7eb53f9e757f1d184b697fc8200f29d3ffa6`, verified **unchanged** after the benchmark and again before committing |
| B-8 | all offline tests pass | **YES** | six suites, all green: 89 / 59 / audit-selftest / 38 / 42 / **103** |
| B-9 | models are neither in Git nor on C: | **YES** | of the **61** files under `models/ocr`, **zero** are committable (exhaustive `git check-ignore`, not sampled); only `.gitkeep` is tracked. `C:\Users\<user>\.paddlex` holds **0 files**, unchanged from before |
| B-10 | `docs/models/local-ocr.md` is complete | **YES** | source chain, four model identities with sizes and SHA256, inference backend incl. the oneDNN caveat, environment, smoke runtime, usage contract, 7 limitations |
| B-11 | the B7 build passes acceptance | **YES** | see §5 |

**No stop condition in the task's §6 was triggered.** In particular: no VRAM sample below 2048 MiB,
no OOM, commit memory never near 4 GB, no system-level change needed, no model or cache on C:, no
model file staged, no antivirus interference, all offline tests green, no verdict regression, frozen
hash intact.

---

## 2. Per-step results

| step | result | evidence |
|---|---|---|
| B0 feasibility | **proceed** | `results/OCR-00-env/env.json`, `pip_dry_run.json` |
| B1 environment + models | 4 models, 186.7 MB, nothing on C: | `results/OCR-01-env/models.json`, `pip_freeze.txt` |
| B1 smoke | both tiers transcribed exactly | `results/OCR-01-env/b1_smoke.json` |
| B2 freeze | 34 images, 9 categories, hash recorded | `benchmarks/ocr/fixtures/manifest.json`, `questions.md` |
| B3 benchmark | mobile selected | `results/OCR-03-bench/bench.json`, `benchmarks/ocr-results.md` |
| B4 service | built + 103 offline checks | `services/ocr/server.py`, `config/local-ocr.yaml`, `scripts/*-local-ocr.ps1` |
| B5 functional | 3 images over HTTP vs in-process | `results/OCR-05-validate/b5_validation.json` |
| B5 coexistence | margin held | same file |
| B7 index rebuild | switched | `results/OCR-07-index/a_index_build.json` |

---

## 3. B3 in brief, including what was MISSED

Full tables in `benchmarks/ocr-results.md`.

| tier | median CER | mean CER | line recall | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| **mobile (selected)** | **0.0000** | **0.0104** | 0.799 | 0.404 s | **6.294 s** |
| server | 0.0000 | 0.0195 | 0.746 | 0.868 s | 12.484 s |

**Selection rule, applied as written:** both medians are **exactly 0.0000**, a gap of **0.00 pp**,
which is inside the pre-registered 1 pp band, so the **latency branch** fired and **`mobile` won**.

**Thresholds: two were MISSED.** "Clear text (english/mixed/ui) median CER <= 2 %":
**mobile/`mixed` 3.17 % FAIL** and **server/`ui` 5.33 % FAIL**. Reported as failures; **no sample,
truth string, rule or threshold was changed and nothing was retuned**. No category exceeded 10 %.

**The dominant error is the recogniser dropping spaces**, verified from the raw blocks: the space is
missing *inside a single recognised block* (`'本地服务local-ocr'` spans x[15,251], one line), so the
detector is right and the recogniser omits it, inconsistently (`'Device: cpu'` keeps its space while
`'Tier:mobile'` loses it).

**A hypothesis of mine was tested and disproved.** I suspected the benchmark itself was at fault for
joining blocks with a newline and discarding a gap. Measured: the artifact is **exactly 0.000 pp**,
because whitespace normalisation already maps `a\nb` and `a b` to the same string. Recorded as a
wrong hypothesis rather than dropped. Evidence: `results/OCR-03-bench/reconstruction_effect.json`.

---

## 4. Deviations, failures and my own mistakes

**1. oneDNN breaks prediction on this build — found, diagnosed, worked around, and documented.** With
the default configuration both tiers *loaded fine and failed at predict*:
`NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>] (onednn_instruction.cc:118)`.
This is a oneDNN/PIR incompatibility in `paddlepaddle 3.3.1`, **not** a bad model or a failed
download — the same files succeed once oneDNN is off. `enable_mkldnn=False` and `FLAGS_use_mkldnn=0`
both work; `FLAGS_enable_pir_api=0` alone does **not**. First failing run preserved as
`results/OCR-01-env/b1_smoke.onednn_default_failed.json`.

**2. `stop-local-ocr.ps1` had a PowerShell parse error, so the service did NOT stop on the first
attempt.** The guard I added to refuse killing a recycled PID used `"$target:"`, and PowerShell read
the colon as a drive qualifier: *"Variable reference is not valid. ':' was not followed by a valid
variable name character."* Fixed to `${target}`. **The service was untouched by the failed attempt**
— the guard is what tripped the parser, and the retry stopped it cleanly (port released, process
gone, pid file removed). First failure preserved in `logs/ocr-b5-stop.txt`.

**3. The benchmark's own prediction for one image was wrong, and B5 caught it.** For
`real_nvidia_smi` the stored B3 prediction contains a stray space (`memory .free`) while the B5 HTTP
call returned `memory.free`, which **matches the frozen truth exactly**. So 2 of the 3 comparison
images are byte-identical and the third differs in one character **because the B3 run was the
inaccurate one**, not the service. The stored B3 record was **not** edited — it is the measurement of
record, and this note is how the discrepancy is disclosed.

**4. That exposed a real property: the recogniser is NOT run-to-run deterministic.** The same
weights, the same image and the same settings produced `memory .free` in B3 and `memory.free` in B5.
The variation is at whitespace level, but it means **a single-image CER carries roughly ±1 percentage
point of run-to-run noise**, which is large next to the 2 % threshold. **This weakens the threshold
result and is stated rather than buried**; a future round should measure the same image several times
before treating any category median as stable.

**5. `real_nvidia_smi` is a snapshot of live GPU state**, so the memory numbers inside it are from the
moment of capture and will not match a later run. This makes it unrepeatable as a command-capture
benchmark item. It is a fixture-design limitation, not a service defect.

**6. My own earlier reference was wrong:** I wrote `benchmarks/ocr/questions.md` as the
pre-registration file before it existed. It now exists and is the actual pre-registration.

**No rollback was needed** — no index rollback (`--switch-only`), no code revert.

---

## 5. B7 — index rebuild

See `results/OCR-07-index/a_index_build.json`. The new OCR documents and code are now corpus material;
the build had to classify the new files correctly by the existing exclusion rules, in particular
excluding `test_`-prefixed tests and the answer-key/benchmark material. Result: `validation_passed`,
`n_answer_key_leaks 0`, both frozen question sets checked, margin held during the build,
`restart_required false`, `corpus_files_not_at_commit` empty, and the switch succeeded with the
rollback target becoming **`488f54d755224c5b`**. File and chunk counts are reported in the commit
message as the authoritative record.

---

## 6. Resource summary

| stage | minimum free VRAM | note |
|---|---:|---|
| A2 regression | **5123 MiB** | 185 samples @200 ms |
| B3 benchmark | **5334 MiB** | 68 samples; OCR on CPU |
| B5 coexistence | **5341 MiB** | 8111+8113+8116 with 20 OCR requests |
| B7 build | reported in the build report | margin preflight + sampling |

Margin 2048 MiB held at every stage. **No OOM.** Commit memory minimum **6824 MB**, well above the
4 GB stop condition. RSS: OCR service 332 MB resident; mobile peak 450 MB; server peak 650 MB.

---

## 7. Boundaries

Not done, as scoped: Phase 7, C (multi-hop), reranker integration, the P3-6 reranker cache release,
P2-7, P2-8, the cross-process ledger (P1-10), any Gateway, PDF text-layer extraction, table
recognition, GPU OCR.

Held: no model added to the QA path or downloaded outside `models/ocr`; no NVIDIA driver, CUDA,
pagefile, security-software or Qwen-Image change; **no antivirus setting or exclusion touched**;
loopback only; the 2048 MiB margin and every default budget unchanged; every frozen set untouched
(`p56` and `heldout_v1` were opened read-only, the OCR set was not modified after freezing);
no historical result overwritten or deleted; subprocess output always to files, never pipes;
`agentvoice/` neither read, written nor staged; all commits by explicit path.
