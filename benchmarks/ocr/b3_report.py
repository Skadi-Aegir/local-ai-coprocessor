import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[2]
d = json.load(open(R / "benchmarks/ocr/results/OCR-03-bench/bench.json", encoding="utf-8"))

print("manifest:", d["manifest_sha256"][:16], "images:", d["n_images"])
print("threads:", d["threads_requested"], "cpu_count:", d["os_cpu_count"],
      "onednn:", d["onednn"])
print(f"nvidia free: before {d['nvidia_free_before_mb']}  min-during "
      f"{d['nvidia_free_min_during_mb']}  after {d['nvidia_free_after_mb']} MiB "
      f"({d['nvidia_free_samples']} samples)")
print(f"commit free: before {d['commit_free_before_mb']}  min {d['commit_free_min_mb']} MB")

for tier in ("mobile", "server"):
    t = d["tiers"][tier]
    o = t["overall"]
    print(f"\n===== {tier} =====")
    print(f"  load {t['load_s']}s   rss idle {t['rss_idle_mb']} MB -> loaded "
          f"{t['rss_after_load_mb']} MB   peak {t['rss_peak_mb']} MB")
    print(f"  overall: n={o['n']}  CER median {o['cer_median']:.4f}  mean {o['cer_mean']:.4f}  "
          f"line recall {o['line_recall_mean']:.3f}  p50 {o['seconds_p50']}s  "
          f"p95 {o['seconds_p95']}s  errors {o['errors']}")
    print(f"  {'category':<15}{'n':>3}{'cer_med':>10}{'cer_mean':>10}{'recall':>9}"
          f"{'p50':>8}{'p95':>8}")
    for c, s in sorted(t["per_category"].items()):
        flag = ""
        if s["cer_median"] is not None and s["cer_median"] > 0.10:
            flag = "  <-- >10%"
        print(f"  {c:<15}{s['n']:>3}{s['cer_median']:>10.4f}{s['cer_mean']:>10.4f}"
              f"{s['line_recall_mean']:>9.3f}{s['seconds_p50']:>8}{s['seconds_p95']:>8}{flag}")

print("\n===== PRE-REGISTERED thresholds =====")
clear = ("english", "mixed", "ui")
print("  clear-text categories (english/mixed/ui), median CER must be <= 2%:")
for tier in ("mobile", "server"):
    for c in clear:
        s = d["tiers"][tier]["per_category"].get(c)
        if s:
            ok = s["cer_median"] is not None and s["cer_median"] <= 0.02
            print(f"    {tier:<7} {c:<9} median {s['cer_median']:.4f}  "
                  f"{'PASS' if ok else 'FAIL'}")
print("  failures recorded by the bench:", d["thresholds"].get("clear_text_failures", "none"))
print("  categories over 10%:",
      d["thresholds"].get("categories_over_10pct_must_be_analysed", "none"))

print("\n===== selection rule =====")
print(" ", json.dumps(d["selection"], ensure_ascii=False, indent=2))

# did the two tiers produce different text at all?
print("\n===== do the tiers disagree on any image? =====")
mo = {r["id"]: r for r in d["tiers"]["mobile"]["images"]}
se = {r["id"]: r for r in d["tiers"]["server"]["images"]}
diff = [i for i in mo if mo[i]["pred"] != se[i]["pred"]]
print(f"  {len(diff)} of {len(mo)} images differ between tiers: {diff}")

print("\n===== worst 8 images (mobile) =====")
for r in sorted(d["tiers"]["mobile"]["images"], key=lambda x: -x["cer"])[:8]:
    print(f"  {r['id']:<22} {r['category']:<14} cer={r['cer']:.4f} recall={r['line_recall']:.2f}")
    print(f"      truth: {r['truth'][:95]!r}")
    print(f"      pred : {r['pred'][:95]!r}")
