"""Summarise a benchmark run JSON: routing misses, handoff gaps, timing stats."""
import json
import sys
from pathlib import Path
from statistics import mean

path = Path(sys.argv[1] if len(sys.argv) > 1 else
            str(Path(__file__).resolve().parents[2] / r"benchmarks\local-fast\results\q4km_8k.json"))
d = json.loads(path.read_text(encoding="utf-8"))
t = d["tests"]

print("=" * 70)
print(f"run: {d['tag']}   params: {d['params']}")
print(f"thinking leaks: {d.get('thinking_leaks_total')}")
print("=" * 70)

# ---------------- A
if "A" in t:
    a = t["A"]
    g = a["grade"]
    print(f"\nTEST A  zh summary   ttft={a['ttft_s']:.3f}s  total={a['total_s']}s  gen={a['gen_tps']} t/s")
    print(f"  bullets            : {g['bullet_points']} (want 5)")
    print(f"  facts covered      : {g['facts_hit']}/{g['facts_total']}")
    for k, v in g["facts_detail"].items():
        if not v:
            print(f"     MISSED: {k}")
    print(f"  hallucination marks: {g['hallucination_markers']}")
    print(f"  numbers not in src : {g['numbers_not_in_source']}")
    print(f"  output chars       : {g['chars']}")

# ---------------- B
if "B" in t:
    b = t["B"]
    print(f"\nTEST B  json extraction  parse={b['parse_rate']}%  schema={b['schema_rate']}%  ({b['schema_ok']}/{b['runs']})")
    print(f"  temp={b['temperature']} grammar={b['grammar']} json_schema={b['json_schema']}")
    print(f"  mean ttft          : {b['mean_ttft_s']}s   mean gen: {b['mean_gen_tps']} t/s")
    if b["field_errors"]:
        print(f"  field errors       : {b['field_errors']}")
    bad = [x for x in b["rows"] if not x["schema_ok"]]
    for x in bad[:5]:
        print(f"   #{x['i']} gold={x['gold']}")
        print(f"        got ={x['parsed']}")
        print(f"        probs={x['problems']}")

# ---------------- C
if "C" in t:
    c = t["C"]
    print(f"\nTEST C  routing  acc={c['accuracy']}% ({c['correct']}/{c['runs']})  parse={c['parse_rate']}%")
    miss = [x for x in c["rows"] if not x["correct"]]
    print(f"  misses: {len(miss)}")
    for x in miss:
        print(f"   task     : {x['task']}")
        print(f"   expected : {x['expected']}")
        print(f"   got      : {x['got']}   conf={x['confidence']}")
        print(f"   reason   : {(x['reason'] or '')[:100]}")
        print()

# ---------------- D
if "D" in t:
    dd = t["D"]
    print(f"\nTEST D  traceback  ttft={dd['ttft_s']:.3f}s  total={dd['total_s']}s  gen={dd['gen_tps']} t/s")
    print(f"  grade: {dd['grade']}")
    print("  --- output ---")
    print("\n".join("   " + ln for ln in dd["output"].splitlines()[:14]))

# ---------------- E
if "E" in t:
    print("\nTEST E  code")
    for row in t["E"]["rows"]:
        st = f"exec={row['exec_pass']}" if row["exec_pass"] is not None else "structural"
        print(f"  {row['name']:<24} {st:<16} gen={row['gen_tps']} t/s  total={row['total_s']}s")
        if row.get("exec_pass") is False:
            print(f"      detail: {row['exec_detail'][-260:]}")
        if row.get("structural"):
            for s in row["structural"]:
                print(f"      {'OK ' if s['pass'] else 'MISS'} {s['check']}")

# ---------------- F
if "F" in t:
    f = t["F"]
    print(f"\nTEST F  handoff  sections={f['sections_ok']}/{f['sections_total']}  "
          f"ratio={f['compression_ratio']}  ttft={f['ttft_s']:.3f}s")
    for k, v in f["sections_present"].items():
        print(f"   {'OK  ' if v else 'MISS'} {k}")
    print(f"  anchors hit: {f['anchors_hit']}")
    print(f"  prompt_tok={f['prompt_tokens']} completion_tok={f['completion_tokens']} out_chars={f['output_chars']}")

# ---------------- latency summary
print("\n" + "=" * 70)
print("LATENCY / THROUGHPUT SUMMARY")
print("=" * 70)
ttfts = []
for k in "ABDEF":
    if k in t and t[k].get("ttft_s"):
        ttfts.append((k, t[k]["ttft_s"], t[k].get("total_s")))
for k, ttft, tot in ttfts:
    print(f"  {k}: ttft={ttft:.3f}s  total={tot}s")
if "B" in t:
    print(f"  B: mean ttft over {t['B']['runs']} runs = {t['B']['mean_ttft_s']}s")
gens = [t[k]["gen_tps"] for k in t if t[k].get("gen_tps")]
if gens:
    print(f"  gen t/s across tests: {gens}")
