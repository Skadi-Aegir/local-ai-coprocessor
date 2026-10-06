import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
QH = R / "benchmarks/qa_hardening/results/QH-01"
REF = "I could not find enough evidence in the local corpus to answer this."


def cls(r):
    if r.get("error"):
        return "ERROR"
    if r.get("refused"):
        return "REFUSED"
    a = (r.get("answer") or "").strip()
    return "REFUSED" if a.lower() == REF.lower() else "ANSWERED"


def table(label, recs, ids):
    print(f"\n=== {label} ===")
    print(f"  {'id':<5}{'finish_reason':<15}{'completion':<12}{'trunc':<8}"
          f"{'verdict':<10}{'cites':<7}{'verify':<12}{'e2e_s'}")
    any_len = []
    for i, r in enumerate(recs):
        u = r.get("usage") or {}
        fr = r.get("finish_reason")
        tr = r.get("answer_truncated")
        ver = sorted({s for s in (r.get("citation_verification") or [])})
        print(f"  {ids[i]:<5}{str(fr):<15}{str(u.get('completion_tokens')):<12}{str(tr):<8}"
              f"{cls(r):<10}{len(r.get('citations') or []):<7}{','.join(ver) or '-':<12}"
              f"{r.get('e2e_s')}")
        if fr == "length":
            any_len.append(ids[i])
    return any_len


p56 = json.load(open(QH / "p56.json", encoding="utf-8"))
held = json.load(open(QH / "heldout_v1.json", encoding="utf-8"))
len56 = table("P56 regression (QH-01)", p56, [f"q{i:02d}" for i in range(1, 11)])
lenh = table("heldout_v1 regression (QH-01)", held, [f"h{i:02d}" for i in range(1, 13)])

print(f"\nremaining finish_reason == 'length': p56 {len56 or 'none'}; heldout {lenh or 'none'}")

# verdict comparison against the previous round
prev_h = json.load(open(R / "benchmarks/phaseA/heldout_v1/results/run1.json", encoding="utf-8"))
print("\n=== heldout verdicts vs run1 ===")
diffs = []
for i, (a, b) in enumerate(zip(prev_h, held), 1):
    ca, cb = cls(a), cls(b)
    if ca != cb:
        diffs.append((f"h{i:02d}", ca, cb))
    print(f"  h{i:02d}: run1={ca:<10} QH-01={cb:<10} {'same' if ca == cb else 'CHANGED'}")
print("  verdict changes:", diffs or "NONE")
print("  expected pattern (h01-h09 ANSWERED, h10-h12 REFUSED):",
      [cls(r) for r in held] == ["ANSWERED"] * 9 + ["REFUSED"] * 3)

# citation + truncation integrity
bad = [(i, s) for i, r in enumerate(held, 1)
       for s in (r.get("citation_verification") or []) if s != "verified"]
print("  non-verified citations anywhere:", bad or "none")
missing = [f"h{i:02d}" for i, r in enumerate(held, 1)
           if "finish_reason" not in r or "answer_truncated" not in r]
print("  records missing the two fields:", missing or "none")
missing56 = [f"q{i:02d}" for i, r in enumerate(p56, 1)
             if "finish_reason" not in r or "answer_truncated" not in r]
print("  p56 records missing the two fields:", missing56 or "none")

lat = [r.get("e2e_s") or 0 for r in held]
print(f"\n  heldout latency: min={min(lat)} max={max(lat)}")
prev_lat = [r.get("e2e_s") or 0 for r in prev_h]
print(f"  run1    latency: min={min(prev_lat)} max={max(prev_lat)}")
pk = [r.get("pack") or {} for r in held]
print("  prompt_tokens vs new budget 7616:",
      [( (r.get('pack') or {}).get('prompt_tokens'), (r.get('pack') or {}).get('budget_tokens'))
       for r in held[:3]])
print("  max completion tokens used:", max((r.get("usage") or {}).get("completion_tokens") or 0
                                          for r in held + p56))
