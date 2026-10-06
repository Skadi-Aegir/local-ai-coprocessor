"""P57-C0 - compare the 8115 re-verification against the reference B run."""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
ref = json.load(open(R / "benchmarks/phaseAB/results/AB-05-b/b_e2e.json", encoding="utf-8"))
new = json.load(open(R / "benchmarks/p57c0/results/b_reverify.json", encoding="utf-8"))


def outcome(d, key):
    s = d.get("scenarios", {}).get(key) or {}
    b = s.get("body") or {}
    return {"http": s.get("http"), "outcome": b.get("outcome"),
            "kind": (b.get("error") or {}).get("kind"), "elapsed_s": s.get("elapsed_s")}


print(f"{'scenario':<26}{'reference':<30}{'P57-C0':<30}consistent")
keys = ["health", "normal_answer", "answer_absent", "index_override_rejected",
        "stale_index", "service_unavailable", "question_too_long"]
allok = True
for k in keys:
    a, b = outcome(ref, k), outcome(new, k)
    same = (a["http"] == b["http"] and a["outcome"] == b["outcome"] and a["kind"] == b["kind"])
    allok = allok and same
    fa = f"{a['http']} {a['outcome'] or a['kind'] or ''}"
    fb = f"{b['http']} {b['outcome'] or b['kind'] or ''}"
    print(f"{k:<26}{fa:<30}{fb:<30}{'yes' if same else 'NO'}")

# index version must be the current one
print("\nindex version in the re-verification:",
      (new.get("scenarios", {}).get("health", {}).get("body") or {}).get("index_version"))
print("index version in the reference       :",
      (ref.get("scenarios", {}).get("health", {}).get("body") or {}).get("index_version"))

print("\nlimits_verified (P57-C0):", json.dumps(new.get("limits_verified")))
print("limits_verified (reference):", json.dumps(ref.get("limits_verified")))

va = ref.get("vram_overall", {}).get("min_across_scenarios")
vb = new.get("vram_overall", {}).get("min_across_scenarios")
print(f"\nmin free VRAM across scenarios: reference {va} MiB, P57-C0 {vb} MiB, "
      f"margin 2048 held: {vb is not None and vb >= 2048}")
print(f"concurrency codes: reference {ref['scenarios']['concurrency']['http_codes']}, "
      f"P57-C0 {new['scenarios']['concurrency']['http_codes']}")
print(f"\nOVERALL response consistency: {'CONSISTENT' if allok else 'DIFFERS'}")
