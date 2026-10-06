"""Compare two code-review runs: which planted defects each model found."""
import json
import sys
from pathlib import Path

a_path, b_path = sys.argv[1], sys.argv[2]
a = json.loads(Path(a_path).read_text(encoding="utf-8"))
b = json.loads(Path(b_path).read_text(encoding="utf-8"))

print(f"A = {a['model']}   recall {a['recall_pct']}%  fp {a['false_positives']}")
print(f"B = {b['model']}   recall {b['recall_pct']}%  fp {b['false_positives']}\n")

fa = {r["fixture"]: r for r in a["fixtures"]}
fb = {r["fixture"]: r for r in b["fixtures"]}

for fx in sorted(set(fa) | set(fb)):
    ra, rb = fa.get(fx, {}), fb.get(fx, {})
    print("=" * 74)
    print(f"{fx}")
    ids = [x["id"] for x in ra.get("found", [])] + [x["id"] for x in ra.get("missed", [])]
    found_a = {x["id"] for x in ra.get("found", [])}
    found_b = {x["id"] for x in rb.get("found", [])}
    allids = sorted(found_a | found_b | {x["id"] for x in ra.get("missed", [])}
                    | {x["id"] for x in rb.get("missed", [])})
    for i in allids:
        ma = "FOUND" if i in found_a else "miss "
        mb = "FOUND" if i in found_b else "miss "
        mark = "  <-- differs" if (i in found_a) != (i in found_b) else ""
        print(f"  {i:26} A[{ma}]  B[{mb}]{mark}")
    fpa = [x["id"] for x in ra.get("false_positives", [])]
    fpb = [x["id"] for x in rb.get("false_positives", [])]
    if fpa or fpb:
        print(f"  false positives  A={fpa}  B={fpb}")
    print(f"  issues reported  A={ra.get('n_issues_reported')}  B={rb.get('n_issues_reported')}")
    print(f"  top severity     A={ra.get('top_severity')}  B={rb.get('top_severity')}")
    print(f"  needs_cloud      A={ra.get('needs_cloud_review')}  B={rb.get('needs_cloud_review')}")
