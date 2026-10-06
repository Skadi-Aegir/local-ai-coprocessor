"""Show why a model failed a specific repair fixture."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
fixture = sys.argv[2] if len(sys.argv) > 2 else None
d = json.loads(p.read_text(encoding="utf-8"))

for row in d["fixtures"]:
    if fixture and row["fixture"] != fixture:
        continue
    print("=" * 72)
    print(f"fixture: {row['fixture']}   pass@1={row.get('pass_at_1')}  pass@2={row.get('pass_at_2')}")
    for a in row.get("attempts", []):
        print(f"\n--- attempt {a.get('attempt')} ---")
        print(f"  json_ok      : {a.get('json_ok')}")
        print(f"  applied      : {a.get('applied')}")
        print(f"  apply_msgs   : {a.get('apply_msgs')}")
        print(f"  files_touched: {a.get('files_touched')}")
        print(f"  tests_pass   : {a.get('tests_pass')}")
        print(f"  bug_fixed    : {a.get('bug_fixed')}")
        print(f"  success      : {a.get('success')}")
        if a.get("tests_tail"):
            print(f"  tests_tail   : {a['tests_tail']}")
        if a.get("demo_tail"):
            print(f"  demo_tail    : {a['demo_tail']}")
        if a.get("explanation"):
            print(f"  explanation  : {a['explanation'][:300]}")
        if a.get("raw") and not a.get("success"):
            print(f"  raw (first 700):\n{a['raw'][:700]}")
