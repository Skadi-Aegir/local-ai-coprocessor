"""Show the generated code and execution failure for a specific Test E case."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
name = sys.argv[2]
d = json.loads(p.read_text(encoding="utf-8"))
for row in d["tests"]["E"]["rows"]:
    if row["name"] != name:
        continue
    print(f"case: {row['name']}  kind={row['kind']}  exec_pass={row.get('exec_pass')}")
    print(f"detail: {row.get('exec_detail')}")
    print("\n--- generated code ---")
    print(row["code"])
    print("\n--- raw model output (first 1500) ---")
    print(row["output"][:1500])
