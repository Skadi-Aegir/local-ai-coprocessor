import json
from pathlib import Path

R = Path(__file__).resolve().parents[2]
rep = json.load(open(R / "benchmarks/ocr/results/OCR-07-index/a_index_build.json",
                    encoding="utf-8"))
man = json.load(open(R / "vector/qa/current/manifest.json", encoding="utf-8"))
prev = json.load(open(R / "vector/qa/current.prev/manifest.json", encoding="utf-8"))
v = rep["manifest"]["validation"]
bv = rep.get("build_vram") or man.get("build_vram") or {}
esm = rep.get("embedding_service_memory") or {}
ret = esm.get("retention") or {}

print("=== B7 acceptance ===")
print(f"  validation_passed          : {v['validation_passed']}")
print(f"  n_answer_key_leaks         : {v['n_answer_key_leaks']}")
print(f"  question_sets_checked      : {json.dumps(v['question_sets_checked'])}")
print(f"  build_vram.margin_held     : {bv.get('margin_held')}  free_min_mb={bv.get('free_min_mb')}")
print(f"  retention.restart_required : {ret.get('restart_required')}  retained={ret.get('retained_mb')}")
print(f"  corpus_files_not_at_commit : {man.get('corpus_files_not_at_commit')} (empty={not man.get('corpus_files_not_at_commit')})")
print(f"  index_version              : {man['index_version']}")
print(f"  n_files / n_chunks         : {man['n_files']} / {man['n_chunks']}")
print(f"  hash_scheme / ivs          : {man.get('hash_scheme')} / {man.get('index_version_scheme')}")
print()
print("=== rollback target (must be 488f54d755224c5b) ===")
print(f"  current.prev index_version : {prev['index_version']}")
print(f"  MATCHES 488f54d755224c5b   : {prev['index_version'] == '488f54d755224c5b'}")
print()
print("=== were the NEW OCR files classified correctly? ===")
inc = set(man["included_files"])
exc = man["excluded_files"]
new_files = [
    "docs/models/local-ocr.md", "services/ocr/server.py", "config/local-ocr.yaml",
    "scripts/start-local-ocr.ps1", "scripts/stop-local-ocr.ps1", "scripts/check-local-ocr.ps1",
    "services/ocr/test_ocr_server.py", "benchmarks/ocr/REPORT.md",
    "benchmarks/ocr/bench.json", "benchmarks/ocr/make_fixtures.py",
    "benchmarks/ocr/questions.md", "benchmarks/ocr-results.md",
    "benchmarks/ocr/fixtures/manifest.json",
]
for f in new_files:
    if f in inc:
        print(f"  INCLUDED   {f}")
    elif f in exc:
        print(f"  excluded   {f}   [{exc[f]}]")
    else:
        print(f"  ABSENT     {f}   (not scanned at all)")
print()
print("=== the test file MUST be excluded (task requirement) ===")
t = "services/ocr/test_ocr_server.py"
print(f"  {t}: {'EXCLUDED correctly' if t in exc else '*** INCLUDED - WRONG ***'}"
      f" {exc.get(t, '')}")
print()
print("=== OCR benchmark/answer material must not be in the corpus ===")
leaks = [f for f in inc if f.startswith("benchmarks/")]
print(f"  corpus files under benchmarks/: {leaks or 'NONE - correct'}")
