"""
test_grading.py - offline self-test for the benchmark grading logic.

Runs without a server, so the harness can be validated (and kept honest) even
when the GPU is busy. Verifies JSON extraction tolerance, schema strictness
(including the bool-vs-int trap), and that the fixed task set has the required
shape and variety.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import check_schema, extract_json  # noqa: E402
import tasks as T  # noqa: E402

FAILS: list[str] = []


def estimate_tokens_cjk(text: str) -> int:
    """
    Rough token estimate, used only where the real tokenizer is not available.

    Calibrated against `llama-tokenize` with this model's own vocab on the actual
    fixtures (mixed Chinese + markdown/code):

        fixture A : 1616 chars ->  944 tokens  (0.584 tok/char, 73% han)
        fixture F : 10622 chars -> 5609 tokens  (0.528 tok/char, 52% han)

    So Chinese runs about 0.6 tok/char (not 1.0 - the vocab has many multi-char
    CJK merges) and Latin/markdown about 0.45 tok/char. The authoritative check is
    test_grading's fixture assertions plus `llama-tokenize`; this is a sanity gate.
    """
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f")
    other = len(text) - han
    return int(han * 0.60 + other * 0.45)


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {label}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


# ---------------------------------------------------------------- extraction
print("--- JSON extraction tolerance ---")
GOOD = json.dumps(
    {
        "task_id": "T-200", "project": "x", "priority": 80,
        "requires_gpu": True, "worker": "local-smart", "status": "pending",
    }
)
extract_cases = [
    ("bare json", GOOD, True),
    ("fenced json", "```json\n" + GOOD + "\n```", True),
    ("prose wrapped", "Here is the result: " + GOOD + " hope that helps", True),
    ("garbage", "not json at all", False),
    ("empty", "", False),
]
for label, text, should_parse in extract_cases:
    parsed = extract_json(text)
    check(f"extract: {label}", (parsed is not None) == should_parse, repr(text[:32]))

# ---------------------------------------------------------------- schema
print("\n--- schema strictness ---")
gold = json.loads(GOOD)
ok, probs = check_schema(gold, gold)
check("schema: exact match passes", ok, str(probs))

trap = dict(gold); trap["requires_gpu"] = 1          # int is NOT bool
ok, probs = check_schema(trap, gold)
check("schema: bool->int rejected", not ok, str(probs))

trap2 = dict(gold); trap2["priority"] = "80"          # str is NOT int
ok, probs = check_schema(trap2, gold)
check("schema: int->str rejected", not ok, str(probs))

trap3 = dict(gold); trap3.pop("status")
ok, probs = check_schema(trap3, gold)
check("schema: missing field rejected", not ok, str(probs))

trap4 = dict(gold); trap4["extra"] = 1
ok, probs = check_schema(trap4, gold)
check("schema: extra field rejected", not ok, str(probs))

trap5 = dict(gold); trap5["project"] = "X"            # case-insensitive compare
ok, probs = check_schema(trap5, gold)
check("schema: string case-insensitive", ok, str(probs))

trap6 = dict(gold); trap6["priority"] = 81
ok, probs = check_schema(trap6, gold)
check("schema: wrong value rejected", not ok, str(probs))

# ---------------------------------------------------------------- task set
print("\n--- fixed task set ---")
cases = T.make_extraction_cases(20)
check("extraction: 20 cases", len(cases) == 20, str(len(cases)))
ids = {c["gold"]["task_id"] for c in cases}
check("extraction: all task_ids unique", len(ids) == 20, f"{len(ids)} unique")
texts = {c["text"] for c in cases}
check("extraction: all prompts unique", len(texts) == 20, f"{len(texts)} unique")
bools = {c["gold"]["requires_gpu"] for c in cases}
check("extraction: both bool values exercised", bools == {True, False}, str(bools))

check("routing: >= 20 cases", len(T.ROUTING_CASES) >= 20, str(len(T.ROUTING_CASES)))
routes_used = {e for _, e, _ in T.ROUTING_CASES}
check("routing: expected routes all valid", routes_used <= set(T.ROUTES), str(routes_used - set(T.ROUTES)))
check("routing: at least 5 distinct routes covered", len(routes_used) >= 5, str(sorted(routes_used)))

check("code: python+shell+sql covered",
      {c["kind"] for c in T.CODE_CASES} >= {"python", "shell", "sql"},
      str([c["kind"] for c in T.CODE_CASES]))
executable = [c["name"] for c in T.CODE_CASES if c["test"]]
check("code: at least 2 cases are auto-executed", len(executable) >= 2, str(executable))

# ---------------------------------------------------------------- memory model
# The Phase 2 VRAM budget published in quant-analysis.md / local-fast.md rests on
# arithmetic derived from the weights metadata. Pin it down here so a future edit
# cannot silently contradict the documented numbers.
print("\n--- memory model (qwen35 hybrid KV cache) ---")
_ARCH = {
    "n_layers": 32,
    "full_attention_interval": 4,
    "n_kv_heads": 4,
    "key_length": 256,
    "value_length": 256,
}
_n_full = _ARCH["n_layers"] // _ARCH["full_attention_interval"]
check("8 full-attention layers of 32", _n_full == 8, str(_n_full))

_kv_bytes_per_token = _n_full * _ARCH["n_kv_heads"] * 256 * 2 * 2  # width x (K,V) x f16
check("KV cache = 32768 B/token (32 KiB)", _kv_bytes_per_token == 32768, f"{_kv_bytes_per_token} B")

for ctx_k, expected_mb in ((4, 134.2), (8, 268.4), (16, 536.9), (32, 1073.7)):
    got = round(_kv_bytes_per_token * (ctx_k * 1024) / 1e6, 1)
    check(f"KV cache at {ctx_k}K = {expected_mb} MB", abs(got - expected_mb) < 0.1, f"got {got}")

# The hybrid saving comes ONLY from having 8 full-attention layers instead of 32,
# holding KV geometry constant. It is 4x, not 8x. A conventional 4B with 8 KV heads
# and 128-dim heads would differ for a second reason (different geometry), so it is
# not a like-for-like comparison - do not quote that as the saving.
check(
    "hybrid KV saving vs all-attention = 4x (not 8x)",
    round(_ARCH["n_layers"] / _n_full) == 4,
    f"{_ARCH['n_layers']}/{_n_full}",
)

# --- measured breakdown vs the analytical model ---------------------------------
# CPU sweep (llama-cli -ngl 0, fit_memory_breakdown host row). This validates the
# per-token KV model without needing the GPU, which matters because the GPU was
# unavailable for most of Phase 2.
print("\n--- KV model validated against measurement (CPU sweep) ---")
_CPU_SWEEP = {4096: 178, 8192: 306, 16384: 562, 32768: 1074}  # ctx -> context MiB
_ctxs = sorted(_CPU_SWEEP)
for a, b in zip(_ctxs, _ctxs[1:]):
    implied = (_CPU_SWEEP[b] - _CPU_SWEEP[a]) * 1048576 / (b - a)
    check(
        f"measured slope {a}->{b} equals {_kv_bytes_per_token} B/token",
        abs(implied - _kv_bytes_per_token) < 1.0,
        f"{implied:.0f} B/token",
    )

# least-squares fit must reproduce the sweep, confirming slope AND intercept
_n = len(_ctxs)
_sx = sum(_ctxs)
_sy = sum(_CPU_SWEEP[c] for c in _ctxs)
_sxx = sum(c * c for c in _ctxs)
_sxy = sum(c * _CPU_SWEEP[c] for c in _ctxs)
_slope = (_n * _sxy - _sx * _sy) / (_n * _sxx - _sx * _sx)
_intercept = (_sy - _slope * _sx) / _n
check("fitted slope = 32768 B/token", abs(_slope * 1048576 - _kv_bytes_per_token) < 1.0,
      f"{_slope * 1048576:.0f} B/token")
check("fitted fixed overhead = 50 MiB", abs(_intercept - 50.0) < 0.5, f"{_intercept:.1f} MiB")
for c in _ctxs:
    pred = _intercept + _slope * c
    check(f"model predicts ctx {c} within 1 MiB", abs(pred - _CPU_SWEEP[c]) < 1.0,
          f"pred {pred:.1f} vs measured {_CPU_SWEEP[c]}")

# --- GPU figures: 4-slot vs 1-slot, and the resolution -------------------------
# Two GPU measurements exist. The 4-slot one was the default and overallocated the
# context pool; the 1-slot one matches the CPU-validated model exactly. Both are
# pinned so the resolution stays visible and the retracted mechanism cannot return.
print("\n--- GPU context: 4-slot vs 1-slot (resolved) ---")
_GPU_4SLOT_CTX = 457    # MiB at ctx 8192, n_slots=4, kv_unified=true
_GPU_1SLOT_CTX = 306    # MiB at ctx 8192, n_slots=1, kv_unified=false
_predicted_at_8192 = _intercept + _slope * 8192

check("1-slot GPU context matches the CPU-validated model exactly",
      abs(_GPU_1SLOT_CTX - _predicted_at_8192) < 1.0,
      f"gpu {_GPU_1SLOT_CTX} vs model {_predicted_at_8192:.0f} MiB")
check("4-slot GPU context exceeded the model (pool overallocation)",
      _GPU_4SLOT_CTX > _predicted_at_8192,
      f"gpu {_GPU_4SLOT_CTX} vs model {_predicted_at_8192:.0f} MiB")
_overalloc = _GPU_4SLOT_CTX / _predicted_at_8192
check("multi-slot overallocation factor ~1.49x", abs(_overalloc - 1.49) < 0.05,
      f"{_overalloc:.2f}x")

# Guard the retracted claim: the pool is NOT simply 4 x context.
_four_slot_mib = _kv_bytes_per_token * 4 * 8192 / 1048576
check("excess is NOT 4 x context (retracted claim)", abs(_GPU_4SLOT_CTX - _four_slot_mib) > 100,
      f"4 x context would be {_four_slot_mib:.0f} MiB")

# Total GPU footprint with the deployed single-slot setting.
check("deployed GPU footprint ~2981 MiB (model+context+compute)",
      2603 + _GPU_1SLOT_CTX + 72 == 2981,
      f"{2603 + _GPU_1SLOT_CTX + 72} MiB")

# The deployed config must ask for a single slot: with -c 8192 and 4 slots each
# request would only get 2048 tokens, too small for the 4K/8K tasks.
import importlib.util as _ilu  # noqa: E402

_cfg_path = (Path(__file__).resolve().parents[2] / r"config\local-fast.yaml")
if _cfg_path.exists():
    _spec = _ilu.spec_from_file_location("lf_config", str(Path(__file__).resolve().parents[2] / r"scripts\lf_config.py"))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _cfg = _mod.load_config(_cfg_path)
    check("config runtime.parallel == 1",
          _mod.dig(_cfg, "runtime.parallel") == 1,
          str(_mod.dig(_cfg, "runtime.parallel")))
    check("config context == 8192",
          _mod.dig(_cfg, "runtime.context") == 8192,
          str(_mod.dig(_cfg, "runtime.context")))

check("handoff: 9 sections defined", len(T.HANDOFF_SECTIONS) == 9, str(len(T.HANDOFF_SECTIONS)))

# --- Test E executable graders: prove they accept a CORRECT answer -------------
# A grader that rejects a correct solution is worse than no grader, so each
# auto-executed case is checked against a known-good reference answer here.
print("\n--- executable graders accept correct answers ---")

_REF_SQL = (
    "SELECT project, COUNT(*) AS task_count, AVG(priority) AS avg_priority "
    "FROM tasks GROUP BY project HAVING COUNT(*) > 1 ORDER BY task_count DESC;"
)
_REF_PY = (
    "import csv\n"
    "def to_csv(records, path):\n"
    "    if not records:\n"
    "        open(path, 'w', encoding='utf-8').close()\n"
    "        return\n"
    "    with open(path, 'w', newline='', encoding='utf-8') as fh:\n"
    "        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))\n"
    "        w.writeheader()\n"
    "        w.writerows(records)\n"
)

from run_bench import run_python_check  # noqa: E402

for case in T.CODE_CASES:
    if not case["test"]:
        continue
    ref = _REF_SQL if case["kind"] == "sql" else _REF_PY
    ok, detail = run_python_check(ref, case["test"])
    check(f"grader accepts correct answer: {case['name']}", ok, detail[-200:] if not ok else "")

# and prove the grader actually rejects a wrong answer (no false passes)
_wrong_sql = "SELECT project, COUNT(*) AS task_count, AVG(priority) AS avg_priority FROM tasks GROUP BY project;"
_sql_case = next(c for c in T.CODE_CASES if c["kind"] == "sql")
ok_wrong, _ = run_python_check(_wrong_sql, _sql_case["test"])
check("grader rejects missing HAVING clause", not ok_wrong)

# fixtures exist and meet the size requirements in the spec
fix = HERE / "fixtures"
zh = (fix / "test_a_zh_summary.txt").read_text(encoding="utf-8")
han = sum(1 for ch in zh if "\u4e00" <= ch <= "\u9fff")
check("fixture A: 1000-2000 Chinese chars", 1000 <= han <= 2000, f"{han} han chars")

handoff = (fix / "test_f_handoff_context.md").read_text(encoding="utf-8")
est_tokens = estimate_tokens_cjk(handoff)
check(
    "fixture F: ~5000-8000 token context (estimated)",
    4500 <= est_tokens <= 9000,
    f"{len(handoff)} chars, ~{est_tokens} est. tokens",
)

# ---------------------------------------------------------------- verdict
print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): {FAILS}")
    raise SystemExit(1)
print("all grading self-tests passed")
raise SystemExit(0)
