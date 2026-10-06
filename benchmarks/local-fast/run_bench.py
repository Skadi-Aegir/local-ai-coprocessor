"""
run_bench.py - fixed benchmark suite for the local-fast worker (Phase 2).

Runs Tests A-F against a live OpenAI-compatible endpoint and writes raw JSON
results plus a readable summary. Grading is deliberately mechanical wherever
possible: Test B checks a strict schema, Test C checks a hand-labelled answer
key, and Test E actually executes the generated Python/SQL.

Usage:
  python run_bench.py --out ../results --tag q4km
  python run_bench.py --only B --repeat 20 --grammar
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import Client, check_schema, extract_json, mean, pct, thinking_leak  # noqa: E402
import tasks as T  # noqa: E402

FIX = HERE / "fixtures"


def _leak_fields(text: str) -> dict:
    """Attach thinking-leak diagnostics to any test row."""
    m = thinking_leak(text or "")
    return {"thinking_leak": m} if m else {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, lang: str | None = None) -> str:
    """Pull the first fenced code block (optionally matching a language)."""
    blocks = FENCE_RE.findall(text or "")
    if not blocks:
        return (text or "").strip()
    if lang:
        for tag, body in blocks:
            if tag.lower() == lang.lower():
                return body.strip()
    return blocks[0][1].strip()


def run_python_check(code: str, harness: str, timeout: int = 30) -> tuple[bool, str]:
    """Write generated code to a temp file and run the harness against it."""
    exe = sys.executable
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "gen.py"
        src.write_text(code, encoding="utf-8")
        h = Path(d) / "harness.py"
        h.write_text(harness, encoding="utf-8")
        try:
            p = subprocess.run(
                [exe, str(h), str(src)],
                capture_output=True, text=True, timeout=timeout, cwd=d,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        out = (p.stdout or "") + (p.stderr or "")
        return (p.returncode == 0 and "PASS" in (p.stdout or "")), out.strip()[-500:]


# --------------------------------------------------------------------------
# Test A - Chinese summary
# --------------------------------------------------------------------------

HALLUCINATION_MARKERS = [
    "Kafka", "Kubernetes", "k8s", "Docker", "Redis", "MongoDB", "gRPC",
    "微服务", "K8S", "TensorFlow", "PyTorch", "Kubernetes",
]

# facts that a faithful summary may reasonably mention
A_EXPECTED_FACTS = {
    "新旧架构": ["旧架构", "Redis", "队列", "三层", "优先级"],
    "延迟问题": ["延迟", "排队", "6.8", "200"],
    "pgvector": ["pgvector", "向量", "FAISS"],
    "缓存策略": ["缓存", "23%", "语义缓存"],
    "显存问题": ["显存", "KV", "16K", "9.2"],
    "日志量": ["日志", "240", "24", "采样"],
    "灰度状态": ["灰度", "0.15", "0.8", "60%"],
    "未解决问题": ["40 毫秒", "40毫秒", "租户", "调度"],
    "成本": ["4200", "5100", "成本", "21%"],
    "回滚": ["回滚", "90 秒", "90秒", "开关"],
}


def grade_summary(text: str, src: str) -> dict:
    points = [ln for ln in (text or "").splitlines() if re.match(r"^\s*(\d+[.、)]|[-*])\s*\S", ln)]
    facts_hit = {k: any(m in text for m in ms) for k, ms in A_EXPECTED_FACTS.items()}
    hallucinations = [m for m in HALLUCINATION_MARKERS if m in text]
    # numbers present in the summary but absent from the source are suspicious
    src_nums = set(re.findall(r"\d+(?:\.\d+)?", src))
    out_nums = set(re.findall(r"\d+(?:\.\d+)?", text or ""))
    invented = sorted(n for n in (out_nums - src_nums) if len(n) >= 2)
    return {
        "bullet_points": len(points),
        "bullet_count_ok": len(points) == 5,
        "facts_hit": sum(1 for v in facts_hit.values() if v),
        "facts_total": len(facts_hit),
        "facts_detail": facts_hit,
        "hallucination_markers": hallucinations,
        "numbers_not_in_source": invented[:10],
        "chars": len(text or ""),
    }


# --------------------------------------------------------------------------
# runners per test
# --------------------------------------------------------------------------

def test_a(c: Client, args) -> dict:
    src = (FIX / "test_a_zh_summary.txt").read_text(encoding="utf-8")
    r = c.chat(
        [
            {"role": "system", "content": "你是一个严谨的技术文档摘要助手。"},
            {
                "role": "user",
                "content": (
                    "请把下面的技术文本压缩为恰好 5 个要点。"
                    "只使用原文中出现的信息，绝对不要添加原文没有的内容。"
                    "每个要点用一行，以数字编号开头。\n\n" + src
                ),
            },
        ],
        max_tokens=700, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        enable_thinking=args.thinking,
    )
    g = grade_summary(r.text, src)
    return {
        "test": "A_zh_summary", "ok": r.ok, "error": r.error,
        "ttft_s": r.ttft_s, "total_s": round(r.total_s, 3),
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "gen_tps": round(r.gen_tps, 2), "prompt_tps": round(r.prompt_tps, 2),
        "output": r.text, "grade": g, **_leak_fields(r.text),
    }


def test_b(c: Client, args) -> dict:
    n = args.repeat or 10
    cases = T.make_extraction_cases(n)
    rows = []
    for i, case in enumerate(cases, 1):
        kw: dict = {}
        if args.grammar:
            kw["grammar"] = T.EXTRACTION_GRAMMAR
        if args.json_schema:
            kw["json_schema"] = T.EXTRACTION_SCHEMA
        r = c.chat(
            [
                {"role": "system", "content": T.EXTRACTION_SYSTEM},
                {"role": "user", "content": case["text"]},
            ],
            max_tokens=200, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            enable_thinking=args.thinking, **kw,
        )
        parsed = extract_json(r.text)
        schema_ok, problems = check_schema(parsed, case["gold"])
        rows.append({
            "i": i, "input": case["text"], "gold": case["gold"],
            "output": r.text, "parsed": parsed,
            "parse_ok": parsed is not None,
            "schema_ok": schema_ok, "problems": problems,
            "ttft_s": r.ttft_s, "total_s": round(r.total_s, 3),
            "gen_tps": round(r.gen_tps, 2), "error": r.error,
            **_leak_fields(r.text),
        })
    n_parse = sum(1 for x in rows if x["parse_ok"])
    n_schema = sum(1 for x in rows if x["schema_ok"])
    leaks = [x for x in rows if x.get("thinking_leak")]
    field_errors: dict[str, int] = {}
    for x in rows:
        for p in x["problems"]:
            key = p.split(":")[0]
            field_errors[key] = field_errors.get(key, 0) + 1
    return {
        "test": "B_json_extraction", "runs": n,
        "parse_ok": n_parse, "parse_rate": pct(n_parse, n),
        "schema_ok": n_schema, "schema_rate": pct(n_schema, n),
        "field_errors": field_errors,
        "thinking_leaks": len(leaks),
        "thinking_leak_rate": pct(len(leaks), n),
        "mean_ttft_s": mean([x["ttft_s"] for x in rows if x["ttft_s"]]),
        "mean_gen_tps": mean([x["gen_tps"] for x in rows if x["gen_tps"]]),
        "rows": rows,
        "grammar": bool(args.grammar), "json_schema": bool(args.json_schema),
        "temperature": args.temperature,
    }


def test_c(c: Client, args) -> dict:
    rows = []
    for desc, expect, why in T.ROUTING_CASES:
        kw: dict = {}
        if args.grammar:
            kw["grammar"] = T.ROUTING_GRAMMAR
        r = c.chat(
            [
                {"role": "system", "content": T.ROUTING_SYSTEM},
                {"role": "user", "content": desc},
            ],
            max_tokens=200, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            enable_thinking=args.thinking, **kw,
        )
        parsed = extract_json(r.text) or {}
        got = str(parsed.get("route", "")).strip()
        conf = parsed.get("confidence")
        rows.append({
            "task": desc, "expected": expect, "got": got, "why": why,
            "correct": got == expect,
            "confidence": conf,
            "reason": parsed.get("reason", ""),
            "parse_ok": parsed != {},
            "raw": r.text if parsed == {} else None,
            **_leak_fields(r.text),
        })
    n = len(rows)
    n_ok = sum(1 for x in rows if x["correct"])
    n_parse = sum(1 for x in rows if x["parse_ok"])
    leaks = [x for x in rows if x.get("thinking_leak")]
    return {
        "test": "C_routing", "runs": n,
        "correct": n_ok, "accuracy": pct(n_ok, n),
        "parse_ok": n_parse, "parse_rate": pct(n_parse, n),
        "thinking_leaks": len(leaks),
        "rows": rows,
    }


def _keyword_score(text: str, kws: list[str]) -> bool:
    low = (text or "").lower()
    return any(k.lower() in low for k in kws)


def test_d(c: Client, args) -> dict:
    r = c.chat(
        [
            {"role": "system", "content": T.TRACEBACK_SYSTEM},
            {"role": "user", "content": "请分析下面的 Python traceback：\n\n" + T.TRACEBACK},
        ],
        max_tokens=400, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        enable_thinking=args.thinking,
    )
    e = T.TRACEBACK_EXPECT
    return {
        "test": "D_traceback", "ok": r.ok, "error": r.error,
        "ttft_s": r.ttft_s, "total_s": round(r.total_s, 3),
        "gen_tps": round(r.gen_tps, 2),
        "output": r.text,
        "grade": {
            "exception_ok": _keyword_score(r.text, e["exception"]),
            "origin_file_ok": _keyword_score(r.text, e["origin_file"]),
            "origin_line_ok": _keyword_score(r.text, e["origin_line_any"]),
            "cause_ok": _keyword_score(r.text, e["keywords"]),
        },
        **_leak_fields(r.text),
    }


def test_e(c: Client, args) -> dict:
    rows = []
    for case in T.CODE_CASES:
        r = c.chat(
            [
                {"role": "system", "content": case["system"]},
                {"role": "user", "content": case["prompt"]},
            ],
            max_tokens=900, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            enable_thinking=args.thinking,
        )
        code = extract_code(r.text, case["extract"])
        entry: dict = {
            "name": case["name"], "kind": case["kind"], "ok": r.ok, "error": r.error,
            "ttft_s": r.ttft_s, "total_s": round(r.total_s, 3),
            "gen_tps": round(r.gen_tps, 2), "code": code, "output": r.text,
            "exec_pass": None, "exec_detail": "",
            **_leak_fields(r.text),
        }
        if case["kind"] == "python" and case["test"]:
            passed, detail = run_python_check(code, case["test"])
            entry["exec_pass"] = passed
            entry["exec_detail"] = detail
        elif case["kind"] == "sql":
            # wrap the SQL into the harness
            harness = case["test"].replace(
                "src = open(sys.argv[1], encoding='utf-8').read()",
                "src = open(sys.argv[1], encoding='utf-8').read()",
            )
            passed, detail = run_python_check(code, harness)
            entry["exec_pass"] = passed
            entry["exec_detail"] = detail
        else:
            # structural checks for shell / pytest module
            structural = []
            if case["kind"] == "shell":
                structural.append(("has_shebang_or_set", bool(re.search(r"(#!|set -e)", code))))
                structural.append(("loops_or_glob", bool(re.search(r"(\*\.jpeg|for\s+\w+\s+in)", code))))
                structural.append(("guards_existing", bool(re.search(r"(-e\s|\-f\s|test\s+-e|\[\[)", code))))
                structural.append(("does_not_overwrite", "mv -n" in code or "-e" in code or "-f" in code))
            if "pytest" in case["name"]:
                structural.append(("imports_pytest", "import pytest" in code))
                structural.append(("uses_raises", "pytest.raises" in code))
                structural.append(("has_tests", code.count("def test_") >= 4))
            entry["structural"] = [{"check": k, "pass": v} for k, v in structural]
        rows.append(entry)
    return {"test": "E_code", "rows": rows}


def test_f(c: Client, args) -> dict:
    ctx = (FIX / "test_f_handoff_context.md").read_text(encoding="utf-8")
    r = c.chat(
        [
            {"role": "system", "content": T.HANDOFF_SYSTEM},
            {"role": "user", "content": "请把下面的项目记录压缩成结构化 handoff：\n\n" + ctx},
        ],
        # 1400 was too low: the response hit the cap exactly (completion_tok == max_tokens)
        # and was truncated mid-generation, so the final section was never emitted. A
        # truncated answer must not be scored as a missing capability.
        max_tokens=3000, temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        enable_thinking=args.thinking,
    )
    text = r.text or ""
    truncated = r.completion_tokens >= 3000
    present = {s: bool(re.search(rf"^#{{1,4}}\s*{re.escape(s)}\b", text, re.M | re.I))
               for s in T.HANDOFF_SECTIONS}
    # concrete anchors that indicate real compression rather than vague prose
    anchors = ["ivt.py", "scheduler.py", "calibrate.py", "pipeline.yaml", "Parquet", "seed"]
    return {
        "test": "F_handoff_compression", "ok": r.ok, "error": r.error,
        "ttft_s": r.ttft_s, "total_s": round(r.total_s, 3),
        "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
        "gen_tps": round(r.gen_tps, 2), "prompt_tps": round(r.prompt_tps, 2),
        "input_chars": len(ctx),
        "output_chars": len(text),
        "compression_ratio": round(len(text) / max(len(ctx), 1), 3),
        "sections_present": present,
        "sections_ok": sum(1 for v in present.values() if v),
        "sections_total": len(T.HANDOFF_SECTIONS),
        "anchors_hit": [a for a in anchors if a.lower() in text.lower()],
        "truncated": truncated,
        "output": text,
        **_leak_fields(text),
    }


TESTS = {"A": test_a, "B": test_b, "C": test_c, "D": test_d, "E": test_e, "F": test_f}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--model", default="local-fast")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--only", default="ABCDEF", help="subset of ABCDEF")
    ap.add_argument("--repeat", type=int, default=0, help="override repetition count (Test B)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260401)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen thinking mode")
    ap.add_argument("--grammar", action="store_true", help="use GBNF grammar for structured tests")
    ap.add_argument("--json-schema", action="store_true", help="use response_format json_schema")
    args = ap.parse_args()

    c = Client(args.base_url)
    try:
        h = c.health()
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: server not reachable at {args.base_url}: {e}")
        return 2
    print(f"server health: {h}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results: dict = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model": args.model,
        "params": {
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "thinking": args.thinking, "grammar": args.grammar, "json_schema": args.json_schema,
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": {},
    }

    for key in args.only.upper():
        if key not in TESTS:
            print(f"skip unknown test {key}")
            continue
        print(f"--- Test {key} ---", flush=True)
        t0 = time.time()
        try:
            res = TESTS[key](c, args)
        except Exception as e:  # noqa: BLE001
            res = {"test": key, "fatal": f"{type(e).__name__}: {e}"}
        res["elapsed_s"] = round(time.time() - t0, 2)
        results["tests"][key] = res
        print(f"    done in {res['elapsed_s']}s", flush=True)

    results["elapsed_s"] = round(time.time() - started, 2)

    # global thinking-leak tally: recursive scan so a new test cannot forget to
    # report it. See common.thinking_leak for why this matters.
    def count_leaks(node) -> int:
        n = 0
        if isinstance(node, dict):
            if node.get("thinking_leak"):
                n += 1
            for v in node.values():
                n += count_leaks(v)
        elif isinstance(node, list):
            for v in node:
                n += count_leaks(v)
        return n

    results["thinking_leaks_total"] = count_leaks(results["tests"])
    results["thinking_enabled"] = bool(args.thinking)

    dest = outdir / f"{args.tag}.json"
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")

    # compact console summary
    print("\n=== SUMMARY ===")
    if results["thinking_leaks_total"]:
        print(
            f"!! THINKING LEAK: {results['thinking_leaks_total']} response(s) contain reasoning "
            f"markers while enable_thinking={args.thinking}. Quality scores below are unreliable "
            f"until this is fixed."
        )
    else:
        print(f"thinking mode: clean (enable_thinking={args.thinking}, 0 leaks)")
    for k, v in results["tests"].items():
        if k == "A":
            g = v.get("grade", {})
            print(f"A summary   bullets={g.get('bullet_points')} facts={g.get('facts_hit')}/{g.get('facts_total')} "
                  f"hallu={g.get('hallucination_markers')} ttft={v.get('ttft_s')} tps={v.get('gen_tps')}")
        elif k == "B":
            print(f"B json      parse={v.get('parse_rate')}% schema={v.get('schema_rate')}% "
                  f"({v.get('schema_ok')}/{v.get('runs')}) errors={v.get('field_errors')}")
        elif k == "C":
            print(f"C routing   acc={v.get('accuracy')}% ({v.get('correct')}/{v.get('runs')}) parse={v.get('parse_rate')}%")
        elif k == "D":
            print(f"D traceback {v.get('grade')}")
        elif k == "E":
            for row in v.get("rows", []):
                st = row.get("exec_pass")
                st = "exec=" + str(st) if st is not None else "struct"
                print(f"E code      {row['name']:<24} {st} tps={row.get('gen_tps')}")
        elif k == "F":
            print(f"F handoff   sections={v.get('sections_ok')}/{v.get('sections_total')} "
                  f"ratio={v.get('compression_ratio')} anchors={len(v.get('anchors_hit', []))} ttft={v.get('ttft_s')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
