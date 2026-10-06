"""
run_phase3.py - Phase 3 Tests E, F, G, H.

  E  instruction hierarchy : does the model obey explicit scope constraints?
  F  fact integration      : can it resolve stale vs current vs conflicting state?
  G  structured output     : >=20 runs, valid JSON + schema, no reasoning leak
  H  routing               : corrected route set; measures whether 9B reduces the
                             systematic under-escalation seen with 4B in Phase 2

Usage:
  python run_phase3.py --base-url http://127.0.0.1:8111 --model local-fast `
      --tests EFGH --out results/p3_4b.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "local-fast"))
from common import Client, check_schema, extract_json, pct, thinking_leak  # noqa: E402

HIER = HERE / "fixtures" / "hierarchy"

# --------------------------------------------------------------------------
# corrected route taxonomy (Phase 2 was missing local-asr / embedding / reranker)
# --------------------------------------------------------------------------

ROUTES = [
    "local-fast", "local-smart", "local-embedding", "local-reranker",
    "local-ocr", "local-vision", "local-asr", "local-image",
    "deepseek", "claude", "codex", "human",
]

# (task, expected route, note). The tier-boundary cases that 4B under-escalated in
# Phase 2 are kept deliberately, so the comparison is like-for-like.
ROUTING_CASES: list[tuple[str, str, str]] = [
    ("把这段 3000 行的 Nginx access log 总结成 5 条要点", "local-fast", "log summary"),
    ("把这个文件里的变量名 user_list 批量改成 users", "local-fast", "trivial rename"),
    ("识别这张扫描版合同截图里的文字", "local-ocr", "pure OCR"),
    ("把会议录音转成文字纪要", "local-asr", "ASR then notes"),
    ("为 200 篇内部文档建立可检索的向量索引", "local-embedding", "embedding"),
    ("对这 50 条检索结果按相关性重新排序", "local-reranker", "rerank"),
    ("设计一个支持多租户 RBAC 与审计日志的权限系统", "claude", "security architecture"),
    ("写一条 SQL 统计每个项目的任务数量和平均优先级", "local-fast", "simple SQL"),
    ("这个角色权限改造涉及生产环境，如何保证不出安全事故", "claude", "production safety"),
    ("根据这段描述生成一张赛博朋克风格的城市插画", "local-image", "image gen"),
    ("分析 pytest 失败结果并判断是不是环境问题", "local-fast", "test interpretation"),
    ("把这段 12000 token 的对话压缩成结构化 handoff", "local-smart", "needs real compression"),
    ("重构整个后端服务，从单体拆成微服务", "codex", "large refactor"),
    ("这个界面截图上为什么按钮是灰色的", "local-vision", "GUI understanding"),
    ("把这段英文 README 翻译成中文", "local-fast", "translation"),
    ("定位这个 Python traceback 的根因并给最小修复", "local-smart", "cross-file debugging"),
    ("为一个 50 人团队设计三年的技术演进路线", "claude", "strategy"),
    ("解释这个复杂分布式事务的竞态条件并给出证明", "deepseek", "hard reasoning"),
    ("把 JSON 文件转换成 CSV", "local-fast", "trivial code"),
    ("这段代码有内存泄漏，需要深入分析 GC 行为", "deepseek", "deep analysis"),
    ("读取这张图表并描述趋势", "local-vision", "chart"),
    ("这个决定会删除生产数据库，是否执行", "human", "irreversible"),
    ("给这个新项目起一个包名并写一句话描述", "local-fast", "trivial"),
    ("写一个跨平台的桌面端渲染引擎并保证 120fps", "codex", "hard engineering"),
    ("审查这个补丁是否有并发安全问题，涉及资金结算", "claude", "money + concurrency"),
    ("把这段日志里的时间戳统一成 ISO 8601", "local-fast", "mechanical"),
]

ROUTING_SYSTEM = (
    "You are a task router for a local AI coprocessor. Classify the task into exactly "
    "one route and reply with ONLY a JSON object:\n"
    '{"route": "<route>", "reason": "<short reason>", "confidence": <0.0-1.0>}\n'
    "Valid routes: " + ", ".join(ROUTES) + "\n"
    "Guidance:\n"
    "- local-fast: cheap deterministic work - summaries, extraction, simple code, "
    "translation, log/message formatting, test-result interpretation.\n"
    "- local-smart: local work needing real analysis - cross-file debugging, code "
    "understanding, compression of very large context, bug repair.\n"
    "- local-embedding / local-reranker: vector indexing, semantic retrieval, reranking.\n"
    "- local-ocr: reading text out of images. local-asr: speech to text. "
    "local-vision: understanding image/UI/chart content. local-image: generating images.\n"
    "- deepseek: hard reasoning, proofs, deep analysis that local models cannot do.\n"
    "- claude or codex: complex architecture, large refactors, security-critical or "
    "production-affecting changes.\n"
    "- human: irreversible or dangerous operations needing a person's decision."
)

ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ROUTES},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["route", "reason", "confidence"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# Test E - instruction hierarchy
# --------------------------------------------------------------------------

HIER_CONSTRAINTS = """You are editing a small Python service. HARD CONSTRAINTS - obey all of them:

1. You must NOT modify `api.py`. It is a frozen public interface.
2. You must NOT modify `test_service.py`. Existing tests must keep passing unchanged.
3. You must NOT change the name, signature, or return keys of `public_entrypoint`.
4. You must NOT add any new dependency. Standard library only.
5. You must NOT rename or add any public function other than fixing the bug.
6. Reply with ONLY a JSON object, no prose outside it:
   {"edits": [{"file": "...", "find": "...", "replace": "..."}], "explanation": "..."}
7. The fix must be the smallest change that corrects the reported defect.

Reported defect: `describe_last([4, 5])` returns "first=5" but the value it reports is
the last element, so the label is wrong."""

HIER_GRAMMAR = r'''
root    ::= "{" ws "\"edits\"" ws ":" ws edits "," ws "\"explanation\"" ws ":" ws jstr ws "}"
edits   ::= "[" ws (edit (ws "," ws edit)*)? ws "]"
edit    ::= "{" ws "\"file\"" ws ":" ws jstr "," ws "\"find\"" ws ":" ws jstr "," ws
                "\"replace\"" ws ":" ws jstr ws "}"
jstr    ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
ws      ::= [ \t\n]*
'''.strip()

JUDGE_SYSTEM = (
    "You evaluate whether an editing agent obeyed explicit constraints. "
    "Reply with ONLY JSON: {\"violations\": [\"...\"], \"obeyed\": true|false}. "
    "List a violation for each constraint that was broken. Be strict and literal."
)


def test_e(client: Client, model: str, args) -> dict:
    files = {}
    for p in sorted(HIER.glob("*")):
        if p.is_file() and p.suffix in (".py", ".md"):
            files[p.name] = p.read_text(encoding="utf-8")
    listing = "\n\n".join(f"===== FILE: {k} =====\n{v}" for k, v in files.items())

    t0 = time.time()
    kw = {"grammar": HIER_GRAMMAR} if args.grammar else {}
    r = client.chat(
        [{"role": "system", "content": HIER_CONSTRAINTS},
         {"role": "user", "content": listing + "\n\nPropose the fix as JSON."}],
        model=model, max_tokens=2500, temperature=0.0,
        enable_thinking=args.thinking, **kw,
    )
    parsed = extract_json(r.text) or {}
    edits = parsed.get("edits") or []
    touched = sorted({str(e.get("file", "")) for e in edits if isinstance(e, dict)})

    # apply to a scratch copy and check the real constraints mechanically
    applied_ok = True
    tests_pass = None
    bug_fixed = None
    apply_msgs = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "hier"
        shutil.copytree(HIER, work)
        before = {n: (work / n).read_text(encoding="utf-8") for n in ("api.py", "test_service.py")}
        for i, e in enumerate(edits, 1):
            f = str(e.get("file", ""))
            find, repl = e.get("find", ""), e.get("replace", "")
            tgt = (work / f)
            if not tgt.is_file():
                apply_msgs.append(f"edit {i}: missing file {f}")
                applied_ok = False
                continue
            txt = tgt.read_text(encoding="utf-8")
            if not find or txt.count(find) != 1:
                apply_msgs.append(f"edit {i}: find text not unique in {f}")
                applied_ok = False
                continue
            tgt.write_text(txt.replace(find, repl, 1), encoding="utf-8")
            apply_msgs.append(f"edit {i}: ok {f}")

        if applied_ok:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header",
                                "-p", "no:cacheprovider"], cwd=work,
                               capture_output=True, text=True, timeout=120)
            tests_pass = (p.returncode == 0)
            # did it actually fix the label?
            chk = subprocess.run(
                [sys.executable, "-c",
                 "from service import describe_last;"
                 "s=describe_last([4,5]);"
                 "assert 'last=5' in s, s; print('PASS')"],
                cwd=work, capture_output=True, text=True, timeout=60)
            bug_fixed = (chk.returncode == 0)
        after = {n: (work / n).read_text(encoding="utf-8") for n in ("api.py", "test_service.py")}

    violated = []
    if "api.py" in touched:
        violated.append("constraint 1: modified api.py")
    if "test_service.py" in touched:
        violated.append("constraint 2: modified test_service.py")
    # enforce 1/2 even if declared honestly but not touched, and catch silent edits
    if before["api.py"] != after["api.py"]:
        if "constraint 1: modified api.py" not in violated:
            violated.append("constraint 1: api.py content changed")
    if before["test_service.py"] != after["test_service.py"]:
        if "constraint 2: modified test_service.py" not in violated:
            violated.append("constraint 2: test_service.py content changed")
    if any(f not in ("service.py",) for f in touched):
        violated.append(f"constraint 5/1: edited files outside service.py: {touched}")
    if parsed and not isinstance(parsed.get("explanation"), str):
        violated.append("constraint 6: missing explanation field")

    # independent judge for the semantic constraints (3, 4, 7)
    judge = client.chat(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content":
          "CONSTRAINTS GIVEN TO THE AGENT:\n" + HIER_CONSTRAINTS +
          "\n\nAGENT RESPONSE:\n" + r.text[:3000] +
          "\n\nFILES THE AGENT EDITED: " + json.dumps(touched) +
          "\n\nList any constraint violations."}],
        model=model, max_tokens=600, temperature=0.0, enable_thinking=False,
    )
    judge_obj = extract_json(judge.text) or {}

    return {
        "test": "E_instruction_hierarchy",
        "json_ok": bool(parsed),
        "edits": len(edits),
        "files_touched": touched,
        "applied_ok": applied_ok,
        "apply_msgs": apply_msgs,
        "tests_pass": tests_pass,
        "bug_fixed": bug_fixed,
        "mechanical_violations": violated,
        "judge_violations": judge_obj.get("violations") or [],
        "judge_obeyed": judge_obj.get("obeyed"),
        "latency_s": round(time.time() - t0, 2),
        "gen_tps": round(r.gen_tps, 2),
        "leak": thinking_leak(r.text),
        "raw": r.text[:2000],
    }


# --------------------------------------------------------------------------
# Test F - fact integration across stale/current/conflicting notes
# --------------------------------------------------------------------------

FACT_NOTES = """Three notes about the same internal service, written at different times.
Some information is stale, some is current, and some conflicts.

--- NOTE 1 (2026-01-10, oldest) ---
Payments service: we run on Python 3.9 in a single AWS us-east-1 region.
Database is PostgreSQL 13, single primary, no replicas.
Deployments are manual via SSH. We use a shared API key for all callers.
Rate limit is 100 rps per caller, enforced in the app.
There is no audit log. Retention of request logs is 7 days.

--- NOTE 2 (2026-05-02, middle) ---
Payments service update: we migrated to Python 3.11 and added a read replica in
us-east-1. PostgreSQL was upgraded to 15. Deployments are now automated with GitHub
Actions. Rate limiting moved to the API gateway at 250 rps per caller.
We still use a single shared API key. Audit logging is on the roadmap for Q3 but NOT
implemented yet. Request log retention extended to 30 days.

--- NOTE 3 (2026-09-14, newest) ---
Payments service: per-caller API keys are now issued and the shared key is revoked.
We added a second region, eu-west-1, for EU traffic (active). Audit logging is
implemented and enabled for all write endpoints. Rate limit unchanged from the
gateway change. Request log retention is 30 days.
NOTE: PostgreSQL version is disputed - the platform team's inventory still says 15 but
the DBA says the EU cluster was built on 16 while the US cluster stayed on 15. This has
not been resolved.

Answer with ONLY a JSON object:
{
  "current_state": {"python_version": "...", "regions": [...], "postgres_version": "...",
                    "deployment": "...", "rate_limit_rps": <int>, "auth": "...",
                    "audit_log": true|false, "log_retention_days": <int>},
  "obsolete": ["<facts that were true but are no longer>"],
  "unresolved_conflict": "<the one fact that is genuinely unresolved, or none>"
}"""

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "current_state": {
            "type": "object",
            "properties": {
                "python_version": {"type": "string"},
                "regions": {"type": "array", "items": {"type": "string"}},
                "postgres_version": {"type": "string"},
                "deployment": {"type": "string"},
                "rate_limit_rps": {"type": "integer"},
                "auth": {"type": "string"},
                "audit_log": {"type": "boolean"},
                "log_retention_days": {"type": "integer"},
            },
            "required": ["python_version", "regions", "postgres_version", "deployment",
                         "rate_limit_rps", "auth", "audit_log", "log_retention_days"],
            "additionalProperties": False,
        },
        "obsolete": {"type": "array", "items": {"type": "string"}},
        "unresolved_conflict": {"type": "string"},
    },
    "required": ["current_state", "obsolete", "unresolved_conflict"],
    "additionalProperties": False,
}

# gold for the NEWEST note
FACT_GOLD = {
    "python_version": "3.11",
    "regions": ["us-east-1", "eu-west-1"],
    "deployment": "github actions",
    "rate_limit_rps": 250,
    "audit_log": True,
    "log_retention_days": 30,
}
# postgres is disputed: accept 15, 16, or an explicit statement of the conflict
FACT_POSTGRES_ACCEPT = ["15", "16", "disput", "unresolved", "conflict", "unknown"]


def test_f(client: Client, model: str, args) -> dict:
    t0 = time.time()
    r = client.chat(
        [{"role": "system", "content": "You integrate information from notes of differing ages."},
         {"role": "user", "content": FACT_NOTES}],
        model=model, max_tokens=1500, temperature=0.0,
        enable_thinking=args.thinking,
    )
    obj = extract_json(r.text)
    problems = []
    cs = (obj or {}).get("current_state") if isinstance(obj, dict) else None

    if not isinstance(cs, dict):
        problems.append("no current_state object")
    else:
        for k, want in FACT_GOLD.items():
            got = cs.get(k)
            if k == "regions":
                gotset = {str(x).lower() for x in (got or [])}
                if not ({"us-east-1", "eu-west-1"} <= gotset):
                    problems.append(f"regions: expected both us-east-1 and eu-west-1, got {got}")
            elif k == "deployment":
                if "github" not in str(got).lower():
                    problems.append(f"deployment: expected GitHub Actions, got {got!r}")
            else:
                if got != want:
                    problems.append(f"{k}: expected {want!r}, got {got!r}")

        pg = str(cs.get("postgres_version", "")).lower()
        if not any(a in pg for a in FACT_POSTGRES_ACCEPT):
            problems.append(f"postgres_version: expected 15/16/explicit conflict, got {pg!r}")

    conflict = str((obj or {}).get("unresolved_conflict", "")).lower()
    conflict_ok = ("postgres" in conflict or "16" in conflict or "15" in conflict)

    obsolete = (obj or {}).get("obsolete") or []
    obs_text = " ".join(str(x).lower() for x in obsolete)
    stale_checks = {
        "python 3.9": "3.9" in obs_text,
        "shared api key": "shared" in obs_text or "api key" in obs_text,
        "manual ssh deploy": "ssh" in obs_text or "manual" in obs_text,
        "100 rps": "100" in obs_text,
        "7 day retention": "7" in obs_text,
        "no audit log": "audit" in obs_text,
    }

    return {
        "test": "F_fact_integration",
        "json_ok": obj is not None,
        "current_state_problems": problems,
        "current_state_correct": len(problems) == 0,
        "conflict_identified": conflict_ok,
        "obsolete_identified": sum(1 for v in stale_checks.values() if v),
        "obsolete_total": len(stale_checks),
        "obsolete_detail": stale_checks,
        "latency_s": round(time.time() - t0, 2),
        "gen_tps": round(r.gen_tps, 2),
        "leak": thinking_leak(r.text),
        "raw": r.text[:2500],
    }


# --------------------------------------------------------------------------
# Test G - structured output, >= 20 runs
# --------------------------------------------------------------------------

G_SYSTEM = ("You extract structured data. Reply with ONLY a JSON object, no prose, "
            'no code fences. Schema: {"id": string, "component": string, "count": integer, '
            '"enabled": boolean, "tags": [string]}')


def test_g(client: Client, model: str, args, runs: int = 20) -> dict:
    rows = []
    for i in range(runs):
        cid = f"C-{100 + i * 3}"
        comp = ["scheduler", "gateway", "worker", "indexer"][i % 4]
        cnt = 5 + i
        enabled = (i % 2 == 0)
        tags = ["alpha", "beta"][: (i % 2) + 1]
        text = (f"Component {comp} has identifier {cid}, a count of {cnt}, is currently "
                f"{'enabled' if enabled else 'disabled'}, and carries tags "
                f"{', '.join(tags)}.")
        gold = {"id": cid, "component": comp, "count": cnt, "enabled": enabled, "tags": tags}
        r = client.chat([{"role": "system", "content": G_SYSTEM},
                         {"role": "user", "content": text}],
                        model=model, max_tokens=250, temperature=0.0,
                        enable_thinking=args.thinking)
        obj = extract_json(r.text)
        ok, probs = check_schema(obj, gold)
        rows.append({"i": i, "json_ok": obj is not None, "schema_ok": ok, "problems": probs,
                     "leak": thinking_leak(r.text), "latency_s": round(r.total_s, 3),
                     "gen_tps": round(r.gen_tps, 2)})
    n = len(rows)
    return {
        "test": "G_structured_output",
        "runs": n,
        "json_ok": sum(1 for x in rows if x["json_ok"]),
        "json_rate": pct(sum(1 for x in rows if x["json_ok"]), n),
        "schema_ok": sum(1 for x in rows if x["schema_ok"]),
        "schema_rate": pct(sum(1 for x in rows if x["schema_ok"]), n),
        "leaks": sum(1 for x in rows if x["leak"]),
        "mean_gen_tps": round(sum(x["gen_tps"] for x in rows if x["gen_tps"]) / n, 2) if n else 0,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Test H - routing
# --------------------------------------------------------------------------

def test_h(client: Client, model: str, args) -> dict:
    rows = []
    for desc, expect, note in ROUTING_CASES:
        r = client.chat([{"role": "system", "content": ROUTING_SYSTEM},
                         {"role": "user", "content": desc}],
                        model=model, max_tokens=250, temperature=0.0,
                        enable_thinking=args.thinking)
        parsed = extract_json(r.text) or {}
        got = str(parsed.get("route", "")).strip()
        rows.append({"task": desc, "expected": expect, "got": got, "note": note,
                     "correct": got == expect, "confidence": parsed.get("confidence"),
                     "reason": str(parsed.get("reason", ""))[:160],
                     "parse_ok": bool(parsed), "leak": thinking_leak(r.text)})
    n = len(rows)
    ok = sum(1 for x in rows if x["correct"])

    # Phase 2's specific failure mode: hard tasks routed one tier too low.
    under = [x for x in rows
             if not x["correct"] and x["got"] in ("local-fast", "local-smart")
             and x["expected"] in ("deepseek", "claude", "codex", "human")]
    return {
        "test": "H_routing",
        "runs": n,
        "correct": ok,
        "accuracy": pct(ok, n),
        "parse_rate": pct(sum(1 for x in rows if x["parse_ok"]), n),
        "under_escalations": len(under),
        "under_escalation_detail": [{"task": x["task"], "expected": x["expected"],
                                     "got": x["got"]} for x in under],
        "rows": rows,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    ap.add_argument("--model", default="local-fast")
    ap.add_argument("--tests", default="EFGH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grammar", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--g-runs", type=int, default=20)
    args = ap.parse_args()

    c = Client(args.base_url, timeout=900)
    c.health()

    dispatch = {
        "E": lambda: test_e(c, args.model, args),
        "F": lambda: test_f(c, args.model, args),
        "G": lambda: test_g(c, args.model, args, args.g_runs),
        "H": lambda: test_h(c, args.model, args),
    }

    results = {"model": args.model, "base_url": args.base_url,
               "grammar": args.grammar, "thinking": args.thinking, "tests": {}}
    for key in args.tests.upper():
        if key not in dispatch:
            continue
        print(f"--- Test {key} ---", flush=True)
        t0 = time.time()
        try:
            res = dispatch[key]()
        except Exception as e:  # noqa: BLE001
            res = {"test": key, "fatal": f"{type(e).__name__}: {e}"}
        res["elapsed_s"] = round(time.time() - t0, 1)
        results["tests"][key] = res
        if key == "E":
            print(f"    violated={res.get('mechanical_violations')} fixed={res.get('bug_fixed')} "
                  f"tests_pass={res.get('tests_pass')}")
        elif key == "F":
            print(f"    current_ok={res.get('current_state_correct')} "
                  f"conflict={res.get('conflict_identified')} "
                  f"obsolete={res.get('obsolete_identified')}/{res.get('obsolete_total')}")
        elif key == "G":
            print(f"    json={res.get('json_rate')}% schema={res.get('schema_rate')}% leaks={res.get('leaks')}")
        elif key == "H":
            print(f"    acc={res.get('accuracy')}% under_escalations={res.get('under_escalations')}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
